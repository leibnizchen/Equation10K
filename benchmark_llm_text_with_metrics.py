#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text-only benchmark for mathematical step grading.

This is the text-input counterpart of the original VLM benchmark.
The model never receives ground-truth correctness/error labels.

For each sample, each solution step's s[0] is used as the textual step input.
By default, the model sees:
  - the original question
  - all solution-step TEXT up to the current step
and is asked to grade ONLY the current step.

This preserves the reasoning context needed to identify propagated errors while
preventing label leakage. For a stricter independent-step ablation, use:
    --context-mode current-only

Supported backends:
  1) transformers-local   : ordinary text LLMs (Qwen/Llama/Mistral/etc.)
  2) qwen-vl-text-local   : reuse a Qwen-VL-style multimodal checkpoint in TEXT-ONLY mode
  3) openai-compatible    : OpenAI-compatible Chat Completions API

Examples:
    python benchmark_text_with_metrics.py --all \
        --backend transformers-local \
        --model /path/to/Qwen2.5-7B-Instruct

    python benchmark_text_with_metrics.py --type fraction \
        --backend qwen-vl-text-local \
        --model /home/shared/winston/model/Qwen/Qwen2___5-VL-7B-Instruct

    python benchmark_text_with_metrics.py --file steps/fraction.json \
        --backend openai-compatible \
        --model gpt-4.1-mini \
        --api-key "$OPENAI_API_KEY"

    # Strict s[0]-only input:
    python benchmark_text_with_metrics.py --type fraction \
        --context-mode current-only
"""


import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("benchmark_text")

ERROR_TYPE_NAMES = {
    "wrong_value": "Carelessly wrote down the wrong value.",
    "wrong_operator": "Wrong operation symbol",
    "wrong_order": "Incorrect parentheses or order of operations",
    "calculation_error": "Calculation error",
    "propagated": "Pre-order error propagation",
}
VALID_ERRORS = set(ERROR_TYPE_NAMES.values())

CORRECT_CLASS = "__CORRECT__"
MISSING_CLASS = "__MISSING__"
INVALID_CLASS = "__INVALID__"

SYSTEM_PROMPT = """You are an evaluator of mathematical solution steps.
Your task is to judge whether the CURRENT solution step is correct given the
problem and any preceding textual solution steps that are provided.

For an incorrect current step, use exactly one of these error labels:
- "Carelessly wrote down the wrong value."
- "Wrong operation symbol"
- "Incorrect parentheses or order of operations"
- "Calculation error"
- "Pre-order error propagation"

Definitions:
- "Carelessly wrote down the wrong value.": a value, number, or symbol was
  copied, transcribed, substituted, or written incorrectly.
- "Wrong operation symbol": an operation sign/operator was changed or written
  incorrectly.
- "Incorrect parentheses or order of operations": parentheses, precedence, or
  operation order is incorrect.
- "Calculation error": the intended operation is appropriate, but the
  arithmetic result is wrong.
- "Pre-order error propagation": an earlier displayed step already introduced
  an error, and the current transformation is locally valid but continues that
  previous error.

Rules:
- Grade ONLY the current step requested by the user.
- Do not invent or infer any hidden correctness/error labels.
- If the current step introduces a new independent error, use the concrete
  error type rather than "Pre-order error propagation".
- A correct step must have "error_type": null.
- Preserve the current step text exactly in "step_text".
- Return JSON only, without Markdown fences or explanations.

Required JSON structure:
{
  "step_text": "the current mathematical expression",
  "is_correct": true,
  "error_type": null
}
"""


def build_user_prompt(
    sample: dict[str, Any],
    step_index: int,
    context_mode: str,
) -> str:
    """Build text-only inference input without exposing GT labels.

    IMPORTANT: only step[0] text is read from the steps list here.
    step[1] and step[2] are never read in this function.
    """
    question = str(sample.get("question", "")).strip()
    steps = sample.get("steps", [])
    current_text = str(steps[step_index][0])

    if context_mode == "current-only":
        return (
            "Grade the following CURRENT solution step.\n\n"
            f"CURRENT STEP:\n{current_text}"
        )

    if context_mode == "question-current":
        return (
            "Grade the CURRENT solution step for the following problem.\n\n"
            f"PROBLEM:\n{question}\n\n"
            f"CURRENT STEP:\n{current_text}"
        )

    # Default: prefix context. We use only step text s[0], never labels.
    prefix_texts = [str(steps[i][0]) for i in range(step_index)]
    if prefix_texts:
        previous = "\n".join(
            f"Step {i + 1}: {text}" for i, text in enumerate(prefix_texts)
        )
    else:
        previous = "(none)"

    return (
        "Grade ONLY the CURRENT solution step. Earlier solution-step text is "
        "provided only as reasoning context.\n\n"
        f"PROBLEM:\n{question}\n\n"
        f"PREVIOUS STEPS:\n{previous}\n\n"
        f"CURRENT STEP {step_index + 1}:\n{current_text}"
    )


class ModelRunner:
    """Text-only local model or OpenAI-compatible API model."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = args.backend
        self.model_name = args.model
        self.max_new_tokens = args.max_new_tokens

        if self.backend == "transformers-local":
            self._init_transformers(args)
        elif self.backend == "qwen-vl-text-local":
            self._init_qwen_vl_text(args)
        else:
            self._init_api(args)

    def _common_model_kwargs(self, args: argparse.Namespace) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        # Newer Transformers prefers dtype; older versions may still use torch_dtype.
        if args.dtype != "auto":
            import torch
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            kwargs["torch_dtype"] = dtype_map[args.dtype]
        else:
            kwargs["torch_dtype"] = "auto"

        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        return kwargs

    def _init_transformers(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=args.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=args.trust_remote_code,
            **self._common_model_kwargs(args),
        )
        self.model.eval()

    def _init_qwen_vl_text(self, args: argparse.Namespace) -> None:
        """Load the same Qwen-VL-family checkpoint but perform text-only inference."""
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=args.trust_remote_code,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            trust_remote_code=args.trust_remote_code,
            **self._common_model_kwargs(args),
        )
        self.model.eval()

    def _init_api(self, args: argparse.Namespace) -> None:
        from openai import OpenAI

        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        base_url = args.api_base or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError(
                "OpenAI-compatible backend requires --api-key or OPENAI_API_KEY."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def generate(self, user_prompt: str) -> str:
        if self.backend == "transformers-local":
            return self._generate_transformers(user_prompt)
        if self.backend == "qwen-vl-text-local":
            return self._generate_qwen_vl_text(user_prompt)
        return self._generate_api(user_prompt)

    def _generate_transformers(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nAnswer:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.pad_token_id is None
                    else self.tokenizer.pad_token_id
                ),
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_qwen_vl_text(self, user_prompt: str) -> str:
        # No image placeholder and no images= argument: this is genuinely text-only.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_api(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=self.max_new_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty response.")
        return content.strip()


def load_label_file(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    """Load label JSON while retaining its top-level structure."""
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "question_id" in raw:
        records = [raw]
    elif isinstance(raw, dict):
        records = None
        for key in ("data", "samples", "items", "records"):
            if isinstance(raw.get(key), list):
                records = raw[key]
                break
        if records is None:
            raise ValueError(f"Unsupported JSON structure: {path}")
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")

    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Every sample must be a JSON object: {path}")
    return raw, records


def restore_structure(original: Any, predictions: list[dict[str, Any]]) -> Any:
    if isinstance(original, list):
        return predictions
    if isinstance(original, dict) and "question_id" in original:
        return predictions[0] if predictions else {}

    result = dict(original)
    for key in ("data", "samples", "items", "records"):
        if isinstance(original.get(key), list):
            result[key] = predictions
            return result
    return predictions


def extract_json_value(text: str) -> Any:
    """Extract the first valid JSON object or array from model output."""
    cleaned = text.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        cleaned = fence.group(1).strip()

    decoder = json.JSONDecoder()
    for position, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object or array found in model output.")


def normalize_error(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "correct"}:
        return None
    if text in ERROR_TYPE_NAMES:
        return ERROR_TYPE_NAMES[text]
    for code, label in ERROR_TYPE_NAMES.items():
        if text.lower() in {code.lower(), label.lower()}:
            return label
    return text


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "yes", "1"}:
            return True
        if lowered in {"false", "incorrect", "wrong", "no", "0"}:
            return False
    raise ValueError(f"Cannot parse correctness value: {value!r}")


def parse_single_prediction(text: str, expected_step_text: str) -> list[Any]:
    """Parse one current-step grading result as [text, bool, error]."""
    value = extract_json_value(text)

    # Harmless wrappers accepted for robustness.
    if isinstance(value, dict) and isinstance(value.get("steps"), list):
        if not value["steps"]:
            raise ValueError("Model returned an empty steps list.")
        value = value["steps"][0]
    elif isinstance(value, list):
        if not value:
            raise ValueError("Model returned an empty list.")
        # Could be [text, bool, error] or [{...}]
        if isinstance(value[0], (dict, list, tuple)):
            value = value[0]

    if isinstance(value, dict):
        step_text = value.get(
            "step_text",
            value.get("text", value.get("expression", value.get("step", expected_step_text))),
        )
        correct_value = value.get("is_correct", value.get("correct"))
        error_value = value.get("error_type", value.get("error"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        if len(value) >= 3:
            step_text, correct_value, error_value = value[:3]
        else:
            step_text = expected_step_text
            correct_value, error_value = value[:2]
    else:
        raise ValueError(f"Unrecognized model JSON: {value!r}")

    is_correct = parse_bool(correct_value)
    error_type = normalize_error(error_value)
    if is_correct:
        error_type = None
    elif error_type not in VALID_ERRORS:
        error_type = str(error_type) if error_type is not None else None

    # For metrics, the textual field is not used to determine correctness.
    # Still preserve model output when available; fall back to the exact s[0].
    if step_text is None or str(step_text).strip() == "":
        step_text = expected_step_text

    return [str(step_text), is_correct, error_type]


def predict_one_step(
    sample: dict[str, Any],
    step_index: int,
    runner: ModelRunner,
    context_mode: str,
    max_attempts: int,
) -> tuple[list[Any] | None, bool, str]:
    """Predict one step. GT labels are never accessed here."""
    expected_step_text = str(sample["steps"][step_index][0])
    user_prompt = build_user_prompt(sample, step_index, context_mode)

    last_output = ""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            last_output = runner.generate(user_prompt)
            pred = parse_single_prediction(last_output, expected_step_text)
            return pred, True, last_output
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "question_id=%s step=%d attempt=%d/%d parse failed: %s",
                sample.get("question_id", "?"),
                step_index + 1,
                attempt,
                max_attempts,
                exc,
            )

    LOGGER.error(
        "question_id=%s step=%d failed after %d attempts. Last error=%s | output=%r",
        sample.get("question_id", "?"),
        step_index + 1,
        max_attempts,
        last_error,
        last_output[:500],
    )
    return None, False, last_output


def predict_sample(
    sample: dict[str, Any],
    runner: ModelRunner,
    context_mode: str,
    max_attempts: int,
    save_raw_outputs: bool,
) -> tuple[dict[str, Any], bool]:
    """Grade all steps in one sample, one model call per s[0]."""
    gt_steps = sample.get("steps", [])
    if not isinstance(gt_steps, list):
        raise ValueError("Sample `steps` must be a list.")

    prediction = {key: value for key, value in sample.items() if key != "steps"}
    predicted_steps: list[list[Any]] = []
    raw_outputs: list[str] = []
    all_parsed = True

    for step_index, step in enumerate(gt_steps):
        if not isinstance(step, (list, tuple)) or len(step) < 1:
            raise ValueError(f"Invalid step format at position {step_index + 1}: {step!r}")

        pred, parsed, raw = predict_one_step(
            sample=sample,
            step_index=step_index,
            runner=runner,
            context_mode=context_mode,
            max_attempts=max_attempts,
        )
        all_parsed = all_parsed and parsed
        raw_outputs.append(raw)

        if pred is None:
            # Keep positional alignment for scoring. This is explicitly invalid.
            predicted_steps.append([str(step[0]), None, None])
        else:
            predicted_steps.append(pred)

    prediction["steps"] = predicted_steps
    if save_raw_outputs:
        prediction["_raw_model_outputs"] = raw_outputs
    return prediction, all_parsed


def _step_class(step: list[Any] | tuple[Any, ...] | None) -> str:
    if step is None or len(step) < 3:
        return MISSING_CLASS if step is None else INVALID_CLASS
    if step[1] is True:
        return CORRECT_CLASS
    if step[1] is False:
        error_type = normalize_error(step[2])
        return error_type if error_type in VALID_ERRORS else INVALID_CLASS
    return INVALID_CLASS


def _first_error_index(steps: list[list[Any]]) -> int | None:
    for index, step in enumerate(steps, start=1):
        if len(step) >= 2 and step[1] is False:
            return index
    return None


class MetricsAccumulator:
    """Compute all metrics only after text-only inference has completed."""

    def __init__(self) -> None:
        self.num_samples = 0
        self.num_steps = 0
        self.parse_successes = 0
        self.step_count_matches = 0
        self.correctness_hits = 0
        self.error_type_hits_on_gt_errors = 0
        self.gt_error_steps = 0
        self.joint_hits = 0
        self.exact_sample_hits = 0
        self.first_error_hits = 0
        self.missing_prediction_steps = 0
        self.extra_prediction_steps = 0
        self.gt_class_counts: Counter[str] = Counter()
        self.gt_type_hits: Counter[str] = Counter()
        self.confusion: dict[str, Counter[str]] = defaultdict(Counter)

    @staticmethod
    def _safe_div(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def update(
        self,
        ground_truth: dict[str, Any],
        prediction: dict[str, Any],
        parse_success: bool,
    ) -> None:
        gt_steps = ground_truth.get("steps", [])
        pred_steps = prediction.get("steps", [])

        if not isinstance(gt_steps, list):
            raise ValueError("Ground-truth `steps` must be a list.")
        if not isinstance(pred_steps, list):
            pred_steps = []

        self.num_samples += 1
        self.num_steps += len(gt_steps)
        self.parse_successes += int(parse_success)
        self.step_count_matches += int(len(pred_steps) == len(gt_steps))
        self.missing_prediction_steps += max(0, len(gt_steps) - len(pred_steps))
        self.extra_prediction_steps += max(0, len(pred_steps) - len(gt_steps))
        sample_exact = len(pred_steps) == len(gt_steps)

        for index, gt_step in enumerate(gt_steps):
            if not isinstance(gt_step, (list, tuple)) or len(gt_step) < 3:
                raise ValueError(f"Invalid GT step at position {index + 1}: {gt_step!r}")

            gt_step = list(gt_step)
            pred_step = pred_steps[index] if index < len(pred_steps) else None
            gt_correct = gt_step[1]
            gt_error = normalize_error(gt_step[2])
            pred_correct = (
                pred_step[1]
                if pred_step is not None and len(pred_step) >= 2
                else None
            )
            pred_error = (
                normalize_error(pred_step[2])
                if pred_step is not None and len(pred_step) >= 3
                else None
            )

            if pred_correct is gt_correct:
                self.correctness_hits += 1

            if gt_correct is False:
                self.gt_error_steps += 1
                if pred_correct is False and pred_error == gt_error:
                    self.error_type_hits_on_gt_errors += 1
                    if gt_error in VALID_ERRORS:
                        self.gt_type_hits[str(gt_error)] += 1

            joint_hit = pred_correct is gt_correct and pred_error == gt_error
            if joint_hit:
                self.joint_hits += 1
            else:
                sample_exact = False

            gt_class = _step_class(gt_step)
            pred_class = _step_class(pred_step)
            self.gt_class_counts[gt_class] += 1
            self.confusion[gt_class][pred_class] += 1

        if sample_exact:
            self.exact_sample_hits += 1

        gt_first_error = _first_error_index([list(step) for step in gt_steps])
        if len(pred_steps) < len(gt_steps):
            pred_first_error: int | None | str = MISSING_CLASS
        else:
            pred_first_error = _first_error_index(pred_steps)
        if pred_first_error == gt_first_error:
            self.first_error_hits += 1

    def merge(self, other: "MetricsAccumulator") -> None:
        for attr in (
            "num_samples",
            "num_steps",
            "parse_successes",
            "step_count_matches",
            "correctness_hits",
            "error_type_hits_on_gt_errors",
            "gt_error_steps",
            "joint_hits",
            "exact_sample_hits",
            "first_error_hits",
            "missing_prediction_steps",
            "extra_prediction_steps",
        ):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))

        self.gt_class_counts.update(other.gt_class_counts)
        self.gt_type_hits.update(other.gt_type_hits)
        for gt_class, counts in other.confusion.items():
            self.confusion[gt_class].update(counts)

    def to_dict(self) -> dict[str, Any]:
        per_error_type = {}
        for error_name in ERROR_TYPE_NAMES.values():
            count = self.gt_class_counts[error_name]
            hits = self.gt_type_hits[error_name]
            per_error_type[error_name] = {
                "count": count,
                "accuracy": self._safe_div(hits, count),
            }

        confusion_dict = {
            gt_class: dict(pred_counts)
            for gt_class, pred_counts in sorted(self.confusion.items())
        }
        return {
            "num_samples": self.num_samples,
            "num_steps": self.num_steps,
            "parse_success_rate": self._safe_div(self.parse_successes, self.num_samples),
            "step_count_match_rate": self._safe_div(self.step_count_matches, self.num_samples),
            "step_correctness_accuracy (SCA)": self._safe_div(
                self.correctness_hits, self.num_steps
            ),
            "error_type_accuracy_on_gt_error_steps (ETA)": self._safe_div(
                self.error_type_hits_on_gt_errors, self.gt_error_steps
            ),
            "joint_step_accuracy (JSA)": self._safe_div(self.joint_hits, self.num_steps),
            "sample_exact_match (SEM)": self._safe_div(
                self.exact_sample_hits, self.num_samples
            ),
            "first_error_index_accuracy (FEIA)": self._safe_div(
                self.first_error_hits, self.num_samples
            ),
            "ground_truth_error_steps": self.gt_error_steps,
            "missing_prediction_steps": self.missing_prediction_steps,
            "extra_prediction_steps": self.extra_prediction_steps,
            "per_error_type": per_error_type,
            "confusion_matrix": confusion_dict,
        }


def evaluate_file(
    label_file: Path,
    result_root: Path,
    runner: ModelRunner,
    limit: int | None,
    max_attempts: int,
    context_mode: str,
    save_raw_outputs: bool,
) -> MetricsAccumulator:
    category = label_file.stem
    output_file = result_root / f"{category}.json"
    metrics_file = result_root / f"{category}_metrics.json"

    original, samples = load_label_file(label_file)
    if limit is not None:
        samples = samples[:limit]

    predictions: list[dict[str, Any]] = []
    metrics = MetricsAccumulator()

    LOGGER.info(
        "Testing %s | samples=%d | input=text | context_mode=%s",
        category,
        len(samples),
        context_mode,
    )

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=category, unit="sample")
    except ImportError:
        iterator = samples

    for sample in iterator:
        question_id = sample.get("question_id", "?")
        parse_success = False
        try:
            prediction, parse_success = predict_sample(
                sample=sample,
                runner=runner,
                context_mode=context_mode,
                max_attempts=max_attempts,
                save_raw_outputs=save_raw_outputs,
            )
        except Exception as exc:
            LOGGER.error(
                "%s: question_id=%s failed during text inference: %s",
                category,
                question_id,
                exc,
            )
            prediction = {
                key: value for key, value in sample.items() if key != "steps"
            }
            # Positional invalid predictions make failures visible in metrics.
            safe_steps = sample.get("steps", [])
            prediction["steps"] = [
                [str(step[0]), None, None]
                for step in safe_steps
                if isinstance(step, (list, tuple)) and len(step) >= 1
            ]

        # Ground truth labels are consulted only here, after inference.
        metrics.update(sample, prediction, parse_success)
        predictions.append(prediction)

    result_root.mkdir(parents=True, exist_ok=True)
    output = restore_structure(original, predictions) if limit is None else predictions

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(), file, ensure_ascii=False, indent=2)

    LOGGER.info("Saved predictions: %s", output_file)
    LOGGER.info("Saved metrics: %s", metrics_file)
    return metrics


def select_label_files(args: argparse.Namespace) -> list[Path]:
    label_root = Path(args.label_root)
    if args.all:
        files = sorted(label_root.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {label_root}")
        return files

    path = label_root / f"{args.type}.json" if args.type else Path(args.file)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    return [path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only LLM benchmark for mathematical step grading."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--type", help="Example: --type fraction")
    mode.add_argument("--file", help="Example: --file steps/fraction.json")

    parser.add_argument("--label-root", default="steps")
    parser.add_argument("--result-root", default="results_text")

    parser.add_argument(
        "--backend",
        choices=("transformers-local", "qwen-vl-text-local", "openai-compatible"),
        default="transformers-local",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model path/name or API model name.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--context-mode",
        choices=("prefix", "question-current", "current-only"),
        default="prefix",
        help=(
            "prefix: question + previous s[0] + current s[0] (recommended); "
            "question-current: question + current s[0]; "
            "current-only: strictly current s[0] only."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default=None,
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry only when model output cannot be parsed; GT labels are never used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples for debugging.",
    )
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Save each raw model response under _raw_model_outputs.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    label_files = select_label_files(args)

    LOGGER.info("Loading model once: %s", args.model)
    LOGGER.info("Backend: %s", args.backend)
    LOGGER.info("Context mode: %s", args.context_mode)
    runner = ModelRunner(args)

    overall = MetricsAccumulator()
    by_category: dict[str, dict[str, Any]] = {}

    for label_file in label_files:
        category_metrics = evaluate_file(
            label_file=label_file.resolve(),
            result_root=result_root,
            runner=runner,
            limit=args.limit,
            max_attempts=args.max_attempts,
            context_mode=args.context_mode,
            save_raw_outputs=args.save_raw_outputs,
        )
        overall.merge(category_metrics)
        by_category[label_file.stem] = category_metrics.to_dict()

    summary = {
        "benchmark": "text-only mathematical step grading",
        "context_mode": args.context_mode,
        "backend": args.backend,
        "model": args.model,
        "overall": overall.to_dict(),
        "by_category": by_category,
    }

    summary_file = result_root / "metrics.json"
    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    LOGGER.info("Saved summary metrics: %s", summary_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
