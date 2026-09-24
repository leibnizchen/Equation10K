import re
import json
import copy
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional


# ==================== AST ====================

@dataclass
class Number:
    value: Fraction
    text: Optional[str] = None

@dataclass
class Var:
    name: str

@dataclass
class Group:
    child: Any

@dataclass
class Unary:
    op: str
    operand: Any

@dataclass
class Percent:
    operand: Any

@dataclass
class Binary:
    op: str
    left: Any
    right: Any


# ==================== 预处理 / 分词 ====================

def normalize_expr(s):
    trans = str.maketrans({
        "（": "(", "）": ")", "【": "(", "】": ")", "［": "(", "］": ")",
        "｛": "(", "｝": ")", "×": "*", "÷": "/", "％": "%", "＋": "+",
        "－": "-", "−": "-", "—": "-"
    })
    return s.translate(trans).replace("²", "**2").replace("³", "**3").strip()


TOKEN_RE = re.compile(r"""
    (?P<SPACE>\s+)
  | (?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+))
  | (?P<ID>[A-Za-z_]\w*)
  | (?P<POW>\*\*|\^)
  | (?P<OP>[+\-*/()%])
  | (?P<MISMATCH>.)
""", re.X)


def tokenize(s):
    result = []
    for m in TOKEN_RE.finditer(s):
        kind, text = m.lastgroup, m.group()
        if kind == "SPACE": continue
        if kind == "MISMATCH": raise ValueError(f"无法识别字符: {text!r}")
        if kind == "POW": result.append(("OP", "**"))
        elif kind == "OP": result.append(("OP", text))
        else: result.append((kind, text))
    result.append(("EOF", ""))
    return result


# ==================== Parser ====================

class Parser:
    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def cur(self):
        return self.tokens[self.i]

    def accept(self, kind, text=None):
        k, t = self.cur()
        if k == kind and (text is None or t == text):
            self.i += 1
            return t
        return None

    def expect(self, kind, text=None):
        r = self.accept(kind, text)
        if r is None: raise ValueError(f"期望 {text or kind}，实际得到 {self.cur()[1]!r}")
        return r

    def parse(self):
        node = self.parse_add()
        if self.cur()[0] != "EOF": raise ValueError(f"表达式后存在多余内容: {self.cur()[1]!r}")
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.cur() in (("OP", "+"), ("OP", "-")):
            op = self.cur()[1]
            self.i += 1
            node = Binary(op, node, self.parse_mul())
        return node

    def parse_mul(self):
        node = self.parse_unary()
        while self.cur() in (("OP", "*"), ("OP", "/")):
            op = self.cur()[1]
            self.i += 1
            node = Binary(op, node, self.parse_unary())
        return node

    def parse_unary(self):
        if self.accept("OP", "+") is not None: return Unary("+", self.parse_unary())
        if self.accept("OP", "-") is not None: return Unary("-", self.parse_unary())
        return self.parse_power()

    def parse_power(self):
        node = self.parse_postfix()
        if self.accept("OP", "**") is not None: node = Binary("**", node, self.parse_unary())
        return node

    def parse_postfix(self):
        node = self.parse_primary()
        while self.accept("OP", "%") is not None: node = Percent(node)
        return node

    def parse_primary(self):
        kind, text = self.cur()
        if kind == "NUMBER":
            self.i += 1
            return Number(Fraction(text), text)
        if kind == "ID":
            self.i += 1
            return Var(text)
        if self.accept("OP", "(") is not None:
            node = self.parse_add()
            self.expect("OP", ")")
            return Group(node)
        raise ValueError(f"此处无法解析: {text!r}")


# ==================== 数字格式 ====================

def is_finite_decimal(x):
    d = x.denominator
    while d % 2 == 0: d //= 2
    while d % 5 == 0: d //= 5
    return d == 1


def format_number(x):
    if x.denominator == 1: return str(x.numerator)
    if not is_finite_decimal(x): return f"{x.numerator}/{x.denominator}"
    sign = "-" if x < 0 else ""
    n, d = abs(x.numerator), x.denominator
    integer, remainder = n // d, n % d
    digits = []
    while remainder:
        remainder *= 10
        digits.append(str(remainder // d))
        remainder %= d
    return f"{sign}{integer}." + "".join(digits)


# ==================== 表达式渲染 ====================

def precedence(node):
    if isinstance(node, (Number, Var, Group)): return 100
    if isinstance(node, Percent): return 90
    if isinstance(node, Binary) and node.op == "**": return 80
    if isinstance(node, Unary): return 70
    if isinstance(node, Binary) and node.op in ("*", "/"): return 60
    if isinstance(node, Binary) and node.op in ("+", "-"): return 50
    return 0


OP_SYMBOLS = {"+": "+", "-": "-", "*": "×", "/": "÷", "**": "^"}


def render(node):
    if isinstance(node, Number): return node.text if node.text is not None else format_number(node.value)
    if isinstance(node, Var): return node.name
    if isinstance(node, Group): return f"({render(node.child)})"

    if isinstance(node, Percent):
        text = render(node.operand)
        if precedence(node.operand) < precedence(node): text = f"({text})"
        return text + "%"

    if isinstance(node, Unary):
        text = render(node.operand)
        if precedence(node.operand) < precedence(node): text = f"({text})"
        return node.op + text

    if isinstance(node, Binary):
        p = precedence(node)
        left, right = render(node.left), render(node.right)
        lp, rp = precedence(node.left), precedence(node.right)
        negative_base = isinstance(node.left, Number) and node.left.value < 0

        if lp < p or (node.op == "**" and (lp == p or negative_base)): left = f"({left})"

        need_right = rp < p
        if node.op == "-" and rp == p: need_right = True
        if node.op == "/" and rp == p: need_right = True
        if need_right: right = f"({right})"

        return left + f" {OP_SYMBOLS[node.op]} " + right

    raise TypeError(f"未知节点: {node!r}")


# ==================== 变量 ====================

def to_fraction(value):
    if isinstance(value, Fraction): return value
    if isinstance(value, bool): raise ValueError("True/False不能作为数字")
    if isinstance(value, int): return Fraction(value)
    if isinstance(value, float): return Fraction(str(value))

    text = str(value).strip()
    if text.endswith("%"): return Fraction(text[:-1].strip()) / 100

    if "/" in text:
        a, b = text.split("/", 1)
        return Fraction(a.strip()) / Fraction(b.strip())

    return Fraction(text)


def substitute(node, variables):
    if isinstance(node, Var):
        if node.name not in variables: return node
        value = to_fraction(variables[node.name])
        return Number(value, format_number(value))
    if isinstance(node, Group): return Group(substitute(node.child, variables))
    if isinstance(node, Unary): return Unary(node.op, substitute(node.operand, variables))
    if isinstance(node, Percent): return Percent(substitute(node.operand, variables))
    if isinstance(node, Binary): return Binary(node.op, substitute(node.left, variables), substitute(node.right, variables))
    return node


def collect_vars(node, result=None):
    if result is None: result = set()
    if isinstance(node, Var): result.add(node.name)
    elif isinstance(node, Group): collect_vars(node.child, result)
    elif isinstance(node, Unary): collect_vars(node.operand, result)
    elif isinstance(node, Percent): collect_vars(node.operand, result)
    elif isinstance(node, Binary):
        collect_vars(node.left, result)
        collect_vars(node.right, result)
    return result


# ==================== 清理 AST ====================

def cleanup(node):
    if isinstance(node, Group):
        child = cleanup(node.child)
        return child if isinstance(child, Number) else Group(child)

    if isinstance(node, Unary):
        child = cleanup(node.operand)
        if isinstance(child, Number):
            return Number(child.value if node.op == "+" else -child.value)
        return Unary(node.op, child)

    if isinstance(node, Percent): return Percent(cleanup(node.operand))
    if isinstance(node, Binary): return Binary(node.op, cleanup(node.left), cleanup(node.right))
    return node


# ==================== 正确计算 ====================

def calc_binary(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b

    if op == "/":
        if b == 0: raise ZeroDivisionError("除数不能为0")
        return a / b

    if op == "**":
        if b.denominator != 1: raise ValueError("当前幂运算只支持整数指数")
        exponent = b.numerator
        if a == 0 and exponent < 0: raise ZeroDivisionError("0不能取负整数次幂")
        return a ** exponent

    raise ValueError(f"未知运算符: {op}")


@dataclass
class Candidate:
    node: Any
    bracket_depth: int
    op_priority: int
    order: int


def collect_candidates(node, depth=0, order_box=None, result=None):
    if order_box is None: order_box = [0]
    if result is None: result = []

    my_order = order_box[0]
    order_box[0] += 1

    if isinstance(node, Group):
        collect_candidates(node.child, depth + 1, order_box, result)
        return result

    if isinstance(node, Percent):
        collect_candidates(node.operand, depth, order_box, result)
        if isinstance(node.operand, Number): result.append(Candidate(node, depth, 50, my_order))
        return result

    if isinstance(node, Unary):
        collect_candidates(node.operand, depth, order_box, result)
        return result

    if isinstance(node, Binary):
        collect_candidates(node.left, depth, order_box, result)
        collect_candidates(node.right, depth, order_box, result)
        if isinstance(node.left, Number) and isinstance(node.right, Number):
            priority = {"**": 40, "*": 30, "/": 30, "+": 20, "-": 20}[node.op]
            result.append(Candidate(node, depth, priority, my_order))
        return result

    return result


def choose_candidate(node):
    candidates = collect_candidates(node)
    if not candidates: return None
    return max(candidates, key=lambda x: (x.bracket_depth, x.op_priority, -x.order))


def evaluate_candidate(node):
    if isinstance(node, Percent):
        value = node.operand.value / 100
        return Number(value), f"{render(node)} = {format_number(value)}"

    if isinstance(node, Binary):
        result = calc_binary(node.op, node.left.value, node.right.value)

        left, right = render(node.left), render(node.right)
        if node.op == "**" and node.left.value < 0: left = f"({left})"
        return Number(result), f"{left} {OP_SYMBOLS[node.op]} {right} = {format_number(result)}"

    raise TypeError("该节点不能直接计算")


def replace_target(node, target, replacement):
    if node is target: return replacement
    if isinstance(node, Group): return Group(replace_target(node.child, target, replacement))
    if isinstance(node, Unary): return Unary(node.op, replace_target(node.operand, target, replacement))
    if isinstance(node, Percent): return Percent(replace_target(node.operand, target, replacement))
    if isinstance(node, Binary):
        return Binary(
            node.op,
            replace_target(node.left, target, replacement),
            replace_target(node.right, target, replacement)
        )
    return node


def reduce_correctly(ast):
    ast = copy.deepcopy(ast)
    while True:
        candidate = choose_candidate(ast)
        if candidate is None: break
        replacement, _ = evaluate_candidate(candidate.node)
        ast = cleanup(replace_target(ast, candidate.node, replacement))
    return ast


def count_correct_steps(ast):
    ast = copy.deepcopy(ast)
    count = 0

    while True:
        candidate = choose_candidate(ast)
        if candidate is None: break
        replacement, _ = evaluate_candidate(candidate.node)
        ast = cleanup(replace_target(ast, candidate.node, replacement))
        count += 1

    return count


# ==================== 错误配置 ====================

@dataclass
class ErrorConfig:
    # 每个可计算步骤主动出错概率
    error_probability: float = 0.25

    # 一道题最多主动制造多少次新错误，通常建议1
    max_primary_errors: int = 1

    # 是否保证至少有一次错误
    force_one_error: bool = False

    # 第一次错误尽量不要在第一步和最后一步
    avoid_first_last: bool = False

    # 固定随机种子可复现结果
    seed: Optional[int] = None

    # 各类错误相对概率
    error_type_weights: Optional[dict] = None

    def get_weights(self):
        return self.error_type_weights or {
            "wrong_value": 0.25,
            "wrong_operator": 0.25,
            "wrong_order": 0.25,
            "calculation_error": 0.25
        }


ERROR_TYPE_NAMES = {
    "wrong_value": "Carelessly wrote down the wrong value.",
    "wrong_operator": "Wrong operation symbol",
    "wrong_order": "Incorrect parentheses or order of operations",
    "calculation_error": "Calculation error",
    "propagated": "Pre-order error propagation"
}


# ==================== 错误数字 ====================

def make_wrong_number(value, rng):
    """制造一个与正确结果不同、比较像学生马虎产生的错误数字。"""

    if value.denominator == 1:
        n = value.numerator
        sign = -1 if n < 0 else 1
        text = str(abs(n))

        # 两位及以上整数：尝试交换相邻数字
        if len(text) >= 2:
            positions = list(range(len(text) - 1))
            rng.shuffle(positions)

            for i in positions:
                chars = list(text)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                if chars[0] == "0": continue
                changed = sign * int("".join(chars))
                if changed != n: return Fraction(changed)

        # 个位数或无法交换时
        choices = [
            value + 1, value - 1, value + 2, value - 2,
            value + 10, value - 10
        ]
        return rng.choice([x for x in choices if x != value])

    choices = [
        value + Fraction(1, 10), value - Fraction(1, 10),
        value + Fraction(1, 100), value - Fraction(1, 100),
        value + 1, value - 1
    ]
    return rng.choice([x for x in choices if x != value])


# ==================== 错误1：计算错误 ====================

def inject_calculation_error(ast, candidate, rng):
    node = candidate.node
    correct_replacement, correct_description = evaluate_candidate(node)
    correct_value = correct_replacement.value
    wrong_value = make_wrong_number(correct_value, rng)

    while wrong_value == correct_value:
        wrong_value = make_wrong_number(correct_value, rng)

    if isinstance(node, Binary):
        operation = f"{render(node.left)} {OP_SYMBOLS[node.op]} {render(node.right)} = {format_number(wrong_value)}"
    else:
        operation = f"{render(node)} = {format_number(wrong_value)}"

    ast = cleanup(replace_target(ast, node, Number(wrong_value)))

    return ast, {
        "error_type": "calculation_error",
        "operation": operation,
        "produced_value": format_number(wrong_value),
        "correct_value": format_number(correct_value),
        "correct_operation": correct_description
    }


# ==================== 错误2：马虎写错数字 ====================

def inject_wrong_value(ast, candidate, rng):
    node = candidate.node

    if isinstance(node, Percent):
        original = node.operand.value
        changed = make_wrong_number(original, rng)
        result = changed / 100

        operation = (
            f"将 {format_number(original)}% 误写为 {format_number(changed)}%，"
            f"{format_number(changed)}% = {format_number(result)}"
        )

        ast = cleanup(replace_target(ast, node, Number(result)))

        return ast, {
            "error_type": "wrong_value",
            "operation": operation,
            "produced_value": format_number(result),
            "wrong_from": format_number(original),
            "wrong_to": format_number(changed)
        }

    if not isinstance(node, Binary):
        return inject_calculation_error(ast, candidate, rng)

    a, b = node.left.value, node.right.value
    change_left = rng.choice([True, False])

    if change_left:
        changed = make_wrong_number(a, rng)
        wa, wb = changed, b
        old = a
    else:
        changed = make_wrong_number(b, rng)
        wa, wb = a, changed
        old = b

    try:
        result = calc_binary(node.op, wa, wb)
    except Exception:
        return inject_calculation_error(ast, candidate, rng)

    # 如果错误写数刚好造成与正确结果相同，则重新退化成普通计算错误
    correct_result = calc_binary(node.op, a, b)
    if result == correct_result:
        return inject_calculation_error(ast, candidate, rng)

    operation = (
        f"将 {format_number(old)} 误写为 {format_number(changed)}，"
        f"{format_number(wa)} {OP_SYMBOLS[node.op]} {format_number(wb)} = {format_number(result)}"
    )

    ast = cleanup(replace_target(ast, node, Number(result)))

    return ast, {
        "error_type": "wrong_value",
        "operation": operation,
        "produced_value": format_number(result),
        "wrong_from": format_number(old),
        "wrong_to": format_number(changed)
    }


# ==================== 错误3：运算符写错 ====================

def inject_wrong_operator(ast, candidate, rng):
    node = candidate.node
    if not isinstance(node, Binary): return inject_calculation_error(ast, candidate, rng)

    a, b = node.left.value, node.right.value
    correct_result = calc_binary(node.op, a, b)

    alternatives = ["+", "-", "*", "/"]
    if node.op in alternatives: alternatives.remove(node.op)
    rng.shuffle(alternatives)

    for wrong_op in alternatives:
        try:
            wrong_result = calc_binary(wrong_op, a, b)
        except Exception:
            continue

        if wrong_result == correct_result: continue

        operation = (
            f"将运算符 {OP_SYMBOLS[node.op]} 误写为 {OP_SYMBOLS[wrong_op]}，"
            f"{format_number(a)} {OP_SYMBOLS[wrong_op]} {format_number(b)} = {format_number(wrong_result)}"
        )

        ast = cleanup(replace_target(ast, node, Number(wrong_result)))

        return ast, {
            "error_type": "wrong_operator",
            "operation": operation,
            "produced_value": format_number(wrong_result),
            "correct_operator": node.op,
            "wrong_operator": wrong_op
        }

    return inject_calculation_error(ast, candidate, rng)


# ==================== 错误4：括号/运算顺序错误 ====================

@dataclass
class WrongOrderAction:
    target: Any
    replacement: Any
    operation: str
    produced_value: Fraction


def collect_wrong_order_actions(node, result=None):
    if result is None: result = []

    if isinstance(node, Group):
        collect_wrong_order_actions(node.child, result)
        return result

    if isinstance(node, Unary):
        collect_wrong_order_actions(node.operand, result)
        return result

    if isinstance(node, Percent):
        collect_wrong_order_actions(node.operand, result)
        return result

    if not isinstance(node, Binary): return result

    collect_wrong_order_actions(node.left, result)
    collect_wrong_order_actions(node.right, result)

    if node.op not in ("+", "-", "*", "/"): return result

    # 例如：(2+3)×4 错误地先算 3×4，变成 2+12
    if (
        isinstance(node.left, Group)
        and isinstance(node.left.child, Binary)
        and isinstance(node.left.child.left, Number)
        and isinstance(node.left.child.right, Number)
        and isinstance(node.right, Number)
    ):
        inner = node.left.child

        if inner.op in ("+", "-", "*", "/"):
            b, c = inner.right.value, node.right.value

            try:
                wrong_value = calc_binary(node.op, b, c)
                correct_value = calc_binary(node.op, calc_binary(inner.op, inner.left.value, b), c)
                replacement = Binary(inner.op, inner.left, Number(wrong_value))

                # 仅保留真正能改变结果的顺序错误
                simulated = calc_binary(inner.op, inner.left.value, wrong_value)
                if simulated != correct_value:
                    operation = (
                        f"没有先计算括号，错误地先算 "
                        f"{format_number(b)} {OP_SYMBOLS[node.op]} {format_number(c)} = "
                        f"{format_number(wrong_value)}"
                    )
                    result.append(WrongOrderAction(node, replacement, operation, wrong_value))
            except Exception:
                pass

    # 例如：4×(2+3) 错误地先算 4×2，变成 8+3
    if (
        isinstance(node.right, Group)
        and isinstance(node.right.child, Binary)
        and isinstance(node.right.child.left, Number)
        and isinstance(node.right.child.right, Number)
        and isinstance(node.left, Number)
    ):
        inner = node.right.child

        if inner.op in ("+", "-", "*", "/"):
            c, a = node.left.value, inner.left.value

            try:
                wrong_value = calc_binary(node.op, c, a)
                correct_inner = calc_binary(inner.op, a, inner.right.value)
                correct_value = calc_binary(node.op, c, correct_inner)
                replacement = Binary(inner.op, Number(wrong_value), inner.right)

                simulated = calc_binary(inner.op, wrong_value, inner.right.value)
                if simulated != correct_value:
                    operation = (
                        f"没有先计算括号，错误地先算 "
                        f"{format_number(c)} {OP_SYMBOLS[node.op]} {format_number(a)} = "
                        f"{format_number(wrong_value)}"
                    )
                    result.append(WrongOrderAction(node, replacement, operation, wrong_value))
            except Exception:
                pass

    return result


def inject_wrong_order(ast, candidate, rng):
    actions = collect_wrong_order_actions(ast)

    if not actions:
        # 当前表达式无法合理制造括号错误时，退化为普通计算错误
        return inject_calculation_error(ast, candidate, rng)

    action = rng.choice(actions)
    ast = cleanup(replace_target(ast, action.target, action.replacement))

    return ast, {
        "error_type": "wrong_order",
        "operation": action.operation,
        "produced_value": format_number(action.produced_value)
    }


# ==================== 错误类型选择 ====================

def choose_error_type(ast, candidate, config, rng):
    weights = config.get_weights()
    available = ["calculation_error"]

    if isinstance(candidate.node, (Binary, Percent)): available.append("wrong_value")
    if isinstance(candidate.node, Binary): available.append("wrong_operator")
    if collect_wrong_order_actions(ast): available.append("wrong_order")

    types, ws = [], []

    for t in available:
        w = weights.get(t, 0)
        if w > 0:
            types.append(t)
            ws.append(w)

    if not types: return "calculation_error"
    return rng.choices(types, weights=ws, k=1)[0]


def inject_error_once(ast, candidate, error_type, rng):
    if error_type == "wrong_value": return inject_wrong_value(ast, candidate, rng)
    if error_type == "wrong_operator": return inject_wrong_operator(ast, candidate, rng)
    if error_type == "wrong_order": return inject_wrong_order(ast, candidate, rng)
    return inject_calculation_error(ast, candidate, rng)

def can_finish_safely(ast):
    """检查当前错误表达式能否继续计算到底，避免随机错误制造 ÷0 等非法状态。"""
    ast = copy.deepcopy(ast)
    try:
        while True:
            candidate = choose_candidate(ast)
            if candidate is None: break
            replacement, _ = evaluate_candidate(candidate.node)
            ast = cleanup(replace_target(ast, candidate.node, replacement))
        return True
    except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError):
        return False
def inject_error_safely(ast, error_type, rng, max_retry=50):
    """
    尝试制造错误，但保证错误后的表达式仍能继续计算。
    若指定错误类型一直无法安全产生，则尝试其他错误类型。
    """
    error_types = [error_type] + [x for x in ("wrong_value", "wrong_operator", "wrong_order", "calculation_error") if x != error_type]

    for current_type in error_types:
        for _ in range(max_retry):
            trial_ast = copy.deepcopy(ast)
            trial_candidate = choose_candidate(trial_ast)
            if trial_candidate is None: return None, None

            try:
                new_ast, error_info = inject_error_once(trial_ast, trial_candidate, current_type, rng)
            except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError):
                continue

            # 必须真的发生变化
            if render(new_ast) == render(trial_ast): continue

            # 最关键：错误结果必须能够继续算到底
            if can_finish_safely(new_ast): return new_ast, error_info

    return None, None


# ==================== 主函数 ====================

def calculate_steps_with_errors(expr, variables=None, config=None):
    """
    带随机错误的逐步计算。

    correct:
        从原题整体角度，该步骤是否正确。
        一旦前面发生错误，后面的步骤全部为False。

    local_correct:
        只考虑当前步骤输入的数据，这一步算术是否正确。
        例如前一步错误得到18，本步18+2=20：
        correct=False，但local_correct=True。

    is_primary_error:
        当前步骤是否为真正新产生的错误。

    error_source_step:
        最早错误来自第几步。
    """

    config = config or ErrorConfig()

    if not 0 <= config.error_probability <= 1:
        raise ValueError("error_probability必须在0~1之间")

    if config.max_primary_errors < 0:
        raise ValueError("max_primary_errors不能小于0")

    rng = random.Random(config.seed)

    normalized = normalize_expr(expr)
    ast = Parser(tokenize(normalized)).parse()
    original_expr = render(ast)

    if variables: ast = substitute(ast, variables)
    ast = cleanup(ast)
    substituted_expr = render(ast)

    baseline_ast = copy.deepcopy(ast)

    try:
        correct_final_result = render(reduce_correctly(baseline_ast))
        total_correct_steps = count_correct_steps(baseline_ast)
    except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError) as e:
        return {
            "input": expr,
            "normalized_expression": original_expr,
            "substituted_expression": substituted_expr if variables else None,
            "steps": [],
            "final_result": None,
            "correct_final_result": None,
            "final_correct": False,
            "has_error": True,
            "invalid_expression": True,
            "error": str(e),
            "error_type": type(e).__name__,
            "first_error_step": None,
            "primary_error_count": 0,
            "unassigned_variables": sorted(collect_vars(ast))
        }

    forced_error_step = None

    if config.force_one_error and config.max_primary_errors > 0 and total_correct_steps > 0:
        if config.avoid_first_last and total_correct_steps >= 3:
            possible = list(range(2, total_correct_steps))
        else:
            possible = list(range(1, total_correct_steps + 1))
        forced_error_step = rng.choice(possible)

    steps = []
    step_number = 0
    primary_error_count = 0
    error_chain_active = False
    first_error_step = None

    while True:
        candidate = choose_candidate(ast)
        if candidate is None: break

        step_number += 1
        expression_before = render(ast)
        expected_replacement, expected_description = evaluate_candidate(candidate.node)
        expected_value = expected_replacement.value

        can_inject = primary_error_count < config.max_primary_errors

        # 第一次错误尽量避开首尾；第二个及以后错误不再受此限制
        if config.avoid_first_last and total_correct_steps >= 3 and primary_error_count == 0:
            good_position = 1 < step_number < total_correct_steps
        else:
            good_position = True

        should_inject = False

        if can_inject and good_position:
            if forced_error_step is not None and primary_error_count == 0 and step_number == forced_error_step:
                should_inject = True
            elif rng.random() < config.error_probability:
                should_inject = True

        # ---------- 本步骤主动制造错误 ----------
        if should_inject:
            requested_type = choose_error_type(ast, candidate, config, rng)
            new_ast, error_info = inject_error_safely(ast, requested_type, rng)

            # 成功制造一个可继续计算的错误
            if error_info is not None:
                ast = new_ast
                actual_type = error_info["error_type"]
                primary_error_count += 1

                if first_error_step is None: first_error_step = step_number
                error_chain_active = True

                steps.append({
                    "step": step_number,
                    "expression_before": expression_before,
                    "operation": error_info["operation"],
                    "expression_after": render(ast),
                    "correct": False,
                    "local_correct": False,
                    "status": "错误",
                    "error_type": ERROR_TYPE_NAMES[actual_type],
                    "error_type_code": actual_type,
                    "is_primary_error": True,
                    "error_source_step": first_error_step,
                    "expected_operation": expected_description,
                    "expected_value": format_number(expected_value),
                    "actual_value": error_info.get("produced_value")
                })
                continue

            # 当前步骤实在找不到安全错误，则不制造错误，继续按正确步骤计算

        # ---------- 本步骤正常计算 ----------
        replacement, description = evaluate_candidate(candidate.node)
        ast = cleanup(replace_target(ast, candidate.node, replacement))

        if error_chain_active:
            steps.append({
                "step": step_number,
                "expression_before": expression_before,
                "operation": description,
                "expression_after": render(ast),

                # 前面已经错，因此整体标错
                "correct": False,

                # 但本步骤基于当前错误结果算术本身正确
                "local_correct": True,
                "status": "错误",

                "error_type": ERROR_TYPE_NAMES["propagated"],
                "error_type_code": "propagated",
                "is_primary_error": False,
                "error_source_step": first_error_step,

                "expected_operation": None,
                "expected_value": None,
                "actual_value": format_number(replacement.value)
            })
        else:
            steps.append({
                "step": step_number,
                "expression_before": expression_before,
                "operation": description,
                "expression_after": render(ast),

                "correct": True,
                "local_correct": True,
                "status": "正确",

                "error_type": None,
                "error_type_code": None,
                "is_primary_error": False,
                "error_source_step": None,

                "expected_operation": description,
                "expected_value": format_number(replacement.value),
                "actual_value": format_number(replacement.value)
            })

    remaining_vars = sorted(collect_vars(ast))
    actual_final_result = render(ast)

    return {
        "input": expr,
        "normalized_expression": original_expr,
        "substituted_expression": substituted_expr if variables else None,

        "steps": steps,

        # 实际按照错误过程得到的最终答案
        "final_result": actual_final_result,

        # 正确答案
        "correct_final_result": correct_final_result,

        "final_correct": not error_chain_active,
        "has_error": error_chain_active,
        "first_error_step": first_error_step,
        "primary_error_count": primary_error_count,
        "unassigned_variables": remaining_vars,

        "error_config": {
            "error_probability": config.error_probability,
            "max_primary_errors": config.max_primary_errors,
            "force_one_error": config.force_one_error,
            "avoid_first_last": config.avoid_first_last,
            "seed": config.seed,
            "error_type_weights": config.get_weights()
        }
    }


# ==================== 普通正确计算 ====================

def calculate_steps(expr, variables=None):
    """不制造错误，方便需要正确计算时使用。"""
    return calculate_steps_with_errors(
        expr,
        variables,
        ErrorConfig(
            error_probability=0,
            max_primary_errors=0,
            force_one_error=False
        )
    )


# ==================== 测试 ====================

# if __name__ == "__main__":
#     from pprint import pprint
#
#     config = ErrorConfig(
#         error_probability=0.3,
#
#         # 最多主动制造1个错误，之后只进行错误传播
#         max_primary_errors=1,
#
#         # 保证本题至少出现一次错误
#         force_one_error=True,
#
#         # 第一次错误尽量不出现在首尾
#         avoid_first_last=True,
#
#         # 固定后每次运行结果相同；生产环境可设为None
#         seed=123,
#
#         error_type_weights={
#             "wrong_value": 0.30,
#             "wrong_operator": 0.25,
#             "wrong_order": 0.25,
#             "calculation_error": 0.20
#         }
#     )
#
#     result = calculate_steps_with_errors(
#         "（-3*（a+b）+c）/d",
#         variables={
#             "a": 2,
#             "b": "1/2",
#             "c": 5,
#             "d": 2
#         },
#         config=config
#     )
#
#     pprint(result, sort_dicts=False)


if __name__ == '__main__':
    config = ErrorConfig(
                error_probability=0.3,

                # 最多主动制造1个错误，之后只进行错误传播
                max_primary_errors=1,

                # 保证本题至少出现一次错误
                force_one_error=True,

                # 第一次错误尽量不出现在首尾
                avoid_first_last=True,

                # 固定后每次运行结果相同；生产环境可设为Nonec
                seed=123,

                error_type_weights={
                    "wrong_value": 0.20,
                    "wrong_operator": 0.15,
                    "wrong_order": 0.15,
                    "calculation_error": 0.10
                }
            )

    QUESTION_TYPES=["addition","subtraction","multiplication","division","mixed_operations","parentheses","fraction","decimal","exponentiation","random_combination"]
    for q_t in QUESTION_TYPES:
        with open(f"{"question_data"}/{q_t}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for data in dataset:
            t = calculate_steps_with_errors( data["question"])
            step=[]
            for s in t['steps']:
                 step.append([s['expression_after'],s['correct'],s['error_type']])
            data["steps"]= step
        with open(f"{'steps'}/{q_t}.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)