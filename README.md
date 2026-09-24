Equation10K: Teacher-Style Error Diagnosis in Handwritten Arithmetic Solutions
Equation10K is a benchmark construction project designed to evaluate
whether open-source and closed-source vision-language models (VLMs) can
identify errors in students' handwritten arithmetic solutions in a
teacher-like manner.
The benchmark covers the full pipeline from arithmetic expression
generation to step-by-step solution synthesis, simulated student errors,
handwriting rendering, and VLM evaluation.
---
1. Automatic Arithmetic Expression Generation
1.1 Expression Generation
Expressions are generated using a three-stage pipeline:
Generate the expression structure
Fill in numerical values
Validate the final expression
The template layer describes only the computational structure and is
independent of specific numbers. The value layer fills templates
with integers, decimals, fractions, or mixed numeric types according to
the problem category. Finally, the validation layer converts the
displayed expression into a Python-evaluable form and filters out
invalid cases such as division by zero, syntax errors, or abnormal
results.
Templates use `N` as a placeholder for a numeric value. For example:
``` text
N + N × N
```
represents a mixed addition-and-multiplication expression containing
three numbers. The template specifies only the order-of-operations
structure; concrete values such as `8`, `2.5`, or `1/3` are inserted
later.
Expressions Without Parentheses
For an expression containing `n` numbers, `n - 1` operators are
required. All operator sequences are automatically enumerated by taking
the Cartesian product of:
``` text
+, -, ×, ÷
```
This removes the need to manually define templates and prevents operator
arrangements from being accidentally omitted.
Expressions With Parentheses
Parenthesized structures are generated using binary expression
trees:
Each internal node represents an operator.
Each leaf node represents `N`.
`tree_shapes(left, right)` recursively enumerates all possible
left/right subtree partitions.
Each tree is converted back into an arithmetic expression.
This approach effectively covers all valid binary association patterns.
---
       Number of     Structures    Parenthesis     Structures          Total
          Values        Without         Shapes           With 
                    Parentheses                   Parentheses 
---
               3             16              2             32             48

               4             64              5            320            384

               5            256             14          3,584          3,840

       **Total**        **336**         **12**      **3,936**      **4,272**
---
For expressions containing 3--5 values, the base structure library
therefore contains 4,272 automatically generated structures.
Extending the generator to six values produces substantially more
structures without requiring manually added templates.
---
1.2 Difficulty Levels
In educational settings, arithmetic errors often arise from confusion
about rules such as:
left-to-right evaluation,
multiplication/division before addition/subtraction,
parentheses taking precedence.
Equation10K defines five difficulty levels according to the
complexity of the required operations.
---
                    Difficulty Main Characteristics      Example
---
                             1 Single operation type;    `8 + 3 + 6`
                               straightforward order     

                             2 Mixed operators at the    `18 - 7 + 5`,
                               same precedence level;    `48 ÷ 6 × 3`
                               evaluated left to right   

                             3 Mixed                     `8 + 6 × 4 - 3`
                               multiplication/division   
                               and addition/subtraction  

                             4 Single-level parentheses  `8 × (6 - 4) + 3`
                               that change evaluation    
                               order                     

                             5 Nested parentheses and    `6 + 24 ÷ (8 - 4) × 3`,
                               multiple interacting      `5 × (12 - (8 - 3))`
                               rules                     
---
---
1.3 Problem Categories
Problem category and difficulty level are independent
dimensions:
The problem category determines which expression structures are
allowed.
The difficulty level determines which complexity tier is sampled
from those structures.
The benchmark defines 10 problem categories.
---
Category                            Filtering Rule
---
Addition                            Operator set must be exactly `{+}`
Subtraction                         Operator set must be exactly `{-}`
Multiplication                      Operator set must be exactly `{×}`
Division                            Operator set must be exactly `{÷}`
Mixed Operations                    At least two operator types and no
parentheses
Parenthesized Operations            Template must contain parentheses
Fraction Arithmetic                 Use all structures available at the
current difficulty; fill all values
with fractions
Decimal Arithmetic                  Use all structures available at the
current difficulty; fill all values
with decimals
Exponentiation                      Expression contains square or cube
operations
Random Combination                  Random combination of the nine
categories above
---
1.4 Expression Conversion and Validation
Every generated expression is checked for validity before being included
in the dataset. The validation stage filters out invalid or unusable
expressions, including cases such as division by zero, malformed syntax,
and abnormal evaluation results.
---
1.5 Dataset Size
Each category covers difficulty levels 1--5, with 1,000 samples
per difficulty level.
Category                    Difficulty Levels    Samples per Level    Total
---
Addition                          1--5                       1,000    5,000
Subtraction                       1--5                       1,000    5,000
Multiplication                    1--5                       1,000    5,000
Division                          1--5                       1,000    5,000
Mixed Operations                  1--5                       1,000    5,000
Parenthesized Operations          1--5                       1,000    5,000
Fraction Arithmetic               1--5                       1,000    5,000
Decimal Arithmetic                1--5                       1,000    5,000
Exponentiation                    1--5                       1,000    5,000
Random Combination                1--5                       1,000   50,000
> **Note:** The table above preserves the quantities stated in the
> original project README.
---
2. Step-by-Step Solution Generation
The solution generator parses an arithmetic expression into a sequence
of calculation steps. It can also probabilistically inject simulated
student errors, producing training examples labeled as correct or
incorrect.
2.1 Processing Pipeline
``` text
Input Expression
      ↓
Expression Normalization
      ↓
Tokenizer
      ↓
Parser → Abstract Syntax Tree (AST)
      ↓
Locate the Next Highest-Priority Operation
      ↓
Execute One Step / Randomly Inject an Error
      ↓
Write the Result Back to the AST
      ↓
Continue Until Evaluation Is Complete
      ↓
Output the Step Dictionary
```
---
2.2 Supported Error Types
---
Error Type                          Description
---
`wrong_value`                       Carelessly writes down an incorrect
value
`wrong_operator`                    Uses the wrong operation symbol
`wrong_order`                       Applies parentheses or the order of
operations incorrectly
`calculation_error`                 Makes an arithmetic calculation
error
`propagated`                        Propagates an error from an earlier
step
---
2.3 Simulated Student Error Propagation
The generator models the common pattern in which an early mistake
affects all subsequent calculations.
Once an error is introduced at a particular step, later steps are
treated as being affected by that preceding error and are labeled:
``` json
{
  "correct": false
}
```
This allows the benchmark to represent both the initial error and
its downstream propagation through the student's solution.
---
3. Handwritten Solution Generation
The generated solution steps are rendered into handwritten form to
create visual inputs for VLM evaluation.
3.1 Handwriting Styles
The project includes 12 handwriting styles to increase visual
diversity.
---
4. Vision-Language Model Evaluation
The benchmark evaluates whether a vision-language model can determine
whether the calculation steps shown in a handwritten arithmetic solution
are correct.
Both open-source and closed-source / API-based VLM backends are
supported.
---
5. Running the Project
Step 1 --- Generate Arithmetic Problems
``` bash
python gen_questions.py
```
Step 2 --- Generate Step-by-Step Solutions
``` bash
python gen_steps.py
```
Step 3 --- Render Handwritten Solutions
``` bash
python various_handwriting.py
```
Step 4 --- Evaluate an Open-Source VLM
Evaluate all files:
``` bash
python benchmark_vlm_with_metrics.py --backend qwen-local --all
```
Evaluate a specific problem category:
``` bash
python benchmark_vlm_with_metrics.py --backend qwen-local --type fraction
```
Step 5 --- Evaluate a Closed-Source / OpenAI-Compatible VLM
Configure the API endpoint:
``` bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-api-host/v1"
```
Evaluate all files:
``` bash
python benchmark_vlm_with_metrics.py --backend openai-compatible --all
```
Evaluate a specific problem category:
``` bash
python benchmark_vlm_with_metrics.py --backend openai-compatible --type fraction
```
---
6. Ablation Experiments
Oracle-Text Input
Run the text-based benchmark with a local Qwen-VL backend:
``` bash
python benchmark_text_with_metrics.py --type fraction \
    --backend qwen-vl-text-local \
    --model Qwen/Qwen2___5-VL-7B-Instruct
```
Run the text-based benchmark with an OpenAI-compatible backend:
``` bash
python benchmark_text_with_metrics.py --file steps/fraction.json \
    --backend openai-compatible \
    --model gpt-5.6-sol \
    --api-key "$OPENAI_API_KEY"
```
---
7. Generated Dataset
The dataset generated by this project is available here:
https://docs.google.com/spreadsheets/d/1LhPTykJGEQucwKXFUYcSizweWGgGYdIOeYbEcNJJbLw/edit?usp=drive_link
---
Project Overview
``` text
Arithmetic Structure Generation
            ↓
      Value Sampling
            ↓
   Expression Validation
            ↓
 Step-by-Step AST Evaluation
            ↓
 Student Error Injection
            ↓
 Handwriting Rendering
            ↓
 Vision-Language Model Evaluation
```
Equation10K is designed to test not only whether a model can recognize
handwritten arithmetic, but also whether it can diagnose where and how
a student's reasoning goes wrong across multi-step solutions.
