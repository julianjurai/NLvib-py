"""OpenAI-powered validation and assumption testing.

Uses the OpenAI API for two purposes:
  1. Assumption sub-agent (o3/o4-mini): mathematical reasoning about algorithms
     before committing to an implementation.
  2. Cross-validation (GPT-4o code interpreter): independently runs Python
     numerical code to cross-check results without needing MATLAB.

Usage:
    # Test a mathematical assumption
    python tools/openai_validator.py assume \
        "Is the Newmark beta=0.25, gamma=0.5 scheme identical to average \
         constant acceleration for linear second-order ODEs?"

    # Cross-validate a numerical result
    python tools/openai_validator.py crossval \
        --result-file tests/fixtures/duffing_python.npz \
        --reference-code tools/reference_scripts/duffing_reference.py

    # Validate a Jacobian derivation
    python tools/openai_validator.py jacobian \
        --module src/nlvib/nonlinearities/elements.py \
        --function cubic_spring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

SYSTEM_ASSUME = textwrap.dedent("""
    You are an expert in nonlinear structural dynamics and numerical methods.
    Your role is to answer mathematical and algorithmic questions precisely,
    with references to literature where relevant.

    Context: We are building a Python port of the NLvib MATLAB toolbox
    (Krack & Gross 2019, Harmonic Balance for Nonlinear Vibration Problems, Springer).
    The toolbox implements: harmonic balance (AFT method), Newmark shooting,
    and arc-length continuation.

    Answer the assumption question clearly. State:
    1. The answer (yes/no/it depends)
    2. The mathematical justification
    3. Any edge cases or caveats
    4. The recommended Python/scipy implementation approach
    5. Relevant equation numbers from Krack & Gross (2019) if applicable
""").strip()

SYSTEM_CROSSVAL = textwrap.dedent("""
    You are a numerical methods expert and Python programmer.
    You will receive Python code implementing a numerical calculation.
    Run it, verify the result is mathematically correct, and report:
    1. Whether the result matches the expected analytical or reference value
    2. The relative error
    3. Any numerical issues observed
    4. Pass or Fail verdict
""").strip()


def get_client():
    """Initialise OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def assume(question: str, model: str = "o3") -> str:
    """Send an assumption question to o3 for mathematical reasoning."""
    client = get_client()
    print(f"Querying {model} for assumption: {question[:80]}...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_ASSUME},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content
    return answer


def crossval_code(code: str, model: str = "gpt-4o") -> str:
    """Send Python code to GPT-4o code interpreter for independent execution."""
    client = get_client()
    print(f"Cross-validating via {model} code interpreter...")

    # Use Responses API with code_interpreter tool
    response = client.responses.create(
        model=model,
        tools=[{"type": "code_interpreter"}],
        input=[
            {
                "role": "user",
                "content": (
                    f"{SYSTEM_CROSSVAL}\n\n"
                    f"Run the following Python code and validate the result:\n\n"
                    f"```python\n{code}\n```"
                ),
            }
        ],
    )
    # Extract text output
    for block in response.output:
        if hasattr(block, "content"):
            for item in block.content:
                if hasattr(item, "text"):
                    return item.text
    return str(response)


def validate_jacobian(module_path: str, function_name: str) -> str:
    """Ask o3 to verify a Jacobian derivation by reading the source."""
    source = Path(module_path).read_text()
    question = (
        f"Review the following Python implementation of '{function_name}' "
        f"from the NLvib toolbox. Verify that the Jacobian (df_dq and df_ddq) "
        f"is analytically correct by deriving it from the force expression. "
        f"Report: correct/incorrect, the correct expression if wrong, "
        f"and any edge cases.\n\n"
        f"Source:\n```python\n{source}\n```"
    )
    return assume(question, model="o3")


def save_result(result: str, output_path: Path | None = None) -> None:
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
    if output_path:
        output_path.write_text(result)
        print(f"\nResult saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # assume
    p_assume = sub.add_parser("assume", help="Mathematical assumption question (uses o3)")
    p_assume.add_argument("question", help="The assumption to test")
    p_assume.add_argument("--model", default="o3")
    p_assume.add_argument("--output", type=Path, default=None)

    # crossval
    p_cv = sub.add_parser("crossval", help="Cross-validate Python code via GPT-4o code interpreter")
    p_cv.add_argument("--code-file", type=Path, required=True, help="Python file to execute")
    p_cv.add_argument("--model", default="gpt-4o")
    p_cv.add_argument("--output", type=Path, default=None)

    # jacobian
    p_jac = sub.add_parser("jacobian", help="Verify a Jacobian derivation via o3")
    p_jac.add_argument("--module", required=True, help="Path to Python module")
    p_jac.add_argument("--function", required=True, help="Function name to verify")
    p_jac.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "assume":
        result = assume(args.question, model=args.model)
        save_result(result, args.output)

    elif args.command == "crossval":
        code = args.code_file.read_text()
        result = crossval_code(code, model=args.model)
        save_result(result, args.output)

    elif args.command == "jacobian":
        result = validate_jacobian(args.module, args.function)
        save_result(result, args.output)


if __name__ == "__main__":
    main()
