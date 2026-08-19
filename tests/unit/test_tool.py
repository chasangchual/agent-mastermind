"""Runnable check for calculator's AST evaluator - literal_eval alone can't
handle real expressions with operators (BinOp nodes), which is what broke it.
"""

from mastermind.agent.tool import calculator


def test_calculator_evaluates_arithmetic() -> None:
    assert calculator.invoke("3 * (4 + 5)") == "27"
    assert calculator.invoke("100 / 4") == "25.0"
    assert calculator.invoke("-2 + 3") == "1"


def test_calculator_rejects_unsupported_input() -> None:
    assert calculator.invoke("__import__('os')").startswith("Error evaluating expression")
    assert calculator.invoke("2 ** 3").startswith("Error evaluating expression")
