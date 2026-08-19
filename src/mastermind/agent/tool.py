import ast
import operator

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# ast.literal_eval only parses literals (numbers, tuples, ...), not expressions
# with operators - "3 * (4 + 5)" is a BinOp node, which it rejects outright.
# Walking the AST ourselves, restricted to this whitelist, evaluates real
# arithmetic without eval()'s arbitrary-code-execution risk.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@tool
def calculator(query: str) -> str:
    """A calculator that can evaluate mathematical expressions."""
    try:
        result = _eval_node(ast.parse(query, mode="eval").body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


search = DuckDuckGoSearchRun()

tools = [search, calculator]