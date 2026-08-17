import ast 

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def calculator(query: str) -> str:
    """A calculator that can evaluate mathematical expressions."""
    try:
        # Evaluate the expression and return the result
        result = ast.literal_eval(query)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
search = DuckDuckGoSearchRun()

tools = [search, calculator]