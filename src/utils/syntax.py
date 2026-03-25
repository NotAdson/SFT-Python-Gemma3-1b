import ast

def check_syntax(code: str) -> bool:
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

