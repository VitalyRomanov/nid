import ast


def has_valid_syntax(function_body: str) -> bool:
    try:
        ast.parse(function_body.lstrip())
        return True
    except SyntaxError:
        return False