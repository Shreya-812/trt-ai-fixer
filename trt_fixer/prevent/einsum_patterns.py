def parse_einsum_equation(equation: str):
    """
    Parse einsum equation like:
    'ij,jk->ik'
    """
    equation = equation.replace(" ", "")
    lhs, rhs = equation.split("->")
    inputs = lhs.split(",")
    return inputs, rhs


def is_matmul_einsum(equation: str) -> bool:
    """
    Detects:
      ij,jk->ik
      bij,bjk->bik
    """
    inputs, output = parse_einsum_equation(equation)

    if len(inputs) != 2:
        return False

    a, b = inputs

    # last dim of A == second-last dim of B
    if len(a) >= 2 and len(b) >= 2:
        return a[-1] == b[-2] and output[-2:] == (a[-2] + b[-1])

    return False
