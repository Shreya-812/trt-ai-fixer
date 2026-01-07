# trt_fixer/prevent/einsum_rules.py

from typing import Dict, Any, Union


def analyze_einsum(equation: Union[str, bytes]) -> Dict[str, Any]:
    """
    Analyze einsum equation and determine rewrite strategy.
    Handles both bytes and str (ONNX stores attributes as bytes).
    """

    if isinstance(equation, bytes):
        equation = equation.decode("utf-8")

    eq = equation.replace(" ", "")

    # Simple matrix multiplication
    if eq == "ij,jk->ik":
        return {
            "rewriteable": True,
            "strategy": "matmul",
            "description": "Standard matrix multiplication"
        }

    # Batched matrix multiplication
    if eq == "bij,bjk->bik":
        return {
            "rewriteable": True,
            "strategy": "batched_matmul",
            "description": "Batched matrix multiplication"
        }

    # Anything else is unsafe
    return {
        "rewriteable": False,
        "strategy": None,
        "description": f"Unsupported einsum pattern: {eq}"
    }

