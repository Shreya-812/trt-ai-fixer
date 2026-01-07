import onnx
from typing import List, Dict, Callable

from trt_fixer.prevent.gelu_rewrite import rewrite_gelu_onnx
from trt_fixer.prevent.einsum_rewrite import rewrite_einsum_onnx


# --------------------------------------------------
# Rewrite registry (frozen)
# --------------------------------------------------
REWRITE_REGISTRY: Dict[str, Callable] = {
    "GELU": rewrite_gelu_onnx,
    "EINSUM": rewrite_einsum_onnx,
}


def apply_rewrites(
    onnx_path: str,
    rewrite_plan: List[Dict],
) -> str:
    """
    Apply ONNX-level rewrites based on a rewrite plan.
    """

    model = onnx.load(onnx_path)
    applied = False

    for action in rewrite_plan:
        op = action["op"]
        node_name = action["node_name"]

        if op not in REWRITE_REGISTRY:
            continue

        rewrite_fn = REWRITE_REGISTRY[op]
        model = rewrite_fn(model, node_name)
        applied = True

    if not applied:
        raise RuntimeError("Rewrite plan contained no applicable actions")

    out_path = onnx_path.replace(".onnx", "_rewritten.onnx")
    onnx.save(model, out_path)
    return out_path
