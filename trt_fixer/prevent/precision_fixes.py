# trt_fixer/prevent/precision_fixes.py

from trt_fixer.scanner.graph_repr import NodeInfo


def suggest_precision_fix(node: NodeInfo):
    """
    Suggest precision-level fixes to improve TensorRT compatibility.
    """
    if node.op_type in ["LayerNormalization", "ReduceMean"]:
        return {
            "node": node.name,
            "fix": "force_fp32",
            "reason": f"{node.op_type} is numerically sensitive in FP16/INT8"
        }

    return None
