from typing import Dict

from trt_fixer.scanner.graph_repr import IRNode
from trt_fixer.detector.gelu_patterns import is_erf_gelu_pattern


def classify_node(node: IRNode, support_db) -> Dict:
    # --------------------------------------------------
    # GELU (Erf-based composite)
    # --------------------------------------------------
    if is_erf_gelu_pattern(node.graph, node):
        return {
            "status": "rewrite_supported",
            "reason": (
                "Detected GELU implemented as an Erf-based composite "
                "subgraph. This pattern can be rewritten to TensorRT-"
                "friendly approximate GELU."
            ),
        }

    # --------------------------------------------------
    # Native TensorRT ops
    # --------------------------------------------------
    if node.op_type in support_db.native_ops:
        return {
            "status": "native_supported",
            "reason": "Operator supported by TensorRT",
        }

    # --------------------------------------------------
    # Rewrite-supported ops
    # --------------------------------------------------
    if node.op_type in support_db.rewrite_supported_ops:
        return {
            "status": "rewrite_supported",
            "reason": "Operator can be rewritten to TensorRT-supported ops",
        }

    # --------------------------------------------------
    # Plugin-required ops
    # --------------------------------------------------
    if node.op_type in support_db.plugin_required_ops:
        return {
            "status": "plugin_required",
            "reason": "Operator requires TensorRT plugin",
        }

    # --------------------------------------------------
    # Blocked ops
    # --------------------------------------------------
    if node.op_type in support_db.blocked_ops:
        return {
            "status": "blocked",
            "reason": "Operator not supported by TensorRT",
        }

    # --------------------------------------------------
    # Default fallback
    # --------------------------------------------------
    return {
        "status": "unknown",
        "reason": "Unknown operator support",
    }
