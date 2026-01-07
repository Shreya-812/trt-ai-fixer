# trt_fixer/suggest/plugin_advisor.py

from typing import Dict
from trt_fixer.scanner.graph_repr import NodeInfo


def advise_plugin(node: NodeInfo) -> Dict[str, str]:
    """
    Provide guidance when a TensorRT plugin is required.
    """

    if node.op_type == "Einsum":
        return {
            "action": "write_plugin",
            "difficulty": "medium",
            "summary": "Implement Einsum as a TensorRT plugin",
            "details": (
                "This Einsum pattern cannot be safely rewritten. "
                "A custom TensorRT plugin implementing the contraction "
                "using a CUDA kernel is required."
            )
        }
    if node.op_type == "LayerNormalization":
        return {
            "action": "write_plugin",
            "difficulty": "medium",
            "summary": "Implement LayerNormalization as a TensorRT plugin",
            "details": (
                "LayerNormalization is commonly implemented as a fused plugin "
                "in TensorRT. NVIDIA provides reference plugins in some samples, "
                "or it can be fused from ReduceMean, Sub, Pow, Div, and Mul."
            )
        }

    return {
        "action": "write_plugin",
        "difficulty": "unknown",
        "summary": f"Plugin required for {node.op_type}",
        "details": "Refer to TensorRT PluginV2DynamicExt documentation."
    }
