from typing import List, Dict
from trt_fixer.scanner.graph_repr import IRNode


def rank_fixes(node: IRNode) -> List[Dict]:
    suggestions = []

    if node.op_type == "LayerNormalization":
        suggestions.append({
            "rank": 1,
            "type": "plugin",
            "summary": "Use TensorRT LayerNorm plugin",
        })
    else:
        suggestions.append({
            "rank": 1,
            "type": "fallback",
            "summary": "Fallback to ONNX Runtime for this subgraph",
        })

    return suggestions
