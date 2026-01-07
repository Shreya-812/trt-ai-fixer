from trt_fixer.scanner.graph_repr import IRNode


def explain_failure(node: IRNode, trt_version: str):
    reasons = []

    if node.dynamic:
        reasons.append("Dynamic shapes detected")

    if node.op_type in ("LayerNormalization", "InstanceNormalization"):
        reasons.append("Normalization ops typically require plugins in TensorRT")

    if not reasons:
        reasons.append("Operator not supported by TensorRT")

    return {
        "node": node.name,
        "op_type": node.op_type,
        "trt_version": trt_version,
        "explanation": "; ".join(reasons),
    }
