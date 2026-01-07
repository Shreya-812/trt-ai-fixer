def can_rewrite_gelu(node):
    return node.op_type == "Gelu"


def gelu_rewrite_info():
    return {
        "strategy": "exact_to_approx_gelu",
        "safe_for_inference": True,
        "max_error": "< 1e-3",
        "note": "Standard tanh-based approximation used in inference engines"
    }
