# trt_fixer/explain/trt_error_parser.py

from typing import Dict


def parse_trt_error(raw_error: str) -> Dict[str, str]:
    """
    Convert raw TensorRT error logs into structured failure reasons.
    """

    error = raw_error.lower()

    if "unsupported" in error and "einsum" in error:
        return {
            "type": "unsupported_op",
            "summary": "Einsum operation is not supported by TensorRT",
            "detail": "TensorRT parser does not support Einsum directly"
        }

    if "dynamic" in error and "shape" in error:
        return {
            "type": "dynamic_shape",
            "summary": "Dynamic shapes are not supported in this context",
            "detail": "TensorRT requires static or bounded shapes for kernel selection"
        }

    if "plugin" in error:
        return {
            "type": "missing_plugin",
            "summary": "Required TensorRT plugin not found",
            "detail": "Custom implementation is required for this operator"
        }

    return {
        "type": "unknown",
        "summary": "Unknown TensorRT failure",
        "detail": raw_error
    }
