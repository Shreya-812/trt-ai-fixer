# trt_fixer/scanner/shape_infer.py

import onnx
from onnx import shape_inference


def infer_shapes_safe(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Runs ONNX shape inference safely.
    If inference fails, returns the original model.
    """
    try:
        return shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] Shape inference failed: {e}")
        return model
