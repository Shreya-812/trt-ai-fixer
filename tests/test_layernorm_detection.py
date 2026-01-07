from trt_fixer.scanner.onnx_scanner import scan_onnx_model
from trt_fixer.detector.support_db import TensorRTSupportDB
from trt_fixer.detector.classify_node import classify_node

def test_layernorm_detection():
    graph = scan_onnx_model("examples/models/layernorm_fail.onnx")
    db = TensorRTSupportDB("TensorRT-8.6")

    node = graph.nodes[0]
    result = classify_node(node, db)

    assert result["status"] == "plugin_required"
    assert "LayerNormalization" in result["reason"]
