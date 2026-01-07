from trt_fixer.scanner.onnx_scanner import scan_onnx_model
from trt_fixer.detector.support_db import TensorRTSupportDB
from trt_fixer.detector.classify_node import classify_node


def test_einsum_is_rewrite_supported():
    graph = scan_onnx_model("examples/models/einsum_fail.onnx")
    db = TensorRTSupportDB("TensorRT-8.6")

    einsum_nodes = [n for n in graph.nodes if n.op_type == "Einsum"]
    assert len(einsum_nodes) > 0

    result = classify_node(einsum_nodes[0], db)

    assert result["status"] in ["rewrite_supported", "plugin_required"]
