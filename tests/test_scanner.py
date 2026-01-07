import os
from trt_fixer.scanner.onnx_scanner import scan_onnx_model


def test_onnx_scanner_detects_nodes():
    model_path = "examples/models/einsum_fail.onnx"
    assert os.path.exists(model_path), "Test ONNX model not found"

    graph = scan_onnx_model(model_path)

    assert graph.model.format == "onnx"
    assert len(graph.nodes) > 0

    # At least one Einsum node should exist
    einsum_nodes = [n for n in graph.nodes if n.op_type == "Einsum"]
    assert len(einsum_nodes) >= 1
