# trt_fixer/utils/validation.py

from trt_fixer.scanner.graph_repr import ModelGraph


def validate_model_graph(graph: ModelGraph):
    assert graph.model.format in ["onnx"], "Unsupported model format"

    for node in graph.nodes:
        assert node.op_type, "Node missing op_type"
        assert isinstance(node.inputs, list)
        assert isinstance(node.outputs, list)
