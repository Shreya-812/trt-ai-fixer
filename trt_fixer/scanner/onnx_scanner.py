import onnx
from onnx import numpy_helper

from trt_fixer.scanner.graph_repr import IRNode, IRGraph


def scan_onnx_model(path: str) -> IRGraph:
    model = onnx.load(path)
    graph = model.graph

    model_info = {
        "name": graph.name or "unknown_model",
        "opset": model.opset_import[0].version if model.opset_import else None,
        "producer": model.producer_name,
        "format": "onnx",
    }

    nodes = []
    for idx, n in enumerate(graph.node):
        attrs = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                attrs[a.name] = numpy_helper.to_array(a.t)
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = a.f
            elif a.type == onnx.AttributeProto.INT:
                attrs[a.name] = a.i
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8")
            else:
                attrs[a.name] = None

        node = IRNode(
            name=n.name or f"node_{idx}",
            op_type=n.op_type,
            inputs=list(n.input),
            outputs=list(n.output),
            attributes=attrs,
            dynamic=False,   # conservative default
            shape=None,      # can be filled later by shape inference
        )
        nodes.append(node)

    return IRGraph(nodes=nodes, model_info=model_info)
