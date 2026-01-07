import onnx
from onnx import helper

from trt_fixer.prevent.einsum_patterns import is_matmul_einsum


def rewrite_einsum_onnx(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """
    Rewrite supported Einsum ops into MatMul.
    """

    graph = model.graph
    nodes = list(graph.node)

    einsum_node = None
    for n in nodes:
        if n.name == node_name and n.op_type == "Einsum":
            einsum_node = n
            break

    if einsum_node is None:
        raise RuntimeError("Einsum node not found")

    # Extract equation
    equation = None
    for attr in einsum_node.attribute:
        if attr.name == "equation":
            equation = attr.s.decode("utf-8")

    if equation is None:
        raise RuntimeError("Einsum equation missing")

    # Check if supported
    if not is_matmul_einsum(equation):
        raise RuntimeError(f"Unsupported Einsum pattern: {equation}")

    # Create MatMul
    matmul_node = helper.make_node(
        "MatMul",
        inputs=list(einsum_node.input),
        outputs=list(einsum_node.output),
        name=f"{einsum_node.name}_matmul",
    )

    # Replace node
    graph.node.remove(einsum_node)
    graph.node.append(matmul_node)

    return model
