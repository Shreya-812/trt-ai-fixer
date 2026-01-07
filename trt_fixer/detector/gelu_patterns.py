from trt_fixer.scanner.graph_repr import IRGraph, IRNode


def is_erf_gelu_pattern(graph: IRGraph, node: IRNode) -> bool:
    """
    Detects PyTorch-style exact GELU implemented via:
    Erf -> Add -> Mul

    Conservative, read-only detection.
    """

    if node.op_type != "Erf":
        return False

    # Erf output tensor
    erf_out = node.outputs[0]

    # Erf -> Add
    add_nodes = [
        n for n in graph.consumers_of(erf_out)
        if n.op_type == "Add"
    ]
    if not add_nodes:
        return False

    # Add -> Mul
    for add in add_nodes:
        mul_nodes = [
            n for n in graph.consumers_of(add.outputs[0])
            if n.op_type == "Mul"
        ]
        if mul_nodes:
            return True

    return False
