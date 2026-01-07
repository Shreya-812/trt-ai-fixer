import onnx
import numpy as np
from onnx import helper


def rewrite_gelu_onnx(model: onnx.ModelProto, erf_node_name: str) -> onnx.ModelProto:
    """
    Fully replace exact GELU subgraph:
        Div -> Erf -> Add -> Mul
    with tanh-approx GELU.

    Correctly unwraps x from x * 0.5 if needed.
    """

    graph = model.graph
    nodes = list(graph.node)

    # --------------------------------------------------
    # 1. Locate Erf node
    # --------------------------------------------------
    erf = None
    for n in nodes:
        if n.name == erf_node_name and n.op_type == "Erf":
            erf = n
            break
    if erf is None:
        raise RuntimeError("Erf node not found")

    # --------------------------------------------------
    # 2. Find Div feeding Erf
    # --------------------------------------------------
    div = None
    for n in nodes:
        if erf.input[0] in n.output and n.op_type == "Div":
            div = n
            break
    if div is None:
        raise RuntimeError("Div feeding Erf not found")

    # --------------------------------------------------
    # 3. Find Add consuming Erf
    # --------------------------------------------------
    add = None
    for n in nodes:
        if erf.output[0] in n.input and n.op_type == "Add":
            add = n
            break
    if add is None:
        raise RuntimeError("Add after Erf not found")

    # --------------------------------------------------
    # 4. Find final Mul producing GELU output
    # --------------------------------------------------
    mul = None
    for n in nodes:
        if add.output[0] in n.input and n.op_type == "Mul":
            mul = n
            break
    if mul is None:
        raise RuntimeError("Final Mul for GELU not found")

    gelu_out = mul.output[0]

    # --------------------------------------------------
    # 5. Correctly identify raw input x
    # --------------------------------------------------
    x = None

    for inp in mul.input:
        # Check if this input comes from a Mul with a scalar constant (x * 0.5)
        producer = next((n for n in nodes if inp in n.output), None)

        if producer and producer.op_type == "Mul":
            # Check if one input is a scalar constant
            other = None
            for p_inp in producer.input:
                if p_inp != inp:
                    other = p_inp

            # If other input is a scalar initializer, unwrap
            init = next((i for i in graph.initializer if i.name == other), None)
            if init is not None:
                x = inp if False else producer.input[0] if producer.input[1] == other else producer.input[1]
                break

        # Otherwise, this is raw x
        x = inp
        break

    if x is None:
        raise RuntimeError("Failed to identify raw GELU input tensor")

    # --------------------------------------------------
    # 6. Remove exact GELU subgraph
    # --------------------------------------------------
    for n in [div, erf, add, mul]:
        graph.node.remove(n)

    # --------------------------------------------------
    # 7. Create constants
    # --------------------------------------------------
    def const(name, value):
        return helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[value],
        )

    graph.initializer.extend([
        const("gelu_half", 0.5),
        const("gelu_one", 1.0),
        const("gelu_044715", 0.044715),
        const("gelu_sqrt_2_pi", np.sqrt(2 / np.pi)),
    ])

    # --------------------------------------------------
    # 8. Build tanh-approx GELU
    # --------------------------------------------------
    x2 = f"{x}_sq"
    x3 = f"{x}_cube"

    n1 = helper.make_node("Mul", [x, x], [x2])
    n2 = helper.make_node("Mul", [x2, x], [x3])

    x3_scaled = f"{x}_cube_scaled"
    n3 = helper.make_node("Mul", [x3, "gelu_044715"], [x3_scaled])

    inner = f"{x}_inner"
    n4 = helper.make_node("Add", [x, x3_scaled], [inner])

    scaled = f"{x}_scaled"
    n5 = helper.make_node("Mul", [inner, "gelu_sqrt_2_pi"], [scaled])

    tanh = f"{x}_tanh"
    n6 = helper.make_node("Tanh", [scaled], [tanh])

    tanh_plus = f"{x}_tanh_plus"
    n7 = helper.make_node("Add", ["gelu_one", tanh], [tanh_plus])

    x_half = f"{x}_half"
    n8 = helper.make_node("Mul", [x, "gelu_half"], [x_half])

    n9 = helper.make_node("Mul", [x_half, tanh_plus], [gelu_out])

    graph.node.extend([n1, n2, n3, n4, n5, n6, n7, n8, n9])

    return model
