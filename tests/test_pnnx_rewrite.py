from trt_fixer.prevent.rewrite_engine import apply_onnx_rewrite
import onnx
import os


def test_einsum_to_matmul_rewrite(tmp_path):
    out = tmp_path / "rewritten.onnx"

    ok = apply_onnx_rewrite(
        "examples/models/einsum_fail.onnx",
        str(out)
    )

    assert ok is True
    assert os.path.exists(out)

    model = onnx.load(str(out))
    ops = [n.op_type for n in model.graph.node]

    assert "Einsum" not in ops
    assert "MatMul" in ops
