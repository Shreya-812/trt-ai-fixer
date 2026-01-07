from trt_fixer.prevent.einsum_rules import analyze_einsum


def test_simple_matmul_einsum():
    result = analyze_einsum("ij,jk->ik")
    assert result["rewriteable"] is True
    assert result["strategy"] == "matmul"


def test_batched_matmul_einsum():
    result = analyze_einsum("bij,bjk->bik")
    assert result["rewriteable"] is True
    assert result["strategy"] == "batched_matmul"


def test_complex_einsum_rejected():
    result = analyze_einsum("abc,cd->abd")
    assert result["rewriteable"] is False
