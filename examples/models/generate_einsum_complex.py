import torch
import torch.nn as nn


class ComplexEinsumModel(nn.Module):
    def forward(self, x, w):
        # NOT rewriteable by simple MatMul
        return torch.einsum("abc,cd->abd", x, w)


model = ComplexEinsumModel().eval()

x = torch.randn(1, 8, 16)
w = torch.randn(16, 32)

torch.onnx.export(
    model,
    (x, w),
    "einsum_complex.onnx",
    opset_version=17,
    input_names=["X", "W"],
    output_names=["Y"],
)
