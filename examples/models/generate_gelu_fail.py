import torch
import torch.nn as nn
import torch.nn.functional as F


class GeluModel(nn.Module):
    def forward(self, x):
        return F.gelu(x)  # exact GELU


model = GeluModel().eval()
x = torch.randn(1, 128)

torch.onnx.export(
    model,
    x,
    "gelu_fail.onnx",
    opset_version=17,
    input_names=["X"],
    output_names=["Y"],
)
