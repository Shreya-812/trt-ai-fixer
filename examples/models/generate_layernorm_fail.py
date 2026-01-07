import torch
import torch.nn as nn

class LNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(128)

    def forward(self, x):
        return self.ln(x)

model = LNModel().eval()
x = torch.randn(1, 16, 128)

torch.onnx.export(
    model,
    x,
    "layernorm_fail.onnx",
    opset_version=17,
    input_names=["X"],
    output_names=["Y"]
)
