import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        # Einsum
        x = torch.einsum("ij,jk->ik", x, self.weight)

        # GELU (exact, Erf-based)
        x = torch.nn.functional.gelu(x)

        return x


if __name__ == "__main__":
    model = CombinedModel().eval()
    dummy = torch.randn(4, 16)

    torch.onnx.export(
        model,
        dummy,
        "combined_einsum_gelu.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
