import torch
import torch.nn as nn


class EinsumFailModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        # Batched einsum: bij,bjk->bik
        return torch.einsum("bij,bjk->bik", x, w)


def main():
    model = EinsumFailModel()
    model.eval()

    # Fixed input shapes
    x = torch.randn(1, 16, 32)
    w = torch.randn(1, 32, 64)

    torch.onnx.export(
        model,
        (x, w),
        "einsum_fail.onnx",
        opset_version=17,
        input_names=["X", "W"],
        output_names=["Y"],
        dynamic_axes=None
    )

    print("Generated einsum_fail.onnx")


if __name__ == "__main__":
    main()
