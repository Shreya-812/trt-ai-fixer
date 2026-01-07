# trt_fixer/detector/support_db.py

import json
from pathlib import Path
from typing import Dict, List


class TensorRTSupportDB:
    def __init__(self, trt_version: str):
        self.trt_version = trt_version

        self.native_ops = {
            "Conv", "MatMul", "Gemm", "Relu", "Add", "Mul",
            "Sub", "Div", "Transpose", "Reshape", "Softmax"
        }

        # 🔥 rewrite-supported ops
        self.rewrite_supported_ops = {
            "Einsum",
            "Gelu",
        }

        self.plugin_required_ops = {
            "LayerNormalization",
            "InstanceNormalization",
            "GroupNormalization",
        }

        self.blocked_ops = {
            "Loop", "If", "Scan"
        }
