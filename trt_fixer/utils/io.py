# trt_fixer/utils/io.py

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)
