#!/usr/bin/env python3
"""Print a machine-readable summary of the evaluation runtime."""

from __future__ import annotations

import importlib.metadata
import json
import platform


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    import torch

    payload = {
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("radeval", "torch", "torchvision", "transformers", "pyarrow")
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
