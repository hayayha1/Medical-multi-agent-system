"""Inspect the extracted IU X-Ray layout without modifying the dataset."""

from collections import Counter
from pathlib import Path

from app.config import get_settings


def main() -> None:
    root = Path(get_settings().iu_xray_dataset_path)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    counts = Counter(
        path.suffix.lower() or "<no-extension>"
        for path in root.rglob("*") if path.is_file()
    )
    print(f"root: {root}")
    print("file types:")
    for suffix, count in counts.most_common(20):
        print(f"  {suffix}: {count}")
    print("sample images:")
    images = [
        path for path in root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    for path in images[:10]:
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()

