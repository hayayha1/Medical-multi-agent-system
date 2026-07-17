from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from evaluation.models import EvaluationCase


NORMAL_MARKERS = (
    "normal",
    "no acute",
    "no finding",
    "clear lungs",
    "negative",
)


def deterministic_split(case_id: str, development_percent: int = 20) -> str:
    bucket = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 100
    return "development" if bucket < development_percent else "test"


def infer_stratum(problems: str, findings: str, impression: str) -> str:
    combined = " ".join((problems, findings, impression)).strip().lower()
    if not combined or any(marker in combined for marker in NORMAL_MARKERS):
        abnormal_terms = (
            "opacity", "effusion", "edema", "atelect", "pneumothorax",
            "cardiomeg", "consolid", "fracture", "nodule", "mass",
        )
        if not any(term in combined for term in abnormal_terms):
            return "normal"
    problem_count = len([value for value in re.split(r"[,;|/]", problems) if value.strip()])
    return "multi_abnormal" if problem_count >= 2 else "abnormal"


def load_huggingface_cases(
    parquet_dir: str | Path,
    image_root: str | Path,
    development_percent: int = 20,
    include_empty_references: bool = False,
) -> Iterable[EvaluationCase]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read IU X-Ray Parquet files. "
            "Install: pip install -e '.[evaluation]'"
        ) from exc
    dataset = ds.dataset(str(parquet_dir), format="parquet")
    columns = ["uid", "MeSH", "Problems", "indication", "comparison", "findings", "impression"]
    root = Path(image_root)
    for batch in dataset.to_batches(columns=columns, batch_size=256):
        for row in batch.to_pylist():
            uid = str(row.get("uid", "")).strip()
            case_id = uid if uid.lower().startswith("cxr") else f"CXR{uid}"
            findings = str(row.get("findings") or "").strip()
            impression = str(row.get("impression") or "").strip()
            if not include_empty_references and not findings and not impression:
                continue
            problems = str(row.get("Problems") or "").strip()
            images = sorted(root.glob(f"{case_id}_*.png"))
            yield EvaluationCase(
                case_id=case_id,
                split=deterministic_split(case_id, development_percent),
                stratum=infer_stratum(problems, findings, impression),
                image_paths=[str(path.relative_to(root.parent)).replace("\\", "/") for path in images],
                reference_findings=findings,
                reference_impression=impression,
                indication=str(row.get("indication") or "").strip(),
                comparison=str(row.get("comparison") or "").strip(),
                problems=problems,
                mesh=str(row.get("MeSH") or "").strip(),
                metadata={"uid": uid, "image_count": len(images)},
            )


def select_reader_cases(
    cases: list[EvaluationCase],
    sample_size: int,
    seed: int = 20260714,
) -> list[EvaluationCase]:
    import random

    rng = random.Random(seed)
    grouped: dict[str, list[EvaluationCase]] = {"normal": [], "abnormal": [], "multi_abnormal": []}
    for case in cases:
        grouped.setdefault(case.stratum, []).append(case)
    for items in grouped.values():
        rng.shuffle(items)
    targets = {
        "normal": round(sample_size * 0.4),
        "abnormal": round(sample_size * 0.4),
    }
    targets["multi_abnormal"] = sample_size - targets["normal"] - targets["abnormal"]
    selected = [case for name, count in targets.items() for case in grouped.get(name, [])[:count]]
    if len(selected) < sample_size:
        used = {case.case_id for case in selected}
        remaining = [case for case in cases if case.case_id not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[:sample_size - len(selected)])
    rng.shuffle(selected)
    return selected
