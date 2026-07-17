from __future__ import annotations

from copy import deepcopy
import hashlib
import random
import re
from typing import Any, Callable

from evaluation.models import EvaluationRecord


ERROR_TYPES = (
    "false_prediction",
    "omission",
    "incorrect_location",
    "incorrect_severity",
    "spurious_comparison",
    "omitted_comparison",
)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if part.strip()]


def _false_prediction(text: str) -> str | None:
    candidates = (
        ("pneumothorax", "There is a small right apical pneumothorax."),
        ("pleural effusion", "There is a moderate left pleural effusion."),
        ("pulmonary edema", "There is moderate pulmonary edema."),
        ("focal consolidation", "There is focal right lower lobe consolidation."),
    )
    lowered = text.lower()
    for keyword, sentence in candidates:
        if keyword not in lowered:
            return (text.rstrip() + " " + sentence).strip()
    return None


def _omission(text: str) -> str | None:
    sentences = _sentences(text)
    if len(sentences) < 2:
        return None
    negative_markers = ("no ", "without", "normal", "unremarkable", "clear")
    for index, sentence in enumerate(sentences):
        if not any(marker in sentence.lower() for marker in negative_markers):
            remaining = sentences[:index] + sentences[index + 1:]
            return " ".join(remaining)
    return " ".join(sentences[1:])


def _swap_location(text: str) -> str | None:
    replacements = (
        (r"\bleft\b", "right"),
        (r"\bright\b", "left"),
        (r"\bupper\b", "lower"),
        (r"\blower\b", "upper"),
        ("左", "右"),
        ("右", "左"),
        ("上叶", "下叶"),
        ("下叶", "上叶"),
    )
    for pattern, replacement in replacements:
        changed, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count:
            return changed
    return None


def _swap_severity(text: str) -> str | None:
    replacements = (
        (r"\bmild\b", "severe"),
        (r"\bminimal\b", "marked"),
        (r"\bsmall\b", "large"),
        (r"\bmoderate\b", "severe"),
        (r"\bsevere\b", "mild"),
        ("轻度", "重度"),
        ("少量", "大量"),
        ("重度", "轻度"),
    )
    for pattern, replacement in replacements:
        changed, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count:
            return changed
    return None


def _spurious_comparison(text: str) -> str:
    return (text.rstrip() + " Compared with the prior examination, the findings have improved.").strip()


def _omit_comparison(text: str) -> str | None:
    markers = ("compared", "comparison", "interval", "previous", "prior", "improved", "worsened", "stable")
    sentences = _sentences(text)
    remaining = [sentence for sentence in sentences if not any(marker in sentence.lower() for marker in markers)]
    return " ".join(remaining) if len(remaining) < len(sentences) and remaining else None


INJECTORS: dict[str, Callable[[str], str | None]] = {
    "false_prediction": _false_prediction,
    "omission": _omission,
    "incorrect_location": _swap_location,
    "incorrect_severity": _swap_severity,
    "spurious_comparison": _spurious_comparison,
    "omitted_comparison": _omit_comparison,
}


def inject_text(text: str, error_type: str) -> str | None:
    if error_type not in INJECTORS:
        raise ValueError(f"Unsupported error type: {error_type}")
    return INJECTORS[error_type](text)


def _mutate_state(state: dict[str, Any], error_type: str) -> dict[str, Any] | None:
    mutated = deepcopy(state)
    draft = mutated.get("report_draft") or {}
    for section in ("impression", "findings"):
        items = draft.get(section) or []
        combined = " ".join(str(item.get("text", "")) for item in items).strip()
        changed = inject_text(combined, error_type) if combined else None
        if changed and changed != combined:
            if items:
                items[0]["text"] = changed
                del items[1:]
            return mutated
    return None


def build_audit_challenges(
    records: list[EvaluationRecord],
    per_error_type: int = 20,
    controls: int = 100,
    seed: int = 20260714,
) -> list[dict[str, Any]]:
    candidates = [record for record in records if record.success and record.state.get("report_draft")]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    output: list[dict[str, Any]] = []
    for error_type in ERROR_TYPES:
        created = 0
        for record in candidates:
            mutated = _mutate_state(record.state, error_type)
            if mutated is None:
                continue
            digest = hashlib.sha256(f"{record.case_id}:{error_type}:{created}".encode()).hexdigest()[:12]
            output.append({
                "challenge_id": f"AUD-{digest}",
                "case_id": record.case_id,
                "method_id": record.method_id,
                "error_type": error_type,
                "is_error": True,
                "state": mutated,
            })
            created += 1
            if created >= per_error_type:
                break
    for index, record in enumerate(candidates[:controls]):
        digest = hashlib.sha256(f"{record.case_id}:control:{index}".encode()).hexdigest()[:12]
        output.append({
            "challenge_id": f"AUD-{digest}",
            "case_id": record.case_id,
            "method_id": record.method_id,
            "error_type": "no_error",
            "is_error": False,
            "state": deepcopy(record.state),
        })
    rng.shuffle(output)
    return output


def aggregate_auditor_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(bool(row.get("is_error")) and bool(row.get("flagged")) for row in rows)
    fn = sum(bool(row.get("is_error")) and not bool(row.get("flagged")) for row in rows)
    fp = sum(not bool(row.get("is_error")) and bool(row.get("flagged")) for row in rows)
    tn = sum(not bool(row.get("is_error")) and not bool(row.get("flagged")) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    by_type: dict[str, dict[str, float | int]] = {}
    for error_type in ERROR_TYPES:
        items = [row for row in rows if row.get("error_type") == error_type]
        detected = sum(bool(row.get("flagged")) for row in items)
        by_type[error_type] = {
            "n": len(items),
            "detected": detected,
            "recall": detected / len(items) if items else 0.0,
        }
    return {
        "n": len(rows),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "by_error_type": by_type,
    }

