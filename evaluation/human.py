from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean
from typing import Any


ERROR_CATEGORIES = (
    "false_prediction",
    "omission",
    "incorrect_location",
    "incorrect_severity",
    "spurious_comparison",
    "omitted_comparison",
)


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def enrich_annotation(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    significant = sum(_number(row.get(f"{name}_significant")) for name in ERROR_CATEGORIES)
    insignificant = sum(_number(row.get(f"{name}_insignificant")) for name in ERROR_CATEGORIES)
    enriched["significant_errors"] = significant
    enriched["insignificant_errors"] = insignificant
    enriched["total_errors"] = significant + insignificant
    enriched["no_significant_error"] = float(significant == 0)
    enriched["error_free"] = float(significant + insignificant == 0)
    return enriched


def aggregate_reader_annotations(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        method = str(raw.get("method_id") or raw.get("blind_report_id") or "unknown")
        grouped[method].append(enrich_annotation(raw))
    summaries: dict[str, dict[str, float | int]] = {}
    for method, items in grouped.items():
        result: dict[str, float | int] = {
            "n_ratings": len(items),
            "n_cases": len({str(item.get("case_id")) for item in items}),
            "significant_errors_per_report": mean(item["significant_errors"] for item in items),
            "insignificant_errors_per_report": mean(item["insignificant_errors"] for item in items),
            "total_errors_per_report": mean(item["total_errors"] for item in items),
            "no_significant_error_rate": mean(item["no_significant_error"] for item in items),
            "error_free_rate": mean(item["error_free"] for item in items),
        }
        for field in ("usable_without_edit", "completeness", "clarity", "edit_seconds"):
            values = [_number(item.get(field), float("nan")) for item in items]
            finite = [value for value in values if math.isfinite(value)]
            result[f"mean_{field}"] = mean(finite) if finite else float("nan")
        for category in ERROR_CATEGORIES:
            result[f"{category}_significant_per_report"] = mean(
                _number(item.get(f"{category}_significant")) for item in items
            )
            result[f"{category}_insignificant_per_report"] = mean(
                _number(item.get(f"{category}_insignificant")) for item in items
            )
        summaries[method] = result
    return summaries


def cohen_kappa(left: list[int], right: list[int], weights: str | None = None) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    labels = sorted(set(left) | set(right))
    index = {label: position for position, label in enumerate(labels)}
    size = len(labels)
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(left, right, strict=True):
        observed[index[a]][index[b]] += 1 / len(left)
    row = [sum(values) for values in observed]
    column = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    if weights == "linear" and size > 1:
        weight = [[abs(i - j) / (size - 1) for j in range(size)] for i in range(size)]
    else:
        weight = [[float(i != j) for j in range(size)] for i in range(size)]
    numerator = sum(weight[i][j] * observed[i][j] for i in range(size) for j in range(size))
    denominator = sum(weight[i][j] * row[i] * column[j] for i in range(size) for j in range(size))
    return 1 - numerator / denominator if denominator else 1.0


def icc_2_1(matrix: list[list[float]]) -> float:
    """Two-way random, absolute-agreement, single-measure ICC(2,1)."""
    if len(matrix) < 2 or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        return float("nan")
    n = len(matrix)
    k = len(matrix[0])
    if k < 2:
        return float("nan")
    grand = mean(value for row in matrix for value in row)
    row_means = [mean(row) for row in matrix]
    column_means = [mean(matrix[i][j] for i in range(n)) for j in range(k)]
    ms_rows = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    ms_columns = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    residual = sum(
        (matrix[i][j] - row_means[i] - column_means[j] + grand) ** 2
        for i in range(n) for j in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return (ms_rows - ms_error) / denominator if denominator else float("nan")


def inter_rater_agreement(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("method_id") or row.get("blind_report_id")))
        grouped[key].append(enrich_annotation(row))
    complete = [sorted(items, key=lambda item: str(item.get("rater_id"))) for items in grouped.values() if len(items) >= 2]
    if not complete:
        return {"n_double_rated": 0, "icc_significant_errors": float("nan"), "kappa_no_significant_error": float("nan"), "weighted_kappa_completeness": float("nan"), "weighted_kappa_clarity": float("nan")}
    matrices = [[float(item["significant_errors"]) for item in pair[:2]] for pair in complete]
    no_error_left = [int(pair[0]["no_significant_error"]) for pair in complete]
    no_error_right = [int(pair[1]["no_significant_error"]) for pair in complete]
    completeness_left = [int(_number(pair[0].get("completeness"))) for pair in complete]
    completeness_right = [int(_number(pair[1].get("completeness"))) for pair in complete]
    clarity_left = [int(_number(pair[0].get("clarity"))) for pair in complete]
    clarity_right = [int(_number(pair[1].get("clarity"))) for pair in complete]
    return {
        "n_double_rated": len(complete),
        "icc_significant_errors": icc_2_1(matrices),
        "kappa_no_significant_error": cohen_kappa(no_error_left, no_error_right),
        "weighted_kappa_completeness": cohen_kappa(completeness_left, completeness_right, "linear"),
        "weighted_kappa_clarity": cohen_kappa(clarity_left, clarity_right, "linear"),
    }

