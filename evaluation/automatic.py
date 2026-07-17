from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import importlib
from statistics import mean
import sys
from typing import Any

from evaluation.basic_metrics import per_case_basic
from evaluation.models import EvaluationRecord
from evaluation.statistics import bootstrap_mean_ci, paired_comparison, percentile


LOWER_IS_BETTER = {
    "radcliq", "fineradscore", "error_count", "green_error_count", "latency_seconds",
}


def _section_text(record: EvaluationRecord, section: str, reference: bool) -> str:
    prefix = "reference" if reference else "candidate"
    if section == "findings":
        return getattr(record, f"{prefix}_findings")
    if section == "impression":
        return getattr(record, f"{prefix}_impression")
    return record.reference_report if reference else record.candidate_report


@lru_cache(maxsize=16)
def _radeval_instance(metrics: tuple[str, ...]) -> Any:
    try:
        radeval_package = importlib.import_module("radeval")
        # RadEval 2.2.1 vendors RadGraph with case-sensitive imports from
        # ``RadEval.*``.  Linux installs the distribution as ``radeval``;
        # registering the package alias keeps the vendor code portable without
        # modifying site-packages.
        sys.modules.setdefault("RadEval", radeval_package)
        RadEval = radeval_package.RadEval
    except ImportError as exc:
        raise RuntimeError(
            "RadEval is not installed. Install the evaluation extra: pip install -e '.[evaluation]'"
        ) from exc
    return RadEval(metrics=list(metrics), per_sample=True)


def run_radeval(
    references: list[str],
    hypotheses: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    evaluator = _radeval_instance(tuple(metrics))
    return evaluator(refs=references, hyps=hypotheses)


def evaluate_records(
    records: list[EvaluationRecord],
    external_metrics: list[str] | None = None,
    sections: tuple[str, ...] = ("findings", "impression", "combined"),
    bootstrap_iterations: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = [record for record in records if record.success]
    per_case: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for record in successful:
        for section in sections:
            grouped[(record.method_id, section)].append(record)

    for (method_id, section), items in grouped.items():
        references = [_section_text(item, section, True) for item in items]
        hypotheses = [_section_text(item, section, False) for item in items]
        rows = []
        for item, reference, hypothesis in zip(items, references, hypotheses, strict=True):
            rows.append({
                "case_id": item.case_id,
                "method_id": method_id,
                "section": section,
                "split": item.split,
                "stratum": item.stratum,
                "latency_seconds": item.latency_seconds,
                **per_case_basic(reference, hypothesis),
            })
        if external_metrics:
            external = run_radeval(references, hypotheses, external_metrics)
            for metric_name, values in external.items():
                if isinstance(values, (list, tuple)) and len(values) == len(rows):
                    for row, value in zip(rows, values, strict=True):
                        row[metric_name] = float(value)
                elif hasattr(values, "tolist"):
                    converted = values.tolist()
                    if isinstance(converted, list) and len(converted) == len(rows):
                        for row, value in zip(rows, converted, strict=True):
                            row[metric_name] = float(value)
        per_case.extend(rows)

    summary: list[dict[str, Any]] = []
    numeric_by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    reserved = {"case_id", "method_id", "section", "split", "stratum", "latency_seconds"}
    for row in per_case:
        group = numeric_by_group[(str(row["method_id"]), str(row["section"]))]
        for key, value in row.items():
            if key not in reserved and isinstance(value, (int, float)) and value is not None:
                group[key].append(float(value))
    for (method_id, section), metrics in numeric_by_group.items():
        for metric_name, values in metrics.items():
            stats = bootstrap_mean_ci(values, iterations=bootstrap_iterations)
            summary.append({
                "method_id": method_id,
                "section": section,
                "metric": metric_name,
                "higher_is_better": not any(
                    token in metric_name.lower() for token in LOWER_IS_BETTER
                ),
                **stats,
            })
    return per_case, summary


def summarize_subgroups(
    per_case: list[dict[str, Any]],
    bootstrap_iterations: int = 1000,
) -> list[dict[str, Any]]:
    reserved = {"case_id", "method_id", "section", "split", "stratum", "latency_seconds"}
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_case:
        target = grouped[(str(row["method_id"]), str(row["section"]), str(row["stratum"]))]
        for key, value in row.items():
            if key not in reserved and isinstance(value, (int, float)) and value is not None:
                target[key].append(float(value))
    output = []
    for (method_id, section, stratum), metrics in grouped.items():
        for metric_name, values in metrics.items():
            output.append({
                "method_id": method_id,
                "section": section,
                "stratum": stratum,
                "metric": metric_name,
                "higher_is_better": not any(token in metric_name.lower() for token in LOWER_IS_BETTER),
                **bootstrap_mean_ci(values, iterations=bootstrap_iterations),
            })
    return output


def summarize_execution(records: list[EvaluationRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.method_id].append(record)
    output = []
    for method_id, items in sorted(grouped.items()):
        successful = [item for item in items if item.success]
        latencies = [float(item.latency_seconds) for item in successful if item.latency_seconds is not None]
        output.append({
            "method_id": method_id,
            "n_total": len(items),
            "n_success": len(successful),
            "n_failed": len(items) - len(successful),
            "failure_rate": (len(items) - len(successful)) / len(items) if items else 0.0,
            "mean_latency_seconds": mean(latencies) if latencies else float("nan"),
            "p95_latency_seconds": percentile(latencies, 0.95) if latencies else float("nan"),
        })
    return output


def compare_methods(
    per_case: list[dict[str, Any]],
    baseline_method: str,
    candidate_method: str,
    iterations: int = 5000,
) -> list[dict[str, Any]]:
    reserved = {"case_id", "method_id", "section", "split", "stratum"}
    indexed = {
        (str(row["method_id"]), str(row["section"]), str(row["case_id"])): row
        for row in per_case
    }
    sections = sorted({str(row["section"]) for row in per_case})
    output: list[dict[str, Any]] = []
    for section in sections:
        cases = sorted({
            case_id for method, row_section, case_id in indexed
            if method == baseline_method and row_section == section
            and (candidate_method, section, case_id) in indexed
        })
        if not cases:
            continue
        metric_names = sorted(
            key for key, value in indexed[(baseline_method, section, cases[0])].items()
            if key not in reserved and isinstance(value, (int, float))
        )
        for metric_name in metric_names:
            baseline = [float(indexed[(baseline_method, section, case)][metric_name]) for case in cases]
            candidate = [float(indexed[(candidate_method, section, case)][metric_name]) for case in cases]
            comparison = paired_comparison(baseline, candidate, iterations=iterations)
            output.append({
                "baseline_method": baseline_method,
                "candidate_method": candidate_method,
                "section": section,
                "metric": metric_name,
                "higher_is_better": not any(token in metric_name.lower() for token in LOWER_IS_BETTER),
                "baseline_mean": mean(baseline),
                "candidate_mean": mean(candidate),
                **comparison,
            })
    return output
