from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "NA" if value != value else f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def generate_markdown_report(
    automatic_summary: list[dict[str, Any]],
    output_path: str | Path,
    execution_summary: list[dict[str, Any]] | None = None,
    human_summary: dict[str, Any] | None = None,
    agreement: dict[str, Any] | None = None,
    evidence_summary: dict[str, Any] | None = None,
    auditor_summary: dict[str, Any] | None = None,
    comparisons: list[dict[str, Any]] | None = None,
) -> None:
    sections = [
        "# 医疗影像报告生成评价结果",
        "",
        "## 自动指标",
        "",
        markdown_table(automatic_summary, ["method_id", "section", "metric", "n", "mean", "ci_low", "ci_high", "higher_is_better"]),
    ]
    if execution_summary:
        sections.extend(["", "## 运行效率与失败率", "", markdown_table(execution_summary, ["method_id", "n_total", "n_success", "n_failed", "failure_rate", "mean_latency_seconds", "p95_latency_seconds"])])
    if comparisons:
        sections.extend(["", "## 配对方法比较", "", markdown_table(comparisons, ["baseline_method", "candidate_method", "section", "metric", "n", "mean_difference", "ci_low", "ci_high", "p_value"])])
    if human_summary:
        rows = [{"method_id": key, **value} for key, value in human_summary.items()]
        sections.extend(["", "## 医生盲评", "", markdown_table(rows, ["method_id", "n_cases", "significant_errors_per_report", "total_errors_per_report", "no_significant_error_rate", "error_free_rate", "mean_usable_without_edit", "mean_edit_seconds"])])
    if agreement:
        sections.extend(["", "## 评价者一致性", "", "```json", json.dumps(agreement, ensure_ascii=False, indent=2), "```"])
    if evidence_summary:
        rows = [{"method_id": key, **value} for key, value in evidence_summary.items()]
        sections.extend(["", "## 证据引用评价", "", markdown_table(rows, ["method_id", "n_citations", "citation_support_precision", "full_support_rate", "source_appropriateness_rate", "unsupported_required_claim_rate"])])
    if auditor_summary:
        sections.extend(["", "## 审计智能体挑战集", "", "```json", json.dumps(auditor_summary, ensure_ascii=False, indent=2), "```"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sections) + "\n", encoding="utf-8")
