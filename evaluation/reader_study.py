from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from evaluation.dataset import select_reader_cases
from evaluation.evidence import evidence_annotation_rows
from evaluation.human import ERROR_CATEGORIES
from evaluation.io import write_csv
from evaluation.models import EvaluationCase, EvaluationRecord


def _blind_id(case_id: str, method_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{case_id}:{method_id}".encode()).hexdigest()[:12]
    return f"RPT-{digest.upper()}"


def create_reader_study(
    records: list[EvaluationRecord],
    output_dir: str | Path,
    sample_size: int = 200,
    rater_ids: list[str] | None = None,
    seed: int = 20260714,
) -> dict[str, Any]:
    raters = rater_ids or ["RATER-1", "RATER-2"]
    successful = [record for record in records if record.success]
    methods = sorted({record.method_id for record in successful})
    by_key = {(record.case_id, record.method_id): record for record in successful}
    common_ids = sorted({record.case_id for record in successful if all((record.case_id, method) in by_key for method in methods)})
    case_models = []
    first_by_case = {record.case_id: record for record in successful}
    for case_id in common_ids:
        record = first_by_case[case_id]
        case_models.append(EvaluationCase(
            case_id=case_id,
            split=record.split,
            stratum=record.stratum,
            image_paths=record.image_paths,
            reference_findings=record.reference_findings,
            reference_impression=record.reference_impression,
        ))
    selected = select_reader_cases(case_models, min(sample_size, len(case_models)), seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    salt = hashlib.sha256(str(seed).encode()).hexdigest()
    key: dict[str, dict[str, str]] = {}
    form_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for case in selected:
        candidates = [by_key[(case.case_id, method)] for method in methods]
        rng.shuffle(candidates)
        for candidate in candidates:
            blind_id = _blind_id(case.case_id, candidate.method_id, salt)
            key[blind_id] = {"case_id": case.case_id, "method_id": candidate.method_id}
            base = {
                "blind_report_id": blind_id,
                "case_id": case.case_id,
                "stratum": case.stratum,
                "image_paths": json.dumps(case.image_paths, ensure_ascii=False),
                "reference_report": candidate.reference_report,
                "candidate_report": candidate.candidate_report,
            }
            for rater_id in raters:
                row = {"rater_id": rater_id, **base}
                for category in ERROR_CATEGORIES:
                    row[f"{category}_significant"] = ""
                    row[f"{category}_insignificant"] = ""
                row.update({
                    "usable_without_edit": "",
                    "completeness": "",
                    "clarity": "",
                    "edit_seconds": "",
                    "notes": "",
                })
                form_rows.append(row)
            evidence_rows.extend(evidence_annotation_rows(candidate, blind_id))
    write_csv(output / "reader_study_form.csv", form_rows)
    write_csv(output / "evidence_review_form.csv", evidence_rows)
    (output / "blinding_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(READER_INSTRUCTIONS, encoding="utf-8")
    return {
        "cases": len(selected),
        "methods": methods,
        "raters": raters,
        "reader_rows": len(form_rows),
        "evidence_rows": len(evidence_rows),
    }


READER_INSTRUCTIONS = """# 放射科医生盲评说明

1. 评价者不得查看 `blinding_key.json`；该文件仅由统计人员保管。
2. 结合胸片、参考报告和候选报告，填写六类错误数量。
3. 每类错误分别记录临床显著与临床不显著错误；没有错误填写 `0`，不要留空。
4. `usable_without_edit` 填 `1` 或 `0`；完整性和清晰度按 1–5 分填写。
5. `edit_seconds` 记录将候选报告修改到可签发状态的时间。
6. 两名医生独立完成；不得在首次评分前讨论病例。

六类错误：虚假发现、遗漏发现、位置/侧别错误、严重程度错误、虚构比较、遗漏比较。
"""

