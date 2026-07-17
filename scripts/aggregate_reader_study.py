import argparse
import json
from pathlib import Path

from evaluation.evidence import aggregate_evidence_annotations
from evaluation.human import aggregate_reader_annotations, inter_rater_agreement
from evaluation.io import read_csv


def apply_blinding_key(rows, key):
    for row in rows:
        mapping = key.get(str(row.get("blind_report_id")), {})
        if mapping:
            row["method_id"] = mapping["method_id"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate reader-study and citation-review results.")
    parser.add_argument("--reader-form", required=True)
    parser.add_argument("--blinding-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-form")
    args = parser.parse_args()
    key = json.loads(Path(args.blinding_key).read_text(encoding="utf-8"))
    reader_rows = apply_blinding_key(read_csv(args.reader_form), key)
    result = {
        "human_summary": aggregate_reader_annotations(reader_rows),
        "agreement": inter_rater_agreement(reader_rows),
    }
    if args.evidence_form:
        evidence_rows = apply_blinding_key(read_csv(args.evidence_form), key)
        result["evidence_summary"] = aggregate_evidence_annotations(evidence_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

