import argparse
import json
from pathlib import Path

from evaluation.automatic import (
    compare_methods,
    evaluate_records,
    summarize_execution,
    summarize_subgroups,
)
from evaluation.evidence import structural_evidence_metrics
from evaluation.io import read_models, write_csv
from evaluation.models import EvaluationRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute report-generation metrics and bootstrap CIs.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", default="bleu,rouge,bertscore,f1chexbert,radgraph,radcliq")
    parser.add_argument(
        "--sections",
        default="findings,impression,combined",
        help="Comma-separated subset of findings, impression, combined.",
    )
    parser.add_argument("--include-green", action="store_true")
    parser.add_argument("--baseline-method")
    parser.add_argument("--candidate-method")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    metrics = [] if args.metrics.lower() in {"", "none"} else [item.strip() for item in args.metrics.split(",") if item.strip()]
    if args.include_green and "green" not in metrics:
        metrics.append("green")
    sections = tuple(item.strip() for item in args.sections.split(",") if item.strip())
    invalid_sections = sorted(set(sections) - {"findings", "impression", "combined"})
    if not sections or invalid_sections:
        parser.error(f"invalid --sections value: {','.join(invalid_sections) or '(empty)'}")
    records = read_models(args.records, EvaluationRecord)
    per_case, summary = evaluate_records(
        records,
        metrics,
        sections=sections,
        bootstrap_iterations=args.bootstrap,
    )
    output = Path(args.output_dir)
    write_csv(output / "automatic_per_case.csv", per_case)
    write_csv(output / "automatic_summary.csv", summary)
    write_csv(output / "automatic_subgroup_summary.csv", summarize_subgroups(per_case, args.bootstrap))
    write_csv(output / "execution_summary.csv", summarize_execution(records))
    evidence = [structural_evidence_metrics(record) for record in records if record.success]
    write_csv(output / "evidence_structural_per_case.csv", evidence)
    comparisons = []
    if args.baseline_method and args.candidate_method:
        comparisons = compare_methods(per_case, args.baseline_method, args.candidate_method)
        write_csv(output / "paired_comparisons.csv", comparisons)
    (output / "automatic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"evaluated {len(records)} records; outputs: {output}")


if __name__ == "__main__":
    main()
