import argparse

from evaluation.dataset import load_huggingface_cases
from evaluation.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-safe IU X-Ray evaluation manifest.")
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclusions-output")
    parser.add_argument("--development-percent", type=int, default=20)
    args = parser.parse_args()
    all_cases = list(load_huggingface_cases(
        args.parquet_dir,
        args.image_root,
        args.development_percent,
        include_empty_references=True,
    ))
    excluded = [
        {"case_id": case.case_id, "reason": "empty_reference_report"}
        for case in all_cases if not case.reference_report.strip()
    ]
    cases = [case for case in all_cases if case.reference_report.strip()]
    write_jsonl(args.output, cases)
    exclusions_output = args.exclusions_output or f"{args.output}.excluded.jsonl"
    write_jsonl(exclusions_output, excluded)
    print(f"wrote {len(cases)} cases to {args.output}; excluded {len(excluded)} to {exclusions_output}")


if __name__ == "__main__":
    main()
