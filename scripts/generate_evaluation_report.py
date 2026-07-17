import argparse
import json
from pathlib import Path

from evaluation.io import read_csv
from evaluation.reporting import generate_markdown_report


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine automatic and human results into Markdown.")
    parser.add_argument("--automatic-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-summary")
    parser.add_argument("--reader-summary")
    parser.add_argument("--auditor-summary")
    parser.add_argument("--comparisons")
    args = parser.parse_args()
    reader = load_json(args.reader_summary) or {}
    generate_markdown_report(
        read_csv(args.automatic_summary),
        args.output,
        execution_summary=read_csv(args.execution_summary) if args.execution_summary else None,
        human_summary=reader.get("human_summary"),
        agreement=reader.get("agreement"),
        evidence_summary=reader.get("evidence_summary"),
        auditor_summary=load_json(args.auditor_summary),
        comparisons=read_csv(args.comparisons) if args.comparisons else None,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
