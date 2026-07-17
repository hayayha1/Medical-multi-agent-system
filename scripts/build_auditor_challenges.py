import argparse

from evaluation.error_injection import build_audit_challenges
from evaluation.io import read_models, write_jsonl
from evaluation.models import EvaluationRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ReXVal-style synthetic auditor challenges.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-error-type", type=int, default=20)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    records = read_models(args.records, EvaluationRecord)
    challenges = build_audit_challenges(records, args.per_error_type, args.controls, args.seed)
    write_jsonl(args.output, challenges)
    print(f"wrote {len(challenges)} challenges to {args.output}")


if __name__ == "__main__":
    main()

