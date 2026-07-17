import argparse
import json

from evaluation.io import read_models
from evaluation.models import EvaluationRecord
from evaluation.reader_study import create_reader_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Create blinded radiologist and evidence-review forms.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--raters", default="RATER-1,RATER-2")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    records = read_models(args.records, EvaluationRecord)
    result = create_reader_study(
        records,
        args.output_dir,
        args.sample_size,
        [item.strip() for item in args.raters.split(",") if item.strip()],
        args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

