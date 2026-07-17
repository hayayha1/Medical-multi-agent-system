import argparse
from collections import Counter
import json

from evaluation.io import read_models
from evaluation.models import EvaluationCase


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an evaluation manifest before inference.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    cases = read_models(args.manifest, EvaluationCase)
    identifiers = Counter(case.case_id for case in cases)
    duplicate_ids = sorted(case_id for case_id, count in identifiers.items() if count > 1)
    missing_images = sorted(case.case_id for case in cases if not case.image_paths)
    empty_references = sorted(case.case_id for case in cases if not case.reference_report.strip())
    unsafe_paths = sorted({
        path for case in cases for path in case.image_paths
        if path.startswith(("/", "\\")) or ".." in path.replace("\\", "/").split("/")
    })
    summary = {
        "cases": len(cases),
        "unique_cases": len(identifiers),
        "split_counts": dict(Counter(case.split for case in cases)),
        "stratum_counts": dict(Counter(case.stratum for case in cases)),
        "duplicate_case_ids": duplicate_ids,
        "missing_image_cases": missing_images,
        "empty_reference_cases": empty_references,
        "unsafe_image_paths": unsafe_paths,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if duplicate_ids or missing_images or empty_references or unsafe_paths:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

