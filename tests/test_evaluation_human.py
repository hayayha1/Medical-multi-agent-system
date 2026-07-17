import pytest

from evaluation.human import aggregate_reader_annotations, cohen_kappa, icc_2_1, inter_rater_agreement


def annotation(rater, significant=0, completeness=5):
    return {
        "case_id": "CXR1",
        "method_id": "full",
        "rater_id": rater,
        "false_prediction_significant": significant,
        "false_prediction_insignificant": 0,
        "completeness": completeness,
        "clarity": 5,
        "usable_without_edit": int(significant == 0),
        "edit_seconds": 10,
    }


def test_human_summary_and_agreement():
    rows = [annotation("A"), annotation("B")]
    summary = aggregate_reader_annotations(rows)["full"]
    assert summary["no_significant_error_rate"] == 1.0
    agreement = inter_rater_agreement(rows)
    assert agreement["n_double_rated"] == 1
    assert agreement["kappa_no_significant_error"] == 1.0


def test_agreement_helpers():
    assert cohen_kappa([0, 1, 1], [0, 1, 1]) == 1.0
    assert icc_2_1([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)

