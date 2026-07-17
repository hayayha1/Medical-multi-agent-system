import pytest

from evaluation.basic_metrics import rouge_l_f1, sentence_bleu4, token_f1
from evaluation.statistics import benjamini_hochberg, bootstrap_mean_ci, paired_comparison


def test_basic_metrics_reward_identical_text():
    text = "No focal consolidation or pleural effusion."
    assert token_f1(text, text) == 1.0
    assert rouge_l_f1(text, text) == 1.0
    assert sentence_bleu4(text, text) == pytest.approx(1.0)


def test_basic_metrics_penalize_negation_change():
    reference = "No pleural effusion."
    hypothesis = "Pleural effusion."
    assert token_f1(reference, hypothesis) < 1.0
    assert rouge_l_f1(reference, hypothesis) < 1.0


def test_bootstrap_and_paired_statistics_are_deterministic():
    result = bootstrap_mean_ci([1, 2, 3], iterations=100, seed=7)
    assert result["mean"] == 2
    comparison = paired_comparison([1, 1, 1], [2, 2, 2], iterations=100, seed=7)
    assert comparison["mean_difference"] == 1
    adjusted = benjamini_hochberg([0.01, 0.04, 0.2])
    assert adjusted == sorted(adjusted)

