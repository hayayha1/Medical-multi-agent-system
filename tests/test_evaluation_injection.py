from evaluation.error_injection import ERROR_TYPES, aggregate_auditor_results, inject_text


def test_error_injectors_cover_rexval_categories():
    text = "There is a mild left basilar opacity. Compared with prior, it is stable."
    for error_type in ERROR_TYPES:
        changed = inject_text(text, error_type)
        assert changed is not None
        assert changed != text


def test_auditor_aggregation():
    summary = aggregate_auditor_results([
        {"is_error": True, "flagged": True, "error_type": "omission"},
        {"is_error": True, "flagged": False, "error_type": "omission"},
        {"is_error": False, "flagged": False, "error_type": "no_error"},
        {"is_error": False, "flagged": True, "error_type": "no_error"},
    ])
    assert summary["recall_sensitivity"] == 0.5
    assert summary["specificity"] == 0.5

