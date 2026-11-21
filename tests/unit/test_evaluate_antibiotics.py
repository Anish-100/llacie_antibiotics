"""Unit tests for antibiotics label evaluation (llacie/evaluate_antibiotics.py)."""
import pandas as pd
import pytest

from llacie.evaluate_antibiotics import evaluate_antibiotic_labels, match_courses

pytestmark = pytest.mark.unit


def labels(rows):
    """antibiotic_labels-like rows from (episode, drug_name, drug_name_raw, start, relevance)."""
    return pd.DataFrame([{
        "id": i, "FK_episode_id": ep, "drug_name": name, "drug_name_raw": raw,
        "start_date": pd.Timestamp(start).date() if start else None, "end_date": None,
        "duration_days": None, "is_current": None, "relevance": relevance,
    } for i, (ep, name, raw, start, relevance) in enumerate(rows)])


def test_repeat_courses_pair_by_start_date():
    truth = labels([(1, "cephalexin", "cephalexin", "2023-11-20", "relevant"),
        (1, "cephalexin", "cephalexin", "2023-10-01", "not_relevant")])
    pred = labels([(1, "cephalexin", "Keflex", "2023-10-02", "not_relevant"),
        (1, "cephalexin", "cephalexin", "2023-11-20", "relevant")])
    courses = match_courses(truth, pred)
    assert list(courses["found"]) == ["both", "both"]
    assert [str(d) for d in courses["start_date_true"]] == ["2023-10-01", "2023-11-20"]
    assert [str(d) for d in courses["start_date_pred"]] == ["2023-10-02", "2023-11-20"]


def test_missed_extra_and_unlisted_drugs():
    truth = labels([(1, None, "Colistin", None, "not_relevant"),
        (1, "amoxicillin", "amoxicillin", None, "relevant")])
    pred = labels([(1, None, "colistin", None, "not_relevant"),
        (1, "doxycycline", "doxycycline", None, "relevant")])
    found = match_courses(truth, pred).set_index("drug_key")["found"].to_dict()
    # Drugs not in drugs.db still match on their normalized raw name
    assert found == {"colistin": "both", "amoxicillin": "true_only", "doxycycline": "pred_only"}


def test_report_counts_unlabeled_episodes_as_nothing_found():
    truth = labels([(1, "amoxicillin", "amoxicillin", None, "relevant"),
        (2, None, None, None, "not_relevant")])  # episode 2: no antibiotics
    pred = labels([])
    pred = pred.reindex(columns=truth.columns)
    report, courses = evaluate_antibiotic_labels(truth, pred, [1, 2], n_resamples=20)
    assert "1 labeled, 0 predicted, 0 found, 1 missed, 0 extra" in report
    assert list(courses["found"]) == ["true_only"]
