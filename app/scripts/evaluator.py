import json
from typing import Dict, Any, List, Optional
from app.scripts.config import TEST_CASES_PATH

# Weights for unified composite triage accuracy
WEIGHT_CATEGORY = 0.40
WEIGHT_ESCALATION = 0.30
WEIGHT_PRIORITY = 0.20
WEIGHT_SENTIMENT = 0.10


def load_ground_truths() -> Dict[int, Dict[str, Any]]:
    """Loads ground truths mapped by test case ID from test_cases.json."""
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {c["id"]: c for c in cases}


def score_single_prediction(prediction: Optional[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scores a single model prediction against ground truth.
    Returns composite score (0.0 to 1.0) and individual matches.
    """
    if not prediction:
        return {
            "score": 0.0,
            "category_match": False,
            "escalation_match": False,
            "priority_match": False,
            "sentiment_match": False,
        }

    cat_pred = str(prediction.get("category", "")).strip().lower()
    cat_true = str(ground_truth.get("category", "")).strip().lower()
    category_match = cat_pred == cat_true

    esc_pred = bool(prediction.get("requires_human", False))
    esc_true = bool(ground_truth.get("expected_requires_human", False))
    escalation_match = esc_pred == esc_true

    prio_pred = str(prediction.get("priority", "")).strip().lower()
    prio_true = str(ground_truth.get("expected_priority", "")).strip().lower()
    priority_match = prio_pred == prio_true

    sent_pred = str(prediction.get("sentiment", "")).strip().lower()
    sent_true = str(ground_truth.get("expected_sentiment", "")).strip().lower()
    sentiment_match = sent_pred == sent_true

    composite_score = (
        (WEIGHT_CATEGORY if category_match else 0.0)
        + (WEIGHT_ESCALATION if escalation_match else 0.0)
        + (WEIGHT_PRIORITY if priority_match else 0.0)
        + (WEIGHT_SENTIMENT if sentiment_match else 0.0)
    )

    return {
        "score": composite_score,
        "category_match": category_match,
        "escalation_match": escalation_match,
        "priority_match": priority_match,
        "sentiment_match": sentiment_match,
    }


def evaluate_report_accuracy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a full list of benchmark records against ground truth test cases.
    Returns single unified composite accuracy percentage and sub-component breakdown.
    """
    ground_truths = load_ground_truths()
    total_cases = len(records)
    if total_cases == 0:
        return {
            "overall_accuracy_pct": 0.0,
            "category_accuracy_pct": 0.0,
            "escalation_accuracy_pct": 0.0,
            "priority_accuracy_pct": 0.0,
            "sentiment_accuracy_pct": 0.0,
        }

    total_score = 0.0
    cat_matches = 0
    esc_matches = 0
    prio_matches = 0
    sent_matches = 0

    for rec in records:
        q_id = rec.get("question_id")
        gt = ground_truths.get(q_id, {})
        pred = rec.get("output") if rec.get("validation_passed") else None

        result = score_single_prediction(pred, gt)
        total_score += result["score"]
        if result["category_match"]:
            cat_matches += 1
        if result["escalation_match"]:
            esc_matches += 1
        if result["priority_match"]:
            prio_matches += 1
        if result["sentiment_match"]:
            sent_matches += 1

    overall_accuracy = (total_score / total_cases) * 100

    return {
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "category_accuracy_pct": round((cat_matches / total_cases) * 100, 2),
        "escalation_accuracy_pct": round((esc_matches / total_cases) * 100, 2),
        "priority_accuracy_pct": round((prio_matches / total_cases) * 100, 2),
        "sentiment_accuracy_pct": round((sent_matches / total_cases) * 100, 2),
    }
