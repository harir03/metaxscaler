from typing import Dict, Any


def _norm(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("approve", "approved", "accept", "yes"):
        return "approve"
    if raw in ("reject", "rejected", "deny", "denied", "no"):
        return "reject"
    if raw in ("conditional", "conditionally", "conditional_approve", "conditions"):
        return "conditional"
    return raw


def _match_score(pred, gt):
    p, g = _norm(pred), _norm(gt)
    if p == g:
        return 1.0
    if {p, g} == {"approve", "conditional"}:
        return 0.5  # partial credit
    return 0.0


def _reasoning_score(text, gt_decision):
    if not text or len(text.strip()) < 10:
        return 0.0

    score = 0.0
    low = text.lower()

    # length component
    words = len(text.split())
    score += min(words / 50, 1.0) * 0.3

    # financial terminology
    keywords = [
        "dscr", "debt", "equity", "ratio", "margin", "revenue", "growth",
        "cash flow", "profit", "loss", "risk", "compliance", "rating",
        "npa", "default", "pledge", "audit", "gst", "turnover", "leverage",
        "coverage", "working capital", "collateral", "promoter",
    ]
    hits = sum(1 for kw in keywords if kw in low)
    score += min(hits * 0.05, 0.4)

    # directional alignment
    if gt_decision == "reject":
        neg = ["reject", "deny", "risk", "concern", "flag", "weak", "negative", "default"]
        if any(t in low for t in neg):
            score += 0.3
    elif gt_decision == "approve":
        pos = ["approve", "strong", "healthy", "good", "solid", "positive"]
        if any(t in low for t in pos):
            score += 0.3
    elif gt_decision == "conditional":
        cond = ["conditional", "condition", "monitor", "review", "caution", "mitigat"]
        if any(t in low for t in cond):
            score += 0.3

    return min(score, 1.0)


def _confidence_score(conf, correct, diff):
    if correct:
        if diff == "easy":
            return conf
        elif diff == "medium":
            return 1.0 - abs(conf - 0.6) * 2
        else:
            return 0.5 if conf > 0.8 else min(conf + 0.3, 1.0)
    return max(1.0 - conf, 0.0)


def grade_easy(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    d = _match_score(decision, ground_truth_decision)
    r = _reasoning_score(reasoning, ground_truth_decision)
    c = _confidence_score(confidence, d >= 0.5, "easy")

    total = 0.70 * d + 0.15 * r + 0.15 * c
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {"decision_score": round(d, 4), "reasoning_score": round(r, 4),
                      "confidence_score": round(c, 4)},
        "weights": {"decision": 0.70, "reasoning": 0.15, "confidence": 0.15},
    }


def grade_medium(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    d = _match_score(decision, ground_truth_decision)
    r = _reasoning_score(reasoning, ground_truth_decision)
    c = _confidence_score(confidence, d >= 0.5, "medium")

    total = 0.50 * d + 0.30 * r + 0.20 * c
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {"decision_score": round(d, 4), "reasoning_score": round(r, 4),
                      "confidence_score": round(c, 4)},
        "weights": {"decision": 0.50, "reasoning": 0.30, "confidence": 0.20},
    }


def grade_hard(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    d = _match_score(decision, ground_truth_decision)
    r = _reasoning_score(reasoning, ground_truth_decision)
    c = _confidence_score(confidence, d >= 0.5, "hard")

    fraud_kw = [
        "wilful defaulter", "circular trad", "revenue inflation",
        "evergreening", "related party", "audit qualification",
        "promoter pledge", "nclt", "suspicious", "fraud",
        "inflated", "manipulat", "concealment",
    ]
    low = reasoning.lower()
    fraud_bonus = min(sum(1 for kw in fraud_kw if kw in low) * 0.05, 0.15)

    total = 0.40 * d + 0.40 * r + 0.20 * c + fraud_bonus
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {
            "decision_score": round(d, 4), "reasoning_score": round(r, 4),
            "confidence_score": round(c, 4), "fraud_detection_bonus": round(fraud_bonus, 4),
        },
        "weights": {"decision": 0.40, "reasoning": 0.40, "confidence": 0.20,
                    "fraud_bonus": "up to 0.15"},
    }


def grade(task_name, decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    graders = {
        "credit-approval-easy": grade_easy,
        "credit-approval-medium": grade_medium,
        "credit-approval-hard": grade_hard,
    }
    fn = graders.get(task_name)
    if not fn:
        raise ValueError(f"unknown task: {task_name}")
    return fn(decision, reasoning, confidence, ground_truth_decision, **kw)
