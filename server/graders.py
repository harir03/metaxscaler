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


def _keyword_reasoning_score(text, gt_decision):
    if not text or len(text.strip()) < 10:
        return 0.0

    score = 0.0
    low = text.lower()

    words = len(text.split())
    score += min(words / 50, 1.0) * 0.3

    keywords = [
        "dscr", "debt", "equity", "ratio", "margin", "revenue", "growth",
        "cash flow", "profit", "loss", "risk", "compliance", "rating",
        "npa", "default", "pledge", "audit", "gst", "turnover", "leverage",
        "coverage", "working capital", "collateral", "promoter",
    ]
    hits = sum(1 for kw in keywords if kw in low)
    score += min(hits * 0.05, 0.4)

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


def _structural_reasoning_score(text, obs_dict):
    """Check if the reasoning cites actual values from the observation."""
    if not text or not obs_dict:
        return 0.0

    score = 0.0
    fi = obs_dict.get("financials", {})
    ri = obs_dict.get("risk", {})

    # does the reasoning cite specific numeric values from the observation?
    value_hits = 0
    for key in ["dscr", "current_ratio", "debt_equity_ratio", "net_profit_margin",
                "revenue_growth_yoy", "interest_coverage_ratio"]:
        val = fi.get(key)
        if val is None:
            continue
        # check both 1-decimal and 2-decimal representations
        patterns = [f"{val:.1f}", f"{val:.2f}", str(round(val, 1))]
        if any(p in text for p in patterns):
            value_hits += 1

    score += min(value_hits * 0.08, 0.4)

    # contradiction detection — penalize wrong direction claims
    low = text.lower()
    dscr = fi.get("dscr", 0)
    if dscr < 1.0 and any(w in low for w in ["strong dscr", "healthy dscr", "excellent dscr"]):
        score -= 0.15
    if dscr > 2.0 and any(w in low for w in ["weak dscr", "poor dscr", "inadequate dscr"]):
        score -= 0.15

    npm = fi.get("net_profit_margin", 0)
    if npm < 0 and any(w in low for w in ["profitable", "strong margin", "healthy margin"]):
        score -= 0.15
    if npm > 10 and any(w in low for w in ["unprofitable", "negative margin", "loss-making"]):
        score -= 0.15

    # key risk factor identification
    if ri.get("wilful_defaulter") and "wilful" in low:
        score += 0.25
    if ri.get("nclt_active") and "nclt" in low:
        score += 0.15
    if ri.get("related_party_transactions_flagged") and any(w in low for w in ["rpt", "related party"]):
        score += 0.1

    # does reasoning mention the credit rating?
    rating = ri.get("credit_rating", "")
    if rating and rating.lower() != "not disclosed" and rating.lower() in low:
        score += 0.1

    return max(min(score, 1.0), 0.0)


def _blended_reasoning(text, gt_decision, obs_dict, difficulty):
    kw = _keyword_reasoning_score(text, gt_decision)
    st = _structural_reasoning_score(text, obs_dict) if obs_dict else 0.0

    if difficulty == "easy":
        return 0.7 * kw + 0.3 * st
    elif difficulty == "medium":
        return 0.5 * kw + 0.5 * st
    else:
        return 0.3 * kw + 0.7 * st


def _confidence_score(conf, correct, diff):
    if correct:
        if diff == "easy":
            return conf
        elif diff == "medium":
            return 1.0 - abs(conf - 0.6) * 2
        else:
            return 0.5 if conf > 0.8 else min(conf + 0.3, 1.0)
    return max(1.0 - conf, 0.0)


def _info_bonus(steps_taken):
    # reward agents that gathered info before deciding (max +0.10)
    if steps_taken <= 1:
        return 0.0
    return min((steps_taken - 1) * 0.05, 0.10)


def grade_easy(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    obs = kw.get("observation", {})
    steps = kw.get("steps_taken", 1)

    d = _match_score(decision, ground_truth_decision)
    r = _blended_reasoning(reasoning, ground_truth_decision, obs, "easy")
    c = _confidence_score(confidence, d >= 0.5, "easy")
    ib = _info_bonus(steps)

    total = 0.70 * d + 0.15 * r + 0.15 * c + ib
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {"decision_score": round(d, 4), "reasoning_score": round(r, 4),
                      "confidence_score": round(c, 4), "info_bonus": round(ib, 4)},
        "weights": {"decision": 0.70, "reasoning": 0.15, "confidence": 0.15,
                    "info_gathering": "up to 0.10"},
    }


def grade_medium(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    obs = kw.get("observation", {})
    steps = kw.get("steps_taken", 1)

    d = _match_score(decision, ground_truth_decision)
    r = _blended_reasoning(reasoning, ground_truth_decision, obs, "medium")
    c = _confidence_score(confidence, d >= 0.5, "medium")
    ib = _info_bonus(steps)

    total = 0.50 * d + 0.30 * r + 0.20 * c + ib
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {"decision_score": round(d, 4), "reasoning_score": round(r, 4),
                      "confidence_score": round(c, 4), "info_bonus": round(ib, 4)},
        "weights": {"decision": 0.50, "reasoning": 0.30, "confidence": 0.20,
                    "info_gathering": "up to 0.10"},
    }


def grade_hard(decision, reasoning, confidence, ground_truth_decision, **kw) -> Dict[str, Any]:
    obs = kw.get("observation", {})
    steps = kw.get("steps_taken", 1)

    d = _match_score(decision, ground_truth_decision)
    r = _blended_reasoning(reasoning, ground_truth_decision, obs, "hard")
    c = _confidence_score(confidence, d >= 0.5, "hard")
    ib = _info_bonus(steps)

    fraud_kw = [
        "wilful defaulter", "circular trad", "revenue inflation",
        "evergreening", "related party", "audit qualification",
        "promoter pledge", "nclt", "suspicious", "fraud",
        "inflated", "manipulat", "concealment",
    ]
    low = reasoning.lower()
    fraud_bonus = min(sum(1 for kw in fraud_kw if kw in low) * 0.05, 0.15)

    total = 0.40 * d + 0.40 * r + 0.20 * c + fraud_bonus + ib
    return {
        "score": round(min(max(total, 0.01), 0.99), 4),
        "breakdown": {
            "decision_score": round(d, 4), "reasoning_score": round(r, 4),
            "confidence_score": round(c, 4), "fraud_detection_bonus": round(fraud_bonus, 4),
            "info_bonus": round(ib, 4),
        },
        "weights": {"decision": 0.40, "reasoning": 0.40, "confidence": 0.20,
                    "fraud_bonus": "up to 0.15", "info_gathering": "up to 0.10"},
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
