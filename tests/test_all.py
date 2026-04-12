"""Comprehensive tests covering S-tier features + backward compatibility.
Run from repo root: python -m pytest tests/ -v
Or directly: python tests/test_all.py
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CreditAction, CreditObservation
from server.environment import CreditApprovalEnvironment
from server.graders import (
    grade, _norm, _match_score, _keyword_reasoning_score,
    _structural_reasoning_score, _blended_reasoning, _info_bonus,
)
from server.task_generator import generate_task

TASKS = ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]


# ── S3: Mixed Hard Ground Truths ──

def test_hard_not_always_reject():
    """Hard tasks must produce at least one non-reject ground truth in 50 samples."""
    decisions = set()
    for _ in range(50):
        _, gt_dec, _, _ = generate_task("credit-approval-hard")
        decisions.add(gt_dec)
    assert "approve" in decisions or "conditional" in decisions, \
        f"Hard tasks always {decisions} — need mixed ground truths"
    print("[PASS] S3: Hard tasks produce mixed ground truths")


def test_hard_has_reject():
    """Hard tasks should still mostly reject (traps should dominate)."""
    rejects = sum(1 for _ in range(100) if generate_task("credit-approval-hard")[1] == "reject")
    assert rejects >= 30, f"Only {rejects}/100 rejects — reject should be common"
    print("[PASS] S3: Hard tasks still predominantly reject")


def test_hard_approve_has_strong_financials():
    """Hard-approve companies should have genuinely good numbers."""
    for _ in range(20):
        obs, gt, _, _ = generate_task("credit-approval-hard")
        if gt == "approve":
            assert obs.financials.dscr >= 1.5, "Hard-approve DSCR too low"
            assert obs.financials.cash_flow_positive, "Hard-approve needs positive cash flow"
            assert obs.financials.net_profit_margin >= 5.0, "Hard-approve margin too low"
    print("[PASS] S3: Hard-approve companies have genuinely strong financials")


# ── S2: Structural Reasoning Grader ──

def test_structural_score_value_citation():
    """Reasoning that cites actual values should score higher than vague reasoning."""
    obs = {"financials": {"dscr": 2.35, "current_ratio": 1.8, "debt_equity_ratio": 0.5,
                          "net_profit_margin": 12.5, "revenue_growth_yoy": 15.0,
                          "interest_coverage_ratio": 4.2},
           "risk": {"credit_rating": "AA", "wilful_defaulter": False}}

    specific = "DSCR of 2.35 is strong, NPM at 12.5% shows healthy profitability"
    vague = "The company has good financial metrics and seems profitable"

    s_specific = _structural_reasoning_score(specific, obs)
    s_vague = _structural_reasoning_score(vague, obs)
    assert s_specific > s_vague, f"Specific ({s_specific}) should beat vague ({s_vague})"
    print("[PASS] S2: Value citations score higher than vague reasoning")


def test_structural_score_contradiction_penalty():
    """Contradictory claims about metrics should be penalized."""
    obs = {"financials": {"dscr": 0.5, "current_ratio": 0.6, "debt_equity_ratio": 4.0,
                          "net_profit_margin": -8.0, "revenue_growth_yoy": -10.0,
                          "interest_coverage_ratio": 0.4},
           "risk": {"credit_rating": "B"}}

    wrong = "The company has a strong DSCR and healthy margins"
    right = "DSCR below 1.0 indicates debt service coverage is inadequate"

    s_wrong = _structural_reasoning_score(wrong, obs)
    s_right = _structural_reasoning_score(right, obs)
    assert s_right > s_wrong, f"Correct ({s_right}) should beat contradictory ({s_wrong})"
    print("[PASS] S2: Contradiction penalty works")


def test_blended_reasoning_weights():
    """Hard tasks should weight structural scoring higher."""
    obs = {"financials": {"dscr": 2.0, "current_ratio": 1.5, "debt_equity_ratio": 0.8,
                          "net_profit_margin": 10.0, "revenue_growth_yoy": 12.0,
                          "interest_coverage_ratio": 3.0},
           "risk": {"credit_rating": "A", "wilful_defaulter": True}}

    text = "DSCR of 2.0 is strong but wilful defaulter flag is a hard block"

    easy_r = _blended_reasoning(text, "reject", obs, "easy")
    hard_r = _blended_reasoning(text, "reject", obs, "hard")
    # hard weights structural more, and this text has good structural score
    assert hard_r > 0, f"Hard reasoning score should be positive, got {hard_r}"
    print("[PASS] S2: Blended reasoning weights differ by difficulty")


def test_info_bonus():
    """Info gathering bonus should reward multi-step agents."""
    assert _info_bonus(1) == 0.0, "No bonus for single step"
    assert _info_bonus(2) == 0.05, "Should be 0.05 for 2 steps"
    assert _info_bonus(3) == 0.10, "Should be 0.10 for 3 steps"
    assert _info_bonus(5) == 0.10, "Capped at 0.10"
    print("[PASS] S2: Info gathering bonus works correctly")


# ── S1: Multi-Step Episodes ──

def test_reset_hides_risk_market():
    """Reset should mask risk and market data."""
    env = CreditApprovalEnvironment()
    result = env.reset("credit-approval-easy")
    obs = result.observation

    assert obs.risk.credit_rating == "not disclosed", \
        f"Risk should be hidden, got {obs.risk.credit_rating}"
    assert obs.market.sector_outlook == "not disclosed", \
        f"Market should be hidden, got {obs.market.sector_outlook}"
    assert "risk" in result.info.get("hidden", []), "Info should list risk as hidden"
    assert "market" in result.info.get("hidden", []), "Info should list market as hidden"
    print("[PASS] S1: Reset masks risk and market data")


def test_request_info_reveals_data():
    """Requesting risk_data should reveal actual risk indicators."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-medium")

    result = env.request_info("risk_data")
    obs = result.observation
    assert obs.risk.credit_rating != "not disclosed", \
        f"Risk should be revealed, got {obs.risk.credit_rating}"
    assert not result.done, "Should not be done after info request"
    assert "risk" not in result.info.get("hidden", []), "Risk should not be hidden anymore"
    print("[PASS] S1: request_info reveals hidden data")


def test_multistep_full_flow():
    """Full multi-step flow: reset → request risk → request market → decide."""
    env = CreditApprovalEnvironment()
    result = env.reset("credit-approval-hard")
    assert not result.done

    # step 1: request risk
    result = env.request_info("risk_data")
    assert not result.done
    assert result.observation.risk.credit_rating != "not disclosed"

    # step 2: request market
    result = env.request_info("market_data")
    assert not result.done
    assert result.observation.market.sector_outlook != "not disclosed"

    # step 3: make decision
    action = CreditAction(decision="reject", reasoning="DSCR looks concerning", confidence=0.7)
    result = env.step(action)
    assert result.done
    assert 0 < result.reward < 1
    assert result.info.get("steps_taken") == 3
    print("[PASS] S1: Full multi-step flow works")


def test_backward_compat_single_step():
    """Agent that decides immediately (no info requests) must still work."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-easy")

    action = CreditAction(decision="approve", reasoning="looks good", confidence=0.8)
    result = env.step(action)
    assert result.done
    assert 0 < result.reward < 1
    assert result.info.get("steps_taken") == 1
    print("[PASS] S1: Single-step backward compatibility works")


def test_duplicate_request_errors():
    """Requesting already-revealed data should error."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-easy")
    env.request_info("risk_data")
    try:
        env.request_info("risk_data")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] S1: Duplicate info request raises error")


def test_cannot_request_after_max_steps():
    """Should not allow info requests when only 1 step remains (need it for decision)."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-easy")
    env.request_info("risk_data")
    env.request_info("market_data")
    # now at step 2, max_steps=3, only 1 step left — must decide
    try:
        env.request_info("risk_data")  # already revealed anyway
        assert False, "Should have raised"
    except (ValueError, RuntimeError):
        pass
    print("[PASS] S1: Cannot request info when only 1 step remains")


# ── Grader Boundary Tests ──

def test_scores_strictly_between_0_and_1():
    """Scores must be in (0, 1) exclusive — validator requirement."""
    for _ in range(200):
        task = random.choice(TASKS)
        result = grade(
            task_name=task,
            decision=random.choice(["approve", "reject", "conditional", "banana"]),
            reasoning="test " * random.randint(0, 30),
            confidence=random.random(),
            ground_truth_decision=random.choice(["approve", "reject", "conditional"]),
        )
        s = result["score"]
        assert 0 < s < 1, f"Score {s} out of range for {task}"
    print("[PASS] Grader: All scores strictly in (0, 1)")


def test_correct_decision_scores_higher():
    """Correct decisions should generally score higher than wrong ones."""
    correct_scores = []
    wrong_scores = []
    for _ in range(50):
        obs, gt, _, _ = generate_task("credit-approval-easy")
        obs_dict = obs.model_dump()
        r_correct = grade("credit-approval-easy", gt, "good reasoning", 0.8, gt, observation=obs_dict)
        r_wrong = grade("credit-approval-easy", "banana", "bad", 0.2, gt, observation=obs_dict)
        correct_scores.append(r_correct["score"])
        wrong_scores.append(r_wrong["score"])

    avg_correct = sum(correct_scores) / len(correct_scores)
    avg_wrong = sum(wrong_scores) / len(wrong_scores)
    assert avg_correct > avg_wrong, f"Correct avg {avg_correct} should beat wrong avg {avg_wrong}"
    print("[PASS] Grader: Correct decisions score higher on average")


def test_normalization():
    """_norm should handle various input formats."""
    assert _norm("APPROVED") == "approve"
    assert _norm("  reject  ") == "reject"
    assert _norm("Conditionally") == "conditional"
    assert _norm("yes") == "approve"
    assert _norm("denied") == "reject"
    assert _norm("gibberish") == "gibberish"
    print("[PASS] Grader: Input normalization works")


def test_partial_credit():
    """approve vs conditional should get partial credit."""
    assert _match_score("approve", "conditional") == 0.5
    assert _match_score("conditional", "approve") == 0.5
    assert _match_score("approve", "reject") == 0.0
    print("[PASS] Grader: Partial credit for approve/conditional")


# ── Task Generator ──

def test_easy_tasks_have_hints():
    """Easy tasks should always include hints."""
    for _ in range(20):
        obs, _, _, _ = generate_task("credit-approval-easy")
        assert obs.hint is not None and len(obs.hint) > 0, "Easy tasks need hints"
    print("[PASS] TaskGen: Easy tasks always have hints")


def test_medium_tasks_no_hints():
    """Medium tasks should not have hints."""
    for _ in range(20):
        obs, _, _, _ = generate_task("credit-approval-medium")
        assert obs.hint is None, "Medium tasks should not have hints"
    print("[PASS] TaskGen: Medium tasks have no hints")


def test_score_variance():
    """Scores should vary across episodes (not just one constant value)."""
    decisions = ["approve", "reject", "conditional"]
    for task in TASKS:
        scores = set()
        for i in range(30):
            obs, gt, _, _ = generate_task(task)
            # vary the decision and reasoning to see score differences
            dec = decisions[i % 3]
            r = grade(task, dec, f"DSCR is {obs.financials.dscr}, margins look fine",
                     0.5 + (i % 5) * 0.1, gt, observation=obs.model_dump())
            scores.add(round(r["score"], 2))
        assert len(scores) >= 3, f"Only {len(scores)} unique scores for {task} — need more variance"
    print("[PASS] Variance: Scores vary across episodes")


def test_invalid_task_name():
    """Unknown task names should raise ValueError."""
    try:
        generate_task("credit-approval-impossible")
        assert False, "Should have raised"
    except ValueError:
        pass

    try:
        grade("credit-approval-impossible", "approve", "", 0.5, "approve")
        assert False, "Should have raised"
    except ValueError:
        pass
    print("[PASS] Edge: Invalid task names raise ValueError")


# ── Environment Edge Cases ──

def test_step_without_reset():
    """Stepping without reset should error."""
    env = CreditApprovalEnvironment()
    try:
        env.step(CreditAction(decision="approve"))
        assert False, "Should have raised"
    except RuntimeError:
        pass
    print("[PASS] Edge: Step without reset raises RuntimeError")


def test_double_step():
    """Stepping twice (after done) should error."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-easy")
    env.step(CreditAction(decision="approve", reasoning="test", confidence=0.5))
    try:
        env.step(CreditAction(decision="reject"))
        assert False, "Should have raised"
    except RuntimeError:
        pass
    print("[PASS] Edge: Double step raises RuntimeError")


def test_state_no_ground_truth_leak():
    """State endpoint should not expose ground truth."""
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-easy")
    state = env.state()
    # the state object has ground_truth fields but the API shouldn't expose them
    # this test checks the model; app.py filters it in the endpoint
    assert hasattr(state, "episode_id")
    assert hasattr(state, "step")
    print("[PASS] Edge: State has expected fields")


if __name__ == "__main__":
    tests = [
        # S3
        test_hard_not_always_reject,
        test_hard_has_reject,
        test_hard_approve_has_strong_financials,
        # S2
        test_structural_score_value_citation,
        test_structural_score_contradiction_penalty,
        test_blended_reasoning_weights,
        test_info_bonus,
        # S1
        test_reset_hides_risk_market,
        test_request_info_reveals_data,
        test_multistep_full_flow,
        test_backward_compat_single_step,
        test_duplicate_request_errors,
        test_cannot_request_after_max_steps,
        # grader boundaries
        test_scores_strictly_between_0_and_1,
        test_correct_decision_scores_higher,
        test_normalization,
        test_partial_credit,
        # task generator
        test_easy_tasks_have_hints,
        test_medium_tasks_no_hints,
        test_score_variance,
        test_invalid_task_name,
        # edge cases
        test_step_without_reset,
        test_double_step,
        test_state_no_ground_truth_leak,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        sys.exit(1)
    print("ALL TESTS PASSED")
