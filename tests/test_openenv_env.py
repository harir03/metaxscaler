import sys
sys.path.insert(0, ".")

from models import CreditAction
from server.environment import CreditApprovalEnvironment


def test_easy():
    env = CreditApprovalEnvironment()
    r = env.reset("credit-approval-easy")
    print(f"easy: {r.observation.company.name} (diff={r.observation.difficulty})")

    action = CreditAction(decision="approve", reasoning="Strong DSCR and clean record", confidence=0.9)
    s = env.step(action)
    print(f"  reward={s.reward:.4f} gt={s.info.get('ground_truth_decision')}")
    assert 0.0 <= s.reward <= 1.0
    assert s.done is True


def test_medium():
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-medium")
    s = env.step(CreditAction(
        decision="conditional",
        reasoning="Mixed signals, acceptable DSCR but RPT flagged. Conditional approval with quarterly review.",
        confidence=0.6,
    ))
    print(f"medium: reward={s.reward:.4f} gt={s.info.get('ground_truth_decision')}")
    assert 0.0 <= s.reward <= 1.0


def test_hard():
    env = CreditApprovalEnvironment()
    env.reset("credit-approval-hard")
    s = env.step(CreditAction(
        decision="reject",
        reasoning="Despite strong financials, wilful defaulter flag detected. Revenue inflation pattern with "
                  "negative cash flow. Related party transactions flagged with audit qualifications. Reject.",
        confidence=0.7,
    ))
    print(f"hard: reward={s.reward:.4f} gt={s.info.get('ground_truth_decision')}")
    assert 0.0 <= s.reward <= 1.0


def test_all_tasks():
    for task in ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]:
        env = CreditApprovalEnvironment()
        r = env.reset(task)
        assert r.observation.task_name == task
        assert r.done is False
    print("all task names ok")


if __name__ == "__main__":
    test_easy()
    test_medium()
    test_hard()
    test_all_tasks()
    print("\nall tests passed")
