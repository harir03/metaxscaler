import sys, time, subprocess, os

sys.path.insert(0, ".")
from client import CreditApprovalClient
from models import CreditAction


def test_e2e():
    """End-to-end test: starts the server, runs all 3 tasks, validates responses."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    time.sleep(3)

    try:
        c = CreditApprovalClient("http://127.0.0.1:8765")

        # Test health endpoint
        health = c.health()
        assert health["status"] == "healthy", f"Health check failed: {health}"
        assert "tasks" in health, "Health response missing tasks list"

        for task in ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]:
            r = c.reset(task_name=task)
            assert not r.done, f"Reset should return done=False, got {r.done}"
            assert r.reward == 0.0, f"Reset should return reward=0.0, got {r.reward}"
            assert r.observation is not None, "Reset returned None observation"
            assert r.observation.company is not None, "Missing company in observation"
            assert r.observation.financials is not None, "Missing financials in observation"
            assert r.observation.risk is not None, "Missing risk in observation"
            assert r.observation.market is not None, "Missing market in observation"

            dscr = r.observation.financials.dscr
            wilful = r.observation.risk.wilful_defaulter

            if wilful:
                dec, reason = "reject", "wilful defaulter - automatic reject"
            elif dscr >= 1.5:
                dec, reason = "approve", f"DSCR {dscr} shows good capacity"
            elif dscr >= 1.0:
                dec, reason = "conditional", f"Borderline DSCR {dscr}, needs monitoring"
            else:
                dec, reason = "reject", f"DSCR {dscr} below threshold"

            s = c.step(CreditAction(decision=dec, reasoning=reason, confidence=0.7))
            assert s.done, f"Step should return done=True, got {s.done}"
            assert 0.0 <= s.reward <= 1.0, f"Reward out of range: {s.reward}"
            assert "ground_truth_decision" in s.info, "Missing ground_truth_decision in info"
            print(f"{task}: {dec} -> reward={s.reward:.3f} (gt={s.info.get('ground_truth_decision')})")

        c.close()
        print("\ne2e passed")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_invalid_task():
    """Test that an invalid task name returns an error."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8766"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        import httpx
        r = httpx.post("http://127.0.0.1:8766/reset", json={"task_name": "nonexistent-task"})
        assert r.status_code == 400, f"Expected 400 for invalid task, got {r.status_code}"
        print("invalid_task test passed")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_step_before_reset():
    """Test that stepping before reset returns an error."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8767"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        import httpx
        r = httpx.post("http://127.0.0.1:8767/step", json={"decision": "approve", "reasoning": "test", "confidence": 0.5})
        assert r.status_code == 400, f"Expected 400 for step before reset, got {r.status_code}"
        print("step_before_reset test passed")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_double_step():
    """Test that stepping twice on the same episode returns an error."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8768"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        import httpx
        base = "http://127.0.0.1:8768"

        # Reset
        r = httpx.post(f"{base}/reset", json={"task_name": "credit-approval-easy"})
        assert r.status_code == 200, f"Reset failed: {r.status_code}"

        # First step should succeed
        r = httpx.post(f"{base}/step", json={"decision": "approve", "reasoning": "test", "confidence": 0.5})
        assert r.status_code == 200, f"First step failed: {r.status_code}"

        # Second step should fail (episode already done)
        r = httpx.post(f"{base}/step", json={"decision": "reject", "reasoning": "test2", "confidence": 0.5})
        assert r.status_code == 400, f"Expected 400 for double step, got {r.status_code}"
        print("double_step test passed")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    test_e2e()
    test_invalid_task()
    test_step_before_reset()
    test_double_step()
    print("\nAll tests passed!")
