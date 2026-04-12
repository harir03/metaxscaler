"""E2E test: spins up uvicorn, hits all endpoints, validates multi-step flow.
Run: python tests/test_e2e.py
"""
import sys
import os
import time
import subprocess
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"

TASKS = ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]


def _start_server():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{BASE}/health", timeout=2.0)
            if r.status_code == 200:
                return proc
        except httpx.ConnectError:
            pass
    proc.terminate()
    raise RuntimeError("server did not start")


def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[PASS] health endpoint returns ok")


def test_root():
    r = httpx.get(f"{BASE}/")
    d = r.json()
    assert d["version"] == "1.1.0", f"version mismatch: {d['version']}"
    assert len(d["tasks"]) == 3
    print("[PASS] root endpoint returns correct metadata")


def test_single_step_all_tasks():
    for task in TASKS:
        r = httpx.post(f"{BASE}/reset", json={"task_name": task})
        assert r.status_code == 200
        d = r.json()
        assert not d["done"]
        assert d["reward"] == 0.0

        r = httpx.post(f"{BASE}/step", json={
            "decision": "approve", "reasoning": "test reasoning", "confidence": 0.6,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["done"]
        assert 0 < d["reward"] < 1, f"reward {d['reward']} out of (0,1)"
        # ground truth must NOT leak
        assert "ground_truth_decision" not in d.get("info", {}), \
            "ground truth should not be in step info"
        print(f"[PASS] {task}: single-step reward={d['reward']:.3f}")


def test_multi_step():
    r = httpx.post(f"{BASE}/reset", json={"task_name": "credit-approval-hard"})
    d = r.json()
    assert "risk" in d["info"]["hidden"], "risk should be hidden on reset"
    assert "market" in d["info"]["hidden"], "market should be hidden on reset"
    obs = d["observation"]
    assert obs["risk"]["credit_rating"] == "not disclosed"

    # request risk
    r = httpx.post(f"{BASE}/step", json={"request": "risk_data"})
    d = r.json()
    assert not d["done"]
    assert d["observation"]["risk"]["credit_rating"] != "not disclosed"
    assert "risk" not in d["info"].get("hidden", [])

    # request market
    r = httpx.post(f"{BASE}/step", json={"request": "market_data"})
    d = r.json()
    assert not d["done"]
    assert d["observation"]["market"]["sector_outlook"] != "not disclosed"

    # decide
    r = httpx.post(f"{BASE}/step", json={
        "decision": "reject", "reasoning": "DSCR is weak, high pledge", "confidence": 0.8,
    })
    d = r.json()
    assert d["done"]
    assert 0 < d["reward"] < 1
    assert d["info"]["steps_taken"] == 3
    print(f"[PASS] multi-step: reward={d['reward']:.3f}, steps=3")


def test_invalid_task():
    r = httpx.post(f"{BASE}/reset", json={"task_name": "nonexistent"})
    assert r.status_code == 400
    print("[PASS] invalid task returns 400")


def test_step_before_reset():
    # reset first to clear state, then try double step
    httpx.post(f"{BASE}/reset", json={"task_name": "credit-approval-easy"})
    httpx.post(f"{BASE}/step", json={"decision": "approve", "reasoning": "t", "confidence": 0.5})
    # now episode is done — stepping again should fail
    r = httpx.post(f"{BASE}/step", json={"decision": "reject", "reasoning": "t", "confidence": 0.5})
    assert r.status_code == 400
    print("[PASS] step after done returns 400")


def test_no_decision_no_request():
    httpx.post(f"{BASE}/reset", json={"task_name": "credit-approval-easy"})
    r = httpx.post(f"{BASE}/step", json={"reasoning": "just vibing"})
    assert r.status_code == 400
    print("[PASS] step with neither decision nor request returns 400")


def test_masked_gst_not_misleading():
    r = httpx.post(f"{BASE}/reset", json={"task_name": "credit-approval-easy"})
    obs = r.json()["observation"]
    risk = obs["risk"]
    # masked risk should not show alarming values
    assert risk["credit_rating"] == "not disclosed"
    # gst should be neutral (not 0.0 which signals terrible compliance)
    assert risk["gst_compliance_pct"] != 0.0, \
        f"masked gst_compliance_pct should not be 0.0 (misleading), got {risk['gst_compliance_pct']}"
    print(f"[PASS] masked risk uses neutral placeholders (gst={risk['gst_compliance_pct']})")


if __name__ == "__main__":
    proc = _start_server()
    print(f"=== E2E Tests (server on :{PORT}) ===")
    tests = [
        test_health, test_root, test_single_step_all_tasks,
        test_multi_step, test_invalid_task, test_step_before_reset,
        test_no_decision_no_request, test_masked_gst_not_misleading,
    ]
    passed, failed = 0, 0
    try:
        for t in tests:
            try:
                t()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {t.__name__}: {e}")
                failed += 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("ALL E2E TESTS PASSED")
