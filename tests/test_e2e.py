import sys, time, subprocess

sys.path.insert(0, ".")
from client import CreditApprovalClient
from models import CreditAction


def test_e2e():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        c = CreditApprovalClient("http://127.0.0.1:8765")
        assert c.health()["status"] == "healthy"

        for task in ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]:
            r = c.reset(task_name=task)
            assert not r.done

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
            assert s.done
            assert 0.0 <= s.reward <= 1.0
            print(f"{task}: {dec} -> reward={s.reward:.3f} (gt={s.info.get('ground_truth_decision')})")

        c.close()
        print("\ne2e passed")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    test_e2e()
