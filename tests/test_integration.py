"""Quick integration test for env_client.py + Docker container."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_client import DockerEnvClient


async def test():
    print("=== Integration Test: DockerEnvClient ===")

    # 1. Start container
    print("[1] Starting container from credit-approval-env...")
    env = await DockerEnvClient.from_docker_image(
        "credit-approval-env", container_port=7860
    )
    print("[1] PASS: Container started and ready")

    # 2. Test reset
    print("[2] Testing /reset with credit-approval-easy...")
    result = await env.reset(task_name="credit-approval-easy")
    assert result.done is False, f"Expected done=False, got {result.done}"
    assert isinstance(result.observation, dict), "Observation should be dict"
    obs_keys = list(result.observation.keys())
    print(f"[2] PASS: Reset OK, obs keys={obs_keys}")

    # 3. Test step
    print("[3] Testing /step with approve action...")
    action = {
        "decision": "approve",
        "reasoning": "Strong DSCR and clean record",
        "confidence": 0.85,
    }
    result = await env.step(action)
    assert result.done is True, f"Expected done=True, got {result.done}"
    assert 0.0 <= result.reward <= 1.0, f"Reward out of range: {result.reward}"
    print(f"[3] PASS: Step OK, reward={result.reward:.3f}, done={result.done}")

    # 4. Test all 3 tasks
    for task in [
        "credit-approval-easy",
        "credit-approval-medium",
        "credit-approval-hard",
    ]:
        r = await env.reset(task_name=task)
        assert r.done is False
        r = await env.step(action)
        assert r.done is True
        print(f"[4] PASS: {task} -> reward={r.reward:.3f}")

    # 5. Cleanup
    await env.close()
    print("[5] PASS: Container cleaned up")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(test())
