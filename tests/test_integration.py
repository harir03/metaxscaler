"""Integration test against the live Docker container.
Tests multi-step flow, backward compat, and all task difficulties.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_client import DockerEnvClient

IMAGE = "credit-approval-env"
TASKS = ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]


async def main():
    print(f"=== Integration Test: DockerEnvClient (Multi-Step) ===")
    env = await DockerEnvClient.from_docker_image(IMAGE, container_port=7860)
    errors = []

    try:
        # 1. Container startup
        print(f"[1] PASS: Container started and ready")

        # 2. Multi-step flow: reset → request risk → request market → decide
        result = await env.reset(task_name="credit-approval-hard")
        obs = result.observation if isinstance(result.observation, dict) else {}
        info = result.info if isinstance(result.info, dict) else {}

        assert "hidden" in info, "reset should report hidden data"
        assert "risk" in info["hidden"], "risk should be hidden"
        assert "market" in info["hidden"], "market should be hidden"
        assert obs.get("risk", {}).get("credit_rating") == "not disclosed", \
            f"Risk should be masked, got {obs.get('risk', {}).get('credit_rating')}"
        print("[2] PASS: Reset returns masked observation")

        # 3. Request risk data
        result = await env.step({"request": "risk_data"})
        obs = result.observation if isinstance(result.observation, dict) else {}
        info = result.info if isinstance(result.info, dict) else {}
        assert not result.done, "Should not be done after info request"
        assert obs.get("risk", {}).get("credit_rating") != "not disclosed", \
            "Risk should now be revealed"
        assert "risk" not in info.get("hidden", []), "Risk should not be hidden anymore"
        print("[3] PASS: request_info(risk_data) reveals risk")

        # 4. Request market data
        result = await env.step({"request": "market_data"})
        obs = result.observation if isinstance(result.observation, dict) else {}
        assert not result.done
        assert obs.get("market", {}).get("sector_outlook") != "not disclosed"
        print("[4] PASS: request_info(market_data) reveals market")

        # 5. Make decision
        result = await env.step({
            "decision": "reject",
            "reasoning": "Wilful defaulter flag is a hard block despite DSCR of 2.5",
            "confidence": 0.85,
        })
        assert result.done, "Should be done after decision"
        assert 0 < result.reward < 1, f"Reward {result.reward} out of range"
        print(f"[5] PASS: Multi-step decision -> reward={result.reward:.3f}")

        # 6. Backward compat: single-step per task
        for task in TASKS:
            result = await env.reset(task_name=task)
            result = await env.step({
                "decision": "approve", "reasoning": "testing", "confidence": 0.5,
            })
            assert result.done
            assert 0 < result.reward < 1
            print(f"[6] PASS: {task} -> single-step reward={result.reward:.3f}")

        # 7. Cleanup
        await env.close()
        print("[7] PASS: Container cleaned up")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        try:
            await env.close()
        except Exception:
            pass
        sys.exit(1)

    print(f"\n=== ALL INTEGRATION TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
