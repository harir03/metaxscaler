import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import CreditAction
from server.environment import CreditApprovalEnvironment

app = FastAPI(title="Credit Approval Environment")
env = CreditApprovalEnvironment()


class ResetReq(BaseModel):
    task_name: str = "credit-approval-easy"


class StepReq(BaseModel):
    decision: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.5
    suggested_terms: Optional[str] = None
    request: Optional[str] = None


def _env_result_to_dict(result):
    obs = result.observation
    obs_dict = obs.model_dump() if obs else {}
    # strip ground truth from observation
    obs_dict.pop("ground_truth_decision", None)
    obs_dict.pop("ground_truth_score", None)
    return {
        "observation": obs_dict,
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/reset")
async def reset(req: ResetReq):
    try:
        result = env.reset(task_name=req.task_name)
        return _env_result_to_dict(result)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/step")
async def step(req: StepReq):
    try:
        if req.request:
            result = env.request_info(req.request)
            return _env_result_to_dict(result)
        elif req.decision:
            action = CreditAction(
                decision=req.decision,
                reasoning=req.reasoning,
                confidence=req.confidence,
                suggested_terms=req.suggested_terms,
            )
            result = env.step(action)
            return _env_result_to_dict(result)
        else:
            raise HTTPException(400, "provide either 'decision' or 'request'")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.get("/state")
async def get_state():
    s = env.state()
    return {
        "episode_id": s.episode_id,
        "step": s.step,
        "task_name": s.task_name,
        "difficulty": s.difficulty,
        "done": s.done,
        "max_steps": s.max_steps,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "Credit Approval Environment", "version": "1.0.0",
            "endpoints": {"reset": "POST /reset", "step": "POST /step",
                        "state": "GET /state", "health": "GET /health"},
            "tasks": ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
