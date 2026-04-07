import sys, os, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import CreditAction, CreditObservation, CreditState
from server.environment import CreditApprovalEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Credit Approval Environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = CreditApprovalEnvironment()


class ResetReq(BaseModel):
    task_name: Optional[str] = "credit-approval-easy"

class StepReq(BaseModel):
    decision: str
    reasoning: str = ""
    confidence: float = 0.5
    suggested_terms: Optional[str] = None


@app.post("/reset")
async def reset(req: ResetReq = ResetReq()):
    try:
        task = req.task_name or "credit-approval-easy"
        result = env.reset(task_name=task)
        log.info(f"reset: task={task} company={result.observation.company.name}")
        return {"observation": result.observation.model_dump(), "reward": result.reward, "done": result.done, "info": result.info}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/step")
async def step(req: StepReq):
    try:
        action = CreditAction(decision=req.decision, reasoning=req.reasoning, confidence=req.confidence, suggested_terms=req.suggested_terms)
        result = env.step(action)
        log.info(f"step: decision={action.decision} reward={result.reward:.4f}")
        return {"observation": result.observation.model_dump(), "reward": result.reward, "done": result.done, "info": result.info}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/state")
async def state():
    s = env.state()
    return {"episode_id": s.episode_id, "step": s.step, "task_name": s.task_name, "difficulty": s.difficulty, "done": s.done, "company_name": s.company_name}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "credit-approval-env", "version": "1.0.0",
            "tasks": ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]}


@app.get("/")
async def root():
    return {"service": "Credit Approval Environment", "version": "1.0.0",
            "endpoints": {"reset": "POST /reset", "step": "POST /step", "state": "GET /state", "health": "GET /health"},
            "tasks": ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]}
