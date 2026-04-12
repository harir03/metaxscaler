import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CreditAction, CreditObservation, CreditState, EnvResult,
    RiskIndicators, MarketContext,
)
from server.task_generator import generate_task
from server.graders import grade

VALID_TASKS = [
    "credit-approval-easy",
    "credit-approval-medium",
    "credit-approval-hard",
]

VALID_REQUESTS = {"risk_data", "market_data"}


class CreditApprovalEnvironment:
    """Single or multi-step RL env. Agent sees financial data, optionally
    requests more info, then makes a credit decision and gets scored."""

    def __init__(self):
        self._state = None
        self._full_obs = None
        self._revealed = set()
        self._gt_decision = ""
        self._gt_score = 0
        self._gt_reason = ""

    def reset(self, task_name="credit-approval-easy"):
        if task_name not in VALID_TASKS:
            raise ValueError(f"unknown task '{task_name}', pick from {VALID_TASKS}")

        obs, gt_dec, gt_score, gt_reason = generate_task(task_name)

        self._state = CreditState(
            episode_id=str(uuid.uuid4()),
            step=0, task_name=task_name,
            difficulty=obs.difficulty, done=False,
            ground_truth_decision=gt_dec,
            ground_truth_score=gt_score,
            company_name=obs.company.name,
            max_steps=3,
        )
        self._full_obs = obs
        self._revealed = {"company", "financials"}
        self._gt_decision = gt_dec
        self._gt_score = gt_score
        self._gt_reason = gt_reason

        masked = self._mask_obs()
        hidden = []
        if "risk_data" not in self._revealed:
            hidden.append("risk")
        if "market_data" not in self._revealed:
            hidden.append("market")

        return EnvResult(
            observation=masked, reward=0.0, done=False,
            info={
                "episode_id": self._state.episode_id,
                "task": task_name,
                "difficulty": obs.difficulty,
                "disclosed": sorted(self._revealed),
                "hidden": hidden,
                "available_requests": sorted(VALID_REQUESTS - self._revealed),
                "max_steps": self._state.max_steps,
            },
        )

    def request_info(self, category: str):
        if self._state is None:
            raise RuntimeError("call reset() first")
        if self._state.done:
            raise RuntimeError("episode already done, call reset()")
        if category not in VALID_REQUESTS:
            raise ValueError(f"invalid request '{category}', pick from {VALID_REQUESTS}")
        if category in self._revealed:
            raise ValueError(f"'{category}' already disclosed")
        if self._state.step >= self._state.max_steps - 1:
            raise RuntimeError("no steps left for info requests, must submit decision")

        self._state.step += 1
        self._revealed.add(category)

        masked = self._mask_obs()
        hidden = []
        if "risk_data" not in self._revealed:
            hidden.append("risk")
        if "market_data" not in self._revealed:
            hidden.append("market")

        return EnvResult(
            observation=masked, reward=0.0, done=False,
            info={
                "episode_id": self._state.episode_id,
                "disclosed": sorted(self._revealed),
                "hidden": hidden,
                "available_requests": sorted(VALID_REQUESTS - self._revealed),
                "steps_remaining": self._state.max_steps - self._state.step,
            },
        )

    def step(self, action: CreditAction):
        if self._state is None:
            raise RuntimeError("call reset() first")
        if self._state.done:
            raise RuntimeError("episode already done, call reset()")

        self._state.step += 1
        self._state.done = True

        obs_dict = self._full_obs.model_dump() if self._full_obs else {}

        result = grade(
            task_name=self._state.task_name,
            decision=action.decision,
            reasoning=action.reasoning,
            confidence=action.confidence,
            ground_truth_decision=self._gt_decision,
            observation=obs_dict,
            steps_taken=self._state.step,
        )

        if self._full_obs:
            self._full_obs.step = self._state.step

        return EnvResult(
            observation=self._full_obs,
            reward=result["score"],
            done=True,
            info={
                "episode_id": self._state.episode_id,
                "grade_breakdown": result["breakdown"],
                "grade_weights": result["weights"],
                "ground_truth_decision": self._gt_decision,
                "ground_truth_score": self._gt_score,
                "ground_truth_reason": self._gt_reason,
                "agent_decision": action.decision,
                "agent_reasoning_length": len(action.reasoning),
                "steps_taken": self._state.step,
                "data_gathered": sorted(self._revealed),
            },
        )

    def _mask_obs(self):
        data = self._full_obs.model_dump()
        if "risk_data" not in self._revealed:
            data["risk"] = RiskIndicators().model_dump()
        if "market_data" not in self._revealed:
            data["market"] = MarketContext().model_dump()
        return CreditObservation(**data)

    def state(self):
        if self._state is None:
            return CreditState(episode_id="none", step=0, task_name="", done=True)
        return self._state

    def close(self):
        self._state = None
        self._full_obs = None
        self._revealed = set()
