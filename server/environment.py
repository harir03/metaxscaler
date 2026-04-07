import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CreditAction, CreditObservation, CreditState, EnvResult
from server.task_generator import generate_task
from server.graders import grade

VALID_TASKS = [
    "credit-approval-easy",
    "credit-approval-medium",
    "credit-approval-hard",
]


class CreditApprovalEnvironment:
    """
    Single-step RL env. Each episode is one company to evaluate.
    Agent sees financial data, makes a credit decision, gets scored.
    """

    def __init__(self):
        self._state = None
        self._obs = None
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
        )
        self._obs = obs
        self._gt_decision = gt_dec
        self._gt_score = gt_score
        self._gt_reason = gt_reason

        return EnvResult(
            observation=obs, reward=0.0, done=False,
            info={"episode_id": self._state.episode_id, "task": task_name, "difficulty": obs.difficulty},
        )

    def step(self, action: CreditAction):
        if self._state is None:
            raise RuntimeError("call reset() first")
        if self._state.done:
            raise RuntimeError("episode already done, call reset()")

        self._state.step += 1
        self._state.done = True

        result = grade(
            task_name=self._state.task_name,
            decision=action.decision,
            reasoning=action.reasoning,
            confidence=action.confidence,
            ground_truth_decision=self._gt_decision,
        )

        if self._obs:
            self._obs.step = self._state.step

        return EnvResult(
            observation=self._obs,
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
            },
        )

    def state(self):
        if self._state is None:
            return CreditState(episode_id="none", step=0, task_name="", done=True)
        return self._state

    def close(self):
        self._state = None
        self._obs = None
