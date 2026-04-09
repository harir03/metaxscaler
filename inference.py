"""
Credit Approval Environment — Inference Script
================================================
Uses OpenEnv SDK (from_docker_image) + OpenAI Client.
Follows mandatory stdout format: [START] / [STEP] / [END].

MANDATORY ENV VARS:
    API_BASE_URL       The API endpoint for the LLM
    MODEL_NAME         The model identifier
    HF_TOKEN           Your HuggingFace / API key
    IMAGE_NAME         Docker image name (set by evaluator)
"""

import asyncio
import os
import json
import subprocess
import sys
import textwrap
import traceback
from typing import List, Optional


# ── Auto-install missing dependencies ──
def _ensure_installed(package_name: str, pip_name: str = None):
    """Install a package if it's not already available."""
    pip_name = pip_name or package_name
    try:
        __import__(package_name)
    except ImportError:
        print(f"[SETUP] Installing {pip_name}...", file=sys.stderr, flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
            stdout=subprocess.DEVNULL,
        )


_ensure_installed("openai", "openai>=1.0.0")
_ensure_installed("openenv", "openenv-core>=0.1.0")

from openai import OpenAI  # noqa: E402
from openenv import GenericEnvClient, GenericAction  # noqa: E402

# ── Configuration from environment ──
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "credit_approval"

TASKS = ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]
EPISODES_PER_TASK = 3
MAX_STEPS = 1  # Single-step environment
TEMPERATURE = 0.3
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert credit analyst at a major Indian bank. You review corporate
    loan applications and make credit decisions based on financial data.

    You will receive a company profile with financial metrics, risk indicators,
    and market context. Respond with EXACTLY this JSON:
    {"decision": "approve"|"reject"|"conditional", "reasoning": "...", "confidence": 0.0-1.0}

    Key rules:
    - DSCR > 1.5 and clean record -> likely approve
    - DSCR < 1.0 or wilful defaulter -> reject
    - Hard blocks (wilful defaulter, NCLT, criminal cases) -> always reject
    - Watch for fraud: unrealistic growth + low margins, RPT flags, high pledge %
    - Revenue inflation: high growth but negative cash flow is suspicious
    Be thorough. Mention specific metrics.""")


# ── Logging (mandatory stdout format) ──

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    act = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={act} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatting ──

def _fmt_obs(obs: dict) -> str:
    co = obs.get("company", {})
    fi = obs.get("financials", {})
    ri = obs.get("risk", {})
    mk = obs.get("market", {})
    hint = obs.get("hint")

    out = f"""=== CREDIT APPLICATION REVIEW ===
Difficulty: {obs.get('difficulty', '?')}

COMPANY: {co.get('name')} | {co.get('sector')} | Est. {co.get('incorporation_year')}
Turnover: Rs {co.get('annual_turnover_cr', 0):.1f} Cr | Loan: {co.get('loan_type')} Rs {co.get('loan_amount_cr', 0):.1f} Cr

FINANCIALS:
  DSCR: {fi.get('dscr', 0):.2f} | Current: {fi.get('current_ratio', 0):.2f} | D/E: {fi.get('debt_equity_ratio', 0):.2f}
  NPM: {fi.get('net_profit_margin', 0):.1f}% | Rev Growth: {fi.get('revenue_growth_yoy', 0):.1f}% | ICR: {fi.get('interest_coverage_ratio', 0):.2f}
  WC Days: {fi.get('working_capital_days', 0)} | Cash Flow +ve: {fi.get('cash_flow_positive', False)}

RISK:
  Rating: {ri.get('credit_rating')} | Wilful Defaulter: {ri.get('wilful_defaulter', False)}
  Criminal Case: {ri.get('active_criminal_case', False)} | NCLT: {ri.get('nclt_active', False)}
  GST Compliance: {ri.get('gst_compliance_pct', 0):.1f}% | RPT Flagged: {ri.get('related_party_transactions_flagged', False)}
  Audit Quals: {ri.get('audit_qualifications', 0)} | Promoter Pledge: {ri.get('promoter_pledge_pct', 0):.1f}%

MARKET: Outlook={mk.get('sector_outlook')} | NPA={mk.get('sector_npa_rate', 0):.1f}% | Reg Risk={mk.get('regulatory_risk')}"""

    if hint:
        out += f"\n\nHINT: {hint}"
    out += "\n\nAnalyze and respond with the JSON format."
    return out


# ── LLM decision ──

def _get_decision(llm: OpenAI, obs_data: dict) -> dict:
    """Returns a dict with decision, reasoning, confidence."""
    prompt = _fmt_obs(obs_data)
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        decision = str(parsed.get("decision", "reject")).lower().strip()
        if decision not in ("approve", "reject", "conditional"):
            decision = "reject"
        return {
            "decision": decision,
            "reasoning": str(parsed.get("reasoning", "")),
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
        }
    except json.JSONDecodeError:
        low = (text or "").lower()
        dec = "reject"
        if "approve" in low and "reject" not in low:
            dec = "approve"
        elif "conditional" in low:
            dec = "conditional"
        return {"decision": dec, "reasoning": text or "parse error", "confidence": 0.3}
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return {"decision": "reject", "reasoning": f"error: {e}", "confidence": 0.1}


# ── Main ──

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Use OpenEnv SDK to spin up the environment from Docker image
    env = await GenericEnvClient.from_docker_image(IMAGE_NAME)

    overall_rewards: List[float] = []
    overall_steps = 0
    total_score = 0.0

    try:
        for task in TASKS:
            rewards: List[float] = []
            steps = 0

            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

            try:
                for ep in range(EPISODES_PER_TASK):
                    result = await env.reset(task_name=task)
                    obs = result.observation if isinstance(result.observation, dict) else result.observation.dict()

                    action_data = _get_decision(client, obs)
                    action = GenericAction(action_data)

                    result = await env.step(action)

                    reward = result.reward or 0.0
                    reward = max(0.0, min(1.0, float(reward)))
                    done = result.done
                    error = getattr(result, "last_action_error", None)

                    rewards.append(reward)
                    steps += 1

                    log_step(
                        step=steps,
                        action=f"{action_data['decision']}(conf={action_data['confidence']:.2f})",
                        reward=reward,
                        done=done,
                        error=error,
                    )

            except Exception as e:
                print(f"[DEBUG] {task} failed: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

            task_score = sum(rewards) / len(rewards) if rewards else 0.0
            task_score = max(0.0, min(1.0, task_score))
            success = task_score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=success, steps=steps, score=task_score, rewards=rewards)

            overall_rewards.extend(rewards)
            overall_steps += steps
            total_score += task_score

        avg = total_score / len(TASKS) if TASKS else 0.0
        print(
            f"[SUMMARY] overall_score={max(0.0, min(1.0, avg)):.2f} total_steps={overall_steps} tasks={len(TASKS)}",
            file=sys.stderr,
            flush=True,
        )

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
