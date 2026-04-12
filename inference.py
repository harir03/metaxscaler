import asyncio
import os
import json
import subprocess
import sys
import textwrap
import traceback
from typing import List, Optional


def _ensure_installed(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing {pip_name}...", file=sys.stderr, flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--root-user-action=ignore", pip_name],
            stdout=subprocess.DEVNULL,
        )


_ensure_installed("openai", "openai>=1.0.0")

from openai import OpenAI  # noqa: E402
from env_client import DockerEnvClient  # noqa: E402

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "credit_approval"

TASKS = ["credit-approval-easy", "credit-approval-medium", "credit-approval-hard"]
EPISODES_PER_TASK = 3
TEMPERATURE = 0.3
MAX_TOKENS = 500
SUCCESS_THRESHOLD = 0.3

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
    - If data shows "not disclosed", note that in your reasoning
    - Cite specific metric values (e.g. "DSCR of 2.3") in your reasoning
    Be thorough. Mention specific metrics.""")


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = error if error else "null"
    act = action.replace("\n", " ").replace("\r", "")[:200]
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)


def _fmt_obs(obs):
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


def _get_decision(llm, obs_data):
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

        # strip markdown fences if the model wraps json in them
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


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await DockerEnvClient.from_docker_image(IMAGE_NAME, container_port=7860)

    all_rewards = []
    total_steps = 0
    total_score = 0.0

    try:
        for task in TASKS:
            rewards = []
            steps = 0
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

            try:
                for _ in range(EPISODES_PER_TASK):
                    result = await env.reset(task_name=task)
                    obs = result.observation if isinstance(result.observation, dict) else {}
                    info = result.info if isinstance(result.info, dict) else {}

                    # gather hidden data before deciding
                    hidden = info.get("hidden", [])
                    for cat in hidden:
                        req_key = f"{cat}_data" if not cat.endswith("_data") else cat
                        result = await env.step({"request": req_key})
                        obs = result.observation if isinstance(result.observation, dict) else obs
                        steps += 1
                        log_step(
                            step=steps,
                            action=f"request({req_key})",
                            reward=0.0, done=False, error=None,
                        )

                    action = _get_decision(client, obs)
                    result = await env.step(action)

                    reward = max(0.0, min(1.0, float(result.reward or 0.0)))
                    rewards.append(reward)
                    steps += 1

                    log_step(
                        step=steps,
                        action=f"{action['decision']}(conf={action['confidence']:.2f})",
                        reward=reward,
                        done=result.done,
                        error=result.last_action_error,
                    )
            except Exception as e:
                print(f"[DEBUG] {task} failed: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

            score = sum(rewards) / len(rewards) if rewards else 0.0
            score = max(0.0, min(1.0, score))
            log_end(success=score >= SUCCESS_THRESHOLD, steps=steps, score=score, rewards=rewards)

            all_rewards.extend(rewards)
            total_steps += steps
            total_score += score

    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
