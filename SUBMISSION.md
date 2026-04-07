# 📋 Submission Checklist — Credit Approval Environment

> **Team**: Meta  
> **Environment**: `credit_approval`  
> **Deadline**: April 8, 2026, 11:59 PM IST

---

## Pre-Submission Checklist

### ✅ Infrastructure
- [x] `server/Dockerfile` builds successfully
- [x] `docker-compose.yml` references correct Dockerfile
- [x] Docker image is lightweight (python:3.12-slim, no heavy ML deps)
- [x] Build completes in < 600 seconds

### ✅ OpenEnv Spec Compliance
- [x] `openenv.yaml` present at root with valid schema
- [x] 3 tasks defined: `credit-approval-easy`, `credit-approval-medium`, `credit-approval-hard`
- [x] Each task has a grader reference (`server.graders:grade_*`)
- [x] `spec_version: 1` declared
- [x] Runtime type: `docker`, port: `8000`
- [x] Endpoints: `/reset`, `/step`, `/state`

### ✅ API Endpoints
- [x] `POST /reset` — returns 200 with observation, reward=0.0, done=false
- [x] `POST /step` — returns 200 with observation, reward ∈ [0,1], done=true
- [x] `GET /state` — returns episode metadata
- [x] `GET /health` — returns `{"status": "healthy"}`

### ✅ Inference Script
- [x] `inference.py` at project root
- [x] Uses `OpenAI` client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [x] Emits `[START]`, `[STEP]`, `[END]` to stdout in correct format
- [x] `reward` and `rewards` formatted to 2 decimal places
- [x] `done` and `success` are lowercase booleans
- [x] `error` is raw string or `null`
- [x] Score in [0.0, 1.0] per task
- [x] Runs in < 20 minutes

### ✅ Typed Models
- [x] `models.py` with Pydantic v2 models
- [x] `CreditAction` — decision, reasoning, confidence
- [x] `CreditObservation` — company, financials, risk, market
- [x] `CreditState` — episode_id, step, task_name, done

### ✅ Grading
- [x] 3 graders (easy, medium, hard) with different weight distributions
- [x] Scores are deterministic and reproducible
- [x] Scores vary meaningfully — not hardcoded
- [x] Hard task rewards fraud detection with bonus points
- [x] All scores clamped to [0.0, 1.0]

### ✅ Environment Design
- [x] `reset()` produces clean state each time
- [x] Action/Observation types are well-documented
- [x] Reward provides useful varying signal (not sparse)
- [x] Episode boundaries are sensible (single-step per company)
- [x] 3 difficulty levels with meaningful progression

---

## Environment Summary

| Property | Value |
|---|---|
| **Name** | `credit_approval` |
| **Domain** | Corporate credit decisioning (Indian banking) |
| **Tasks** | 3 (easy → medium → hard) |
| **Action Space** | `{decision, reasoning, confidence}` |
| **Observation Space** | Company profile + financials + risk + market context |
| **Reward Range** | [0.0, 1.0] |
| **Episode Length** | 1 step (single decision per company) |
| **Grading Dimensions** | Decision correctness, reasoning quality, confidence calibration |

---

## Known Limitations

1. **Single-step episodes**: Each episode is one company evaluation. Multi-turn negotiation or iterative review is not supported.
2. **Synthetic data only**: Company profiles are procedurally generated, not from real filings.
3. **No persistent memory**: Agent cannot reference past episodes or build sector expertise.
4. **English only**: All observations and actions are in English.

---

## Key Files

| File | Purpose |
|---|---|
| [`inference.py`](inference.py) | Agent script (judges execute this) |
| [`openenv.yaml`](openenv.yaml) | Environment manifest |
| [`models.py`](models.py) | Pydantic Action/Observation/State |
| [`server/app.py`](server/app.py) | FastAPI server |
| [`server/environment.py`](server/environment.py) | Core environment logic |
| [`server/graders.py`](server/graders.py) | Deterministic graders |
| [`server/task_generator.py`](server/task_generator.py) | Synthetic data generator |
| [`server/Dockerfile`](server/Dockerfile) | Container image |

---

## Validation Commands

```bash
# 1. Docker build
docker build -f server/Dockerfile -t credit-approval-env .

# 2. Run server
docker run -p 8000:8000 credit-approval-env

# 3. Test /reset endpoint
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "credit-approval-easy"}'

# 4. OpenEnv validate
pip install openenv-core
openenv validate

# 5. Run inference
export HF_TOKEN=<your-token>
python inference.py
```

---

## Contact

For questions about this environment, reach out to the team via the hackathon platform.
