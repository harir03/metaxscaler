# 🏦 Credit Approval Environment — OpenEnv RL

> An AI agent evaluates corporate loan applications with realistic Indian financial data, risk indicators, and market context. Features 3 difficulty levels with deterministic grading and fraud detection challenges.

---

## 🎯 What This Environment Does

An LLM agent acts as a **credit analyst** reviewing corporate loan applications. Each episode presents a company profile with financial metrics, risk flags, and market context. The agent must decide: **approve**, **reject**, or **conditional** — and justify its reasoning.

| Difficulty | Description | Key Challenge |
|---|---|---|
| **Easy** | Clear approve/reject signals | Basic financial literacy |
| **Medium** | Borderline cases, mixed signals | Nuanced reasoning under uncertainty |
| **Hard** | Adversarial — looks good, hidden fraud | Detecting wilful defaulters, revenue inflation, circular trading |

### Reward Design

Grading evaluates **three dimensions** with varying weights per difficulty:

| Dimension | Easy Weight | Medium Weight | Hard Weight |
|---|---|---|---|
| Decision correctness | 70% | 50% | 40% |
| Reasoning quality | 15% | 30% | 40% |
| Confidence calibration | 15% | 20% | 20% |
| Fraud detection bonus | — | — | up to 15% |

All scores are in `[0.0, 1.0]`. Partial credit for "conditional" when ground truth is "approve" (or vice versa).

---

## 🚀 Quick Start

### 1. Start the environment server

```bash
# Build & run via Docker
docker build -f server/Dockerfile -t credit-approval-env .
docker run -p 8000:8000 credit-approval-env

# Or run locally
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 2. Run inference

```bash
# Set required env vars
export HF_TOKEN=<your-hf-token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:8000

# Run the agent
python inference.py
```

### 3. Validate

```bash
pip install openenv-core
openenv validate
```

---

## 📁 Project Structure

```
credit-approval-env/
├── inference.py          # 🔑 Agent script (judges run this)
├── models.py             # Pydantic Action/Observation/State
├── client.py             # HTTP client for the env
├── openenv.yaml          # OpenEnv manifest (3 tasks)
├── pyproject.toml        # Project metadata & deps
├── __init__.py           # Package exports
├── .env.example          # Environment variables template
├── server/               # 🐳 Environment server (containerized)
│   ├── app.py            # FastAPI endpoints (reset/step/state)
│   ├── environment.py    # Core env logic
│   ├── graders.py        # Deterministic grading (3 graders)
│   ├── task_generator.py # Synthetic company data generator
│   ├── Dockerfile        # Container image
│   └── requirements.txt  # Server dependencies
├── backend/              # Intelli-Credit domain logic (optional)
│   ├── agents/           # Credit analysis agents
│   ├── models/           # Domain schemas
│   └── ...
├── config/               # Scoring constants & settings
└── docs/                 # Documentation
```

---

## 🧪 Tasks & Graders

### Task 1: `credit-approval-easy`
- **Input**: Company with clear financial signals + hints
- **Grading**: 70% decision, 15% reasoning, 15% confidence
- **Example**: DSCR=2.5, rating=AA+, no flags → approve

### Task 2: `credit-approval-medium`
- **Input**: Borderline cases with conflicting metrics, no hints
- **Grading**: 50% decision, 30% reasoning, 20% confidence
- **Example**: DSCR=1.2, rating=BBB, RPT flagged, moderate pledge → conditional

### Task 3: `credit-approval-hard`
- **Input**: Adversarial profiles with hidden traps
- **Grading**: 40% decision, 40% reasoning, 20% confidence + fraud bonus
- **Example**: DSCR=2.5, rating=AAA BUT wilful_defaulter=True → reject

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode (body: `{"task_name": "credit-approval-easy"}`) |
| POST | `/step` | Submit decision (body: `{"decision": "approve", "reasoning": "...", "confidence": 0.8}`) |
| GET | `/state` | Get current episode state |
| GET | `/health` | Health check |

---

## 📊 Observation Schema

```json
{
  "company": {
    "name": "Apex Industries Ltd.",
    "sector": "Steel & Metals",
    "incorporation_year": 2003,
    "annual_turnover_cr": 456.78,
    "loan_type": "Term Loan",
    "loan_amount_cr": 50.0
  },
  "financials": {
    "dscr": 1.85,
    "current_ratio": 1.62,
    "debt_equity_ratio": 0.95,
    "net_profit_margin": 8.5,
    "revenue_growth_yoy": 12.3,
    "interest_coverage_ratio": 3.2,
    "working_capital_days": 72,
    "cash_flow_positive": true
  },
  "risk": {
    "credit_rating": "A+",
    "wilful_defaulter": false,
    "active_criminal_case": false,
    "nclt_active": false,
    "gst_compliance_pct": 97.5,
    "related_party_transactions_flagged": false,
    "audit_qualifications": 0,
    "promoter_pledge_pct": 8.2
  },
  "market": {
    "sector_outlook": "positive",
    "sector_npa_rate": 3.5,
    "gdp_growth_relevant": 6.2,
    "regulatory_risk": "low"
  }
}
```

---

## 🏗️ Built With

- **FastAPI** — Environment server
- **Pydantic v2** — Typed models
- **OpenAI Client** — LLM inference
- **OpenEnv** — Framework compliance
- **Docker** — Containerized deployment

---

> **Note:** This environment is built on top of the [Intelli-Credit](docs/intellicredit.md) credit decisioning engine, adapting its scoring framework and domain logic into an RL-ready format.
