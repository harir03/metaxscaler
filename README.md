---
title: Credit Approval Environment
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - credit-approval
---

# Credit Approval Environment

An RL environment where an LLM agent acts as a credit analyst reviewing corporate loan applications. Each episode presents a company with realistic Indian financial data — the agent has to decide approve, reject, or conditional and explain why.

Three difficulty levels:

- **Easy** — clear signals, hints provided
- **Medium** — mixed/borderline metrics, no hints
- **Hard** — adversarial profiles that look clean but hide fraud patterns (wilful defaulters, revenue inflation, circular trading, evergreening)

## Grading

| Dimension | Easy | Medium | Hard |
|---|---|---|---|
| Decision correctness | 70% | 50% | 40% |
| Reasoning quality | 15% | 30% | 40% |
| Confidence calibration | 15% | 20% | 20% |
| Fraud detection bonus | — | — | up to 15% |

Scores are in `[0, 1]`. Partial credit when predicting "conditional" vs "approve".

## Running locally

```bash
# build & start
docker build -t credit-approval-env .
docker run -p 7860:7860 credit-approval-env

# test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_name": "credit-approval-easy"}'
```

## Inference

```bash
export HF_TOKEN=<your-token>
export IMAGE_NAME=credit-approval-env
python inference.py
```

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/reset` | New episode: `{"task_name": "credit-approval-easy"}` |
| POST | `/step` | Submit decision: `{"decision": "approve", "reasoning": "...", "confidence": 0.8}` |
| GET | `/state` | Current episode state |
| GET | `/health` | Health check |

## Project layout

```
├── inference.py            # agent script
├── env_client.py           # lightweight docker+http client
├── models.py               # pydantic schemas
├── openenv.yaml            # env manifest
├── Dockerfile
├── server/
│   ├── app.py              # fastapi server
│   ├── environment.py      # env logic
│   ├── graders.py          # scoring (3 graders)
│   ├── task_generator.py   # synthetic company data
│   ├── requirements.txt
│   └── Dockerfile
└── tests/
```

## Built with

FastAPI, Pydantic v2, OpenAI client, Docker
