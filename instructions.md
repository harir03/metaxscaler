# 📋 Instructions — Credit Approval Environment

> **Read this file before making ANY changes to this project.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATOR HOST (OpenEnv Platform)                              │
│                                                                 │
│  1. Builds Docker image from Dockerfile                         │
│  2. Runs inference.py OUTSIDE the container                     │
│  3. inference.py calls GenericEnvClient.from_docker_image()     │
│  4. SDK starts the container + connects on port from openenv.yaml│
│                                                                 │
│  ┌─────────────────┐     HTTP      ┌──────────────────────┐    │
│  │  inference.py    │ ──────────── │  Docker Container     │    │
│  │  (OpenAI client) │  /reset      │  (FastAPI server)     │    │
│  │                  │  /step       │  port 7860            │    │
│  │                  │  /state      │                       │    │
│  └─────────────────┘  /health     └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Rules

### 1. TWO REPOS — Always push to BOTH
```bash
git push origin main   # GitHub (github.com/harir03/metaxscaler)
git push hf main       # HuggingFace Space (the evaluator reads from HERE)
```

### 2. Dockerfile must be LEAN
- The Docker container runs ONLY the FastAPI server
- DO NOT install `openai`, `openenv-core`, `gradio`, or any inference deps
- Only `fastapi`, `uvicorn`, `pydantic` go in `server/requirements.txt`
- Target image size: < 300MB
- Target startup time: < 5 seconds

### 3. Port Alignment
- `Dockerfile` EXPOSE + CMD: **7860**
- `openenv.yaml` app.port: **7860**
- These MUST match. The SDK reads the port from openenv.yaml.

### 4. inference.py runs OUTSIDE the container
- It self-installs its deps (openai, openenv-core) at runtime
- It uses `GenericEnvClient.from_docker_image(IMAGE_NAME)`
- IMAGE_NAME is set by the evaluator (ECR image URL)
- DO NOT use `LOCAL_IMAGE_NAME` — the evaluator doesn't set it

### 5. Stdout Format (MANDATORY)
```
[START] task=<name> env=credit_approval model=<model>
[STEP] step=1 action=<action> reward=0.70 done=true error=null
[END] success=true steps=1 score=0.700 rewards=0.70
```
- reward/rewards: 2 decimal places
- done/success: lowercase boolean
- error: raw string or "null"
- score: 3 decimal places, in [0, 1]

---

## Quality Checks (run before EVERY push)

### Infrastructure Check
```bash
# 1. Build Docker image
docker build -f Dockerfile -t credit-approval-env .

# 2. Start container + measure startup time (must be < 5s)
docker run --rm -d --name test-env -p 7860:7860 credit-approval-env
# Wait 2 seconds, then:
curl http://localhost:7860/health

# 3. Test all endpoints
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name":"credit-approval-easy"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"decision":"approve","reasoning":"test","confidence":0.8}'

# 4. Check image size (must be < 300MB)
docker images credit-approval-env
```

### Code Quality Check
- [ ] `inference.py` parses without errors: `python -c "import ast; ast.parse(open('inference.py').read())"`
- [ ] `openenv.yaml` port matches Dockerfile
- [ ] No `openai` or `openenv-core` in server/requirements.txt
- [ ] All 3 tasks defined in openenv.yaml
- [ ] `[START]`, `[STEP]`, `[END]` format matches spec exactly

### Cross-File Impact Check
| File Changed | Check These |
|---|---|
| `Dockerfile` | openenv.yaml port, server/requirements.txt, .dockerignore |
| `inference.py` | stdout format, env var names, GenericEnvClient usage |
| `openenv.yaml` | Dockerfile port, runtime.dockerfile path |
| `server/app.py` | models.py imports, endpoint paths |
| `models.py` | server/app.py, server/environment.py, client.py |
| `server/graders.py` | openenv.yaml grader references |

---

## Evaluation Phases (what the evaluator checks)

1. **Docker Build Creation** — Does `Dockerfile` build successfully?
2. **inference.py Execution** — Does the script run without exceptions?
3. **Output Parsing** — Does stdout match [START]/[STEP]/[END] format?
4. **Task Validation** — Do all tasks produce valid scores in [0,1]?
5. **LLM Criteria Check** — Is the agent reasoning meaningful?

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| `ModuleNotFoundError: openai` | inference.py self-installs deps at top |
| Container timeout (30s) | Keep Dockerfile lean, no heavy deps |
| Port mismatch | openenv.yaml port == Dockerfile port |
| Pushing only to GitHub | The evaluator reads from HuggingFace Space |
| IMAGE_NAME vs LOCAL_IMAGE_NAME | Evaluator sets `IMAGE_NAME`, not `LOCAL_IMAGE_NAME` |
