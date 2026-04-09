# Fix OpenEnv Submission — Inference.py Crash & Preemptive Fixes

## Root Cause Analysis
The OpenEnv evaluator copies the workspace to `/tmp/workspace/` and runs `inference.py`
directly. The error: `ModuleNotFoundError: No module named 'openai'` means the evaluator's
Python environment doesn't have `openai` installed. 

The evaluator expects a **root-level `requirements.txt`** to install inference dependencies,
but we only have `server/requirements.txt` (which only covers server deps: fastapi, uvicorn, pydantic).

## Fixes Required

### Phase 1: Fix the crash (ModuleNotFoundError)
- [ ] 1. Create root `requirements.txt` with `openai` and `openenv-core` (inference deps)
- [ ] 2. Update root `Dockerfile` to copy `inference.py` and install root requirements
- [ ] 3. Ensure both Dockerfiles are consistent

### Phase 2: Preemptive fixes for remaining evaluation phases
- [ ] 4. Output Parsing: Verify stdout format
- [ ] 5. Task Validation: Ensure openenv.yaml alignment
- [ ] 6. LLM Criteria Check: Graceful error handling
- [ ] 7. Port alignment: openenv.yaml vs Dockerfile
- [ ] 8. inference.py hardening
