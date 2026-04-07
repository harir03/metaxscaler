import httpx
from typing import Optional

from models import CreditAction, CreditObservation, CreditState, EnvResult


class CreditApprovalClient:
    def __init__(self, base_url="http://localhost:8000", timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def reset(self, task_name="credit-approval-easy"):
        r = self._http.post("/reset", json={"task_name": task_name})
        r.raise_for_status()
        d = r.json()
        return EnvResult(observation=CreditObservation(**d["observation"]), reward=d.get("reward", 0.0), done=d.get("done", False), info=d.get("info", {}))

    def step(self, action: CreditAction):
        r = self._http.post("/step", json=action.model_dump())
        r.raise_for_status()
        d = r.json()
        return EnvResult(observation=CreditObservation(**d["observation"]), reward=d.get("reward", 0.0), done=d.get("done", True), info=d.get("info", {}))

    def state(self):
        r = self._http.get("/state")
        r.raise_for_status()
        return CreditState(**r.json())

    def health(self):
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class AsyncCreditApprovalClient:
    def __init__(self, base_url="http://localhost:8000", timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def reset(self, task_name="credit-approval-easy"):
        r = await self._http.post("/reset", json={"task_name": task_name})
        r.raise_for_status()
        d = r.json()
        return EnvResult(observation=CreditObservation(**d["observation"]), reward=d.get("reward", 0.0), done=d.get("done", False), info=d.get("info", {}))

    async def step(self, action: CreditAction):
        r = await self._http.post("/step", json=action.model_dump())
        r.raise_for_status()
        d = r.json()
        return EnvResult(observation=CreditObservation(**d["observation"]), reward=d.get("reward", 0.0), done=d.get("done", True), info=d.get("info", {}))

    async def state(self):
        r = await self._http.get("/state")
        r.raise_for_status()
        return CreditState(**r.json())

    async def health(self):
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        await self.close()
