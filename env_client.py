import asyncio
import random
import subprocess
import sys
import time

try:
    import httpx
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--root-user-action=ignore", "httpx>=0.28.0"],
        stdout=subprocess.DEVNULL,
    )
    import httpx


class EnvResult:
    def __init__(self, data: dict):
        self.observation = data.get("observation", {})
        self.reward = float(data.get("reward", 0.0))
        self.done = bool(data.get("done", False))
        self.info = data.get("info", {})
        self.last_action_error = (
            data.get("last_action_error") or data.get("info", {}).get("error")
        )


class DockerEnvClient:
    """Talks to an OpenEnv-compatible container over HTTP."""

    def __init__(self, base_url: str, container_id: str = None):
        self.base_url = base_url.rstrip("/")
        self.container_id = container_id
        self._client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    async def from_docker_image(cls, image_name, container_port=7860, ready_timeout=120.0):
        host_port = random.randint(30000, 59999)

        print(f"[SETUP] Starting container {image_name} on :{host_port}",
              file=sys.stderr, flush=True)

        proc = subprocess.run(
            ["docker", "run", "-d", "-p", f"{host_port}:{container_port}", image_name],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")

        cid = proc.stdout.strip()
        base_url = f"http://localhost:{host_port}"

        # poll /health until it responds
        deadline = time.monotonic() + ready_timeout
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                try:
                    r = await probe.get(f"{base_url}/health")
                    if r.status_code == 200:
                        print(f"[SETUP] Ready at {base_url}", file=sys.stderr, flush=True)
                        return cls(base_url, cid)
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        # timed out — dump logs then bail
        logs = subprocess.run(
            ["docker", "logs", "--tail", "20", cid], capture_output=True, text=True
        )
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        raise TimeoutError(
            f"Container not ready in {ready_timeout}s\n{logs.stdout}\n{logs.stderr}"
        )

    async def reset(self, task_name=None):
        body = {"task_name": task_name} if task_name else {}
        r = await self._client.post(f"{self.base_url}/reset", json=body)
        r.raise_for_status()
        return EnvResult(r.json())

    async def step(self, action):
        body = action if isinstance(action, dict) else action.__dict__
        r = await self._client.post(f"{self.base_url}/step", json=body)
        r.raise_for_status()
        return EnvResult(r.json())

    async def close(self):
        await self._client.aclose()
        if self.container_id:
            subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True)
