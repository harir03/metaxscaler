"""
Lightweight Docker Environment Client
======================================
Replaces openenv-core's GenericEnvClient with zero heavy dependencies.
Uses subprocess for Docker CLI + httpx for HTTP calls.
No gradio, no numpy, no pandas — installs in < 3 seconds.
"""

import asyncio
import json
import os
import random
import subprocess
import sys
import time

# Ensure httpx is available (installs in ~2s, unlike openenv-core's 60s+)
try:
    import httpx
except ImportError:
    print("[SETUP] Installing httpx...", file=sys.stderr, flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--root-user-action=ignore", "httpx>=0.28.0"],
        stdout=subprocess.DEVNULL,
    )
    import httpx


class EnvResult:
    """Minimal result object matching openenv's GenericEnvResult interface."""

    def __init__(self, data: dict):
        self.observation = data.get("observation", {})
        self.reward = float(data.get("reward", 0.0))
        self.done = bool(data.get("done", False))
        self.info = data.get("info", {})
        self.last_action_error = data.get("last_action_error") or data.get("info", {}).get("error")


class DockerEnvClient:
    """
    Lightweight async client for OpenEnv-compatible Docker environments.
    Manages container lifecycle via Docker CLI (no docker-py needed).
    """

    def __init__(self, base_url: str, container_id: str = None):
        self.base_url = base_url.rstrip("/")
        self.container_id = container_id
        self._client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        container_port: int = 7860,
        ready_timeout: float = 120.0,
    ) -> "DockerEnvClient":
        """Start a container from the given image and wait for it to be ready."""
        host_port = random.randint(30000, 59999)

        print(f"[SETUP] Starting container from {image_name} (port {host_port}:{container_port})...",
              file=sys.stderr, flush=True)

        # Start container via Docker CLI
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{host_port}:{container_port}",
                image_name,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        base_url = f"http://localhost:{host_port}"

        print(f"[SETUP] Container {container_id[:12]} started, waiting for ready...",
              file=sys.stderr, flush=True)

        # Wait for health endpoint
        deadline = time.monotonic() + ready_timeout
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                try:
                    r = await probe.get(f"{base_url}/health")
                    if r.status_code == 200:
                        print(f"[SETUP] Container ready at {base_url}",
                              file=sys.stderr, flush=True)
                        return cls(base_url, container_id)
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
                        httpx.RemoteProtocolError, httpx.ReadError, OSError, Exception):
                    pass
                await asyncio.sleep(0.5)

        # Timeout — grab logs for debugging
        logs = subprocess.run(
            ["docker", "logs", "--tail", "20", container_id],
            capture_output=True, text=True,
        )
        # Cleanup failed container
        subprocess.run(["docker", "rm", "-f", container_id],
                       capture_output=True)
        raise TimeoutError(
            f"Container did not become ready within {ready_timeout}s.\n"
            f"Last container logs:\n{logs.stdout}\n{logs.stderr}"
        )

    async def reset(self, task_name: str = None) -> EnvResult:
        """Reset the environment, optionally specifying a task."""
        body = {}
        if task_name:
            body["task_name"] = task_name
        r = await self._client.post(f"{self.base_url}/reset", json=body)
        r.raise_for_status()
        return EnvResult(r.json())

    async def step(self, action: dict) -> EnvResult:
        """Take a step in the environment with the given action."""
        body = action if isinstance(action, dict) else action.__dict__
        r = await self._client.post(f"{self.base_url}/step", json=body)
        r.raise_for_status()
        return EnvResult(r.json())

    async def close(self) -> None:
        """Stop and remove the container, close HTTP client."""
        await self._client.aclose()
        if self.container_id:
            print(f"[SETUP] Stopping container {self.container_id[:12]}...",
                  file=sys.stderr, flush=True)
            subprocess.run(
                ["docker", "rm", "-f", self.container_id],
                capture_output=True,
            )
