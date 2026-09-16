"""Synchronous Python client for the isolated Mineflayer JSONL process."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Self


class MinecraftBridgeError(RuntimeError):
    pass


class MinecraftBridgeClient:
    def __init__(
        self,
        bridge_script: Path | None = None,
        *,
        node_executable: str = "node",
        timeout: float = 30.0,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.bridge_script = bridge_script or (
            project_root / "bridge" / "mineflayer" / "server.cjs"
        )
        if not self.bridge_script.exists():
            raise FileNotFoundError(self.bridge_script)
        if shutil.which(node_executable) is None:
            raise MinecraftBridgeError(
                f"Node.js executable not found: {node_executable}"
            )
        self.timeout = timeout
        self._request_id = 0
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [node_executable, str(self.bridge_script)],
            cwd=self.bridge_script.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise MinecraftBridgeError(
                    f"bridge exited with {self._process.returncode}: {self.stderr_tail}"
                )
            self._request_id += 1
            request_id = self._request_id
            assert self._process.stdin is not None
            self._process.stdin.write(
                json.dumps(
                    {"id": request_id, "method": method, "params": params or {}},
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._process.stdin.flush()
            try:
                response = self._responses.get(timeout=timeout or self.timeout)
            except queue.Empty as exc:
                raise MinecraftBridgeError(
                    f"bridge request timed out: {method}; stderr={self.stderr_tail}"
                ) from exc
            if response.get("id") != request_id:
                raise MinecraftBridgeError("bridge response ID mismatch")
            if not response.get("ok"):
                raise MinecraftBridgeError(
                    str(response.get("error", "unknown bridge error"))
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise MinecraftBridgeError("bridge result must be a JSON object")
            return result

    @property
    def stderr_tail(self) -> str:
        return " | ".join(self._stderr)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self.request("disconnect", timeout=5.0)
        except MinecraftBridgeError:
            pass
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5.0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr.append(f"invalid stdout: {line.rstrip()}")

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip())
