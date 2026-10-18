"""Private per-model residency scopes; no public inference signature changes."""

from contextvars import ContextVar
from pathlib import Path
import json
import subprocess
import sys


ACTIVE_MODEL_SESSION = ContextVar("aging_comparison_model_session", default=None)
_RESULT_PREFIX = "__AGING_RESIDENT_RESULT__"


class ModelSession:
    """Keep only the current baseline resident until its phase finishes."""

    def __init__(self, bundle, name):
        self.bundle = bundle or {}
        self.name = name
        self.worker = None
        self.placements = {}
        self.token = None

    def __enter__(self):
        self.token = ACTIVE_MODEL_SESSION.set(self)
        return self

    def owns(self, model):
        entry = self.bundle.get(self.name) or {}
        return entry.get("model") is model

    def place(self, model, device):
        if self.placements.get(id(model)) != str(device):
            model = model.to(device)
            self.placements[id(model)] = str(device)
        model.eval()
        return model

    def accepts_command(self, cmd, cwd):
        entry = self.bundle.get(self.name) or {}
        return (
            self.name in {"FADING", "HRFAE"} and len(cmd) >= 2
            and Path(cmd[1]).resolve() == Path(entry["script"]).resolve()
        )

    def run_command(self, cmd, cwd, env):
        if self.worker is None or self.worker.poll() is not None:
            runner = Path(__file__).with_name("resident_worker.py")
            self.worker = subprocess.Popen(
                [sys.executable, "-u", str(runner), self.name], cwd=cwd, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            print(f"[{self.name}] Resident worker started", flush=True)
        request = {"cmd": list(map(str, cmd)), "cwd": str(cwd) if cwd else None}
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        for line in self.worker.stdout:
            if line.startswith(_RESULT_PREFIX):
                result = json.loads(line[len(_RESULT_PREFIX):])
                if not result["ok"]:
                    raise RuntimeError(f"{self.name}: {result['error']}")
                return subprocess.CompletedProcess(cmd, 0, stdout="")
            print(line, end="", flush=True)
        raise RuntimeError(f"{self.name} resident worker exited unexpectedly")

    def close(self):
        if self.worker is not None:
            try:
                if self.worker.poll() is None:
                    try:
                        self.worker.stdin.write(json.dumps({"close": True}) + "\n")
                        self.worker.stdin.flush()
                    except OSError:
                        pass
                    try:
                        self.worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.worker.terminate()
                        try:
                            self.worker.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.worker.kill()
                            self.worker.wait(timeout=5)
            finally:
                self.worker.stdin.close()
                self.worker.stdout.close()
                self.worker = None
        entry = self.bundle.get(self.name) or {}
        model = entry.get("model")
        if model is not None and id(model) in self.placements:
            entry["model"] = model.to("cpu")
        self.placements.clear()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            ACTIVE_MODEL_SESSION.reset(self.token)
