#!/usr/bin/env python3
"""PostToolUse hook: after any edit to a .py file in src/ or tests/, run ruff + fast tests.

On failure, exit 2 so the failure output is fed back to Claude immediately.
"""

import json
import os
import subprocess
import sys

data = json.load(sys.stdin)
inp = data.get("tool_input", {}) or {}
path = (inp.get("file_path") or "").replace("\\", "/")

if not path.endswith(".py") or not any(seg in path for seg in ("/src/", "/tests/")):
    sys.exit(0)

root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
# Prefer the project virtualenv (ruff/pytest live there); fall back to the hook's own interpreter.
_venv_py = os.path.join(root, ".venv", "bin", "python")
PY = _venv_py if os.path.exists(_venv_py) else sys.executable


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)


try:
    lint = run([PY, "-m", "ruff", "check", "--quiet", path], 30)
    ruff_missing = "No module named ruff" in lint.stderr
    if lint.returncode == 1 and not ruff_missing:  # 1 = lint errors
        print(f"ruff found issues in {path}:\n{lint.stdout[-3000:]}", file=sys.stderr)
        sys.exit(2)

    tests = run([PY, "-m", "pytest", "-m", "not slow", "-x", "-q", "--no-header"], 180)
    # 5 = no tests collected: fine early in the project
    if tests.returncode not in (0, 5):
        print(f"Fast tests FAILED after editing {path}:\n{tests.stdout[-4000:]}", file=sys.stderr)
        sys.exit(2)
except subprocess.TimeoutExpired:
    print("Fast test suite exceeded 180 s: mark long tests with @pytest.mark.slow.", file=sys.stderr)
    sys.exit(2)

sys.exit(0)
