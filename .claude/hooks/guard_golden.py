#!/usr/bin/env python3
"""PreToolUse hook: block any modification of existing files in tests/golden/.

Creating a NEW golden file is allowed (validator agent); editing or overwriting an existing one is not.
Exit code 2 blocks the tool call and feeds stderr back to Claude.
"""

import json
import os
import sys

data = json.load(sys.stdin)
tool = data.get("tool_name", "")
inp = data.get("tool_input", {}) or {}
path = inp.get("file_path") or inp.get("notebook_path") or ""
norm = path.replace("\\", "/")

if "/tests/golden/" in norm or norm.startswith("tests/golden/"):
    exists = os.path.exists(path)
    if tool in ("Edit", "MultiEdit") or (tool == "Write" and exists):
        print(
            f"BLOCKED: {path} is a frozen golden oracle. Golden values must never be changed to make "
            "a test pass. If you believe the value is wrong, stop and explain the evidence to the owner "
            "(and log it in docs/OPEN_ISSUES.md).",
            file=sys.stderr,
        )
        sys.exit(2)

if tool == "Bash":
    cmd = inp.get("command", "")
    if "tests/golden" in cmd and any(k in cmd for k in ("rm ", "mv ", "sed -i", ">", "truncate", "cp ")):
        print("BLOCKED: shell commands may not modify tests/golden/.", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
