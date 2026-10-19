"""Versioned in-guest runner used by the SPACE backend.

The runner is embedded in the command line so the guest template only needs a
``python3`` interpreter; no manual file staging or template mutation is needed.
It intentionally mirrors ``runner.py``: prefer the ``result`` variable, else
captured stdout, and always emit one JSON object on stdout.
"""
from __future__ import annotations

RUNNER_VERSION = "legal-sandbox-space-runner/1"

SPACE_RUNNER_SOURCE = r'''
import base64
import io
import json
import sys


class CappedOutput(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.used = 0

    def write(self, text):
        self.used += len(text.encode("utf-8"))
        if self.used > self.limit:
            raise ValueError("Output limit exceeded")
        return super().write(text)


def calculate(code, limit):
    output = CappedOutput(limit)
    old = sys.stdout
    namespace = {}
    try:
        sys.stdout = output
        exec(code, namespace)
        value = str(namespace["result"]) if "result" in namespace else output.getvalue().strip()
        if not value:
            raise ValueError("Calculation produced neither result nor output")
        if len(value.encode("utf-8")) > limit:
            raise ValueError("Result output limit exceeded")
        return {"output": value, "error": None, "error_kind": None}
    except Exception as exc:
        return {"output": "", "error": f"{type(exc).__name__}: {exc}"[:4096],
                "error_kind": "execution"}
    finally:
        sys.stdout = old


def main():
    limit = int(sys.argv[2])
    if limit < 1:
        raise ValueError("limit must be positive")
    code = base64.b64decode(sys.argv[1], validate=True).decode("utf-8")
    sys.stdout.write(json.dumps(calculate(code, limit), ensure_ascii=True))


main()
'''.strip()
