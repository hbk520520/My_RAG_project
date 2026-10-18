"""Container-only runner. Importing it does not execute code."""
import base64
import io
import json
import sys


class CappedOutput(io.StringIO):
    def __init__(self, limit=65536):
        super().__init__()
        self.limit = limit
        self.used = 0

    def write(self, text):
        self.used += len(text.encode("utf-8"))
        if self.used > self.limit:
            raise ValueError("Output limit exceeded")
        return super().write(text)


def calculate(code, limit=65536):
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
        return {"output": value, "error": None}
    except Exception as exc:
        return {"output": "", "error": f"{type(exc).__name__}: {exc}"[:4096], "error_kind": "execution"}
    finally:
        sys.stdout = old


if __name__ == "__main__":
    code = base64.b64decode(sys.argv[1], validate=True).decode("utf-8")
    print(json.dumps(calculate(code), ensure_ascii=True))
