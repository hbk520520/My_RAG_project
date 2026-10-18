"""
沙箱执行服务 —— Docker 里的 Python 解释器
=======================================
跑在断网的容器里，接代码→执行→返回结果。persistent_globals 保证多次执行共享变量，
适合"写完代码→执行→拿到 result 变量→继续"的有状态场景。

技术栈: FastAPI / uvicorn / io (stdout 捕获)
"""
import sys, io, traceback
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 常驻内存的全局变量字典 —— 实现状态保持的关键
persistent_globals = {}

class CodeRequest(BaseModel):
    code: str

@app.post("/execute")
def execute_code(req: CodeRequest):
    """在持久化上下文中执行代码，并捕获标准输出和错误堆栈"""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    error_msg = None

    try:
        exec(req.code, persistent_globals)
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return {
        "output": redirected_output.getvalue().strip(),
        "error": error_msg
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)