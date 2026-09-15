#!/bin/sh
# Worker 容器统一入口：由 WORKER_TYPE 决定跑哪一个 Worker。
# 合法取值: planner | retriever | grader | replanner | reasoner
set -e

# 脚本所在目录即应用根目录（镜像中为 /app）
APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKERS_DIR="$APP_ROOT/asynchronization/workers"

# 统一注入 PYTHONPATH：根目录（config_loader/prompts/double_layer_plan）
# 与 asynchronization 层（kafka_utils/state_manager）都要可见
PYTHONPATH="$APP_ROOT:$APP_ROOT/asynchronization:$WORKERS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# -u 关闭输出缓冲，否则 k8s 里看不到实时日志
case "$WORKER_TYPE" in
    planner)
        exec python -u "$WORKERS_DIR/planner_worker.py"
        ;;
    retriever)
        exec python -u "$WORKERS_DIR/retriever_worker.py"
        ;;
    grader)
        exec python -u "$WORKERS_DIR/grader_worker.py"
        ;;
    replanner)
        exec python -u "$WORKERS_DIR/replanner_worker.py"
        ;;
    reasoner)
        exec python -u "$WORKERS_DIR/reasoner_worker.py"
        ;;
    *)
        echo "Unknown WORKER_TYPE: '${WORKER_TYPE}'" >&2
        echo "合法取值: planner | retriever | grader | replanner | reasoner" >&2
        exit 1
        ;;
esac