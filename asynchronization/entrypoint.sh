#!/bin/sh
if [ "$WORKER_TYPE" = "planner" ]; then
    exec python infrastructure/workers/planner_worker.py
elif [ "$WORKER_TYPE" = "retriever" ]; then
    exec python infrastructure/workers/retriever_worker.py
elif [ "$WORKER_TYPE" = "grader" ]; then
    exec python infrastructure/workers/grader_worker.py
elif [ "$WORKER_TYPE" = "replanner" ]; then
    exec python infrastructure/workers/replanner_worker.py
elif [ "$WORKER_TYPE" = "reasoner" ]; then
    exec python infrastructure/workers/reasoner_worker.py
else
    echo "Unknown worker type"
    exit 1
fi