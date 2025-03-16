#!/bin/bash

python -m main --benchmark_task.kind retrieval \
    --benchmark_task.dataset tinyqa \
    --app_endpoint http://127.0.0.1:3000 \
    --save_dir ./results \
