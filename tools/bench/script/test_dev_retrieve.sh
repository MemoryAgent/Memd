#!/bin/bash

python -m main --benchmark_task.kind retrieval \
    --benchmark_task.dataset tinyqa \
    --app_endpoint http://localhost:3000 \
    --save_dir ./results \
