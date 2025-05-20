#!/bin/bash

python -m main --benchmark_task.kind retrieval \
    --benchmark_task.dataset scifact \
    --evaluation_criterion bulk_build_efficiency \
    --app_endpoint http://localhost:3000 \
    --save_dir ./results \
