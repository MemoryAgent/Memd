#!/bin/bash

python -m main --benchmark_task.kind prefeval \
    --benchmark_task.opt.inter_turns 1 \
    --benchmark_task.opt.task zero-shot \
    --benchmark_task.opt.pref_form explicit \
    --benchmark_task.opt.dataset_dir ./prefeval \
    --app_endpoint http://localhost:3000 \
    --save_dir ./results \
