#!/bin/bash

work_dir=./results

mkdir -p $work_dir

python -m main --benchmark_test.kind retrieval \
    --benchmark_test.dataset tinyqa \
    --app_endpoint http://localhost:3000 \
    --save_dir ./results | tee ${work_dir}/tinyqa_retrieval.log

python -m main --benchmark_test.kind qa \
    --benchmark_test.dataset tinyqa \
    --app_endpoint http://localhost:3000 \
    --save_dir ./results | tee ${work_dir}/tinyqa_qa.log
