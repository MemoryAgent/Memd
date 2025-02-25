# Bench

Bench is a toolkit for the evaluation of personalized LLM assistant. It utilizes existing benchmarks
such as [BEIR](https://github.com/beir-cellar/beir) and [PrefEval](https://github.com/amazon-science/PrefEval).

## What is unique about this benchmark?

This benchmark has following features:

1. It is model-agnostic. You connect the model via HTTP.
2. It effectively measures the **personalized** degree of your LLM application.

## HTTP API

 Open --- the benchmark informs the model to prepare for testing
   via network -- GET /open
                  200 - OK

 Store --- the benchmark dumps its corpus(if any) into the model
   via network -- POST /store
                  200 - OK

 Query --- the benchmark queries about the result
   via network -- POST /query
                   200 - OK

 Close --- the benchmark informs the model to stop testing
   via network --- GET /close
