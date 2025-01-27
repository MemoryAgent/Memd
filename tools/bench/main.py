from enum import Enum
from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import requests

import logging
import pathlib, os
import random
import json

from pydantic import BaseModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "scifact"

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=".\\datasets\\tinyscifact").load(split="test")

REMOTE_URL = "http://localhost:3000"

class Seconds(BaseModel):
    seconds: float

class PerformanceMetric(BaseModel):
    embedding_time: Seconds
    query_time: Seconds

def seconds_from_rust_duration(json: dict):
    secs = json['secs']
    nanos = json['nanos']
    return Seconds(seconds=secs + nanos / 1e9)

class RemoteState(Enum):
    CLOSED = 0
    OPEN = 1

# testing interface
# for retrieval methods, there are four methods,
#
# Open --- the benchmark informs the model to prepare for testing
#   via network -- GET /open
#                  200 - OK
#
# Store --- the benchmark dumps its corpus into the model
#   via network -- POST /store
#                  200 - OK
#
# Query --- the benchmark queries about the result
#   via network -- POST /query
#                   200 - OK
# Close --- the benchmark informs the model to stop testing
#   via network --- GET /close
#
class RemoteModel(BaseModel):
    url: str
    state: RemoteState = RemoteState.CLOSED

def rm_open(rm: RemoteModel) -> bool:
    resp = requests.post(f"{rm.url}/open")
    if resp.content.decode("utf-8") == "happy for challenge.":
        rm.state = RemoteState.OPEN
        return True
    return False

def rm_store(rm: RemoteModel, corpus: dict[int, dict[str, str]]) -> bool:
    assert rm.state == RemoteState.OPEN
    texts = [x['text'] for x in corpus.values()]
    resp = requests.post(f"{rm.url}/store", json=texts)
    if resp.content.decode("utf-8") == "added":
        return True
    return False

def rm_query(rm: RemoteModel, query: str) -> str:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/query", query)
    return resp.content.decode("utf-8")

def rm_close(rm: RemoteModel) -> PerformanceMetric:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/close")
    performance = resp.json()
    return PerformanceMetric(
        embedding_time=seconds_from_rust_duration(performance["embedding_cost"]),
        query_time=seconds_from_rust_duration(performance["query_cost"])
    )
    
def evaluate_queries(rm:  RemoteModel, corpus: dict, queries: dict, qrel: dict):
    rm_open(rm) # TODO: context manager?
    
    rm_store(rm, corpus)
    inverted_corpus = {
       v["text"]:k for (k, v) in corpus.items()
    }
    results = {}
    # Fix rust embedding col issue [batch_index, token, dim]
    for (qid, query) in queries.items():
        answer = rm_query(rm, query)
        aid = inverted_corpus.get(answer, -1)
        logging.info(f"getting query {qid} answer digest {answer[:100]} in document {aid}")
        results[f"{qid}"] = { f"{aid}": 100.0 } # TODO: return confidence
    performance = rm_close(rm)
    return EvaluateRetrieval.evaluate(qrels=qrel, results=results, k_values=[1]), performance

# Slow? 2k text x 5 ~ 20s
# speed algorithm or model deployment
#
if __name__ == "__main__":
    bench = RemoteModel(url=REMOTE_URL)
    sol = evaluate_queries(bench, corpus, queries, qrels)
    print(sol)
