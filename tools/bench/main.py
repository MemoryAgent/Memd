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

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

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

def rm_open(rm: RemoteModel):
    pass