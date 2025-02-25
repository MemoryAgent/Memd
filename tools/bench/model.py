import requests

from enum import Enum
from pydantic import BaseModel


class Seconds(BaseModel):
    seconds: float


class PerformanceMetric(BaseModel):
    embedding_time: Seconds
    query_time: Seconds


def seconds_from_rust_duration(json: dict):
    secs = json["secs"]
    nanos = json["nanos"]
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
    texts = [x["text"] for x in corpus.values()]
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
        query_time=seconds_from_rust_duration(performance["query_cost"]),
    )
