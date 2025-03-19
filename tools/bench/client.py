from typing import List, Literal, Optional
import requests

from enum import Enum, StrEnum
from pydantic import BaseModel, TypeAdapter


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
#
# Chat --- the benchmark simply chat with the model
#   via network -- POST /chat
#                  200 - OK
#
# Close --- the benchmark informs the model to stop testing
#   via network --- GET /close
#


class RemoteState(Enum):
    CLOSED = 0
    OPEN = 1


class RemoteModel(BaseModel):
    url: str
    state: RemoteState = RemoteState.CLOSED


class NaiveRAGOptions(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int


class MemdAgentOption(BaseModel):
    chunk_size: int
    chunk_overlap: int
    retrieve_top_k: int
    ranking_top_k: int


MemdOpt = (
    NaiveRAGOptions
    | MemdAgentOption
    | Literal["HippoRAG"]
    | Literal["NoRAG"]
    | Literal["Raptor"]
    | Literal["ReadAgent"]
    | Literal["NoRAG"]
)

DEFAULT_RAG_OPTION = NaiveRAGOptions(
    chunk_size=512,
    chunk_overlap=0,
    top_k=5,
)


def opt_to_json(opt: MemdOpt) -> dict:
    def get_discriminator(opt: NaiveRAGOptions | MemdAgentOption) -> str:
        if isinstance(opt, NaiveRAGOptions):
            return "NaiveRAG"
        if isinstance(opt, MemdAgentOption):
            return "MemdAgent"

    def opt_dump() -> dict | str:
        if isinstance(opt, (NaiveRAGOptions, MemdAgentOption)):
            return {get_discriminator(opt): opt.model_dump()}
        else:
            return opt

    metadata = dict()
    metadata["opt"] = opt_dump()
    return metadata


def rm_open(rm: RemoteModel, additional_metadata: Optional[MemdOpt] = None) -> bool:
    resp = requests.post(
        f"{rm.url}/open",
        json=opt_to_json(additional_metadata or DEFAULT_RAG_OPTION),
    )
    if resp.content.decode("utf-8") == "happy for challenge.":
        rm.state = RemoteState.OPEN
        return True
    raise RuntimeError(
        f"""failed to open remote model {rm.url} with {additional_metadata}.
        error message: {resp}
        """
    )


class StorePayload(BaseModel):
    title: Optional[str]
    content: str


def rm_store(rm: RemoteModel, payload: StorePayload) -> bool:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/store", json=payload.model_dump())
    if resp.content.decode("utf-8") == "added":
        return True
    print(f"model_dump_json {payload.model_dump_json()}, err {resp.content.decode()}")
    assert False


class Document(BaseModel):
    name: str
    content: str


class QueryResult(BaseModel):
    document: Document
    conf_score: float


query_results_adapter = TypeAdapter(List[QueryResult])


def rm_query(rm: RemoteModel, query: str) -> List[QueryResult]:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/query", query)
    decoded_resp = resp.content.decode("utf-8")
    return query_results_adapter.validate_json(decoded_resp)


def rm_chat(rm: RemoteModel, prompt: str) -> str:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/chat", prompt)
    return resp.content.decode("utf-8")


class Seconds(BaseModel):
    seconds: float


class PerformanceMetric(BaseModel):
    embedding_time: Seconds
    query_time: Seconds


class RequestType(StrEnum):
    Store = "Store"
    Query = "Query"
    Chat = "Chat"


class PerRequestMetricData(BaseModel):
    kind: RequestType
    time_cost: Seconds
    storage_memory_usage: int
    total_memory_usage_before: int
    total_memory_usage_after: int


def seconds_from_rust_duration(json: dict):
    secs = json["secs"]
    nanos = json["nanos"]
    return Seconds(seconds=secs + nanos / 1e9)


def rm_close(rm: RemoteModel) -> List[PerRequestMetricData]:
    assert rm.state == RemoteState.OPEN
    resp = requests.post(f"{rm.url}/close")
    performance = resp.json()
    all_metrics = []
    for req_prof in performance["request_metrics"]:
        all_metrics.append(
            PerRequestMetricData(
                kind=req_prof["kind"],
                time_cost=seconds_from_rust_duration(req_prof["time_cost"]),
                storage_memory_usage=req_prof["storage_memory_usage"],
                total_memory_usage_before=req_prof["total_memory_usage_before"],
                total_memory_usage_after=req_prof["total_memory_usage_after"],
            )
        )
    return all_metrics
