import matplotlib.pyplot as plt
import pickle
from typing import List

from client import PerRequestMetricData, RequestType


def read_performance_metrics(path: str) -> List[PerRequestMetricData]:
    with open(path, "rb") as f:
        metrics = pickle.load(f)
    return metrics


def select_by_req_type(
    metrics: List[PerRequestMetricData], req_type: RequestType
) -> List[PerRequestMetricData]:
    return [metric for metric in metrics if metric.kind == req_type]


def plot_index_memory_usage(metrics: List[PerRequestMetricData]) -> None:
    store_metrics = select_by_req_type(metrics, RequestType.Store)
    index_memory_usages = [x.storage_memory_usage for x in store_metrics]
    y = index_memory_usages
    x = list(range(len(y)))

    plt.plot(x, y, label="Index Memory Usage")

def plot_query_time_usage(metrics: List[PerRequestMetricData]) -> None:
    query_metrics = select_by_req_type(metrics, RequestType.Query)
    query_time_usages = [x.time_cost.seconds for x in query_metrics]
    y = query_time_usages
    x = list(range(len(y)))

    plt.plot(x, y, label="Query Time Usage")
