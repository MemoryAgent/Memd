"""
BEIR is a traditional information retrieval benchmark. It measures the performance & ability to
retrieve relevant information from a wide corpus.

Useful metrics:

mAP: accuracy.

time: embedding + query
"""

from beir import util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
from pathlib import Path

import logging

from model import RemoteModel, rm_open, rm_query, rm_store, rm_close


def _download_beir_dataset(dataset: str = "sciface", path: str = "./datasets"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    return util.download_and_unzip(url, path)


def _load_beir_dataset(dataset_path: str, dataset_name: str, download_if_missing: bool):
    path = Path(dataset_path) / dataset_name
    if not path.exists():
        if not download_if_missing:
            raise RuntimeError("no dataset, no benchmark")
        path = _download_beir_dataset(dataset=dataset_name, path=dataset_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split="test")
    return corpus, queries, qrels


def _evaluate_queries(rm: RemoteModel, corpus: dict, queries: dict, qrel: dict):
    rm_open(rm)

    rm_store(rm, corpus)
    inverted_corpus = {v["text"]: k for (k, v) in corpus.items()}
    results = {}

    for qid, query in queries.items():
        answer = rm_query(rm, query)
        aid = inverted_corpus.get(answer, -1)
        logging.info(
            f"getting query {qid} answer digest {answer[:100]} in document {aid}"
        )
        results[f"{qid}"] = {f"{aid}": 100.0}  # TODO: return confidence
    performance = rm_close(rm)
    return (
        EvaluateRetrieval.evaluate(qrels=qrel, results=results, k_values=[1]),
        performance,
    )


def _bench_on_dataset(
    dataset_name: str,
    dataset_path: str,
    rm: RemoteModel,
    download_if_missing=False,
):
    corpus, queries, qrels = _load_beir_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        download_if_missing=download_if_missing,
    )

    return _evaluate_queries(rm=rm, corpus=corpus, queries=queries, qrel=qrels)


def bench_on_scifact(rm: RemoteModel):
    _bench_on_dataset(
        dataset_name="scifact",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=True,
    )
