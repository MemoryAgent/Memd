"""
BEIR is a traditional information retrieval benchmark. It measures the performance & ability to
retrieve relevant information from a wide corpus.

Useful metrics:

mAP: accuracy.

time: embedding + query
"""

from typing import Dict
from beir import util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
from pathlib import Path

import logging

from openai import OpenAI

import client
import config
import llm_judge


def _download_beir_dataset(dataset: str = "scifact", path: str = "./datasets"):
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


def _dump_corpus(rm: client.RemoteModel, corpus: dict[int, dict[str, str]]) -> bool:
    for i, x in enumerate(corpus.values()):
        title = x.get("title", None)
        content = x.get("text")
        result = client.rm_store(rm, client.StorePayload(title=title, content=content))
        if result is False:
            return False
    return True


def _evaluate_retrieves(
    rm: client.RemoteModel, corpus: dict, queries: dict, qrel: dict
):
    client.rm_open(rm)

    _dump_corpus(rm=rm, corpus=corpus)
    inverted_corpus = {v["text"]: k for (k, v) in corpus.items()}
    results: Dict[str, Dict[str, float]] = {}

    for i, (qid, query) in enumerate(queries.items()):
        retrieved_docs = client.rm_query(rm, query)
        for query_result in retrieved_docs:
            aid = inverted_corpus.get(query_result.document.content, -1)
            if aid == -1:
                logging.warning(
                    f"getting unrecognized document {query_result.document}"
                )
                continue
            logging.info(
                f"getting query {qid} answer digest {query_result.document.content[:100]} in document {aid}"
            )
            question_dict = results.setdefault(f"{qid}", {})
            question_dict[f"{aid}"] = query_result.conf_score
    performance = client.rm_close(rm)
    # TODO: make this more explicit. currently use whole message from BEIR as the evaluated performance
    return (
        EvaluateRetrieval.evaluate(qrels=qrel, results=results, k_values=[5]),
        performance,
    )


def _bench_retrieval_on_dataset(
    dataset_name: str,
    dataset_path: str,
    rm: client.RemoteModel,
    download_if_missing=False,
):
    corpus, queries, qrels = _load_beir_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        download_if_missing=download_if_missing,
    )

    return _evaluate_retrieves(rm=rm, corpus=corpus, queries=queries, qrel=qrels)


# TODO: simplify these code.
def bench_on_scifact(rm: client.RemoteModel):
    return _bench_retrieval_on_dataset(
        dataset_name="scifact",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=True,
    )


def bench_on_quora(rm: client.RemoteModel):
    return _bench_retrieval_on_dataset(
        dataset_name="quora",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=True,
    )


def bench_on_hotpotqa(rm: client.RemoteModel):
    return _bench_retrieval_on_dataset(
        dataset_name="hotpotqa",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=True,
    )


def bench_on_natural_questions(rm: client.RemoteModel):
    return _bench_retrieval_on_dataset(
        dataset_name="nq",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=True,
    )


# TinyQA is a tiny subset of hotpotQA. It is used for testing, not a real benchmark, so it is
# excluded from all.
def bench_on_tinyqa(rm: client.RemoteModel):
    return _bench_retrieval_on_dataset(
        dataset_name="tinyqa",
        dataset_path="./datasets",
        rm=rm,
        download_if_missing=False,
    )


def bench_on_all_retrieval(rm: client.RemoteModel):
    return [
        x(rm)
        for x in [
            bench_on_scifact,
            bench_on_quora,
            bench_on_hotpotqa,
            bench_on_natural_questions,
        ]
    ]


bench_retrievals = {
    "all": bench_on_all_retrieval,
    "scifact": bench_on_scifact,
    "quora": bench_on_quora,
    "hotpotqa": bench_on_hotpotqa,
    "nq": bench_on_natural_questions,
    "tinyqa": bench_on_tinyqa,
}


def bench_retrieve_on(task: config.RetrievalTask, rm: client.RemoteModel):
    bench_function = bench_retrievals[task.dataset]
    return bench_function(rm)


# bench QA


class QADataLoader(GenericDataLoader):

    def _load_queries(self):
        import json

        with open(self.query_file, encoding="utf-8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = {
                    "text": line.get("text"),
                    "answer": line.get("metadata").get("answer"),
                }


def _load_qa_dataset(dataset_path: str, dataset_name: str, download_if_missing: bool):
    path = Path(dataset_path) / dataset_name
    if not path.exists():
        if not download_if_missing:
            raise RuntimeError("no dataset, no benchmark")
        path = _download_beir_dataset(dataset=dataset_name, path=dataset_path)

    corpus, queries, qrels = QADataLoader(data_folder=path).load(split="test")
    return corpus, queries, qrels


def normalize(s: str) -> str:
    import re

    def remove_articles(s: str) -> str:
        s = re.sub(r"\b(a|an|the)\b", "", s)
        return s

    def remove_whitespaces(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def remove_punctuations(s: str) -> str:
        s = re.sub(r"[^\w\s]", "", s)
        return s

    def to_lowerspace(s: str) -> str:
        return s.lower()

    return remove_whitespaces(remove_articles(remove_punctuations(to_lowerspace(s))))


def _make_qa_prompt(s: str) -> str:
    prompt = """
    Please answer my question directly. Do not explain. A word or a phrase is enough.
    {question}
    """
    return prompt.format(question=s)


def _evaluate_qa_by_rules(rm: client.RemoteModel, corpus: dict, queries: dict) -> float:
    client.rm_open(rm)

    _dump_corpus(rm=rm, corpus=corpus)
    ground_truth_answers = {k: normalize(v["answer"]) for (k, v) in queries.items()}

    correct_count = 0
    total_count = len(queries)
    for k, v in queries.items():
        prompt = _make_qa_prompt(v["text"])
        results = client.rm_chat(rm=rm, prompt=prompt)
        if normalize(results) == ground_truth_answers[k]:
            correct_count += 1

    return correct_count / total_count


def _evaluate_qa_by_llm(
    rm: client.RemoteModel, corpus: dict, queries: dict, llm: OpenAI
) -> float:
    client.rm_open(rm)

    _dump_corpus(rm=rm, corpus=corpus)
    ground_truth_answers = {k: normalize(v["answer"]) for (k, v) in queries.items()}

    correct_count = 0
    total_count = len(queries)
    for k, v in queries.items():
        prompt = _make_qa_prompt(v["text"])
        results = client.rm_chat(rm=rm, prompt=prompt)
        if llm_judge.judge_qa(
            output_answer=results, ground_truth_answer=ground_truth_answers[k], llm=llm
        ):
            correct_count += 1

    return correct_count / total_count


def bench_on_qa(task: config.QATask, rm: client.RemoteModel):
    corpus, queries, _ = _load_qa_dataset(
        dataset_name=task.dataset, dataset_path="./datasets", download_if_missing=True
    )
    if task.llm_judge_secret is None:
        return _evaluate_qa_by_rules(rm=rm, corpus=corpus, queries=queries)
    return _evaluate_qa_by_llm(
        rm=rm,
        corpus=corpus,
        queries=queries,
        llm=llm_judge.build_llm_judge(secret=task.llm_judge_secret),
    )
