import logging
from enum import StrEnum
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, CliApp
from prefeval.benchmark_classification import PrefevalOptions


class RetrievalTask(BaseModel):
    kind: Literal["retrieval"]
    dataset: Literal["all", "scifact", "quora", "hotpotqa", "nq", "tinyqa"]


class QATask(BaseModel):
    kind: Literal["qa"]
    dataset: Literal["hotpotqa", "tinyqa"]
    llm_judge_secret: Optional[
        str
    ]  # If llm is not available, it uses rule-based evaluator which is very bad.


class PrefEvalTask(BaseModel):
    kind: Literal["prefeval"]
    opt: PrefevalOptions


class EvaluationCriterion(StrEnum):
    BULK_BUILD_EFFICIENCY = "bulk_build_efficiency"
    QUERY_ACCURACY = "query_accuracy"
    QUERY_EFFICIENCY = "query_efficiency"


BenchmarkTask = RetrievalTask | QATask | PrefEvalTask


class BenchConfig(
    BaseSettings,
    cli_parse_args=True,
    cli_prog_name="Personalized LLM model benchmark",
):
    benchmark_task: BenchmarkTask = Field(discriminator="kind")

    evaluation_criterion: EvaluationCriterion = Field(
        EvaluationCriterion.BULK_BUILD_EFFICIENCY
    )

    app_endpoint: str = Field(
        description="the endpoint of your favorite LLM application."
    )

    save_dir: str = Field(
        "./results", description="where to save your benchmark results."
    )


def get_config() -> BenchConfig:
    try:
        config = CliApp.run(BenchConfig)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config
