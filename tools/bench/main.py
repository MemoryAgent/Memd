import logging

from beir.logging import LoggingHandler
from bench_beir import _bench_on_dataset
from model import RemoteModel
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, CliApp
from typing import Literal
from enum import StrEnum


class BenchmarkTarget(StrEnum):
    BEIR = Literal["beir"]
    PERSONA = Literal["persona"]


class BenchConfig(
    BaseSettings,
    cli_parse_args=True,
    cli_prog_name="Personalized LLM model benchmark",
):
    benchmark_target: BenchmarkTarget = Field(description="use beir as a benchmark.")

    app_endpoint: str = Field(
        description="the endpoint of your favorite LLM application."
    )


def get_config() -> BenchConfig:
    try:
        config = CliApp.run(BenchConfig)
    except ValidationError as e:
        logging.fatal(e)
        exit(-1)
    return config


def bench_dispatcher(bench_target: BenchmarkTarget, remote_model: RemoteModel):
    match bench_target:
        case BenchmarkTarget.BEIR:
            return


def main():
    config = get_config()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    remote_model = RemoteModel(url=config.app_endpoint)


if __name__ == "__main__":
    main()
