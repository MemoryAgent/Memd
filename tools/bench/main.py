import logging
import bench_beir

from beir.logging import LoggingHandler
from bench_prefeval import bench_prefeval
from model import RemoteModel
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, CliApp
from enum import StrEnum


class BenchmarkTarget(StrEnum):
    BEIR = "beir"
    PERSONA = "persona"


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


def dispatch_bench(bench_target: BenchmarkTarget, remote_model: RemoteModel):
    match bench_target:
        case BenchmarkTarget.BEIR:
            return bench_beir.bench_on_all(remote_model)
        case BenchmarkTarget.PERSONA:
            return bench_prefeval(remote_model)


def main():
    config = get_config()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    remote_model = RemoteModel(url=config.app_endpoint)
    print(
        dispatch_bench(bench_target=config.benchmark_target, remote_model=remote_model)
    )


if __name__ == "__main__":
    main()
