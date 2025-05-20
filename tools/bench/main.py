import logging
from pathlib import Path
import bench_beir
from beir.logging import LoggingHandler
from bench_prefeval import bench_prefeval
from client import RemoteModel
from config import (
    BenchmarkTask,
    get_config,
    RetrievalTask,
    QATask,
    PrefEvalTask,
    EvaluationCriterion,
)


def dispatch_bench(
    task: BenchmarkTask, rm: RemoteModel, evaluation: EvaluationCriterion
):
    if isinstance(task, RetrievalTask):
        return bench_beir.bench_retrieve_on(task, rm, evaluation)
    if isinstance(task, QATask):
        return bench_beir.bench_on_qa(task, rm, evaluation)
    if isinstance(task, PrefEvalTask):
        return bench_prefeval(rm=rm, opt=task.opt)
    raise NotImplementedError("this is unreachable")


def main():
    config = get_config()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    remote_model = RemoteModel(url=config.app_endpoint)
    benchmark_result = dispatch_bench(
        task=config.benchmark_task,
        rm=remote_model,
        evaluation=config.evaluation_criterion,
    )
    print(benchmark_result)
    save_path = Path(config.save_dir)
    import time

    save_path = save_path / str(time.time_ns())
    save_path.mkdir(exist_ok=True, parents=True)

    json_output_file = save_path / f"{config.benchmark_task.kind}.json"
    with json_output_file.open("w") as f:
        f.write(str(benchmark_result))

    import pickle

    pickle_output_file = save_path / f"{config.benchmark_task.kind}.pkl"
    with pickle_output_file.open("wb") as f:
        pickle.dump(benchmark_result[1], f)


if __name__ == "__main__":
    main()
