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
)


def dispatch_bench(task: BenchmarkTask, rm: RemoteModel):
    if isinstance(task, RetrievalTask):
        return bench_beir.bench_retrieve_on(task, rm)
    if isinstance(task, QATask):
        return bench_beir.bench_on_qa(task, rm)
    if isinstance(task, PrefEvalTask):
        return bench_prefeval(rm)
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
    benchmark_result = dispatch_bench(task=config.benchmark_task, rm=remote_model)
    print(benchmark_result)
    save_path = Path(config.save_dir)
    import datetime

    save_path = save_path / datetime.time().strftime("%Y-%m-%d_%H-%M-%S")
    save_path.mkdir(exist_ok=True, parents=True)
    save_path = save_path / f"{config.benchmark_task.kind}.json"

    with save_path.open("w") as f:
        f.write(str(benchmark_result))


if __name__ == "__main__":
    main()
