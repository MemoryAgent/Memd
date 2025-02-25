import json
import os
import sys
from pydantic import BaseModel, Field

from prefeval.utils.common_utils import extract_multi_turn_message
from prefeval.utils.explicit_utils import create_user_pref_message
from prefeval.utils.utils_mcq import (
    get_question_prompt_mcq,
)
from prefeval.utils.data_loading_utils import (
    load_turns_data,
)
from enum import StrEnum


class Task(StrEnum):
    ZERO_SHOT = "zero-shot"
    COT = "cot"
    REMIND = "remind"
    RAG = "rag"
    SELFCRITIC = "selfcritic"


class ImplicitPrefType(StrEnum):
    PERSONA = "persona"
    CHOICE = "choice"


class PrefForm(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class PrefevalOptions(BaseModel):
    inter_turns: int = Field(description="number of turns during conversation")

    topic: str = Field("travel_restaurant")

    task: Task

    pref_type: ImplicitPrefType

    pref_form: PrefForm

    system_prompt: str = Field("You are a helpful assistant")

    work_dir: str

    dataset_dir: str = Field(description="place of prefeval/benchmark_dataset")

    max_tokens: int = Field(5)


def load_mcq_data(dataset_path: str, topic: str):
    with open(f"{dataset_path}/benchmark_dataset/mcq_options/{topic}.json") as f:
        return json.load(f)


def explicit_mode(opt: PrefevalOptions, turns_data):
    topic_data = load_mcq_data(dataset_path=opt.dataset_dir, topic=opt.topic)
    multi_inter_message, multi_turn_message = extract_multi_turn_message(
        turns_data=turns_data, inter_turns=opt.inter_turns
    )
    for task_id, task in enumerate(topic_data):
        preference = task["preference"]
        prompt = get_question_prompt_mcq(
            preference=preference,
            options=task["classification_task_options"],
            pref_generation="",
            question=task["question"],
            multi_inter_message=multi_inter_message,
            turn_number=opt.inter_turns,
        )
        print(f"{task_id} is {prompt}")


def implicit_mode():
    pass


# dataset path -> topic_data, save_file
def load_dialogue_data():
    pass


def bench_prefeval(opt: PrefevalOptions):
    if opt.task != Task.ZERO_SHOT:
        raise RuntimeError("not implemented.")
    if opt.pref_type != PrefForm.EXPLICIT:
        raise RuntimeError("not implemented.")
    turns_data = load_turns_data(dataset_dir=opt.dataset_dir)
    mcq_data = load_mcq_data(dataset_path=opt.dataset_dir, topic=opt.topic)
