from typing import List
from model import RemoteModel, rm_open, rm_close, rm_chat
from prefeval.benchmark_classification import (
    generate_prompt_sequences,
    PrefevalOptions,
    PrefForm,
    Task,
)
from prefeval.utils.utils_mcq import extract_choice


def test_prompt_sequence(rm: RemoteModel, prompt_sequence: List[str]):
    for noise_prompt in prompt_sequence[:-2]:
        rm_chat(rm=rm, prompt=noise_prompt)
    response = rm_chat(rm=rm, prompt=prompt_sequence[-1])
    answer = extract_choice(response=response)
    return 1 if answer == "A" else 0


def bench_prefeval(rm: RemoteModel):
    opt = PrefevalOptions(
        inter_turns=1,
        task=Task.ZERO_SHOT,
        pref_form=PrefForm.EXPLICIT,
        work_dir=".",
        dataset_dir="./prefeval",
    )
    prompts = generate_prompt_sequences(opt)

    rm_open(rm)
    correct_number = 0
    for prompt_sequence in prompts:
        correct_number += test_prompt_sequence(rm, prompt_sequence)
    performance = rm_close(rm)
    return (correct_number, performance)
