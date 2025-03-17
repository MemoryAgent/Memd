# For QA tasks, LLM as a judge is important.
import pathlib
from pydantic import BaseModel
from openai import OpenAI


# LLMJudge is used to judge QA tasks by an openai protocol compatible LLM.
class LLMConfig(BaseModel):
    endpoint: str
    api_key: str


def build_llm_judge(secret: str = "./secret.json") -> OpenAI:
    secret_path = pathlib.Path(secret)
    if not secret_path.exists():
        raise FileNotFoundError(f"secret file {secret} not found")
    with secret_path.open("r") as f:
        secret = f.read()
    config = LLMConfig.model_validate_json(secret)
    return OpenAI(api_key=config.api_key, base_url=config.endpoint)


def judge_qa(output_answer: str, ground_truth_answer: str, llm: OpenAI) -> int:
    prompt = """
    Here are two answers to the same question.
    Please compare the output answer to the ground truth answer and evaluate their consistency. 
    Your output should be a score:
    - If the output answer is exactly same as the ground truth answer, return 10,
    - if the output answer is totally irrelevant to the ground truth answer, return 0.
    Please do not explain. Your answer should only contain a number between 0 and 10.
    ---
    Example:
    Output answer: the United States
    Ground truth answer: America
    Your answer: 10
    --- 
    Output answer: {output_answer}
    Ground truth answer: {ground_truth_answer}
    """
    question = prompt.format(
        output_answer=output_answer, ground_truth_answer=ground_truth_answer
    )
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": question}],
        stream=False,
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError(f"llm judge failed: {response}")
    print(answer)
    return int(answer)
