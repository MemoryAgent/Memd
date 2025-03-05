import os
import json
import time
import random
from bs4 import BeautifulSoup
import re


def get_mcq_question_format(options):
    """
    Generate the question format with provided options.

    :param options: List of options to be included in the question format
    :return: A formatted MCQ question string
    """
    formatted_options = format_options(options)  # Define formatted_options here

    mcq_question_format = f"""
    I'm trying to decide on this and here are 4 options for my query: \n{formatted_options}\nNow, I'd like you to pick one of them as your top recommendation for me.
    Important instructions for your response:
    1. Choose only one option (A, B, C, or D) that best matches my preferences.
    2. Your answer must be one of these options.
    3. Don't say things like "I can't choose" or suggest alternatives not listed.
    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:
       <choice>[A/B/C/D]</choice>
    """

    return mcq_question_format


def extract_choice(response):
    """
    Extract the choice (A, B, C, or D) from the LLM's response.

    :param response: String containing the LLM's response
    :return: The extracted choice or None if no valid choice is found
    """
    # Create a BeautifulSoup object
    soup = BeautifulSoup(response, "html.parser")
    # Find the choice tag
    choice_tag = soup.find("choice")

    try:
        if choice_tag:
            # Extract the content of the choice tag
            choice_content = choice_tag.string
            # Use regex to find the letter
            match = re.search(r"[ABCD]", choice_content)

            if match:
                return match.group(0)
    except:
        return None


def extract_choice_mistral(response):
    """
    Extract the choice (A, B, C, or D) from the LLM's JSON response.

    :param response: String containing the LLM's response
    :return: The extracted choice or None if no valid choice is found
    """
    try:
        # Try to parse the entire response as JSON
        data = json.loads(response.strip())
        if isinstance(data, dict) and "choice" in data:
            choice = data["choice"].strip().upper()
            if choice in ["A", "B", "C", "D"]:
                return choice
    except json.JSONDecodeError:
        # If parsing fails, try to find a JSON object within the response
        import re

        json_match = re.search(r"\{.*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "choice" in data:
                    choice = data["choice"].strip().upper()
                    if choice in ["A", "B", "C", "D"]:
                        return choice
            except json.JSONDecodeError:
                pass

    # If JSON parsing fails, fall back to searching for A, B, C, or D
    choice_match = re.search(r"\b[ABCD]\b", response, re.IGNORECASE)
    if choice_match:
        return choice_match.group().upper()

    return None


def shuffle_options(options, seed=41):
    """
    Note: In the MCQ datasets, the first choice in the JSON file is the correct answer.
    This function shuffles the options randomly while keeping track of the correct answer.

    :param options: List of options where the first option is the correct answer
    :return: Tuple containing the shuffled options and the index of the correct answer
    """
    if seed is not None:
        random.seed(seed)
    correct_answer = options[0]
    shuffled_options = random.sample(options, len(options))
    correct_index = shuffled_options.index(correct_answer)

    return shuffled_options, correct_index


def convert_top_k_sentences_to_msg(top_k_sentences):
    msg = (
        "Before answering my question, please consider the following context from our previous conversations. "
        "These are the 5 most relevant exchanges that we had previously, which may contain information about "
        "my preferences or prior discussions related to my query:\n\n"
        "#Start of Context#\n"
    )
    for idx, sentence in enumerate(top_k_sentences, 1):
        msg += f"exchange {idx}. {sentence}\n"

    msg += (
        "#End of Context#\n\n"
        "Please use this context to inform your answer and adhere to any preferences I've expressed "
        "that are relevant to the current query. Note that not all contexts are useful for answering "
        "my question and there may be no context that is useful. Now, please address my question:\n\n"
    )
    return msg


def check_file_exists(save_file, total_len):
    if os.path.exists(
        save_file,
    ):
        with open(
            save_file,
            "r",
        ) as infile:
            already_saved_data = json.load(infile)
        if len(already_saved_data) == total_len:
            print(f"Already saved enough data of {total_len}, Skipping evaluation.")
            return True
        else:
            print("only have ", len(already_saved_data))
            return False
    return False


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")
        print()


def load_files_explicit(args):
    # system prompt:
    with open(
        f"{args.dir}/preference_dataset/finished_topics/{args.topic}.json",
        "r",
    ) as infile:
        existing_data = json.load(infile)
    # check if this directory does not exist, create one:
    dir_path = f"{args.dir}/benchmark_results/explicit/{args.task}/{args.topic}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_file = f"{args.dir}/benchmark_results/explicit/{args.task}/{args.topic}/{args.model}_{args.topic}_{args.inter_turns+2}turn.json"
    return existing_data, save_file


def generate_message(
    bedrock_runtime,
    model_id,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
):
    retries = 0
    while retries < max_retries:
        try:
            if model_type == "claude":
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": messages,
                        "temperature": temperature,
                    }
                )
                response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
                response_body = json.loads(response.get("body").read())
                return response_body["content"][0]["text"]

            elif model_type == "mistral":
                native_request = {
                    "prompt": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                outputs = model_response.get("outputs")
                response_text = outputs[0]["text"]
                return response_text

            elif model_type == "llama":
                native_request = {
                    "prompt": messages,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                }
                request = json.dumps(native_request)
                response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                response_text = model_response["generation"]
                return response_text

            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(60)
                print("sleeping 60 seconds")
                retries = 0
            retries += 1
            time.sleep(5)  # Wait for 10 seconds before retrying


def extract_multi_turn_conversation(
    multi_turn_message, turn_number=3, model_type="llama"
):
    message = []
    for turn in multi_turn_message:
        role = turn["role"]
        content = turn["content"]
        if model_type == "llama":
            message.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            )
        elif model_type == "claude":
            message.append({"role": role, "content": content})
        elif model_type == "mistral":
            if role == "user":
                message.append(f"[INST] {content} [/INST]")
            else:
                message.append(f"{content}</s>")
        elif model_type == "gpt":
            message.append({"role": role, "content": content})
        if len(message) == turn_number * 2:
            if role != "assistant":
                raise ValueError("The last turn must be from assistant")
            break
    assert (
        len(message) == turn_number * 2
    ), "The number of turns is less than the specified number"
    if "llama" in model_type or "mistral" in model_type:
        message = "".join(message)
    return message


def extract_conversation_to_messages(conversation, model_type):
    # this is for implicit preference
    messages = []
    role_list = []

    for line in conversation:
        role = line["role"]
        content = line["content"]
        role_list.append(role)

        if model_type == "mistral":
            if role == "user":
                messages.append(f"[INST] {content} [/INST]")
            else:
                messages.append(f"{content}</s>")
        elif model_type == "llama":
            messages.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            )
        elif model_type == "claude":
            messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    return messages, role_list


def format_options(options):
    """
    Format a list of options into a lettered string.

    :param options: List of option strings
    :return: Formatted string with lettered options
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_options = []

    for i, option in enumerate(options):
        formatted_options.append(f"{letters[i]}. {option}")
    return "\n".join(formatted_options)


def get_question_prompt_mcq(
    preference,
    options,
    question,
    multi_inter_message: list = [],
):

    user_message = preference
    messages = [
        user_message,
    ]
    messages.extend(multi_inter_message)
    mcq_question_format = get_mcq_question_format(options)
    messages.append(question + mcq_question_format)

    return messages
