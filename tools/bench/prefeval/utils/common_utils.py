import os
import json
import time
import yaml

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
            elif model_type == "gpt":
                print(model_id)
                completion = bedrock_runtime.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens,
                    # max_completion_tokens=max_tokens,
                )
                print(completion)
                return completion.choices[0].message.content
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(20)
                print("sleeping 20 seconds")
                retries = 0
            retries += 1
            time.sleep(5)  # Wait for 10 seconds before retrying


def extract_multi_turn_conversation(multi_turn_message, turn_number=3):
    message = []
    for turn in multi_turn_message:
        
        role = turn["role"]
        content = turn["content"]
        message.append({"role": role, "content": content})
    return message


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Get the current working directory and remove the last directory in the path
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)  # This removes the last directory

    config["dir_path"] = parent_dir
    return config


def extract_multi_turn_message(turns_data, inter_turns):
    if inter_turns > 0:
        multi_turn_message = []
        for turn_data in turns_data:
            multi_turn_message.extend(turn_data["conversation"])
        return (
            extract_multi_turn_conversation(multi_turn_message, inter_turns),
            multi_turn_message,
        )
    else:
        multi_turn_message = None
    return "", multi_turn_message


ALL_TOPICS = [
    "travel_transportation",
    "shop_motors",
    "lifestyle_beauty",
    "travel_restaurant",
    "shop_fashion",
    "entertain_shows",
    "pet_ownership",
    "lifestyle_fit",
    "entertain_games",
    "shop_home",
    "lifestyle_health",
    "travel_activities",
    "education_learning_styles",
    "entertain_music_book",
    "professional_work_location_style",
    "education_resources",
    "lifestyle_dietary",
    "shop_technology",
    "travel_hotel",
    "entertain_sports",
]


COT_PROMPT = """
When answering a user's question, a good assistant should carefully consider the user's stated preferences and tailor the response accordingly.

Example 1:
User1's preference: "I have an intense phobia of bugs and insects. What outdoor activities would you recommend for the summer?"
Good assistant respnose: "Considering your phobia of bugs and insects, I'll focus on recommending activities that minimize encounters with insects. Let's explore outdoor options with fewer bugs, like beaches, pools, urban areas, well-maintained park trails, and activities during cooler times of day. I'll avoid suggesting things like camping, off-trail hiking, picnicking, or activities in dense forests. Here are some summer outdoor activities you might enjoy: spending a day at the beach swimming and playing sports, exploring outdoor malls or pedestrian areas in the city, hiking on maintained trails in parks (avoiding dense woods), attending evening outdoor concerts or movies, and enjoying water activities like kayaking or paddle boating on lakes or rivers. These options should help you enjoy the outdoors while minimizing encounters with insects."
Example 2:
User2's preference: "I'm a vegan looking for new dinner recipes. Any ideas?"
Good assistant response: "As a vegan, you'll need plant-based recipes without animal products. I'll focus on nutrient-rich, diverse ingredients to ensure balanced meals. Consider: quinoa and black bean burrito bowls, lentil and vegetable curry, roasted vegetable and hummus wrap, or zucchini noodles with avocado pesto. These options offer protein, fiber, and essential nutrients while adhering to vegan principles."
Example 3:
User3's preference: "I'm a beginner guitarist with small hands. What's a good first guitar?"
Good assistant response: "For a beginner with small hands, we'll prioritize comfort and playability. Look for guitars with narrow necks and shorter scale lengths. Consider: Fender Mustang, Ibanez Mikro, or Squier Mini Strat. These models are smaller overall, making them easier to handle. Avoid full-sized dreadnoughts or jumbo acoustics. Try these at a local music store to ensure they feel comfortable before purchasing."
Example 4:
User4's preference: "I'm on a tight budget but want to start a home gym. What equipment should I get?"
Good assistant response: "Working with a tight budget, we'll focus on versatile, cost-effective equipment for a home gym. Prioritize items that offer multiple exercise options. Consider: resistance bands, a set of adjustable dumbbells, a stability ball, and a yoga mat. These allow for a wide range of strength, flexibility, and cardio exercises without breaking the bank. Avoid expensive, single-purpose machines. As your budget allows, you can add more specialized equipment later."
Example 5:
User5's preference: "I'm sensitive to caffeine but need an energy boost in the mornings. Any suggestions?"
Good assistant response: "Given your caffeine sensitivity, we'll explore natural, caffeine-free energy boosters. Focus on lifestyle changes and alternatives to caffeinated drinks. Try: starting your day with a brisk walk or light exercise, drinking a glass of water with lemon, eating a protein-rich breakfast, or trying herbal teas like peppermint or ginger. Avoid coffee, energy drinks, and black tea. These alternatives can help increase alertness without the jitters or crashes associated with caffeine."

Now, please answer the following question while considering my preferences (not the user preferences in the examples above), which I have stated either explicitly or implicitly in our previous conversation:

            """
REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."
