import pandas as pd
from together import Together
import re
from dotenv import load_dotenv
import os
import json

# load in the api key and set up the client
load_dotenv()
api_key = os.getenv('LLAMA_API_KEY')
client = Together(api_key=api_key)

# generate the model's response
def get_model_response(question):
    # prompt the model so it's easy to check the answer
    prompt = f"""
    You are a helpful assistant.
    Question: {question}

    Please show your reasoning, then end your response with:
    "Final Answer: <your concise answer here>"
    """

    # create the response
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None

# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    return match.group(1).strip() if match else model_output.strip()

# check if the final answer matches the gold
def score_response(model_response, gold_answer):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    return int(final_answer.lower().strip() == gold_answer.lower().strip())

# start with an empty list for the overall scores and the list of splits to evaluate
overall_results = []
splits = ["boolean_expressions",
          "causal_judgement",
          "date_understanding",
          "dyck_languages",
          "formal_fallacies",
          "geometric_shapes",
          "logical_deduction_five_objects",
          "logical_deduction_seven_objects",
          "logical_deduction_three_objects",
          "multistep_arithmetic_two",
          "navigate",
          "object_counting",
          "penguins_in_a_table",
          "reasoning_about_colored_objects",
          "temporal_sequences",
          "tracking_shuffled_objects_five_objects",
          "tracking_shuffled_objects_seven_objects",
          "tracking_shuffled_objects_three_objects",
          "web_of_lies",
          "word_sorting"]

# iterate over each of the splits
for split in splits:
    try: 
        with open(f'{split}.json', 'r') as file:
            data = json.load(file)

        dataset = data["examples"]
        results = []
        # iterate through the dataset
        for example in dataset:
            # get the question and gold answer
            q = example["input"]
            gold = example["target"]
            # generate and score the response
            model_resp = get_model_response(q)
            score = score_response(model_resp, gold)

            # append to the results csv for this split
            results.append({
                "question": q,
                "gold_answer": gold,
                "model_response": model_resp,
                "score": score
            })

        results_df = pd.DataFrame(results)

        # save the results on this split
        results_df.to_csv(f"llama_{split}.csv", index=False)
        # append the average score on this split to the overall results
        overall_results.append({
            "dataset": split,
            "average_score": results_df["score"].mean()
        })
    except Exception as e:
        print("Error:", e)
        print("Happened on split:", split)
        # save the overall results
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv("llama_overall_results.csv", index=False)

# save the overall results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("llama_overall_results.csv", index=False)