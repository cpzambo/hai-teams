import pandas as pd
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import json

# load in the api key and set up the client
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# generate the model's response
def get_model_response(question, subject, choices):
    # prompt the model so it's easy to check the answer
    prompt = f"""
    The following is a multiple-choice question on the subject of {subject}. 
    Question: {question}
    Choices: {choices}

    Please respond with the correct answer from the choices, and end your response with:
    "Final Answer: <your answer here>"
    """

    # create the response
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # or another model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
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

overall_results = []
splits = ["Business_ethics",
          "Econometrics",
          "Elementary_math",
          "Formal_logic",
          "Jurisprudence",
          "Logical_fallacies",
          "Management",
          "Marketing",
          "Miscellaneous",
          "Moral_disputes",
          "Moral_scenarios",
          "Philosophy",
          "Professional_accounting"]

for split in splits:
    try: 
        with open(f'{split}.json', 'r') as file:
            data = json.load(file)

        results = []
        # iterate through the dataset
        for example in data:
            # get the question and gold answer
            q = example["question"]
            subject = example["subject"]
            choices = example["choices"]
            answer_index = example["answer"]
            gold = choices[int(answer_index)]
            # generate and score the response
            model_resp = get_model_response(q, subject, choices)
            score = score_response(model_resp, gold)

            # append to the results csv for this split
            results.append({
                "question": q,
                "subject": subject,
                "choices": choices,
                "gold_answer": gold,
                "model_response": model_resp,
                "score": score
            })

        results_df = pd.DataFrame(results)

        # save the results on this split
        results_df.to_csv(f"openai_{split}.csv", index=False)
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
        overall_df.to_csv("openai_overall_results.csv", index=False)

# save the overall results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("openai_overall_results.csv", index=False)