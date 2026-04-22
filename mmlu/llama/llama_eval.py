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
def get_model_response(question, subject, choices):
    # format choices as A/B/C/D
    labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])

    prompt = f"""
    The following is a multiple-choice question on the subject of {subject}.
    Question: {question}
    Choices:
    {formatted_choices}

    Please show your reasoning, then end your response with:
    "Final Answer: <A, B, C, or D>"
    """

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
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`").strip()

# check if the final answer matches — accepts letter, number, or full choice text
def score_response(model_response, gold_text, gold_letter, answer_index):
    final_answer = extract_final_answer(model_response)
    if not final_answer:
        return 0

    fa = final_answer.strip()

    # case 1: exact match with full choice text (same as openai)
    if fa.lower() == gold_text.lower():
        return 1

    # case 2: letter only — "A", "(A)", "A.", "A)" — must be only a letter
    m = re.match(r'^\(?([A-D])\)?\.?\s*$', fa, re.IGNORECASE)
    if m and m.group(1).upper() == gold_letter.upper():
        return 1

    # case 3: index number — model says "2" instead of "C"
    if fa.strip() == str(answer_index):
        return 1

    # case 4: letter + choice text — "A. Cryptocurrencies, Cheap..."
    m = re.match(r'^\(?([A-D])\)?[.):]?\s+(.+)', fa, re.IGNORECASE | re.DOTALL)
    if m and m.group(1).upper() == gold_letter.upper():
        return 1

    # case 5: comma/space variation of full text — "barn, damp" vs "barn damp"
    if fa.lower().replace(",", " ").split() == gold_text.lower().replace(",", " ").split():
        return 1

    return 0

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

labels = ["A", "B", "C", "D"]

for split in splits:
    try:
        with open(f'../{split}.json', 'r') as file:
            data = json.load(file)

        results = []
        for example in data:
            q = example["question"]
            subject = example["subject"]
            choices = example["choices"]
            answer_index = int(example["answer"])
            gold_text = choices[answer_index]
            gold_letter = labels[answer_index]

            model_resp = get_model_response(q, subject, choices)
            score = score_response(model_resp, gold_text, gold_letter, answer_index)

            results.append({
                "question": q,
                "subject": subject,
                "choices": choices,
                "gold_answer": gold_text,
                "model_response": model_resp,
                "score": score
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"llama_{split}.csv", index=False)
        overall_results.append({
            "dataset": split,
            "average_score": round(results_df["score"].mean(), 3)
        })
    except Exception as e:
        print("Error:", e)
        print("Happened on split:", split)
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv("llama_overall_results.csv", index=False)

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("llama_overall_results.csv", index=False)
