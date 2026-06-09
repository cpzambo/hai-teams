import pandas as pd
from together import Together
import re
from dotenv import load_dotenv
import os
import json
import time
import csv

load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')
client = Together(api_key=api_key)

# generate the model's response
def get_model_response(question):
    for attempt in range(5):
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
                model="Qwen/Qwen3.5-9B", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=12500,
                stream=False
            )
            response = completion.choices[0].message.content.strip()
            if response != "":
                return response
            time.sleep(5)
        except Exception as e:
            print("Error:", e)
            return None

        return ""
    
# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`")

# check if the final answer matches the gold
def score_response(model_response, gold_answer, question=""):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    # case 1: accurate matching
    if final_answer.lower().strip() == gold_answer.lower().strip():
        return 1
    # case 2.1 gold_answer: (D), final_answer: "B" or "(B)" or "(B) choices"
    if re.match(r'^\([A-Z]\)$', gold_answer.strip()):
        m = re.match(r'^\(?([A-Z])\)?', final_answer.strip())
        if m and f"({m.group(1)})" == gold_answer.strip():
            return 1
    # case 2.2 gold_answer: (D), final_answer: "choices" with no Alphabet
    if question and re.match(r'^\([A-Z]\)$', gold_answer.strip()):
        options = dict(re.findall(r'\(([A-Z])\)\s*([^\n(]+)', question))
        gold_letter = gold_answer.strip("()")
        gold_content = options.get(gold_letter, "").strip()
        if gold_content and final_answer.lower() == gold_content.lower():
            return 1
    # case 3: deal with "barn, damp" vs "barn damp"
    if final_answer.lower().replace(",", " ").split() == gold_answer.lower().split():
        return 1
    # case 4: deal with complete sequence of parenthesis and brackets
    m = re.search(r'Input:\s*(.+)', question, re.IGNORECASE)
    if m:
        partial = m.group(1).strip()
        full = partial + " " + gold_answer.strip()
        if final_answer.lower().strip() == full.lower().strip():
            return 1
    # case 5: with comma inside "No  ," vs "No"
    if final_answer.lower().replace(",", " ").split() == gold_answer.lower().replace(",", " ").split():
        return 1
    return 0

def update_row(question, gold_answer, response, score, split):
    if response == "nan":
        new_response = get_model_response(question)
        if new_response == "":
            print(f"Empty response on {split}!\n")
        new_score = score_response(new_response, gold_answer, question)
        return new_response, new_score
    
    time.sleep(1)
    return response, int(score)

splits = ["reasoning_about_colored_objects",
          "web_of_lies",
          "word_sorting"]

overall_results = []

for split in splits:
    csv_path = f"qwen_{split}.csv"
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df[["model_response", "score"]] = df.apply(
        lambda row: update_row(str(row["question"]), str(row["gold_answer"]), str(row["model_response"]), str(row["score"]), split),
        axis=1,
        result_type="expand"
    )
    df.to_csv(csv_path, index=False)
    avg = df["score"].mean()
    overall_results.append({"dataset": split, "average_score": round(avg, 3)})
    print(f"{split}: {avg:.3f}")

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("qwen_overall_results.csv", index=False)
print("\nDone. qwen_overall_results.csv updated.")
