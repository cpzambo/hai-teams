import pandas as pd
<<<<<<< HEAD
from together import Together
import re
from dotenv import load_dotenv
import os
import json
import time
import csv

<<<<<<<< HEAD:mmlu/openai/openai_rescore.py
========
load_dotenv()
api_key = os.getenv('TOGETHER_API_KEY')
client = Together(api_key=api_key, timeout=4800)

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
                model="google/gemma-4-31B-it", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=12000,
                stream=False
            )
            response = completion.choices[0].message.content.strip()
            if response != "":
                return response
        except Exception as e:
            print("Error:", e)
            return ""
        return ""
    
>>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2:bbh/gemma/gemma_finish.py
=======
import re
import os

>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
<<<<<<< HEAD
<<<<<<<< HEAD:mmlu/openai/openai_rescore.py
    return result.strip("\"'`.").strip()
========
    return result.strip("\"'`")
>>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2:bbh/gemma/gemma_finish.py

# check if the final answer matches the gold
=======
    return result.strip("\"'`.").strip()

>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
def score_response(model_response, gold_answer, question=""):
    final_answer = extract_final_answer(model_response)
    gold_answer = gold_answer.rstrip(".")
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
<<<<<<< HEAD
    return 0

<<<<<<<< HEAD:mmlu/openai/openai_rescore.py
========
def update_row(question, gold_answer, response, score, split):
    if response == "nan":
        new_response = get_model_response(question)
        if new_response == "":
            print(f"Empty response on {split}!\n")
        new_score = score_response(new_response, gold_answer, question)
        return new_response, new_score
    
    return response, int(score)

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

>>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2:bbh/gemma/gemma_finish.py
=======
    # case 6: sequence spacing "[[<>]]" vs "[ [ < > ] ]"
    if re.sub(r'\s+', '', final_answer.lower()) == re.sub(r'\s+', '', gold_answer.lower()):
        return 1
    return 0

>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
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
<<<<<<< HEAD
<<<<<<<< HEAD:mmlu/openai/openai_rescore.py
    csv_path = f"openai_{split}.csv"
========
    csv_path = f"gemma_{split}.csv"
>>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2:bbh/gemma/gemma_finish.py
=======
    csv_path = f"openai_{split}.csv"
>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
<<<<<<< HEAD
    df[["model_response", "score"]] = df.apply(
        lambda row: update_row(str(row["question"]), str(row["gold_answer"]), str(row["model_response"]), str(row["score"]), split),
        axis=1,
        result_type="expand"
=======
    df["score"] = df.apply(
        lambda row: score_response(str(row["model_response"]), str(row["gold_answer"]), str(row["question"])),
        axis=1
>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
    )
    df.to_csv(csv_path, index=False)
    avg = df["score"].mean()
    overall_results.append({"dataset": split, "average_score": round(avg, 3)})
    print(f"{split}: {avg:.3f}")

overall_df = pd.DataFrame(overall_results)
<<<<<<< HEAD
<<<<<<<< HEAD:mmlu/openai/openai_rescore.py
overall_df.to_csv("openai_overall_results.csv", index=False)
print("\nDone. openai_overall_results.csv updated.")
========
overall_df.to_csv("gemma_overall_results.csv", index=False)
print("\nDone. gemma_overall_results.csv updated.")
>>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2:bbh/gemma/gemma_finish.py
=======
overall_df.to_csv("openai_overall_results.csv", index=False)
print("\nDone. openai_overall_results.csv updated.")
>>>>>>> 55766b89d64fb854b25cf0d756d095992b6e03b2
