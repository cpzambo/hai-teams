import pandas as pd
import re
import os
import json

def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
    else:
        # fallback: use last non-empty line instead of full output
        lines = [l.strip() for l in model_output.strip().splitlines() if l.strip()]
        result = lines[-1] if lines else model_output.strip()
    result = result.strip("\"'`.")
    result = result.removeprefix("The final answer is ")
    result = result.removeprefix("$\\boxed{")
    result = result.removeprefix("\\text{")
    result = result.removesuffix("}$")
    result = result.removesuffix("}")
    result = result.strip("\"'`.")
    return result

# check if the final answer matches — accepts letter, number, or full choice text
def score_response(model_response, gold_text, gold_letter, answer_index):
    final_answer = extract_final_answer(model_response)
    gold_text = gold_text.rstrip(".")
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
    csv_path = f"gemini_{split}.csv"
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    with open(f'{split}.json', 'r') as file:
            data = json.load(file)

    df = pd.read_csv(csv_path)

    for i, example in enumerate(data):
        choices = example["choices"]
        answer_index = int(example["answer"])
        gold_text = choices[answer_index]
        gold_letter = labels[answer_index]
        model_resp = df.loc[i, "model_response"]

        score = score_response(model_resp, gold_text, gold_letter, answer_index)
        df.loc[i, "score"] = score
    df.to_csv(csv_path, index=False)
    avg = df["score"].mean()
    overall_results.append({"dataset": split, "average_score": round(avg, 3)})
    print(f"{split}: {avg:.3f}")

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("gemini_overall_results.csv", index=False)
print("\nDone. gemini_overall_results.csv updated.")