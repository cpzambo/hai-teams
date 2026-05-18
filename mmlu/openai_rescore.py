import pandas as pd
import re
import os

# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`.").strip()

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
    # case 6: sequence spacing "[[<>]]" vs "[ [ < > ] ]"
    if re.sub(r'\s+', '', final_answer.lower()) == re.sub(r'\s+', '', gold_answer.lower()):
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


for split in splits:
    csv_path = f"openai_{split}.csv"
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df["score"] = df.apply(
        lambda row: score_response(str(row["model_response"]), str(row["gold_answer"]), str(row["question"])),
        axis=1
    )
    df.to_csv(csv_path, index=False)
    avg = df["score"].mean()
    overall_results.append({"dataset": split, "average_score": round(avg, 3)})
    print(f"{split}: {avg:.3f}")

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("openai_overall_results.csv", index=False)
print("\nDone. openai_overall_results.csv updated.")
