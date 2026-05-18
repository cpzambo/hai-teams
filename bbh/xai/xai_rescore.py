import pandas as pd
import re
import os

def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    if match:
        result = match.group(1).strip()
    else:
        # fallback: use last non-empty line instead of full output
        lines = [l.strip() for l in model_output.strip().splitlines() if l.strip()]
        result = lines[-1] if lines else model_output.strip()
    result = result.strip("\"'`")
    # handle LaTeX boxed: $\boxed{D}$ → D
    boxed = re.match(r'^\$\\boxed\{([^}]+)\}\$$', result.strip())
    if boxed:
        result = boxed.group(1).strip()
    return result

def score_response(model_response, gold_answer, question=""):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    # case 1: accurate matching
    if final_answer.lower().strip() == gold_answer.lower().strip():
        return 1
    # case 2: deal with multiple choices (obvious gold_answer has)
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

overall_results = []

for split in splits:
    csv_path = f"xai_{split}.csv"
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
overall_df.to_csv("xai_overall_results.csv", index=False)
print("\nDone. xai_overall_results.csv updated.")
