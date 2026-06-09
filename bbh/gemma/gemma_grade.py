import pandas as pd
import re
import os

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
    csv_path = f"gemma_{split}.csv"
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    avg = df["score"].mean()
    overall_results.append({"dataset": split, "average_score": round(avg, 3)})
    print(f"{split}: {avg:.3f}")

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("gemma_overall_results.csv", index=False)
print("\nDone. gemma_overall_results.csv updated.")