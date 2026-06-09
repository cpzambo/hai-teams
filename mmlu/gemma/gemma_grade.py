import pandas as pd
import re
import os

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