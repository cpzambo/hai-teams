"""Merge each CSV produced by gemma_eval.py into one final result file."""
import glob
import pandas as pd

pattern = f"gemma_docvqa_results_half*.csv"
files = sorted(glob.glob(pattern))

if not files:
    raise SystemExit(f"No files matched '{pattern}'")

merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
merged = merged.sort_values("questionId").reset_index(drop=True)
merged.to_csv("gemma_docvqa_results.csv", index=False)

accuracy = merged["score"].mean()
anls = merged["anls"].mean()
print(f"Merged {len(files)} shards → {len(merged)} examples")
print(f"Final accuracy: {accuracy:.4f} ({merged['score'].sum()}/{len(merged)})")
print(f"Final ANLS:     {anls:.4f}")

pd.DataFrame([{
    "dataset": "docvqa_validation",
    "accuracy": accuracy,
    "anls": anls,
}]).to_csv("gemma_docvqa_overall.csv", index=False)
print("Saved → gemma_docvqa_results.csv  gemma_docvqa_overall.csv")
