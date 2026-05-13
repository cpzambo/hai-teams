"""Merge per-shard CSVs produced by openai_eval.py into one final result file."""
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--total-shards", type=int, default=5)
args = parser.parse_args()

pattern = f"openai_docvqa_results_shard*of{args.total_shards}.csv"
files = sorted(glob.glob(pattern))

if len(files) != args.total_shards:
    print(f"WARNING: expected {args.total_shards} shard files, found {len(files)}: {files}")

if not files:
    raise SystemExit(f"No files matched '{pattern}'")

merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
merged = merged.sort_values("questionId").reset_index(drop=True)
merged.to_csv("openai_docvqa_results.csv", index=False)

accuracy = merged["score"].mean()
anls = merged["anls"].mean()
print(f"Merged {len(files)} shards → {len(merged)} examples")
print(f"Final accuracy: {accuracy:.4f} ({merged['score'].sum()}/{len(merged)})")
print(f"Final ANLS:     {anls:.4f}")

pd.DataFrame([{
    "dataset": "docvqa_validation",
    "accuracy": accuracy,
    "anls": anls,
}]).to_csv("openai_docvqa_overall.csv", index=False)
print("Saved → openai_docvqa_results.csv  openai_docvqa_overall.csv")
