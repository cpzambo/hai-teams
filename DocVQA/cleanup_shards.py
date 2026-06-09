"""
Consolidate existing shard CSVs:
- Collect all valid answers across all shard files (deduplicated)
- Redistribute them into the correct per-shard files
- After running this, sbatch resume will only process unanswered questions
"""
import pandas as pd
import glob
import json

DATA_PATH = "./docvqa_output/docvqa_validation.json"
TOTAL_SHARDS = 5

# Collect valid answers from all existing CSVs (deduplicated by questionId)
all_filled = {}
for f in sorted(glob.glob("openai_docvqa_results_shard*.csv")):
    df = pd.read_csv(f)
    df = df[df["model_response"].notna() & (df["model_response"].astype(str).str.strip() != "")]
    for _, row in df.iterrows():
        qid = str(row["questionId"])
        if qid not in all_filled:
            all_filled[qid] = row.to_dict()

print(f"Total valid answers collected: {len(all_filled)} (deduplicated)")

# Redistribute answers into correct shard files
with open(DATA_PATH) as f:
    full_data = json.load(f)

shard_size = (len(full_data) + TOTAL_SHARDS - 1) // TOTAL_SHARDS

total_done = 0
total_remaining = 0
for shard in range(TOTAL_SHARDS):
    start = shard * shard_size
    end = min(start + shard_size, len(full_data))
    shard_qids = {str(ex["questionId"]) for ex in full_data[start:end]}
    shard_rows = [all_filled[q] for q in shard_qids if q in all_filled]

    out = f"openai_docvqa_results_shard{shard}of{TOTAL_SHARDS}.csv"
    pd.DataFrame(shard_rows).to_csv(out, index=False)

    done = len(shard_rows)
    remaining = len(shard_qids) - done
    total_done += done
    total_remaining += remaining
    print(f"  shard{shard}: {done} done, {remaining} remaining -> {out}")

print(f"\nTotal: {total_done} done, {total_remaining} remaining")
