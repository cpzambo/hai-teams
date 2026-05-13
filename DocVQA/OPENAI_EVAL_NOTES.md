# OpenAI DocVQA Evaluation — Issue Log & Fix Notes

## Problem: ~3021 out of 5349 questions have empty responses

### Root Cause

Five shards run in parallel and share the same API key's daily RPD quota (10,000 requests/day).
The old retry logic wasted quota on every failed request:

```
Question 1: success → valid        (costs 1 RPD)
Question 2: fail → retry → retry → retry → empty  (wastes 3 RPD)
Question 3: fail → retry × 3      → empty  (wastes 3 RPD)
...
```

This produced a very regular pattern visible in the CSV data:

```
VALID  × 1
EMPTY  × 6   ← 6 questions failed because quota was burned by retries
VALID  × 1
EMPTY  × 6
...  (repeated ~210 times per shard)
```

Net result: for every 1 question answered, ~6 questions were skipped due to exhausted RPD —
consuming roughly 90 RPD per cycle instead of 7.

### Data State (after the failed runs)

| Shard | Total rows | Valid responses | Empty responses |
|-------|-----------|----------------|----------------|
| shard0 | 1070 | 456 | 614 |
| shard1 | 1070 | 478 | 592 |
| shard2 | 1070 | 459 | 611 |
| shard3 | 1070 | 479 | 591 |
| shard4 | 1069 | 456 | 613 |
| **Total** | **5349** | **2328** | **3021** |

No duplicate questionIds, no cross-shard contamination — only the empty responses need to be retried.

---

## Key Problems Encountered & How They Were Solved

### Problem 1 — API Daily Quota (RPD) Exhaustion

**What happened:**
OpenAI Tier 1 accounts have a hard limit of 10,000 requests/day (RPD). With 5 shards running
in parallel and the old retry logic retrying every failure 3 times, the quota was consumed
much faster than expected — each failed question wasted 3 RPD instead of 1.

**Symptoms:**
- Very regular pattern in CSV: 1 answered question followed by ~6 empty ones, repeating
- Empty responses started appearing from question 3 onwards (quota already nearly gone)

**Fix:**
- Detect RPD errors by string-matching `'requests per day'` in the exception message
- Return `None` immediately without any retry — preserves remaining quota for the next run
- Wait for quota reset at 00:00 UTC before re-submitting

```python
if 'requests per day' in err:
    return None  # no retry — preserve daily quota
```

---

### Problem 2 — Rate Limits Causing Questions to Be Skipped (RPM / TPM)

**What happened:**
Two types of rate limits caused questions to be silently skipped:
- **RPM** (requests per minute): exceeded when too many shards ran simultaneously
- **TPM** (tokens per minute): each document image costs ~850 input tokens with `detail="high"`;
  10 parallel shards × 850 tokens × 30 req/min ≈ 255,000 TPM, exceeding the 200,000 TPM limit

The old code retried on these errors but the retries themselves consumed more quota, causing
a compounding failure cascade.

**Fix:**
- Reduced parallel shards from 10 to 5: `--array=0-9` → `--array=0-4`
- Increased sleep between requests from 1.0s → 2.5s to reduce combined TPM
- Combined TPM after fix: 5 shards × 1050 tokens / 2.5s × 60s ≈ 126,000 TPM (safely under limit)
- Added resume logic so skipped questions are retried on the next run rather than lost permanently

**How resume prevents permanent skips:**
The CSV only counts a question as "done" if `model_response` is non-empty. Any empty row is
automatically retried next time `sbatch` is submitted — no manual intervention needed.

---

### Problem 3 — Hanging Requests / Timeout Issues & Shard Design

**What happened:**
The OpenAI SDK has a default timeout of 600 seconds (10 minutes). A single hung request would
freeze an entire shard for up to 10 minutes, blocking all subsequent questions in that shard.
Without sharding, one hung request near the start could delay thousands of questions.

**Fix — Timeout:**
Set `timeout=60` on every API call. Document images take longer than plain text (upload +
decode + inference), so 60s is a safer ceiling than the original 30s.

```python
completion = client.chat.completions.create(
    ...
    timeout=60,  # images take longer; prevents 10-min hangs
)
```

**Fix — Sharding:**
Split the 5,349-question dataset into N independent shards, each processed by a separate
SLURM array job. Benefits:
- A hang or failure in one shard does not affect others
- Each shard writes to its own CSV file — no file conflicts
- Failed shards can be rerun individually without touching completed shards
- Checkpoint saves every 50 questions within each shard, so partial progress is never lost

Shard size calculation (5 shards, 5349 questions):
```
shard 0: questions   0 – 1069  (1070 q)
shard 1: questions 1070 – 2139  (1070 q)
shard 2: questions 2140 – 3209  (1070 q)
shard 3: questions 3210 – 4279  (1070 q)
shard 4: questions 4280 – 5348  (1069 q)
```

---

### Problem 4 — Improving Model Response Quality

**What happened:**
Early runs returned verbose model outputs that were hard to parse for scoring. The model
sometimes gave multi-sentence explanations instead of a direct answer, making exact-match
scoring unreliable.

**Fix — Structured prompt:**
The prompt explicitly instructs the model to end with a parseable tag:

```
"Give a short, direct answer — a word, number, or brief phrase.
End your response with: 'Final Answer: <your answer here>'"
```

**Fix — Answer extraction:**
`extract_final_answer()` uses regex to pull out only the tagged answer, discarding surrounding
explanation text before scoring.

**Fix — Multi-level scoring:**
`score_response()` applies four levels of matching in order, handling common surface variations
without penalizing correct answers:

| Level | Example |
|-------|---------|
| Exact match (case/whitespace insensitive) | `"California"` = `"california"` |
| Comma/space variation | `"barn, damp"` = `"barn damp"` |
| Punctuation stripped | `"st. louis"` = `"st louis"` |
| Substring containment | `"california"` matches `"University of California"` |

**Metric:**
In addition to binary accuracy, ANLS (Average Normalized Levenshtein Similarity) is computed
as the official DocVQA metric — it gives partial credit for near-correct answers.

---

## Fix Applied

### 1. Stop retrying on RPD errors (`openai_eval.py`)

Old behavior: retry 3 times on any API error, including RPD exhaustion.

New behavior: return `None` immediately when RPD is hit — no wasted quota.

```python
if 'requests per day' in err:
    print("RPD limit exhausted — stopping to preserve quota for resume.")
    return None  # do not retry
```

### 2. Resume logic — skip already-answered questions

On startup, `openai_eval.py` reads the existing CSV and builds `done_ids` from rows that have
a non-empty `model_response`. Only unanswered questions are sent to the API.

```python
if os.path.exists(out_csv):
    existing = pd.read_csv(out_csv)
    existing = existing[existing["model_response"].notna() &
                        (existing["model_response"].astype(str).str.strip() != "")]
    done_ids = set(str(x) for x in existing["questionId"].tolist())
    results = existing.to_dict("records")
```

### 3. Reduced parallel shards (10 → 5)

Running 10 shards simultaneously was pushing combined TPM (tokens/min) over the Tier 1 limit.
Reduced to 5 shards to stay within limits while still parallelizing.

### 4. Added `timeout=30` to API calls

The OpenAI SDK default timeout is 600 seconds. Without a timeout, a hanging request would
block a shard for up to 10 minutes. Now each call times out after 30 seconds.

---

## How to Resume the Run

1. **Wait for RPD reset** — resets at 00:00 UTC (08:00 Beijing time) every day.

2. **Upload the fixed files to Quest:**
   ```bash
   scp openai_eval.py openai_eval_submit_array.sh openai_eval_run_merge.sh merge_openai_results.py \
       <netid>@quest.northwestern.edu:/projects/p32983/.../DocVQA/
   ```

3. **Submit the array job** — resume logic handles the rest automatically:
   ```bash
   sbatch openai_eval_submit_array.sh
   ```
   Each shard will skip its ~456 already-answered questions and only call the API for the
   remaining ~604 empty ones (~3021 total across all shards).

4. **After all shards finish, merge results:**
   ```bash
   sbatch openai_eval_run_merge.sh
   ```
   This produces `openai_docvqa_results.csv` and `openai_docvqa_overall.csv`.

---

## Cost Estimate

| Item | Value |
|------|-------|
| Remaining questions | ~3,021 |
| Avg input tokens/question | ~850 (image + prompt) |
| Avg output tokens/question | ~200 |
| Input cost (gpt-4o-mini) | $0.15 / 1M tokens |
| Output cost (gpt-4o-mini) | $0.60 / 1M tokens |
| **Estimated cost** | **~$0.60** |

Full run from scratch (5,349 questions) costs ~$1.05.
