import argparse
import pandas as pd
from openai import OpenAI
import re
import base64
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, default=0, help="0-indexed shard number")
parser.add_argument("--total-shards", type=int, default=1, help="total number of shards")
parser.add_argument("--save-every", type=int, default=50, help="save checkpoint every N examples")
args = parser.parse_args()

DATA_PATH = "./docvqa_output/docvqa_validation.json"
# path to the repo root so image_path resolves correctly
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def encode_image(image_path: str) -> str:
    abs_path = os.path.join(REPO_ROOT, image_path.lstrip("./"))
    with open(abs_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_model_response(question: str, image_path: str) -> str | None:
    try:
        b64 = encode_image(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return "IMAGE_NOT_FOUND"

    prompt = (
        "You are reading a document image. Answer the question below using only "
        "information visible in the document.\n\n"
        f"Question: {question}\n\n"
        "Give a short, direct answer — a word, number, or brief phrase. "
        'End your response with: "Final Answer: <your answer here>"'
    )

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=256,
                timeout=60,  # images take longer to process; 30s was too tight
            )
            time.sleep(2.5)  # 5 shards × 1050 tokens / 2.5s ≈ 126k TPM, safely under 200k limit
            return completion.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            print(f"API error (attempt {attempt+1}/3): {e}")
            if 'insufficient_quota' in err:
                raise SystemExit("OpenAI quota exhausted — top up billing at platform.openai.com and retry.")
            if 'requests per day' in err:
                print("RPD limit exhausted — stopping to preserve quota for resume.")
                return None  # Do not retry, directly go back
            wait = 5
            m = re.search(r'try again in ([\d.]+)(ms|s)', err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == 'ms' else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None


def extract_final_answer(model_output: str) -> str:
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    return match.group(1).strip() if match else model_output.strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def anls_score(prediction: str, gold_answers: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity — DocVQA official metric."""
    pred = normalize(prediction)
    best = 0.0
    for g in gold_answers:
        g_norm = normalize(g)
        max_len = max(len(pred), len(g_norm))
        if max_len == 0:
            nl = 0.0
        else:
            nl = levenshtein(pred, g_norm) / max_len
        sim = 0.0 if nl > threshold else 1.0 - nl
        best = max(best, sim)
    return best


def score_response(model_output: str, gold_answers: list[str]) -> int:
    fa = extract_final_answer(model_output)
    if not fa:
        return 0

    # case 1: exact match (case/whitespace-insensitive) against any gold answer
    if any(normalize(fa) == normalize(g) for g in gold_answers):
        return 1

    # case 2: comma/space variation — "barn, damp" vs "barn damp"
    def tokenize(t):
        return normalize(t).replace(",", " ").split()
    if any(tokenize(fa) == tokenize(g) for g in gold_answers):
        return 1

    # case 3: punctuation stripped — "st. louis" vs "st louis"
    if any(normalize(strip_punct(fa)) == normalize(strip_punct(g)) for g in gold_answers):
        return 1

    # case 4: predicted answer is fully contained in a gold answer, or vice versa
    # handles truncated/abbreviated responses like "california" matching "university of california"
    fa_norm = normalize(fa)
    if any(fa_norm in normalize(g) or normalize(g) in fa_norm for g in gold_answers):
        return 1

    return 0


with open(DATA_PATH, "r") as f:
    full_data = json.load(f)

# slice this shard's contiguous chunk
total = len(full_data)
shard_size = (total + args.total_shards - 1) // args.total_shards
start = args.shard * shard_size
end = min(start + shard_size, total)
data = full_data[start:end]

shard_tag = f"_shard{args.shard}of{args.total_shards}" if args.total_shards > 1 else ""
out_csv = f"openai_docvqa_results{shard_tag}.csv"
print(f"Shard {args.shard}/{args.total_shards}: processing indices {start}–{end-1} ({len(data)} examples)")

# Resume: load already-finished questionIds from existing checkpoint
done_ids = set()
results = []
if os.path.exists(out_csv):
    existing = pd.read_csv(out_csv)
    existing = existing[existing["model_response"].notna() & (existing["model_response"].astype(str).str.strip() != "")]
    done_ids = set(str(x) for x in existing["questionId"].tolist())
    results = existing.to_dict("records")
    print(f"[shard {args.shard}] Resuming — {len(done_ids)} already done, {len(data) - len(done_ids)} remaining")

def save_checkpoint(rows):
    pd.DataFrame(rows).to_csv(out_csv, index=False)

for i, example in enumerate(data):
    qid = example["questionId"]
    if str(qid) in done_ids:
        continue

    question = example["question"]
    gold_answers = example["answers"]
    image_path = example["image_path"]

    model_resp = get_model_response(question, image_path)
    if model_resp is None:
        score = 0
        anls = 0.0
    else:
        score = score_response(model_resp, gold_answers)
        anls = anls_score(extract_final_answer(model_resp), gold_answers)

    results.append({
        "questionId": qid,
        "question_types": example["question_types"],
        "question": question,
        "gold_answers": gold_answers,
        "model_response": model_resp,
        "score": score,
        "anls": anls,
    })

    completed = len(results)
    if completed % args.save_every == 0:
        save_checkpoint(results)
        acc = sum(r["score"] for r in results) / completed
        print(f"[shard {args.shard}] [{completed}/{len(data)}] acc={acc:.3f} — checkpoint saved")

save_checkpoint(results)
results_df = pd.DataFrame(results)
overall_accuracy = results_df["score"].mean()
overall_anls = results_df["anls"].mean()
print(f"\n[shard {args.shard}] Final accuracy: {overall_accuracy:.4f} ({results_df['score'].sum()}/{len(results_df)})")
print(f"[shard {args.shard}] Final ANLS:     {overall_anls:.4f}")
print(f"[shard {args.shard}] Saved → {out_csv}")
