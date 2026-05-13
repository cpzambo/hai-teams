import argparse
import json
import os
import re
import string
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

LETTERS = string.ascii_uppercase


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def rank_choices(choices):
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build_system_prompt(task, lang):
    prompts = load_yaml("src/configs/prompts.yaml")
    response = load_yaml("src/configs/response.yaml")
    statement = response["base"][lang]
    conditions = response[task][lang]
    fmt = f"\n{statement}\n```json\n    {{\n    {conditions}\n    }}\n```"
    return prompts["sys"][lang] + fmt


def build_user_prompt(task, lang, sample):
    template = load_yaml("src/configs/prompts.yaml")[task][lang]
    if task == "EU":
        return template.format(
            scenario=sample["scenario"],
            subject=sample["subject"],
            emo_choices=rank_choices(sample["emotion_choices"]),
            cause_choices=rank_choices(sample["cause_choices"]),
        )
    else:  # EA
        return template.format(
            scenario=sample["scenario"],
            subject=sample["subject"],
            choices=rank_choices(sample["choices"]),
            q_type=sample["question type"],
        )


def parse_json(text):
    try:
        if "```json" in text:
            text = re.search(r"```json\s*([\s\S]*?)```", text).group(1)
        return json.loads(text.strip())
    except Exception:
        return None


# ── API call with retry ───────────────────────────────────────────────────────

def call_api(messages, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=256,
                timeout=60,
            )
            time.sleep(2.0)  # stay under RPM/TPM limits
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            if "insufficient_quota" in err:
                raise SystemExit("OpenAI quota exhausted — top up billing and retry.")
            if "requests per day" in err:
                print("RPD limit hit — stopping to preserve quota.")
                return None
            wait = 5.0
            m = re.search(r"try again in ([\d.]+)(ms|s)", err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None


# ── evaluation + CSV output ───────────────────────────────────────────────────

def evaluate(results, task, lang, model_name, shard_tag=""):
    if not results:
        return

    df = pd.DataFrame(results)
    out_dir = f"results/{task}"

    # normalize before comparing — mirrors MMLU's .lower().strip()
    for col in ["answer", "emo_answer", "cause_answer", "label", "emo_label", "cause_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    if task == "EA":
        df["score"] = (df["label"] == df["answer"]).astype(int)
        cat_acc = df.groupby("category")["score"].mean()
        overall = df["score"].mean()
    else:  # EU
        df["score"] = ((df["emo_label"] == df["emo_answer"]) & (df["cause_label"] == df["cause_answer"])).astype(int)
        cat_acc = df.groupby("coarse_category")["score"].mean()
        overall = df["score"].mean()

    # per-sample CSV  (matches MMLU per-subject CSV style)
    csv_path = f"{out_dir}/{model_name}_{lang}{shard_tag}.csv"
    df.to_csv(csv_path, index=False)

    # overall results CSV  (matches MMLU overall_results.csv style)
    overall_path = f"{out_dir}/{model_name}_{lang}_overall.csv"
    rows = [{"category": cat, "accuracy": acc} for cat, acc in cat_acc.items()]
    rows.append({"category": "Overall", "accuracy": overall})
    pd.DataFrame(rows).to_csv(overall_path, index=False)

    print(f"\n{'='*50}")
    print(f"[{task}-{lang}] Evaluation for {model_name}")
    print(f"  Overall accuracy: {overall:.4f} ({df['score'].sum()}/{len(df)})")
    for cat, acc in cat_acc.items():
        print(f"  {cat}: {acc:.4f}")
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {overall_path}")
    print(f"{'='*50}\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_task(task, lang, model, shard, total_shards, save_every):
    # Load + filter by language
    data = []
    with open(f"data/{task}.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["language"] == lang:
                data.append(s)

    # Slice shard
    shard_size = (len(data) + total_shards - 1) // total_shards
    start = shard * shard_size
    end = min(start + shard_size, len(data))
    data = data[start:end]

    model_name = model.split("/")[-1].replace(".", "_")
    shard_tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    out_path = f"results/{task}/{model_name}_{lang}{shard_tag}.jsonl"
    os.makedirs(f"results/{task}", exist_ok=True)

    # Resume from checkpoint
    done_ids, results = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(str(r["qid"]))
                results.append(r)
        print(f"[{task}-{lang} shard {shard}] Resuming — {len(done_ids)} done, {len(data)-len(done_ids)} remaining")
    else:
        print(f"[{task}-{lang} shard {shard}] Processing indices {start}–{end-1} ({len(data)} samples)")

    sys_prompt = build_system_prompt(task, lang)

    def save_checkpoint():
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

    for sample in data:
        qid = str(sample["qid"])
        if qid in done_ids:
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": build_user_prompt(task, lang, sample)},
        ]
        raw = call_api(messages, model)
        parsed = parse_json(raw) if raw is not None else None

        if task == "EU":
            res = {
                "qid": sample["qid"],
                "lang": lang,
                "coarse_category": sample["coarse_category"],
                "finegrained_category": sample["finegrained_category"],
                "scenario": sample["scenario"],
                "subject": sample["subject"],
                "emo_label": LETTERS[sample["emotion_choices"].index(sample["emotion_label"])],
                "emo_answer": (parsed or {}).get("answer_q1", ""),
                "cause_label": LETTERS[sample["cause_choices"].index(sample["cause_label"])],
                "cause_answer": (parsed or {}).get("answer_q2", ""),
                "model_response": raw or "",
            }
        else:  # EA
            res = {
                "qid": sample["qid"],
                "lang": lang,
                "category": sample["category"],
                "question_type": sample["question type"],
                "scenario": sample["scenario"],
                "subject": sample["subject"],
                "label": LETTERS[sample["choices"].index(sample["label"])],
                "answer": (parsed or {}).get("answer", ""),
                "model_response": raw or "",
            }

        results.append(res)
        done_ids.add(qid)

        if len(results) % save_every == 0:
            save_checkpoint()
            n_done = len(results)
            print(f"[{task}-{lang} shard {shard}] [{n_done}/{len(data)}] checkpoint saved")

    save_checkpoint()
    print(f"[{task}-{lang} shard {shard}] Done → {out_path}")
    evaluate(results, task, lang, model_name, shard_tag)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=str, default="all", choices=["EU", "EA", "all"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "all"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    langs = ["en", "zh"] if args.lang == "all" else [args.lang]

    for task in tasks:
        for lang in langs:
            run_task(task, lang, args.model, args.shard, args.total_shards, args.save_every)
