import argparse
import json
import os
import re
import string
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user as xai_user, system as xai_system

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
api_key = os.getenv("GROK_API_KEY")
client = Client(api_key=api_key, timeout=3600)
LETTERS = string.ascii_uppercase


# ── helpers ───────────────────────────────────────────────────────────────────

def load_yaml(rel_path):
    with open(os.path.join(ROOT, rel_path)) as f:
        return yaml.safe_load(f)


def rank_choices(choices):
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build_system_prompt(task):
    prompts = load_yaml("src/configs/prompts.yaml")
    response = load_yaml("src/configs/response.yaml")
    statement = response["base"]["en"]
    conditions = response[task]["en"]
    fmt = f"\n{statement}\n```json\n    {{\n    {conditions}\n    }}\n```"
    return prompts["sys"]["en"] + fmt


def build_user_prompt(task, sample):
    template = load_yaml("src/configs/prompts.yaml")[task]["en"]
    if task == "EU":
        return template.format(
            scenario=sample["scenario"],
            subject=sample["subject"],
            emo_choices=rank_choices(sample["emotion_choices"]),
            cause_choices=rank_choices(sample["cause_choices"]),
        )
    else:
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

def call_api(sys_prompt, user_prompt, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            chat = client.chat.create(model=model)
            chat.append(xai_system(sys_prompt))
            chat.append(xai_user(user_prompt))
            response = chat.sample()
            time.sleep(2.0)
            return response.content
        except Exception as e:
            err = str(e)
            print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
            wait = 5.0
            m = re.search(r'try again in ([\d.]+)(ms|s)', err)
            if m:
                wait = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
                wait = max(wait + 1, 1)
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None


# ── evaluation + CSV output ───────────────────────────────────────────────────

def evaluate(results, task, model_name):
    if not results:
        return

    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", task)

    for col in ["answer", "emo_answer", "cause_answer", "label", "emo_label", "cause_label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    if task == "EA":
        df["score"] = (df["label"] == df["answer"]).astype(int)
        cat_acc = df.groupby("category")["score"].mean()
        overall = df["score"].mean()
    else:
        df["score"] = ((df["emo_label"] == df["emo_answer"]) & (df["cause_label"] == df["cause_answer"])).astype(int)
        cat_acc = df.groupby("coarse_category")["score"].mean()
        overall = df["score"].mean()

    csv_path = os.path.join(out_dir, f"{model_name}_en.csv")
    df.to_csv(csv_path, index=False)

    overall_path = os.path.join(out_dir, f"{model_name}_en_overall.csv")
    rows = [{"category": cat, "accuracy": acc} for cat, acc in cat_acc.items()]
    rows.append({"category": "Overall", "accuracy": overall})
    pd.DataFrame(rows).to_csv(overall_path, index=False)

    print(f"\n{'='*50}")
    print(f"[{task}-en] Evaluation for {model_name}")
    print(f"  Overall accuracy: {overall:.4f} ({df['score'].sum()}/{len(df)})")
    for cat, acc in cat_acc.items():
        print(f"  {cat}: {acc:.4f}")
    print(f"  Saved → {csv_path}")
    print(f"  Saved → {overall_path}")
    print(f"{'='*50}\n")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_task(task, model, save_every):
    data = []
    data_path = os.path.join(ROOT, "data", f"{task}.jsonl")
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["language"] == "en":
                data.append(s)

    model_name = model.replace(".", "_").replace("/", "-")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", task)
    out_path = os.path.join(results_dir, f"{model_name}_en.jsonl")

    done_ids, results = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(str(r["qid"]))
                results.append(r)
        print(f"[{task}-en] Resuming — {len(done_ids)} done, {len(data)-len(done_ids)} remaining")
    else:
        print(f"[{task}-en] Processing {len(data)} samples")

    sys_prompt = build_system_prompt(task)

    def save_checkpoint():
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

    for sample in data:
        qid = str(sample["qid"])
        if qid in done_ids:
            continue

        user_prompt = build_user_prompt(task, sample)
        raw = call_api(sys_prompt, user_prompt, model)
        parsed = parse_json(raw) if raw is not None else None

        if task == "EU":
            res = {
                "qid": sample["qid"],
                "lang": "en",
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
        else:
            res = {
                "qid": sample["qid"],
                "lang": "en",
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
            print(f"[{task}-en] [{len(results)}/{len(data)}] checkpoint saved")

    save_checkpoint()
    print(f"[{task}-en] Done → {out_path}")
    evaluate(results, task, model_name)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="grok-3-mini")
    parser.add_argument("--task", type=str, default="all", choices=["EU", "EA", "all"])
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    tasks = ["EU", "EA"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.model, args.save_every)
