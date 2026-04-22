import pandas as pd
from xai_sdk import Client
from xai_sdk.chat import user
import re
import base64
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY')
client = Client(api_key=api_key, timeout=3600)

DATA_PATH = "./docvqa_output/docvqa_validation.json"
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
        return None

    prompt = (
        "You are reading a document image. Answer the question below using only "
        "information visible in the document.\n\n"
        f"Question: {question}\n\n"
        "Give a short, direct answer — a word, number, or brief phrase. "
        'End your response with: "Final Answer: <your answer here>"'
    )

    try:
        chat = client.chat.create(model="grok-2-vision-1212")
        chat.append(user([
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]))
        response = chat.sample()
        return response.content
    except Exception as e:
        print("API error:", e)
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
    fa_norm = normalize(fa)
    if any(fa_norm in normalize(g) or normalize(g) in fa_norm for g in gold_answers):
        return 1

    return 0


with open(DATA_PATH, "r") as f:
    data = json.load(f)

results = []
for i, example in enumerate(data):
    qid = example["questionId"]
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

    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(data)}] running accuracy: {sum(r['score'] for r in results) / len(results):.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("xai_docvqa_results.csv", index=False)

overall_accuracy = results_df["score"].mean()
overall_anls = results_df["anls"].mean()
print(f"\nFinal accuracy: {overall_accuracy:.4f} ({results_df['score'].sum()}/{len(results_df)})")
print(f"Final ANLS:     {overall_anls:.4f}")

pd.DataFrame([{
    "dataset": "docvqa_validation",
    "accuracy": overall_accuracy,
    "anls": overall_anls,
}]).to_csv("xai_docvqa_overall.csv", index=False)
