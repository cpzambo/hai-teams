
######
#test for images

########################
import csv
import json
import os
import re
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

RESULTS_CSV = "./openai_docvqa_results.csv"
DATA_JSON   = "./docvqa_output/docvqa_validation.json"
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
TEST_LIMIT  = 100


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
            )
            time.sleep(1.0)  # proactive rate limit: ~60 req/min
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error (attempt {attempt+1}/3, retrying in 5s): {e}")
            time.sleep(5)
    return None


# find questionIds that had no response in the original run
with open(RESULTS_CSV) as f:
    original = list(csv.DictReader(f))

missing_ids = {
    r["questionId"]
    for r in original
    if not r["model_response"] or not r["model_response"].strip()
}
print(f"Total missing in original run: {len(missing_ids)}")
print(f"Testing retry on first {TEST_LIMIT}...")

# load full data to get image paths
with open(DATA_JSON) as f:
    data = {ex["questionId"]: ex for ex in json.load(f)}

# run on first TEST_LIMIT missing examples
test_ids = list(missing_ids)[:TEST_LIMIT]
success, fail = 0, 0

for i, qid in enumerate(test_ids):
    ex = data[qid]
    resp = get_model_response(ex["question"], ex["image_path"])
    if resp:
        success += 1
    else:
        fail += 1
    print(f"[{i+1}/{TEST_LIMIT}] {'OK' if resp else 'FAIL'} | success rate: {success/(i+1):.1%}")

print(f"\nResult: {success}/{TEST_LIMIT} succeeded, {fail} failed")
