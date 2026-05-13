# EmoBench Data Overview

## Summary

| Task | File | Total | EN | ZH |
|---|---|---|---|---|
| EU (Emotional Understanding) | `data/EU.jsonl` | 400 | 200 | 200 |
| EA (Emotional Application) | `data/EA.jsonl` | 400 | 200 | 200 |

Both tasks are multiple-choice questions in English and Chinese.

---

## EU — Emotional Understanding

Each sample asks **two questions** about a scenario: what emotion the subject feels, and why.

### Fields

| Field | Type | Description |
|---|---|---|
| `qid` | str | Unique question ID |
| `language` | str | `"en"` or `"zh"` |
| `scenario` | str | A short narrative describing a situation |
| `subject` | str | The person whose emotions are being evaluated |
| `emotion_choices` | list[str] | 4–6 emotion options |
| `emotion_label` | str | Correct emotion answer |
| `cause_choices` | list[str] | 4 cause options |
| `cause_label` | str | Correct cause answer |
| `coarse_category` | str | One of 4 top-level categories |
| `finegrained_category` | str | One of 11 fine-grained subcategories |

### Categories

| Coarse | Fine-grained |
|---|---|
| `complex_emotions` | `mixture_of_emotions`, `emotion_transition`, `unexpected_outcome` |
| `emotional_cues` | `visual_cues`, `vocal_cues` |
| `perspective_taking` | `false_belief`, `faux_pas`, `strange_story` |
| `personal_beliefs_and_experiences` | `persona`, `cultural_value`, `sentimental_value` |

### Example

```json
{
  "qid": "1",
  "language": "en",
  "coarse_category": "complex_emotions",
  "finegrained_category": "emotion_transition",
  "scenario": "Dorea was trying to cook a Baklava...",
  "subject": "Dorea",
  "emotion_choices": ["Delight", "Anger", "Embarrassment", "Hopeless", "Pride", "Disappointment"],
  "emotion_label": "Delight",
  "cause_choices": ["Her daughter tried to make her feel better...", "Her daughter enjoyed the Baklava despite it being ruined", "..."],
  "cause_label": "Her daughter enjoyed the Baklava despite it being ruined"
}
```

---

## EA — Emotional Application

Each sample asks **one question**: what is the most effective action or response for the subject.

### Fields

| Field | Type | Description |
|---|---|---|
| `qid` | str | Unique question ID |
| `language` | str | `"en"` or `"zh"` |
| `scenario` | str | An emotionally complex situation |
| `subject` | str | The person who must act or respond |
| `question type` | str | `"Action"` or `"Response"` |
| `choices` | list[str] | 4 options |
| `label` | str | Correct answer |
| `category` | str | One of 4 categories |

### Categories

| Category | Description |
|---|---|
| `Personal-Self` | Managing one's own emotions in personal contexts |
| `Personal-Others` | Responding to others' emotions in personal contexts |
| `Social-Self` | Managing one's own emotions in social/professional contexts |
| `Social-Others` | Responding to others' emotions in social/professional contexts |

### Example

```json
{
  "qid": "1",
  "language": "en",
  "category": "Personal-Others",
  "question type": "Action",
  "scenario": "Sarah found out that her younger brother is being bullied at school...",
  "subject": "Sarah",
  "choices": ["Promise to keep the secret", "Inform their parents anyway", "Confront the bullies herself", "Suggest her brother to talk to a teacher or a school counselor"],
  "label": "Suggest her brother to talk to a teacher or a school counselor"
}
```

---

## Evaluation Metrics

Scoring logic is derived directly from the original authors' code in `src/data.py`.

### EA — Single-question scoring

A sample scores 1 if the model's answer matches the gold label, 0 otherwise:

```python
responses["accuracy"] = responses["label"] == responses["answer"]
```

### EU — Dual-question scoring

A sample scores 1 only if **both** the emotion question and the cause question are correct. If either is wrong, the whole sample scores 0:

```python
responses["accuracy"] = (
    responses["emo_label"] == responses["emo_answer"]
) & (responses["cause_label"] == responses["cause_answer"])
```

### Normalization

Answers are normalized before comparison (`.strip().upper()`) to handle lowercase or extra whitespace in model output — mirrors MMLU's `.lower().strip()` approach.

### Aggregation

Accuracy is computed per category using `groupby(...).mean()`, then an Overall score is added across all samples.

---

## Output Files

For each task + language combination, three files are saved:

| File | Content |
|---|---|
| `results/{task}/{model}_{lang}.jsonl` | Per-sample results (checkpoint, used for resume) |
| `results/{task}/{model}_{lang}.csv` | Per-sample results with `score` column |
| `results/{task}/{model}_{lang}_overall.csv` | Per-category accuracy + Overall |

Example output structure for `gpt-4o-mini` running all tasks and both languages:

```
results/
├── EU/
│   ├── gpt-4o-mini_en.csv
│   ├── gpt-4o-mini_en_overall.csv
│   ├── gpt-4o-mini_zh.csv
│   └── gpt-4o-mini_zh_overall.csv
└── EA/
    ├── gpt-4o-mini_en.csv
    ├── gpt-4o-mini_en_overall.csv
    ├── gpt-4o-mini_zh.csv
    └── gpt-4o-mini_zh_overall.csv
```

---

## Running the Evaluation

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt-4o-mini` | OpenAI model name |
| `--task` | `all` | `EU`, `EA`, or `all` |
| `--lang` | `en` | `en`, `zh`, or `all` |
| `--shard` | `0` | Shard index (0-indexed) |
| `--total-shards` | `1` | Total number of parallel shards |
| `--save-every` | `20` | Checkpoint frequency (every N samples) |

### Commands

```bash
# Run all tasks and both languages locally
python openai_emo_eval.py --model gpt-4o-mini --task all --lang all

# Run on Quest with 4 parallel shards
sbatch run_emobench.sh
```

### Environment

Requires `OPENAI_API_KEY` in a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```
