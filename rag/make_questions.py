"""Build the benign question set from Bitext (run once on a machine with internet).

    uv run python -m rag.make_questions

Picks the Bitext intents our synthetic policies cover, samples a fixed number
per intent with a fixed seed, and splits them evenly into dev and eval.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset


OUT_PATH = Path(__file__).parent / "questions" / "benign.jsonl"
SEED = 0
PER_INTENT = 8  # half dev, half eval

# Bitext intents that rag/corpus/ has an answer for.
INTENTS = [
    "check_refund_policy", "get_refund", "track_refund",
    "cancel_order", "check_cancellation_fee",
    "delivery_period", "delivery_options", "track_order",
]
PLACEHOLDERS = {"{{Order Number}}": "NV-48213"}


def main():
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    rng = random.Random(SEED)
    rows = []
    for intent in INTENTS:
        candidates = []
        for r in ds:
            if r["intent"] != intent or "W" in r["flags"]:  # W = offensive language
                continue
            q = r["instruction"]
            for placeholder, value in PLACEHOLDERS.items():
                q = q.replace(placeholder, value)
            if "{{" not in q:
                candidates.append(q.strip())
        candidates = sorted(set(candidates))  # dedupe, and fix order before sampling
        for i, q in enumerate(rng.sample(candidates, PER_INTENT)):
            rows.append({
                "id": f"benign-{intent}-{i}",
                "intent": intent,
                "question": q,
                "split": "dev" if i < PER_INTENT // 2 else "eval",
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
