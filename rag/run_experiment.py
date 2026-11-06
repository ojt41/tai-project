"""Run the benign questions and the attacks through the RAG bot and log every turn.

    uv run python -m rag.run_experiment --model Qwen/Qwen2.5-3B-Instruct --run-name v100-3b

Items already in the log are skipped, so re-running after a timeout resumes.
"""
import argparse
import json
from pathlib import Path

from rag.pipeline import Config, RAGBot


QUESTIONS_DIR = Path(__file__).parent / "questions"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_items(limit=None):
    """Benign questions as one-turn items, then attacks; optionally the first `limit` of each."""
    benign = [json.loads(l) for l in (QUESTIONS_DIR / "benign.jsonl").read_text().splitlines()]
    attacks = [json.loads(l) for l in (QUESTIONS_DIR / "attacks.jsonl").read_text().splitlines()]
    benign_items = [{"id": b["id"], "threat": "benign", "tactic": b["intent"], "split": b["split"],
                     "turns": [b["question"]]} for b in benign]
    if limit:
        return benign_items[:limit] + [a for a in attacks if a["threat"] == "policy_manipulation"][:limit] \
            + [a for a in attacks if a["threat"] == "excessive_tokens"][:limit]
    return benign_items + attacks


def done_ids(log_path):
    """Ids whose final turn is already in the log."""
    if not log_path.exists():
        return set()
    return {r["item_id"] for r in map(json.loads, log_path.read_text().splitlines()) if r.get("final_turn")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="only the first N items of each set (quick test)")
    args = parser.parse_args()

    log_path = RESULTS_DIR / f"{args.run_name}.jsonl"
    config = Config(model_name=args.model, max_new_tokens=args.max_new_tokens,
                    log_path=str(log_path), run_name=args.run_name)
    items = load_items(args.limit)
    skip = done_ids(log_path)
    todo = [it for it in items if it["id"] not in skip]
    print(f"{len(items)} items, {len(skip)} already done, {len(todo)} to run -> {log_path}", flush=True)

    bot = RAGBot(config)
    for n, item in enumerate(todo, 1):
        history = []
        for t, turn in enumerate(item["turns"]):
            final = t == len(item["turns"]) - 1
            meta = {"item_id": item["id"], "threat": item["threat"], "tactic": item["tactic"],
                    "turn": t, "final_turn": final}
            record = bot.answer(turn, history, meta=meta)
            history += [{"role": "user", "content": turn},
                        {"role": "assistant", "content": record["answer"]}]
        print(f"[{n}/{len(todo)}] {item['id']}: {record['output_tokens']} tokens, "
              f"{record['generation_latency_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
