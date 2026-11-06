import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

# Run from the repo root: `uv run python -m rag.pipeline` (or import rag.pipeline in ipython).
from rag.chatbot import generate_response, load_model
from rag.retrieval import embed_chunks, load_corpus, load_retriever, retrieve


RAG_DIR = Path(__file__).parent


@dataclass
class Config:
    """Everything a teammate may want to swap, in one place."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # use Qwen/Qwen2.5-3B-Instruct on Triton
    retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    corpus_dir: str = str(RAG_DIR / "corpus")
    system_prompt_path: str = str(RAG_DIR / "prompts" / "system.txt")
    k: int = 3
    max_new_tokens: int = 512
    log_path: str = str(RAG_DIR.parent / "results" / "responses.jsonl")
    run_name: str = "dev"  # label written into every record, e.g. "baseline" or "attack-t1"
    extra: dict = field(default_factory=dict)  # free-form notes logged with each record


def format_excerpts(hits):
    """Format retrieved chunks for the prompt, each tagged with its id."""
    return "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in hits)


class RAGBot:
    """Loads the model, retriever and corpus once, then answers many questions."""

    def __init__(self, config=None):
        self.config = config or Config()
        c = self.config
        self.model, self.tokenizer = load_model(c.model_name)
        self.retriever = load_retriever(c.retriever_name)
        self.chunks = load_corpus(c.corpus_dir)
        self.chunk_embeddings = embed_chunks(self.retriever, self.chunks)
        self.system_prompt = Path(c.system_prompt_path).read_text(encoding="utf-8").strip()
        self.setup = {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "dtype": str(self.model.dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def answer(self, question, history=None, log=True):
        """Answer one customer message, optionally continuing a conversation.

        history is a list of earlier {"role": "user"/"assistant", "content": ...}
        turns (without the system message). Retrieval uses only the newest
        question. Returns the response record, which is also appended to the
        JSONL log unless log=False.
        """
        c = self.config
        history = history or []

        start = time.perf_counter()
        hits = retrieve(self.retriever, self.chunks, self.chunk_embeddings, question, k=c.k)
        retrieval_latency = time.perf_counter() - start

        system = f"{self.system_prompt}\n\nPolicy excerpts:\n\n{format_excerpts(hits)}"
        messages = [{"role": "system", "content": system}, *history,
                    {"role": "user", "content": question}]
        result = generate_response(self.model, self.tokenizer, messages,
                                   max_new_tokens=c.max_new_tokens)

        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_name": c.run_name,
            "question": question,
            "history_turns": len(history),
            "answer": result["answer"],
            "retrieved": [{"id": h["id"], "score": round(h["score"], 4)} for h in hits],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "truncated": result["truncated"],
            "retrieval_latency_s": round(retrieval_latency, 4),
            "generation_latency_s": round(result["latency_s"], 4),
            "config": asdict(c),
            "setup": self.setup,
        }
        if log:
            path = Path(c.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def main(config=None):
    """Chat with the RAG bot in the terminal; every reply is logged."""
    bot = RAGBot(config)
    history = []
    print("Ask a question (type 'quit' to exit, 'reset' to start a new conversation).")
    while True:
        question = input("> ")
        if question.lower() in ["exit", "quit"]:
            break
        if question.lower() == "reset":
            history = []
            continue

        record = bot.answer(question, history)
        print(record["answer"])
        truncated = " | TRUNCATED at max_new_tokens" if record["truncated"] else ""
        print(f"[input {record['input_tokens']} tokens (system + retrieved chunks + history + question)"
              f" | output {record['output_tokens']} tokens"
              f" | {record['generation_latency_s']:.2f} s"
              f" | chunks: {', '.join(r['id'] for r in record['retrieved'])}{truncated}]")

        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": record["answer"]}]


if __name__ == "__main__":
    main()
