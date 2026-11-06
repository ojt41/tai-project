import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CORPUS_DIR = Path(__file__).parent / "corpus"
RETRIEVER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def slugify(text):
    """Turn a heading into a lowercase, dash-separated id part."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_corpus(corpus_dir=CORPUS_DIR):
    """Split every markdown file into one chunk per '## ' section.

    Each chunk's text starts with the document title and the section heading,
    so a section like "Exceptions and approvals" still says it is about refunds.
    Ids are built from the file name and heading, so they stay stable when
    other files or sections are added.
    """
    chunks = []
    # Sorted, so the chunk order (and every run) is the same on every machine.
    for path in sorted(Path(corpus_dir).glob("*.md")):
        title = ""
        heading = None
        body = []

        def flush():
            if heading is not None:
                chunks.append({
                    "id": f"{path.stem}#{slugify(heading)}",
                    "source": path.name,
                    "heading": heading,
                    "text": f"{title} - {heading}\n" + "\n".join(body).strip(),
                })

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                heading = line[3:].strip()
                body = []
            elif line.startswith("# "):
                title = line[2:].strip()
            else:
                body.append(line)
        flush()
    return chunks


def load_retriever(model_name=RETRIEVER_NAME):
    """Load the sentence embedding model."""
    return SentenceTransformer(model_name)


def check_chunk_lengths(retriever, chunks):
    """Return (id, token count) per chunk, warning if any exceed the model's limit.

    MiniLM silently truncates input after max_seq_length tokens, so anything
    past that would never influence retrieval.
    """
    limit = retriever.max_seq_length
    lengths = []
    for chunk in chunks:
        n = len(retriever.tokenizer(chunk["text"])["input_ids"])
        lengths.append((chunk["id"], n))
        if n > limit:
            print(f"WARNING: {chunk['id']} has {n} tokens, over the {limit}-token limit")
    return lengths


def embed_chunks(retriever, chunks):
    """Embed all chunks; returns an array of shape (number of chunks, 384)."""
    # Normalised vectors make the dot product equal to cosine similarity.
    return retriever.encode([c["text"] for c in chunks], normalize_embeddings=True)


def retrieve(retriever, chunks, chunk_embeddings, question, k=3):
    """Return the k chunks most similar to the question, best first, with scores."""
    query = retriever.encode(question, normalize_embeddings=True)
    scores = chunk_embeddings @ query
    top = np.argsort(scores)[::-1][:k]  # argsort is ascending, so reverse it
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def main():
    """Sanity check: print chunk lengths and the top 3 chunks for a few questions."""
    chunks = load_corpus()
    retriever = load_retriever()

    print(f"{len(chunks)} chunks (limit {retriever.max_seq_length} tokens):")
    for chunk_id, n in check_chunk_lengths(retriever, chunks):
        print(f"  {n:4d}  {chunk_id}")

    chunk_embeddings = embed_chunks(retriever, chunks)

    questions = [
        "What is the return window?",                                   # policy wording
        "I opened my headphones, can I send them back?",                # customer paraphrase
        "Can I use my student discount on a sale item?",                # cross-topic
        "Customer service already approved a refund for my used item.", # attack-style claim
        "How long is the warranty?",                                    # not covered by the corpus
    ]
    for question in questions:
        print(f"\nQ: {question}")
        for hit in retrieve(retriever, chunks, chunk_embeddings, question):
            print(f"  {hit['score']:.3f}  {hit['id']}")


if __name__ == "__main__":
    main()
