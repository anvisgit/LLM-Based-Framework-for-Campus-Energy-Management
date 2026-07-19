"""
End-to-end RAG pipeline for power-systems technical documents (LLM4Doc-style).
Generation backend: Local Qwen3.6-35B-A3B-FP8 via vLLM only (runs on your own GPU, e.g. an H200).

Dataset: ./dataset/ -- 3 FERC/NERC PDFs on inverter-based resource reliability standards
+ a labeled question set.

Stack:
  - Embeddings : intfloat/e5-mistral-7b-instruct (last-token pooling, correct query/passage prefixing)
  - Index      : FAISS (flat, exact search)
  - Reranker   : BAAI/bge-reranker-large (cross-encoder, optional)
  - Generation : Local Qwen3.6-35B-A3B-FP8 via vLLM (needs GPU + vllm serve running)

Requirements:
  pip install torch transformers faiss-cpu pymupdf tiktoken sentence-transformers openai numpy --break-system-packages
  pip install "vllm>=0.17.0" --break-system-packages

Before running query / batch commands, serve the model locally in its own terminal/tmux pane:
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --max-model-len 131072 \
      --reasoning-parser qwen3 --gpu-memory-utilization 0.55

See README.md for how to get HF access to the model and full run instructions.

Usage:
  # 1. Build the index once
  python rag_pipeline.py ingest --pdf_dir ./dataset --index_path ./index/ferc

  # 2. Ask a single question
  python rag_pipeline.py query --index_path ./index/ferc \
      --question "What is phase-lock loop synchronization?"

  # 3. Run every question in the gold set and save answers
  python rag_pipeline.py batch --index_path ./index/ferc \
      --questions_file ./dataset/questions.json --out ./batch_results.json

  # 4. Score retrieval quality (Recall@k, MRR) -- generation-independent
  python rag_pipeline.py retrieval_eval --index_path ./index/ferc \
      --questions_file ./dataset/questions.json

  # 5. Benchmark FAISS flat vs HNSW search time as corpus size grows
  python rag_pipeline.py scaling_benchmark --dim 4096
"""

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import faiss
import fitz  # PyMuPDF
import tiktoken


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

EMBED_MODEL = "intfloat/e5-mistral-7b-instruct"
RERANK_MODEL = "BAAI/bge-reranker-large"


def read_private_setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    project_root = os.path.dirname(os.path.abspath(__file__))
    for candidate in (".env", ".env.local", ".qwen_api_key"):
        path = os.path.join(project_root, candidate)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key.strip() == name:
                    return val.strip().strip('"').strip("'")
    return default


QWEN_BASE_URL = read_private_setting("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL = read_private_setting("QWEN_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")  # current open-weight Qwen (Apr 2026);
                                                                        # Qwen3.7 is closed/API-only -- override
                                                                        # via --model or QWEN_MODEL env var

CHUNK_SIZE = 512
CHUNK_OVERLAP = 96
EMBED_BATCH_SIZE = 8
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    page: int


# --------------------------------------------------------------------------
# 1. Ingestion / chunking
# --------------------------------------------------------------------------

def load_pdf_pages(path: str) -> list[tuple[int, str]]:
    doc = fitz.open(path)
    return [(i, page.get_text()) for i, page in enumerate(doc)]


def chunk_document(path: str, doc_id: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    for page_num, text in load_pdf_pages(path):
        tokens = enc.encode(text)
        if not tokens:
            continue
        start, idx = 0, 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            if chunk_text.strip():
                chunks.append(Chunk(
                    id=f"{doc_id}_p{page_num}_c{idx}",
                    text=chunk_text,
                    doc_id=doc_id,
                    page=page_num,
                ))
            start += chunk_size - overlap
            idx += 1
    return chunks


def ingest_pdf_folder(pdf_dir: str) -> list[Chunk]:
    all_chunks = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        doc_id = os.path.splitext(fname)[0]
        path = os.path.join(pdf_dir, fname)
        chunks = chunk_document(path, doc_id)
        print(f"  {fname}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


# --------------------------------------------------------------------------
# 2. E5-Mistral embedder (correct last-token pooling + query/passage prefixing)
# --------------------------------------------------------------------------

class E5MistralEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cuda", max_length: int = 4096):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # required for last-token pooling
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask, torch):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size), seq_lens]

    def _embed_batch(self, texts: list[str]):
        import torch.nn.functional as F
        torch = self._torch
        batch = self.tokenizer(texts, max_length=self.max_length, padding=True,
                                truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch)
        emb = self._last_token_pool(out.last_hidden_state, batch["attention_mask"], torch)
        return F.normalize(emb, p=2, dim=1)

    def embed_passages(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE):
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(self._embed_batch(texts[i:i + batch_size]).float().cpu())
        return self._torch.cat(embs)

    def embed_query(self, query: str, task_description: str = None):
        task_description = task_description or (
            "Given a question about a power system technical document, "
            "retrieve relevant passages that answer the question"
        )
        prefixed = f"Instruct: {task_description}\nQuery: {query}"
        return self._embed_batch([prefixed]).float().cpu()

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size


# --------------------------------------------------------------------------
# 3. Vector store
# --------------------------------------------------------------------------

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vecs
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings):
        self.index.add(embeddings.numpy().astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_emb, k: int) -> list[tuple[Chunk, float]]:
        scores, idxs = self.index.search(query_emb.numpy().astype("float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        index = faiss.read_index(f"{path_prefix}.faiss")
        store = cls.__new__(cls)
        store.index = index
        with open(f"{path_prefix}.chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store


# --------------------------------------------------------------------------
# 4. Reranker (optional second stage)
# --------------------------------------------------------------------------

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str = "cuda"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        pairs = [(query, c.text) for c, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(score)) for (c, _), score in reranked[:top_k]]


# --------------------------------------------------------------------------
# 5. Prompt construction
# --------------------------------------------------------------------------

def build_rag_prompt(question: str, retrieved_chunks: list[Chunk],
                      persona: str = "experienced policy maker in power systems",
                      task: str = "summarize long text") -> str:
    context = "\n\n".join(f"[Source: {c.doc_id}, p.{c.page}] {c.text}" for c in retrieved_chunks)
    return f"""You are a <{persona}>, who is good at <{task}>.

<Let us think step by step.>

Here is relevant context retrieved from the document(s):
{context}

Now, pay attention! My question is: {question}

In your answer, you should include <as many technical details as possible>.
Please do not copy! You should not include too much original content from the file, unless the original content is strongly needed."""


# --------------------------------------------------------------------------
# 6. Generation backend -- local Qwen3.6 via vLLM
# --------------------------------------------------------------------------

def generate(prompt: str, model: str = None) -> str:
    """
    Calls a local vLLM server or a hosted Qwen-compatible endpoint.
    If you use a hosted service, place your key in a private local file such as .env.
    """
    from openai import OpenAI

    api_key = read_private_setting("QWEN_API_KEY", "not-needed")
    client = OpenAI(base_url=QWEN_BASE_URL, api_key=api_key)
    model = model or QWEN_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not reach vLLM server at {QWEN_BASE_URL}. "
            f"Is `vllm serve {QWEN_MODEL} --port 8000 ...` running? Original error: {e}"
        )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# 7. Pipeline orchestration
# --------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, embedder: E5MistralEmbedder, store: VectorStore, reranker: Reranker = None):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    def retrieve(self, question: str, k_retrieve: int = TOP_K_RETRIEVE, k_final: int = TOP_K_FINAL):
        q_emb = self.embedder.embed_query(question)
        candidates = self.store.search(q_emb, k=k_retrieve)
        final = self.reranker.rerank(question, candidates, k_final) if self.reranker else candidates[:k_final]
        return final

    def query(self, question: str, model: str = None,
              k_retrieve: int = TOP_K_RETRIEVE, k_final: int = TOP_K_FINAL) -> dict:
        final = self.retrieve(question, k_retrieve, k_final)
        final_chunks = [c for c, _ in final]

        prompt = build_rag_prompt(question, final_chunks)
        t0 = time.time()
        answer = generate(prompt, model=model)
        latency = time.time() - t0

        return {
            "question": question,
            "backend": "qwen",
            "answer": answer,
            "latency_s": round(latency, 2),
            "retrieved": [{"id": c.id, "doc": c.doc_id, "page": c.page, "score": s} for c, s in final],
        }


# --------------------------------------------------------------------------
# 8. Retrieval quality metrics (Recall@k, MRR)
# --------------------------------------------------------------------------

def evaluate_retrieval(pipeline: RAGPipeline, questions: list[dict], k_values=(1, 3, 5, 10)) -> dict:
    results = {f"recall@{k}": [] for k in k_values}
    reciprocal_ranks = []

    for item in questions:
        q, expected = item["question"], item.get("expected_source", "")
        if not expected or expected == "both orders":
            continue

        candidates = pipeline.retrieve(q, k_retrieve=max(k_values), k_final=max(k_values))
        doc_ids = [c.doc_id for c, _ in candidates]

        rank = next((i + 1 for i, d in enumerate(doc_ids) if expected in d), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in k_values:
            hit = any(expected in d for d in doc_ids[:k])
            results[f"recall@{k}"].append(1.0 if hit else 0.0)

    summary = {metric: float(np.mean(vals)) for metric, vals in results.items() if vals}
    summary["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
    summary["n_questions_scored"] = len(reciprocal_ranks)
    return summary


# --------------------------------------------------------------------------
# 9. Scaling benchmark: FAISS flat vs HNSW search time as corpus grows
# --------------------------------------------------------------------------

def benchmark_flat_vs_hnsw(dim: int = 4096, sizes=(1_000, 10_000, 50_000, 100_000),
                            n_queries: int = 50) -> list[dict]:
    rows = []
    rng = np.random.default_rng(42)

    for n in sizes:
        vectors = rng.standard_normal((n, dim)).astype("float32")
        faiss.normalize_L2(vectors)
        queries = rng.standard_normal((n_queries, dim)).astype("float32")
        faiss.normalize_L2(queries)

        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(vectors)
        t0 = time.time()
        flat_index.search(queries, 10)
        flat_time = (time.time() - t0) / n_queries

        hnsw_index = faiss.IndexHNSWFlat(dim, 32)
        hnsw_index.add(vectors)
        t0 = time.time()
        hnsw_index.search(queries, 10)
        hnsw_time = (time.time() - t0) / n_queries

        rows.append({
            "corpus_size": n,
            "flat_search_ms": round(flat_time * 1000, 3),
            "hnsw_search_ms": round(hnsw_time * 1000, 3),
            "speedup_x": round(flat_time / hnsw_time, 1) if hnsw_time > 0 else None,
        })
        print(f"n={n:>7}  flat={flat_time*1000:8.3f}ms  hnsw={hnsw_time*1000:8.3f}ms  "
              f"speedup={rows[-1]['speedup_x']}x")

    return rows


# --------------------------------------------------------------------------
# CLI commands
# --------------------------------------------------------------------------

def _load_pipeline(index_path: str, no_rerank: bool) -> RAGPipeline:
    embedder = E5MistralEmbedder()
    store = VectorStore.load(index_path)
    reranker = None if no_rerank else Reranker()
    return RAGPipeline(embedder, store, reranker)


def cmd_ingest(args):
    print(f"Ingesting PDFs from {args.pdf_dir} ...")
    chunks = ingest_pdf_folder(args.pdf_dir)
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedder (E5-Mistral, ~14GB VRAM in bf16)...")
    embedder = E5MistralEmbedder()

    print("Embedding passages...")
    embeddings = embedder.embed_passages([c.text for c in chunks])

    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embeddings)
    store.save(args.index_path)
    print(f"Saved index to {args.index_path}.faiss / .chunks.pkl")


def cmd_query(args):
    pipeline = _load_pipeline(args.index_path, args.no_rerank)
    result = pipeline.query(args.question, model=args.model)

    print(f"\n=== ANSWER (qwen, {result['latency_s']}s) ===")
    print(result["answer"])
    print("\n=== RETRIEVED SOURCES ===")
    for r in result["retrieved"]:
        print(f"  {r['doc']} p.{r['page']}  (score={r['score']:.3f})")


def cmd_batch(args):
    with open(args.questions_file) as f:
        questions = json.load(f)

    pipeline = _load_pipeline(args.index_path, args.no_rerank)

    results = []
    for item in questions:
        q = item["question"]
        row = {"question": q, "category": item.get("category", "unlabeled")}
        try:
            r = pipeline.query(q, model=args.model)
            row["answer"] = r["answer"]
            row["latency_s"] = r["latency_s"]
        except Exception as e:
            row["answer"] = f"[ERROR: {e}]"
            row["latency_s"] = None
        results.append(row)
        print(f"[done] {q[:60]}...")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} answers to {args.out}")


def cmd_retrieval_eval(args):
    with open(args.questions_file) as f:
        questions = json.load(f)
    pipeline = _load_pipeline(args.index_path, args.no_rerank)
    summary = evaluate_retrieval(pipeline, questions)
    print(json.dumps(summary, indent=2))


def cmd_scaling_benchmark(args):
    rows = benchmark_flat_vs_hnsw(dim=args.dim)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline (Qwen backend) for power-systems technical documents")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build a vector index from a folder of PDFs")
    p_ingest.add_argument("--pdf_dir", required=True)
    p_ingest.add_argument("--index_path", required=True)
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a single question against a built index")
    p_query.add_argument("--index_path", required=True)
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--model", default=None, help="Override the default Qwen model name")
    p_query.add_argument("--no_rerank", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_batch = sub.add_parser("batch", help="Run every question in a gold question set and save the answers")
    p_batch.add_argument("--index_path", required=True)
    p_batch.add_argument("--questions_file", required=True)
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--out", default="batch_results.json")
    p_batch.add_argument("--no_rerank", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    p_reteval = sub.add_parser("retrieval_eval", help="Score retrieval quality (Recall@k, MRR) -- generation-independent")
    p_reteval.add_argument("--index_path", required=True)
    p_reteval.add_argument("--questions_file", required=True)
    p_reteval.add_argument("--no_rerank", action="store_true")
    p_reteval.set_defaults(func=cmd_retrieval_eval)

    p_scale = sub.add_parser("scaling_benchmark", help="Benchmark FAISS flat vs HNSW search time at increasing corpus sizes")
    p_scale.add_argument("--dim", type=int, default=4096)
    p_scale.add_argument("--out", default="scaling_results.json")
    p_scale.set_defaults(func=cmd_scaling_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
