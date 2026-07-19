# RAG Pipeline — Qwen3.6 Backend (Local, vLLM)
### Power Systems Technical Documents (FERC / NERC, inverter-based resources)

Self-contained RAG pipeline: PDF ingestion → chunking → E5-Mistral embeddings → FAISS
retrieval → optional cross-encoder reranking → answer generation via a **locally served
Qwen3.6 model on your own GPU**, using vLLM.

This folder is fully independent — it does not need the OpenAI folder or anything
outside itself to run. No API key is required (the model runs on your machine), but you
do need a GPU with enough VRAM (an H200 or similar) and, once, a Hugging Face login to
download the weights.

---

## What's in this folder

```
rag_demo_qwen/
├── rag_pipeline.py       # everything: ingestion, embedding, retrieval, Qwen generation, eval, CLI
├── dataset/
│   ├── ferc_order_901.pdf         # FERC Order 901 (2023) - the original IBR reliability directive
│   ├── ferc_order_rd26_2026.pdf   # Follow-up order (Feb 2026) approving the actual standards
│   ├── nerc_ibr_intro_guide.pdf   # NERC's plain-language IBR explainer (different register, good test)
│   └── questions.json             # 13 gold questions with expected_source labels
└── README.md
```

**Note on the dataset:** these PDFs are condensed excerpts of the real public documents
(from `elibrary.ferc.gov` and `nerc.com`), not full verbatim reproductions.

---

## 1. Get access to the model (Hugging Face token)

`Qwen/Qwen3.6-35B-A3B-FP8` is downloaded from Hugging Face the first time you run
`vllm serve`. Most Qwen releases are ungated, but if the download errors out with a 401,
do this:

1. Create a Hugging Face account at **https://huggingface.co/join** if you don't have one.
2. Go to **https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8** and accept any license/terms
   shown on the model card.
3. Create a read token at **https://huggingface.co/settings/tokens** → "New token" (role:
   `read`).
4. Log in from the shell that will run `vllm serve`:
   ```bash
   pip install huggingface_hub --break-system-packages
   huggingface-cli login
   # paste the token when prompted
   ```
   or export it directly:
   ```bash
   export HF_TOKEN=hf_...
   ```

No `OPENAI_API_KEY` or any other cloud credential is needed for this folder — generation
happens entirely on your own GPU.

### Private Qwen key setup (recommended for demos)

If you want to use a hosted Qwen-compatible endpoint instead of the local vLLM server,
put your key in a private local file and keep it out of Git:

1. Create a local file named [.env](.env) in this folder.
2. Add your real values to it:
   ```env
   QWEN_API_KEY=your_real_key_here
   QWEN_BASE_URL=http://localhost:8000/v1
   QWEN_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
   ```
3. The pipeline will read the key from this file automatically.

The file [.env](.env) is git-ignored, so it will stay local and can be safely used for your demo.

---

## 2. Install dependencies

```bash
pip install torch transformers faiss-cpu pymupdf tiktoken sentence-transformers openai numpy --break-system-packages
pip install "vllm>=0.17.0" --break-system-packages
```

Notes:
- The `openai` package here is only used as a generic OpenAI-compatible **client** to talk
  to your local vLLM server — no OpenAI account or key involved.
- `Qwen/Qwen3.6-35B-A3B-FP8` is the current open-weight Qwen release (as of Apr 2026);
  it's an MoE model — 35B total params, ~3B active per token — which is why it's fast on
  a single H200. (Qwen3.7 exists but is API-only / closed-weight, so it isn't an option
  for local serving.)

---

## 3. Serve the model locally

In its own terminal / tmux pane, before running any `query` or `batch` command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8000 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.55
```

- `--gpu-memory-utilization 0.55` leaves headroom for E5-Mistral (~14GB) to run alongside
  it on the same GPU during `ingest` / `query` / `batch`.
- `--reasoning-parser qwen3` is required for this model family — it separates the
  model's internal reasoning trace from the final answer, so the pipeline only sees the
  clean answer text.
- Leave this process running; the pipeline talks to it over HTTP at
  `http://localhost:8000/v1` by default (override with `QWEN_BASE_URL` if you serve it
  elsewhere).

---

## 4. Build the index (one-time)

```bash
python rag_pipeline.py ingest --pdf_dir ./dataset --index_path ./index/ferc
```

This loads E5-Mistral, chunks and embeds all PDFs in `./dataset`, and saves a FAISS
index to `./index/ferc.faiss` + `./index/ferc.chunks.pkl`. This step doesn't touch the
vLLM server, so you can run it before or after starting `vllm serve`.

---

## 5. Run it

### Single question
```bash
python rag_pipeline.py query --index_path ./index/ferc \
    --question "What is phase-lock loop synchronization?"
```

### Every question in the gold set, saved to a file
```bash
python rag_pipeline.py batch --index_path ./index/ferc \
    --questions_file ./dataset/questions.json --out batch_results.json
```
Produces one JSON file with the answer and latency for every question in
`questions.json`.

### Retrieval quality (Recall@k, MRR) — doesn't touch vLLM, scores retrieval alone
```bash
python rag_pipeline.py retrieval_eval --index_path ./index/ferc \
    --questions_file ./dataset/questions.json
```

### Scaling benchmark (FAISS flat vs. HNSW search time as corpus size grows)
```bash
python rag_pipeline.py scaling_benchmark --dim 4096
```
No GPU or model loading needed for this one — pure FAISS timing, safe to run any time.

---

## Troubleshooting

- **`Could not reach vLLM server at http://localhost:8000/v1`:** `vllm serve` isn't
  running, or is still loading. Check the server's terminal — the first request after
  startup can be slow while weights load onto the GPU.
- **401 downloading the model:** you haven't accepted the model card terms or logged in
  — see step 1.
- **OOM running `ingest` and `vllm serve` at the same time:** lower
  `--gpu-memory-utilization` on the vLLM side further (e.g. `0.45`), or run `ingest`
  before starting the vLLM server (ingestion doesn't need the generation model at all).
- **Qwen answers include visible "thinking" text:** confirm `--reasoning-parser qwen3`
  is set in your `vllm serve` command — without it, the reasoning trace leaks into
  `message.content` instead of being separated out.
- **"unstable retrieval" / wrong chunks retrieved:** almost always a pooling bug with
  E5-Mistral. Confirm `tokenizer.padding_side = "left"` and last-token pooling (already
  correct in `E5MistralEmbedder` here) — mean-pooling is the most common silent failure
  mode with this model and doesn't throw an error, it just quietly hurts relevance.
