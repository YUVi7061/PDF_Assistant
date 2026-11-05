# PDF Assistant (Streamlit + FastAPI + LangGraph)

```
Streamlit ──HTTP──► FastAPI ──► LangGraph (MemorySaver checkpointer)
                                 input_guard → rewrite → retrieve → generate → output_guard → finalize
                     numpy vector store (data/vectors)    Gemini | Groq (free APIs) | Ollama | Bedrock
```

## Run on Windows (PowerShell) - nothing to install except Python packages

1. Get a **free Gemini API key** at https://aistudio.google.com/apikey (it covers generation *and* embeddings).
   Optional: a free Groq key at https://console.groq.com/keys.
2. Setup:
   ```powershell
   cd C:\Users\yuvraj.paygude\Documents\PDF_Assistant
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements-local.txt
   copy .env.example .env
   notepad .env        # paste GEMINI_API_KEY=...
   ```
3. Two terminals (activate the venv in both):
   ```powershell
   uvicorn backend.main:app --port 8000      # keep ONE worker: MemorySaver is in-process
   streamlit run frontend/app.py
   ```
4. Open http://localhost:8501, upload a PDF, **Process PDF**, chat.

If pip is blocked by a proxy: `pip install -r requirements.txt --proxy http://user:pass@proxy:port` (or use your internal PyPI mirror).

## Deploy with Streamlit

Streamlit Community Cloud hosts the UI, while Render runs the FastAPI backend. Supabase Storage keeps document vectors durable across backend restarts.

1. In Render, choose **New → Blueprint** and select this GitHub repository. Render reads `render.yaml` and creates the FastAPI service.
2. In the Render service's environment settings, enter `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `GROQ_API_KEY`, and `GEMINI_API_KEY`. Keep these values private. Set a long random `APP_ACCESS_TOKEN`; the service creates its private `pdf-assistant-vectors` bucket on first startup.
3. Wait for the Render service to become healthy, then copy its public URL, for example `https://pdf-assistant-api.onrender.com`.
4. On [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository, select branch `main`, and set the main file to `frontend/app.py`.
5. In the Streamlit app's **Settings → Secrets**, add the backend URL and the exact same access token configured in Render:
   ```toml
   BACKEND_URL = "https://your-api.onrender.com"
   APP_ACCESS_TOKEN = "your-long-random-token"
   ```
6. Deploy or reboot the Streamlit app. Upload a PDF and try a question.

The root `requirements.txt` contains the Streamlit UI dependencies; Render installs `requirements-local.txt`, including Uvicorn. Free Render services may sleep while idle, so the first request can take longer. Uploads are limited by the backend's `MAX_UPLOAD_MB` setting. Existing local vector files are not migrated; upload those PDFs again after deployment.

## Privacy
Gemini/Groq are third-party services: your question and the retrieved PDF text are sent to them, and free tiers may be used
for provider training/logging. Only use non-confidential PDFs unless your company policy approves these services.

## Features
- **Memory**: LangGraph `MemorySaver`, `thread_id = session_id`; follow-ups work. "New conversation" clears it.
- **Guardrails** (`backend/guardrails.py`): input (length, prompt injection, blocked topics; blocked prompts never enter memory),
  retrieved-chunk injection filtering, output PII/secret redaction, system-prompt-leak block, grounding warning.
- **Evals** (`backend/evals.py`): offline guardrail tests; dataset evals with hit@k, MRR, keyword recall, refusal accuracy, LLM-judge scores.

## Evals
```powershell
python run_evals.py --guardrails-only
copy evals\sample_dataset.json my_eval.json     # edit questions/answers for your PDF
python run_evals.py --pdf your.pdf --dataset my_eval.json
```
Free tiers are rate-limited (HTTP 429); the client retries with backoff, so large PDFs/evals can be slow, not broken.

## Notes
- Model names change over time. If you get a 404 for a model, update `GEMINI_GEN_MODEL` / `GROQ_GEN_MODEL` / `GEMINI_EMBED_MODEL` in `.env`.
- Changing the embedding model requires re-uploading the PDF (vectors are per-model).
- `MIN_SCORE` (cosine) decides when the bot says "not found" without calling the LLM; tune with the evals.
- Scanned PDFs (no text layer) are rejected; no OCR in this version.
- Output PII redaction also masks legit emails/phones from the PDF; set `REDACT_OUTPUT_PII=false` to disable.
