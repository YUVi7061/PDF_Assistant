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

## Deploy to Vercel

The Vercel deployment serves the browser UI from `public/` through Vercel's CDN and routes API requests under `/api` to the root FastAPI entrypoint. Supabase Storage persists PDF chunks and vectors. The existing Streamlit UI remains available for local use. No internal rewrites are needed.

1. Push the repository to GitHub and import it into Vercel. Keep the project root as the root directory; no build command is needed.
2. Create a Supabase project. Set these environment variables in Vercel for the Production environment:
   - `SUPABASE_URL`: the project URL.
   - `SUPABASE_SERVICE_ROLE_KEY`: the server-side service-role key. Never expose this in browser code.
   - `GROQ_API_KEY`: the Groq API key.
   - `GEMINI_API_KEY`: used for Gemini document embeddings.
   - `APP_ACCESS_TOKEN`: a long random token to protect all API endpoints. Enter it in the app's Deployment access section.
3. Set the model and upload settings:
   - `DEFAULT_GENERATION_MODEL=groq-llama`
   - `GROQ_GEN_MODEL=openai/gpt-oss-20b`
   - `DEFAULT_EMBEDDING_MODEL=gemini-embed`
   - `GEMINI_EMBED_MODEL=gemini-embedding-001`
   - `REWRITE_MODEL=groq-llama`
   - `JUDGE_MODEL=groq-llama`
   - `MAX_UPLOAD_MB=4`
   - `CHUNK_SIZE=1800` and `CHUNK_OVERLAP=300`
4. Redeploy. The backend creates a private `pdf-assistant-vectors` bucket on first startup; override its name with `SUPABASE_STORAGE_BUCKET` if needed.

Vercel's serverless request and execution limits apply. The UI limits PDFs to 4 MB to leave room for multipart upload overhead. Long embedding or generation requests can exceed the function duration allowed by your Vercel plan. Documents stored in a local `data/` directory are not migrated; upload them again after deployment. Chat history is kept in the browser and sent with each request so follow-ups survive function cold starts.

The deployment is not protected unless `APP_ACCESS_TOKEN` is set. Use a unique random value, keep it in Vercel's encrypted environment settings, and do not commit it.

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
