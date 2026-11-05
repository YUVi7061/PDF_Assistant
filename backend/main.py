import hashlib
import hmac
import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from . import config, evals
from .graph import build_graph, turn_input
from .llm_client import LLMClient
from .pdf_utils import PDFError, chunk_pages, extract_pages
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdf-assistant")

app = FastAPI(title="PDF Assistant", version="1.0")


@app.middleware("http")
async def require_access_token(request: Request, call_next):
    if config.APP_ACCESS_TOKEN:
        expected = f"Bearer {config.APP_ACCESS_TOKEN}"
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, expected):
            return JSONResponse({"detail": "Valid app access token required."}, status_code=401)
    return await call_next(request)


bedrock = LLMClient()
store = VectorStore(config.DATA_DIR / "vectors")
graph, checkpointer = build_graph(bedrock, store)  # MemorySaver lives in this process (run 1 worker)


# ------------------------------------------------------------------ schemas
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str
    doc_ids: list[str] = Field(min_length=1)
    generation_model: str = config.DEFAULT_GENERATION_MODEL
    history: list[dict[str, str]] = Field(default_factory=list)


class EvalRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1)
    cases: list[dict] = Field(min_length=1)
    generation_model: str = config.DEFAULT_GENERATION_MODEL
    judge_model: str = config.JUDGE_MODEL


# ------------------------------------------------------------------ meta
@app.get("/health")
def health():
    return {"status": "ok", "documents": len(store.list())}


@app.get("/models")
def models():
    embedding_models = list(config.EMBEDDING_MODELS)
    if os.getenv("VERCEL"):
        embedding_models = [config.DEFAULT_EMBEDDING_MODEL]
    return {
        "generation": list(config.GENERATION_MODELS),
        "embedding": embedding_models,
        "default_generation": config.DEFAULT_GENERATION_MODEL,
        "default_embedding": config.DEFAULT_EMBEDDING_MODEL,
        "bedrock_guardrail": bool(config.GUARDRAIL_ID),
    }


# ------------------------------------------------------------------ documents
@app.post("/documents")
def upload_document(file: UploadFile = File(...), embedding_model: str = Form(config.DEFAULT_EMBEDDING_MODEL)):
    if embedding_model not in config.EMBEDDING_MODELS:
        raise HTTPException(400, f"Unknown embedding model '{embedding_model}'")
    data = file.file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB")

    chunking = f"{embedding_model}:{config.CHUNK_SIZE}:{config.CHUNK_OVERLAP}".encode()
    doc_id = hashlib.sha256(data + chunking).hexdigest()[:16]
    if existing := store.get(doc_id):  # same file + same embedding model -> reuse
        return {**existing.meta(), "cached": True}

    try:
        pages = extract_pages(data)
    except PDFError as e:
        raise HTTPException(422, str(e)) from e
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    try:
        embeddings = bedrock.embed([c["text"] for c in chunks], embedding_model, "document")
    except Exception as e:  # noqa: BLE001
        log.exception("embedding failed")
        raise HTTPException(502, f"Embedding failed: {e}") from e

    rec = store.add(doc_id, file.filename or "document.pdf", len(pages), embedding_model, chunks, embeddings)
    return {**rec.meta(), "cached": False}


@app.get("/documents")
def list_documents():
    return store.list()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not store.delete(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


# ------------------------------------------------------------------ chat
@app.post("/chat")
def chat(req: ChatRequest):
    if req.generation_model not in config.GENERATION_MODELS:
        raise HTTPException(400, f"Unknown generation model '{req.generation_model}'")
    missing = [d for d in req.doc_ids if store.get(d) is None]
    if missing:
        raise HTTPException(404, f"Unknown document(s): {missing}")
    try:
        history = [
            HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"])
            for item in req.history[-config.MAX_HISTORY_MESSAGES :]
            if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str)
        ]
        state = graph.invoke(
            turn_input(req.question, req.doc_ids, req.generation_model, history if req.history else None),
            config={"configurable": {"thread_id": req.session_id}},
        )
    except Exception as e:  # noqa: BLE001
        log.exception("chat failed")
        raise HTTPException(502, f"Model call failed: {e}") from e

    seen, sources = set(), []
    for c in state.get("contexts", []):
        key = (c["doc_id"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": c["filename"], "page": c["page"], "score": round(c["score"], 3),
                            "snippet": c["text"][:300]})
    return {
        "answer": state["answer"],
        "sources": sources,
        "blocked": state.get("blocked", False),
        "flags": state.get("flags", []),
        "standalone_question": state.get("standalone_question", ""),
    }


@app.get("/sessions/{session_id}/history")
def history(session_id: str):
    snap = graph.get_state({"configurable": {"thread_id": session_id}})
    msgs = (snap.values or {}).get("messages", [])
    return [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in msgs]


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    try:
        checkpointer.delete_thread(session_id)
    except AttributeError:  # older langgraph: fall back to dropping the in-memory dicts
        checkpointer.storage.pop(session_id, None)
    return {"cleared": session_id}


# ------------------------------------------------------------------ evals
@app.post("/evals/guardrails")
def eval_guardrails():
    return evals.run_guardrail_evals()


@app.post("/evals/run")
def eval_run(req: EvalRequest):
    for k in (req.generation_model, req.judge_model):
        if k not in config.GENERATION_MODELS:
            raise HTTPException(400, f"Unknown model '{k}'")
    missing = [d for d in req.doc_ids if store.get(d) is None]
    if missing:
        raise HTTPException(404, f"Unknown document(s): {missing}")
    return evals.run_dataset_eval(graph, bedrock, req.doc_ids, req.cases, req.generation_model, req.judge_model)
