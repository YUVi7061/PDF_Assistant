"""Unified LLM client. Routes each model key (see config.py) to Gemini, Groq, Ollama or Amazon Bedrock.
Gemini / Groq / Ollama use plain HTTPS via `requests` - nothing extra to install."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _flat(messages: list[dict]) -> list[dict]:
    """Converse-shape messages -> [{"role", "content": str}]."""
    return [{"role": m["role"], "content": "".join(p["text"] for p in m["content"])} for m in messages]


def _post(url: str, *, headers=None, payload=None, timeout=300, retries=5) -> dict:
    """POST with backoff on 429/5xx (free tiers rate-limit aggressively)."""
    for attempt in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            time.sleep(min(2 ** attempt * 2, 30))
            continue
        if not r.ok:
            raise RuntimeError(f"{url.split('?')[0]} -> HTTP {r.status_code}: {r.text[:400]}")
        return r.json()
    raise RuntimeError("unreachable")


def _need_key(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value


def _gemini_model_fallback(model_id: str) -> str:
    legacy = {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"}
    if model_id in legacy:
        return "gemini-3.8-flash"
    return model_id


def _post_gemini_with_fallback(model_id: str, *, headers: dict, payload: dict):
    candidates = [model_id]
    fallback = _gemini_model_fallback(model_id)
    if fallback != model_id:
        candidates.append(fallback)

    last_error = None
    for candidate in candidates:
        try:
            return _post(f"{GEMINI_BASE}/models/{candidate}:generateContent", headers=headers, payload=payload)
        except RuntimeError as exc:
            last_error = exc
            if "HTTP 404" not in str(exc):
                raise
    raise last_error


class LLMClient:
    def __init__(self):
        self._rt = None  # boto3 client, created lazily so non-AWS setups never touch AWS

    @property
    def rt(self):
        if self._rt is None:
            import boto3
            from botocore.config import Config

            self._rt = boto3.client("bedrock-runtime", region_name=config.AWS_REGION,
                                    config=Config(retries={"max_attempts": 6, "mode": "adaptive"}, read_timeout=120))
        return self._rt

    # ================================================================ embeddings
    def embed(self, texts: list[str], model_key: str, input_type: str = "document") -> list[list[float]]:
        """input_type: 'document' | 'query'."""
        spec = config.EMBEDDING_MODELS[model_key]
        return {
            "gemini": self._embed_gemini,
            "ollama": self._embed_ollama,
            "bedrock-titan": lambda t, m, _i: self._embed_titan(t, m),
            "bedrock-cohere": self._embed_cohere,
        }[spec["provider"]](texts, spec["id"], input_type)

    def _embed_gemini(self, texts, model_id, input_type):
        key = _need_key("GEMINI_API_KEY", config.GEMINI_API_KEY)
        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        out = []
        for i in range(0, len(texts), 100):  # batch limit = 100
            reqs = [{"model": f"models/{model_id}", "content": {"parts": [{"text": t}]},
                     "taskType": task, "outputDimensionality": 768} for t in texts[i : i + 100]]
            data = _post(f"{GEMINI_BASE}/models/{model_id}:batchEmbedContents",
                         headers={"x-goog-api-key": key}, payload={"requests": reqs})
            out.extend(e["values"] for e in data["embeddings"])
        return out

    def _embed_ollama(self, texts, model_id, input_type):
        if "nomic" in model_id:
            prefix = "search_query: " if input_type == "query" else "search_document: "
            texts = [prefix + t for t in texts]
        out = []
        for i in range(0, len(texts), 32):
            data = _post(f"{config.OLLAMA_URL}/api/embed", payload={"model": model_id, "input": texts[i : i + 32]})
            out.extend(data["embeddings"])
        return out

    def _embed_titan(self, texts, model_id):
        def one(t: str):
            body = {"inputText": t}
            if "v2" in model_id:
                body.update({"dimensions": 1024, "normalize": True})
            r = self.rt.invoke_model(modelId=model_id, body=json.dumps(body))
            return json.loads(r["body"].read())["embedding"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(one, texts))

    def _embed_cohere(self, texts, model_id, input_type):
        cohere_type = "search_query" if input_type == "query" else "search_document"
        out = []
        for i in range(0, len(texts), 96):
            body = {"texts": texts[i : i + 96], "input_type": cohere_type, "truncate": "END"}
            r = self.rt.invoke_model(modelId=model_id, body=json.dumps(body))
            out.extend(json.loads(r["body"].read())["embeddings"])
        return out

    # ================================================================ generation
    def generate(self, model_key: str, messages: list[dict], system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """messages use Bedrock Converse shape: [{"role": "user", "content": [{"text": "..."}]}]."""
        spec = config.GENERATION_MODELS[model_key]
        p = spec["provider"]
        if p == "gemini":
            return self._gen_gemini(spec["id"], messages, system, max_tokens, temperature)
        if p == "groq":
            return self._gen_groq(spec["id"], messages, system, max_tokens, temperature)
        if p == "ollama":
            return self._gen_ollama(spec["id"], messages, system, max_tokens, temperature)
        return self._gen_bedrock(spec["id"], messages, system, max_tokens, temperature)

    def _gen_gemini(self, model_id, messages, system, max_tokens, temperature):
        key = _need_key("GEMINI_API_KEY", config.GEMINI_API_KEY)
        payload = {
            "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                         for m in _flat(messages)],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = _post_gemini_with_fallback(
            model_id,
            headers={"x-goog-api-key": key},
            payload=payload,
        )
        cands = data.get("candidates") or []
        if not cands or "content" not in cands[0]:
            reason = (cands[0].get("finishReason") if cands else data.get("promptFeedback", {}).get("blockReason")) or "empty"
            return f"(Model returned no text: {reason})"
        return "".join(part.get("text", "") for part in cands[0]["content"].get("parts", [])).strip()

    def _gen_groq(self, model_id, messages, system, max_tokens, temperature):
        key = _need_key("GROQ_API_KEY", config.GROQ_API_KEY)
        msgs = ([{"role": "system", "content": system}] if system else []) + _flat(messages)
        data = _post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                     payload={"model": model_id, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens})
        return data["choices"][0]["message"]["content"].strip()

    def _gen_ollama(self, model_id, messages, system, max_tokens, temperature):
        msgs = ([{"role": "system", "content": system}] if system else []) + _flat(messages)
        data = _post(f"{config.OLLAMA_URL}/api/chat", timeout=600, payload={
            "model": model_id, "messages": msgs, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}})
        return data["message"]["content"].strip()

    def _gen_bedrock(self, model_id, messages, system, max_tokens, temperature):
        kwargs = {"modelId": model_id, "messages": messages,
                  "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
        if system:
            kwargs["system"] = [{"text": system}]
        resp = self.rt.converse(**kwargs)
        return "".join(p.get("text", "") for p in resp["output"]["message"]["content"]).strip()

    # ================================================================ Bedrock Guardrails (optional)
    def apply_guardrail(self, text: str, source: str) -> tuple[bool, str]:
        """source: 'INPUT' | 'OUTPUT'. No-op unless BEDROCK_GUARDRAIL_ID is set."""
        if not config.GUARDRAIL_ID:
            return False, text
        r = self.rt.apply_guardrail(
            guardrailIdentifier=config.GUARDRAIL_ID, guardrailVersion=config.GUARDRAIL_VERSION,
            source=source, content=[{"text": {"text": text}}],
        )
        if r.get("action") == "GUARDRAIL_INTERVENED":
            outs = r.get("outputs") or []
            return True, outs[0]["text"] if outs else config.NOT_FOUND_MSG
        return False, text
