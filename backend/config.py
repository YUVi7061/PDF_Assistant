import os
from pathlib import Path

from dotenv import load_dotenv

if not os.getenv("VERCEL"):
    load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # free key: https://aistudio.google.com/apikey
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # free key: https://console.groq.com/keys
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "")

# key shown in UI -> {"id": provider model id, "provider": "groq"}
GENERATION_MODELS = {
    "groq-llama": {"id": os.getenv("GROQ_GEN_MODEL", "openai/gpt-oss-20b"), "provider": "groq"},
}

EMBEDDING_MODELS = {
    "gemini-embed": {"id": os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"), "provider": "gemini"},
    "ollama-nomic": {"id": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), "provider": "ollama"},
    "titan-v2": {"id": os.getenv("TITAN_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"), "provider": "bedrock-titan"},
    "cohere-english": {"id": os.getenv("COHERE_EMBED_MODEL_ID", "cohere.embed-english-v3"), "provider": "bedrock-cohere"},
}

DEFAULT_GENERATION_MODEL = os.getenv("DEFAULT_GENERATION_MODEL", "groq-llama")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "gemini-embed")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "groq-llama")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "groq-llama")

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4" if os.getenv("VERCEL") else "25"))
REDACT_OUTPUT_PII = os.getenv("REDACT_OUTPUT_PII", "true").lower() == "true"
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/data" if os.getenv("VERCEL") else "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

NOT_FOUND_MSG = "I couldn't find that information in the provided document(s)."
