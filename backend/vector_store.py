"""Tiny numpy vector store, persisted to disk (one .json + one .npy per document)."""
from __future__ import annotations

import json
import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


@dataclass
class DocRecord:
    doc_id: str
    filename: str
    pages: int
    embedding_model: str
    chunks: list[dict]  # [{"page": int, "text": str}]
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalised

    def meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "pages": self.pages,
            "chunks": len(self.chunks),
            "embedding_model": self.embedding_model,
        }


class VectorStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, DocRecord] = {}
        self._lock = threading.Lock()
        self._supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self._supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self._bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "pdf-assistant-vectors")
        if bool(self._supabase_url) != bool(self._supabase_key):
            raise RuntimeError("Set both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to enable Supabase storage.")
        if self._supabase_url:
            self._ensure_bucket()
            self._load_remote()
        else:
            self._load_all()

    @property
    def _remote_enabled(self) -> bool:
        return bool(self._supabase_url)

    def _storage_request(self, method: str, path: str, *, content_type="application/json", headers=None, **kwargs):
        request_headers = {
            "apikey": self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Content-Type": content_type,
        }
        if headers:
            request_headers.update(headers)
        response = requests.request(
            method,
            f"{self._supabase_url}/storage/v1/{path}",
            headers=request_headers,
            timeout=30,
            **kwargs,
        )
        if not response.ok:
            raise RuntimeError(f"Supabase Storage returned HTTP {response.status_code}: {response.text[:300]}")
        return response

    def _ensure_bucket(self):
        headers = {
            "apikey": self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self._supabase_url}/storage/v1/bucket",
            headers=headers,
            json={"id": self._bucket, "name": self._bucket, "public": False},
            timeout=30,
        )
        if response.ok:
            return

        existing = requests.get(
            f"{self._supabase_url}/storage/v1/bucket/{self._bucket}",
            headers=headers,
            timeout=30,
        )
        if existing.ok:
            return

        detail = response.text[:300]
        raise RuntimeError(
            f"Could not initialize Supabase Storage bucket '{self._bucket}' "
            f"(create HTTP {response.status_code}, lookup HTTP {existing.status_code}): {detail}"
        )

    def _load_remote(self):
        offset = 0
        while True:
            response = self._storage_request(
                "POST",
                f"object/list/{self._bucket}",
                json={"prefix": "", "limit": 100, "offset": offset},
            )
            items = response.json()
            for item in items:
                name = item.get("name", "")
                if not name.endswith(".json"):
                    continue
                doc_id = name[:-5]
                metadata = self._storage_request("GET", f"object/{self._bucket}/{doc_id}.json").json()
                embeddings_response = self._storage_request(
                    "GET", f"object/{self._bucket}/{doc_id}.npy", content_type="application/octet-stream"
                )
                self._docs[doc_id] = DocRecord(
                    doc_id=metadata["doc_id"],
                    filename=metadata["filename"],
                    pages=metadata["pages"],
                    embedding_model=metadata["embedding_model"],
                    chunks=metadata["chunk_data"],
                    embeddings=np.load(io.BytesIO(embeddings_response.content)),
                )
            offset += len(items)
            if len(items) < 100:
                break

    def _save_remote(self, rec: DocRecord):
        embeddings = io.BytesIO()
        np.save(embeddings, rec.embeddings)
        self._storage_request(
            "POST",
            f"object/{self._bucket}/{rec.doc_id}.npy",
            content_type="application/octet-stream",
            data=embeddings.getvalue(),
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Content-Type": "application/octet-stream",
                "x-upsert": "true",
            },
        )
        metadata = {**rec.meta(), "chunk_data": rec.chunks}
        self._storage_request(
            "POST",
            f"object/{self._bucket}/{rec.doc_id}.json",
            json=metadata,
            headers={
                "apikey": self._supabase_key,
                "Authorization": f"Bearer {self._supabase_key}",
                "Content-Type": "application/json",
                "x-upsert": "true",
            },
        )

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.clip(n, 1e-12, None)

    def _load_all(self):
        for meta_path in self.dir.glob("*.json"):
            npy = meta_path.with_suffix(".npy")
            if not npy.exists():
                continue
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            self._docs[m["doc_id"]] = DocRecord(
                doc_id=m["doc_id"],
                filename=m["filename"],
                pages=m["pages"],
                embedding_model=m["embedding_model"],
                chunks=m["chunk_data"],
                embeddings=np.load(npy),
            )

    def add(self, doc_id, filename, pages, embedding_model, chunks, embeddings) -> DocRecord:
        emb = self._normalise(np.asarray(embeddings, dtype=np.float32))
        rec = DocRecord(doc_id, filename, pages, embedding_model, chunks, emb)
        with self._lock:
            if self._remote_enabled:
                self._save_remote(rec)
                self._docs[doc_id] = rec
                return rec
            self._docs[doc_id] = rec
            (self.dir / f"{doc_id}.json").write_text(
                json.dumps({**rec.meta(), "chunk_data": chunks}), encoding="utf-8"
            )
            np.save(self.dir / f"{doc_id}.npy", emb)
        return rec

    def get(self, doc_id: str) -> DocRecord | None:
        return self._docs.get(doc_id)

    def list(self) -> list[dict]:
        return [d.meta() for d in self._docs.values()]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if self._remote_enabled:
                existed = self._docs.pop(doc_id, None) is not None
                self._storage_request(
                    "DELETE",
                    f"object/{self._bucket}",
                    json={"prefixes": [f"{doc_id}.json", f"{doc_id}.npy"]},
                )
                return existed
            existed = self._docs.pop(doc_id, None) is not None
            for ext in (".json", ".npy"):
                (self.dir / f"{doc_id}{ext}").unlink(missing_ok=True)
        return existed

    def search(self, doc_ids: list[str], query_vec, k: int) -> list[dict]:
        """Cosine top-k over the given docs (all must share the query's embedding model)."""
        q = self._normalise(np.asarray(query_vec, dtype=np.float32))
        hits: list[dict] = []
        for did in doc_ids:
            rec = self._docs.get(did)
            if rec is None:
                continue
            scores = rec.embeddings @ q
            for i in np.argsort(-scores)[:k]:
                hits.append(
                    {
                        "doc_id": did,
                        "filename": rec.filename,
                        "page": rec.chunks[i]["page"],
                        "text": rec.chunks[i]["text"],
                        "score": float(scores[i]),
                    }
                )
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]
