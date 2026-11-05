import json
import os
import uuid

import pandas as pd
import requests
import streamlit as st


def setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001  secrets are optional during local development
        value = None
    return value or os.getenv(name, default)


API = setting("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = setting("APP_ACCESS_TOKEN")

st.set_page_config(page_title="PDF Assistant", page_icon="📄", layout="wide")


def api(method: str, path: str, **kw):
    kw.setdefault("timeout", 600)
    headers = dict(kw.pop("headers", {}))
    if API_TOKEN:
        headers.setdefault("Authorization", f"Bearer {API_TOKEN}")
    kw["headers"] = headers
    r = requests.request(method, f"{API}{path}", **kw)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


# ---------------------------------------------------------------- state
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])

try:
    models = api("GET", "/models", timeout=10)
except Exception as e:  # noqa: BLE001
    st.error(f"Backend not reachable at {API}: {e}")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 PDF Assistant")
    gen_model = st.selectbox("Generation model", models["generation"],
                             index=models["generation"].index(models["default_generation"]))
    emb_model = st.selectbox("Embedding model (for new uploads)", models["embedding"],
                             index=models["embedding"].index(models["default_embedding"]))
    st.caption("Bedrock Guardrail: " + ("✅ enabled" if models["bedrock_guardrail"] else "➖ off (built-in guardrails active)"))

    st.divider()
    up = st.file_uploader("Upload PDF", type=["pdf"])
    if up and st.button("Process PDF", type="primary", use_container_width=True):
        with st.spinner("Extracting, chunking and embedding…"):
            try:
                d = api("POST", "/documents", files={"file": (up.name, up.getvalue(), "application/pdf")},
                        data={"embedding_model": emb_model})
                st.success(f"{d['filename']}: {d['pages']} pages, {d['chunks']} chunks" + (" (cached)" if d.get("cached") else ""))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    docs = api("GET", "/documents")
    labels = {f"{d['filename']} · {d['embedding_model']} · {d['doc_id'][:6]}": d["doc_id"] for d in docs}
    selected = st.multiselect("Documents to chat with", list(labels), default=list(labels)[-1:])
    selected_ids = [labels[s] for s in selected]

    if docs and st.button("🗑 Delete selected documents", use_container_width=True):
        for did in selected_ids:
            api("DELETE", f"/documents/{did}")
        st.rerun()

    st.divider()
    if st.button("🧹 New conversation", use_container_width=True):
        api("DELETE", f"/sessions/{st.session_state.session_id}")
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id[:8]}`  (memory: LangGraph MemorySaver)")

tab_chat, tab_eval = st.tabs(["💬 Chat", "🧪 Evals"])

# ---------------------------------------------------------------- chat
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for s in m["sources"]:
                        st.markdown(f"**{s['filename']} — p.{s['page']}** (score {s['score']})")
                        st.caption(s["snippet"] + "…")
            if m.get("flags"):
                st.caption("🛡 guardrails: " + ", ".join(m["flags"]))

    if prompt := st.chat_input("Ask something about your PDF…", disabled=not selected_ids):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    r = api("POST", "/chat", json={
                        "session_id": st.session_state.session_id, "question": prompt,
                        "doc_ids": selected_ids, "generation_model": gen_model,
                    })
                    st.session_state.messages.append({"role": "assistant", "content": r["answer"],
                                                      "sources": r["sources"], "flags": r["flags"]})
                except Exception as e:  # noqa: BLE001
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}"})
        st.rerun()
    if not selected_ids:
        st.info("Upload and select a PDF in the sidebar to start chatting.")

# ---------------------------------------------------------------- evals
with tab_eval:
    st.subheader("Guardrail evals (offline)")
    if st.button("Run guardrail evals"):
        g = api("POST", "/evals/guardrails")
        st.metric("Pass rate", f"{g['passed']}/{g['total']}")
        st.dataframe(pd.DataFrame(g["cases"]), use_container_width=True)

    st.subheader("Answer-quality evals (uses selected documents)")
    st.caption("Upload a JSON list of cases: question, type (answerable|unanswerable), reference, expected_pages, expected_keywords.")
    ds = st.file_uploader("Eval dataset (JSON)", type=["json"], key="ds")
    judge_model = st.selectbox("Judge model", models["generation"], index=0)
    if ds and selected_ids and st.button("Run dataset evals", type="primary"):
        with st.spinner("Running… this calls Bedrock several times per case"):
            try:
                res = api("POST", "/evals/run", json={
                    "doc_ids": selected_ids, "cases": json.load(ds),
                    "generation_model": gen_model, "judge_model": judge_model,
                })
                st.json(res["summary"])
                rows = [{**{k: v for k, v in c.items() if k != "judge"}, **{f"judge_{k}": v for k, v in c.get("judge", {}).items()}}
                        for c in res["cases"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
