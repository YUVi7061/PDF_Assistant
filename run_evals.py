"""CLI eval runner.

  python run_evals.py --guardrails-only
  python run_evals.py --pdf mydoc.pdf --dataset evals/sample_dataset.json \
        --gen-model claude-haiku-4.5 --embed-model titan-v2 --out report.json
"""
import argparse
import hashlib
import json
import sys

from backend import config, evals
from backend.graph import build_graph
from backend.llm_client import LLMClient
from backend.pdf_utils import chunk_pages, extract_pages
from backend.vector_store import VectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails-only", action="store_true")
    ap.add_argument("--pdf")
    ap.add_argument("--dataset", default="evals/sample_dataset.json")
    ap.add_argument("--gen-model", default=config.DEFAULT_GENERATION_MODEL)
    ap.add_argument("--embed-model", default=config.DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    g = evals.run_guardrail_evals()
    print(f"\n== Guardrail evals: {g['passed']}/{g['total']} passed ==")
    for c in g["cases"]:
        if not c["pass"]:
            print("  FAIL:", c)
    report = {"guardrails": g}

    if not args.guardrails_only:
        if not args.pdf:
            sys.exit("--pdf is required unless --guardrails-only")
        bedrock = LLMClient()
        store = VectorStore(config.DATA_DIR / "vectors")
        graph, _ = build_graph(bedrock, store)

        data = open(args.pdf, "rb").read()
        doc_id = hashlib.sha256(data + args.embed_model.encode()).hexdigest()[:16]
        if not store.get(doc_id):
            chunks = chunk_pages(extract_pages(data), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            emb = bedrock.embed([c["text"] for c in chunks], args.embed_model, "document")
            store.add(doc_id, args.pdf, max(c["page"] for c in chunks), args.embed_model, chunks, emb)

        cases = json.load(open(args.dataset, encoding="utf-8"))
        d = evals.run_dataset_eval(graph, bedrock, [doc_id], cases, args.gen_model, args.judge_model)
        report["dataset"] = d
        print("\n== Dataset evals ==")
        for k, v in d["summary"].items():
            print(f"  {k:28s} {v}")

    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
