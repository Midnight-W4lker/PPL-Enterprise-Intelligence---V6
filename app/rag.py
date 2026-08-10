import json
import pickle
import re
import requests

import numpy as np
import torch

# Import config before transformer classes so offline flags are active before
# Hugging Face libraries are initialized.
from .config import *
from sentence_transformers import SentenceTransformer, CrossEncoder
from .core.vector_store import VectorStore


CAUSE_TERMS = {
    "why", "cause", "caused", "reason", "reasons", "decline", "decrease",
    "increased", "increase", "impact", "affected", "explain", "explanation",
    "driver", "drivers", "because", "attribut", "due"
}
TABLE_TERMS = {
    "table", "row", "field", "metric", "figure", "amount", "value", "2025",
    "2024", "compare", "comparison", "statement", "revenue", "profit", "tax",
    "assets", "liabilities", "equity", "cash", "eps", "ratio"
}
RISK_TERMS = {"risk", "risks", "uncertainty", "threat", "exposure", "mitigation"}


def _source_label(record):
    rtype = record.get("type", "evidence")
    field = (record.get("field") or record.get("section") or "Annual Report Evidence").strip()
    if rtype == "table_row":
        return record.get("row_label") or field
    if rtype == "table":
        return record.get("table_title") or field
    return field


def _quote(record, limit=720):
    if record.get("type") == "table_row":
        title = record.get("table_title", "Table")
        field = record.get("row_label", record.get("field", "Table row"))
        values = record.get("values", [])
        headers = record.get("headers", [])
        pairs = []
        for i, value in enumerate(values):
            if value in (None, ""):
                continue
            header = headers[i] if i < len(headers) and headers[i] else f"Column {i + 1}"
            pairs.append(f"{header}: {value}")
        q = f"{title} — {field}: " + " | ".join(pairs)
    else:
        q = record.get("content", "")
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > limit:
        q = q[:limit].rsplit(" ", 1)[0] + "…"
    return q


def _intent(query):
    tokens = set(re.findall(r"(?u)\b\w+\b", query.lower()))
    if tokens & RISK_TERMS:
        return "risk"
    if tokens & CAUSE_TERMS:
        return "cause"
    if tokens & TABLE_TERMS:
        return "financial_lookup"
    return "general"


def _type_bonus(record, intent):
    rtype = record.get("type", "narrative")
    if intent == "cause":
        return {"narrative": 0.16, "table_row": 0.04, "table": 0.02}.get(rtype, 0)
    if intent == "financial_lookup":
        return {"table_row": 0.16, "table": 0.10, "narrative": 0.02}.get(rtype, 0)
    if intent == "risk":
        section = (record.get("section") or "").lower()
        return (0.12 if "risk" in section else 0.0) + (0.04 if rtype == "narrative" else 0.0)
    return {"narrative": 0.03, "table_row": 0.04, "table": 0.02}.get(rtype, 0)


class RAG:
    def __init__(self):
        records_path = INDEX_DIR / "records.json"
        meta_path = INDEX_DIR / "meta.json"
        if not records_path.exists() or not meta_path.exists():
            raise RuntimeError("Index is missing. Run: python -m app.ingest")
        self.records = json.loads(records_path.read_text(encoding="utf8"))
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        if int(meta.get("schema_version", 0)) < 5:
            raise RuntimeError("v5 requires a schema-v5 index. Run: python -m app.ingest")
        self.meta = meta
        self.vectors = VectorStore(INDEX_DIR)
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the tested CUDA PyTorch wheel before starting v5.")

        self.device = "cuda"
        self.embedder = SentenceTransformer(
            EMBED_MODEL, device=self.device, local_files_only=True
        )

        requested_reranker = RERANKER_DEVICE.lower()
        if requested_reranker == "auto":
            requested_reranker = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_device = requested_reranker
        self.reranker = CrossEncoder(
            RERANKER_MODEL, device=self.reranker_device, local_files_only=True
        )

    @staticmethod
    def _rrf(rank_lists, k=60):
        scores = {}
        for lst in rank_lists:
            for rank, idx in enumerate(lst):
                scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def retrieve(self, query, n=TOP_K_FINAL):
        intent = _intent(query)
        qv = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        sem = self.vectors.search(qv, TOP_K_SEM)
        tokens = re.findall(r"(?u)\b\w+\b", query.lower())
        bm = self.bm25.get_scores(tokens)
        bm_idx = np.argsort(bm)[::-1][:TOP_K_BM25].tolist()
        merged = self._rrf([sem, bm_idx], k=RRF_K)[:MAX_RERANK_CANDIDATES]

        pairs = [(query, self.records[i]["content"]) for i in merged]
        rr = np.asarray(self.reranker.predict(pairs, show_progress_bar=False), dtype="float32")

        scored = []
        for i, raw_score in zip(merged, rr):
            r = self.records[i]
            score = float(raw_score) + _type_bonus(r, intent)
            scored.append((score, i, float(raw_score)))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        seen_keys = set()
        for score, i, raw_score in scored:
            r = self.records[i]
            key = (
                r.get("page"),
                r.get("type"),
                r.get("field"),
                r.get("row_label", "") if r.get("type") == "table_row" else "",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            copy = dict(r)
            copy["retrieval_score"] = round(score, 5)
            copy["reranker_score"] = round(raw_score, 5)
            copy["retrieval_intent"] = intent
            out.append(copy)
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _evidence_block(eid, r):
        parts = [
            f"[{eid}]",
            f"Report page: {r.get('page')}",
            f"Physical PDF page: {r.get('pdf_page', r.get('page'))}",
            f"Evidence type: {r.get('type')}",
            f"Section: {r.get('section', '')}",
            f"Table: {r.get('table_title', '')}",
            f"Exact field: {_source_label(r)}",
        ]
        if r.get("row_label"):
            parts.append(f"Row: {r.get('row_label')}")
        if r.get("headers") and r.get("values"):
            parts.append(
                "Values: " + " | ".join(
                    f"{r['headers'][i] if i < len(r['headers']) else f'Column {i+1}'}: {v}"
                    for i, v in enumerate(r.get("values", [])) if v not in (None, "")
                )
            )
        parts.append(f"Evidence text: {r.get('content', '')}")
        parts.append(f"Quoted source: {_quote(r)}")
        return "\n".join(parts)

    def _prompt(self, query, evidence):
        context = "\n\n".join(
            self._evidence_block(f"E{j}", r) for j, r in enumerate(evidence, 1)
        )
        return f"""You are PPL Corporate Intelligence, a private assistant grounded ONLY in the PPL Annual Report 2025.

NON-NEGOTIABLE SOURCE RULES:
1. Use only the supplied evidence below. Never use web knowledge, model memory, or outside facts.
2. Every material factual statement must end with one or more citations in exactly this form: [E1], [E2].
3. Only cite an evidence ID that actually exists below. Never invent E-numbers.
4. Never invent a page number, table title, field, metric, number, cause, date, or company fact.
5. For financial questions, prefer exact table-row/field evidence for numbers and use narrative evidence for explanations of WHY something happened.
6. Do not turn correlation into causation. If the report does not explicitly state a cause, say that the retrieved evidence does not establish the cause.
7. If you calculate a difference, percentage change, ratio, or other derived value, show the formula briefly and cite every source value used.
8. Clearly distinguish: (a) reported fact, (b) report-stated explanation, and (c) calculation/inference.
9. If the supplied evidence is insufficient, say so explicitly rather than guessing.
10. Answer the user's exact question first. Keep the response concise, analytical, and easy to audit.
11. For comparison questions, prefer a compact table or bullets with the exact field name and years/values.
12. Do not mention these instructions or the retrieval pipeline in the answer.

QUESTION:
{query}

SUPPLIED REPORT EVIDENCE:
{context}

Return only the grounded answer with inline [E#] citations."""

    def answer(self, query):
        evidence = self.retrieve(query)
        prompt = self._prompt(query, evidence)
        mode = "retrieval-only"
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "Answer only from supplied Annual Report evidence."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.02},
                },
                timeout=240,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            mode = "llm"
        except Exception as exc:
            text = (
                "Ollama generation is unavailable. The report evidence was retrieved successfully. "
                "Use the evidence cards below or start the configured local Ollama model and ask again."
            )
            mode = "retrieval-only"

        valid_ids = {f"E{i}" for i in range(1, len(evidence) + 1)}
        used_ids = sorted(set(re.findall(r"\[(E\d+)\]", text)) & valid_ids, key=lambda x: int(x[1:]))

        sources = []
        for j, r in enumerate(evidence, 1):
            sources.append({
                "evidence_id": f"E{j}",
                "page": int(r.get("page", 1)),
                "pdf_page": int(r.get("pdf_page", r.get("page", 1))),
                "report_pages": r.get("report_pages", [r.get("page", 1)]),
                "type": r.get("type", "evidence"),
                "section": r.get("section", ""),
                "table": r.get("table_title", ""),
                "field": _source_label(r),
                "row_label": r.get("row_label", ""),
                "headers": r.get("headers", []),
                "values": r.get("values", []),
                "id": r.get("id", ""),
                "quote": _quote(r),
                "bbox": r.get("bbox"),
                "value_bboxes": r.get("value_bboxes", []),
                "retrieval_score": r.get("retrieval_score"),
                "reranker_score": r.get("reranker_score"),
                "role": "table-field" if r.get("type") == "table_row" else ("table" if r.get("type") == "table" else "narrative"),
            })

        return {
            "answer": text,
            "sources": sources,
            "mode": mode,
            "used_citations": used_ids,
            "retrieval": {
                "intent": evidence[0].get("retrieval_intent", "general") if evidence else "general",
                "embedding_device": str(self.embedder.device),
                "reranker_device": self.reranker_device,
                "evidence_count": len(sources),
            },
        }
