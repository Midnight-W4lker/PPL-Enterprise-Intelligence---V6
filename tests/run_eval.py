"""
Minimal golden-set eval runner.

Works with either LLM backend (ollama or vertex) — calls RAG directly.
Set LLM_BACKEND and related env vars before running.

Usage:
    python -m tests.run_eval

For each question in golden_qa.json, calls the running RAG pipeline, prints the
answer, the cited pages/confidence levels, and flags mismatches against
expected_page / expected_answer_contains so a human can quickly scan for
regressions after any ingest.py or rag.py change.

This does NOT replace human review — it's a fast tripwire, not a guarantee.
Expand golden_qa.json with more real questions (aim for 30-50) covering:
  - plain narrative facts
  - clean bordered-table lookups
  - known low-confidence pages (should hedge, not assert)
  - negative controls ("is X mentioned?" for topics that ARE in the report)
  - multi-year comparisons once more reports are ingested
"""
import json
from pathlib import Path
from app.rag import RAG

def main():
    qa = json.loads((Path(__file__).parent / 'golden_qa.json').read_text(encoding='utf8'))
    rag = RAG()
    passed = 0
    for i, item in enumerate(qa, 1):
        res = rag.answer(item['question'])
        pages = sorted({s['page'] for s in res['sources']})
        confs = sorted({s.get('confidence', 'high') for s in res['sources']})
        print(f"\n=== Q{i}: {item['question']}")
        print(f"Answer: {res['answer'][:400]}")
        print(f"Cited pages: {pages}  | confidence levels present: {confs}")
        ok = True
        if item.get('expected_page') and item['expected_page'] not in pages:
            print(f"  [CHECK] expected page {item['expected_page']} not among cited pages")
            ok = False
        for phrase in item.get('expected_answer_contains', []):
            if phrase.lower() not in res['answer'].lower():
                print(f"  [CHECK] expected phrase '{phrase}' not found in answer")
                ok = False
        if item.get('notes'):
            print(f"  Note: {item['notes']}")
        passed += ok
    print(f"\n{passed}/{len(qa)} auto-checks passed (manual review still required)")

if __name__ == '__main__':
    main()
