import re
from rank_bm25 import BM25Okapi
def tokenize(t): return re.findall(r'(?u)\b\w+\b',(t or '').lower())
def build_bm25(records): return BM25Okapi([tokenize(r.get('content','')) for r in records])
