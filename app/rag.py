import json, pickle, re
from collections import defaultdict

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import *
from .core.bm25 import tokenize
from .core.vector_store import VectorStore


def label(r):
 return r.get('row_label') or r.get('field') or r.get('section') or ('Table' if r.get('type') == 'table' else 'Report page')


def quote(r):
 if r.get('type') == 'table_row':
  h = r.get('headers') or []
  v = r.get('values') or []
  pairs = [f"{h[i] if i < len(h) and h[i] else f'Column {i+1}'}: {x}" for i, x in enumerate(v) if x]
  s = f"{r.get('table_title', 'Table')} — {label(r)}: " + ' | '.join(pairs)
 else:
  s = r.get('content', '')
 s = re.sub(r'\s+', ' ', s).strip()
 return s if len(s) <= 700 else s[:700].rsplit(' ', 1)[0] + '…'


def ollama_generate(prompt, temperature=0.05, max_tokens=1500):
 z = requests.post(
  f'{OLLAMA_URL}/api/chat',
  json={
   'model': OLLAMA_MODEL,
   'messages': [{'role': 'user', 'content': prompt}],
   'stream': False,
   'options': {'temperature': temperature, 'num_predict': max_tokens},
  },
  timeout=240,
 )
 z.raise_for_status()
 return z.json()['message']['content']


def vertex_generate(prompt, temperature=0.05, max_tokens=1500):
 if not VERTEX_PROJECT_ID:
  raise RuntimeError('VERTEX_PROJECT_ID is not set')
 from google import genai
 from google.genai.types import HttpOptions, GenerateContentConfig

 client = genai.Client(
  vertexai=True,
  project=VERTEX_PROJECT_ID,
  location=VERTEX_LOCATION,
  http_options=HttpOptions(api_version='v1'),
 )
 resp = client.models.generate_content(
  model=VERTEX_MODEL,
  contents=prompt,
  config=GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
 )
 return resp.text


def llm_generate(prompt, temperature=0.05, max_tokens=1500):
 if LLM_BACKEND == 'ollama':
  return ollama_generate(prompt, temperature, max_tokens)
 if LLM_BACKEND == 'vertex':
  return vertex_generate(prompt, temperature, max_tokens)
 raise RuntimeError(f'Unknown LLM_BACKEND: {LLM_BACKEND}')


class RAG:
 def __init__(self):
  rp = INDEX_DIR / 'records.json'
  mp = INDEX_DIR / 'meta.json'
  if not rp.exists() or not mp.exists():
   raise RuntimeError('Index missing. Run: python -m app.download_models && python -m app.ingest')
  self.records = json.loads(rp.read_text(encoding='utf8'))
  m = json.loads(mp.read_text(encoding='utf8'))
  if int(m.get('schema_version', 0)) < 7:
   raise RuntimeError('v7 index required. Run: python -m app.ingest')
  self.vectors = VectorStore(INDEX_DIR)
  with open(INDEX_DIR / 'bm25.pkl', 'rb') as f:
   self.bm25 = pickle.load(f)
  self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE, local_files_only=True)
  self.reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE, local_files_only=True)

 def rrf(self, lists):
  s = defaultdict(float)
  for a in lists:
   for rank, i in enumerate(a):
    s[i] += 1 / (RRF_K + rank + 1)
  return [i for i, _ in sorted(s.items(), key=lambda z: z[1], reverse=True)]

 def retrieve(self, q):
  v = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
  sem = self.vectors.search(v, TOP_K_SEM)
  bm = np.argsort(self.bm25.get_scores(tokenize(q)))[::-1][:TOP_K_BM25].tolist()
  merged = self.rrf([sem, bm])[:RERANK_CANDIDATES]
  if not merged:
   return []
  scores = self.reranker.predict([(q, self.records[i]['content']) for i in merged], show_progress_bar=False)
  ranked = [i for _, i in sorted(zip(scores, merged), key=lambda z: float(z[0]), reverse=True)]
  out = []
  seen = set()
  for i in ranked:
   r = self.records[i]
   k = (r.get('document_id'), r.get('page'), r.get('type'), r.get('field'), r.get('row_label', ''))
   if k in seen:
    continue
   seen.add(k)
   out.append(r)
   if len(out) >= TOP_K_FINAL:
    break
  return out

 def status(self):
  s = {
   'embedding_device': DEVICE,
   'embedding_model': str(EMBED_MODEL),
   'reranker_device': RERANKER_DEVICE,
   'reranker_model': str(RERANKER_MODEL),
   'llm_backend': LLM_BACKEND,
   'deployment_profile': DEPLOYMENT_PROFILE,
   'index_size': self.vectors.size,
   'vector_dimension': self.vectors.dimension,
  }
  if LLM_BACKEND == 'ollama':
   s['ollama_url'] = OLLAMA_URL
   s['ollama_model'] = OLLAMA_MODEL
  elif LLM_BACKEND == 'vertex':
   s['generation_backend'] = 'vertex_ai'
   s['vertex_project'] = VERTEX_PROJECT_ID
   s['vertex_location'] = VERTEX_LOCATION
   s['vertex_model'] = VERTEX_MODEL
  return s

 def answer(self, q):
  ev = self.retrieve(q)
  if not ev:
   return {
    'answer': 'The local knowledge base did not retrieve sufficient evidence to answer this question.',
    'sources': [],
    'mode': 'retrieval-only',
    'retrieval': self.status(),
   }
  blocks = []
  used = 0
  any_low = False
  for j, r in enumerate(ev, 1):
   conf = r.get('layout_confidence', 'high')
   if conf in ('low', 'none', 'medium'):
    any_low = True
   conf_tag = (
    f"\nExtraction confidence: {conf.upper()}"
    + (
     " — do not state exact figures from this evidence as fact; recommend the user verify on the source page image."
     if conf in ('low', 'none')
     else " — figures inferred from page layout, state with light caveat."
     if conf == 'medium'
     else ''
    )
   )
   b = (
    f"[E{j}]\nDocument: {r.get('source_file')}\nDocument type: {r.get('document_type')}\n"
    f"Fiscal year: {r.get('fiscal_year')}\nReporting period: {r.get('reporting_period')}\n"
    f"Page: {r.get('page')}\nEvidence type: {r.get('type')}\nTable: {r.get('table_title', '')}\n"
    f"Exact field: {label(r)}{conf_tag}\nEvidence: {r.get('content', '')}"
   )
   if used + len(b) > MAX_CONTEXT_CHARS:
    break
   blocks.append(b)
   used += len(b)
  hedge_rule = (
   (
    " Some evidence below is tagged with an Extraction confidence level. For LOW/NONE confidence evidence: never "
    "assert a specific figure as fact — explicitly tell the user the exact value could not be reliably extracted and point "
    "them to the cited page to confirm visually. For MEDIUM confidence evidence: state the figure but note it was "
    "positionally inferred and should be spot-checked. Only treat HIGH confidence evidence as safe to quote outright."
   )
   if any_low
   else ''
  )
  prompt = (
   f'''You are PPL Enterprise Intelligence, a private document-grounded assistant. Use ONLY the supplied evidence. Never use web knowledge or memory. If evidence is insufficient, say so. Never invent figures, pages, tables, fields, causes or citations. Every material factual statement must cite [E1], [E2], etc. For financial figures cite exact evidence. For tables identify table and exact field/row. Distinguish reported facts from inference and show formulas for calculations. For multi-year/quarter comparisons explicitly identify periods.{hedge_rule}\n\nQUESTION:\n{q}\n\nEVIDENCE:\n'''
   + '\n\n'.join(blocks)
  )
  err = None
  try:
   text = llm_generate(prompt)
   text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()
   mode = 'llm'
  except Exception as e:
   backend_label = 'Ollama' if LLM_BACKEND == 'ollama' else 'Vertex AI'
   text = f'{backend_label} could not generate synthesis. Retrieved evidence remains available below.'
   mode = 'retrieval-only'
   err = str(e)
  src = [
   {
    'evidence_id': f'E{j}',
    'document_id': r.get('document_id'),
    'source_file': r.get('source_file'),
    'document_name': r.get('document_name'),
    'document_type': r.get('document_type'),
    'fiscal_year': r.get('fiscal_year'),
    'reporting_period': r.get('reporting_period'),
    'page': int(r.get('page', 1)),
    'pdf_page': int(r.get('pdf_page', r.get('page', 1))),
    'type': r.get('type'),
    'section': r.get('section', ''),
    'table': r.get('table_title', ''),
    'field': label(r),
    'row_label': r.get('row_label', ''),
    'confidence': r.get('layout_confidence', 'high'),
    'id': r.get('id', ''),
    'quote': quote(r),
   }
   for j, r in enumerate(ev, 1)
  ]
  out = {'answer': text, 'sources': src, 'mode': mode, 'retrieval': self.status()}
  if err:
   out[f'{LLM_BACKEND}_error' if LLM_BACKEND == 'ollama' else 'vertex_error'] = err
  return out
