import json, threading

import fitz
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import *
from .rag import RAG

app = FastAPI(title='PPL Enterprise Intelligence', version='8.0.0')
app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')
_rag = None
_rag_lock = threading.Lock()


class ChatRequest(BaseModel):
 query: str = Field(min_length=1, max_length=10000)


def _mode_string():
 if DEPLOYMENT_PROFILE == 'corporate':
  return f'Corporate (in-boundary) · {LLM_BACKEND}'
 if DEPLOYMENT_PROFILE == 'gcp-trial':
  return f'GCP Cloud Run + Vertex AI (trial)'
 if LLM_BACKEND == 'vertex':
  return 'GCP Cloud Run + Vertex AI'
 return 'Local / Ollama · document-grounded'


def rag():
 global _rag
 if _rag is not None:
  return _rag
 with _rag_lock:
  if _rag is None:
   _rag = RAG()
 return _rag


def pdfs():
 return sorted(RAW_DIR.rglob('*.pdf'))


@app.get('/')
def home():
 return FileResponse(ROOT / 'app' / 'static' / 'index.html')


@app.get('/api/health')
def health():
 i = (INDEX_DIR / 'records.json').exists() and (INDEX_DIR / 'semantic.faiss').exists()
 out = {'ok': i, 'index': i, 'llm_backend': LLM_BACKEND, 'documents': len(pdfs())}
 if LLM_BACKEND == 'ollama':
  try:
   o = requests.get(f'{OLLAMA_URL}/api/tags', timeout=3).ok
  except Exception:
   o = False
  out['ollama'] = o
  out['ollama_model'] = OLLAMA_MODEL
 elif LLM_BACKEND == 'vertex':
  out['vertex_project_configured'] = bool(VERTEX_PROJECT_ID)
  out['vertex_model'] = VERTEX_MODEL
 return out


@app.get('/api/system')
def system():
 try:
  r = rag().status()
 except Exception as e:
  r = {'error': str(e)}
 m = json.loads((INDEX_DIR / 'meta.json').read_text()) if (INDEX_DIR / 'meta.json').exists() else {}
 docs = []
 for p in pdfs():
  try:
   with fitz.open(p) as d:
    n = len(d)
  except Exception:
   n = None
  docs.append({'name': p.name, 'pages': n})
 return {
  'application': 'PPL Enterprise Intelligence',
  'version': '8.0.0',
  'mode': _mode_string(),
  'deployment_profile': DEPLOYMENT_PROFILE,
  'llm_backend': LLM_BACKEND,
  'retrieval': r,
  'index': m,
  'documents': docs,
 }


@app.get('/api/documents')
def documents():
 out = []
 for p in pdfs():
  try:
   with fitz.open(p) as d:
    n = len(d)
  except Exception:
   n = None
  out.append({'name': p.name, 'pages': n})
 return {'documents': out}


@app.get('/api/pdf/page/{page}')
def pdf_page(page: int, document: str | None = None):
 p = RAW_DIR / document if document else (pdfs()[0] if pdfs() else None)
 if page < 1 or p is None:
  raise HTTPException(404, 'PDF/page not found')
 try:
  p.resolve().relative_to(RAW_DIR.resolve())
 except ValueError:
  raise HTTPException(404, 'Document not found')
 if not p.exists():
  raise HTTPException(404, 'Document not found')
 try:
  with fitz.open(p) as d:
   if page > len(d):
    raise HTTPException(404, 'Page outside document')
   return Response(
    d[page - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).tobytes('png'),
    media_type='image/png',
   )
 except HTTPException:
  raise
 except Exception as e:
  raise HTTPException(500, str(e))


@app.post('/api/chat')
def chat(req: ChatRequest):
 try:
  return rag().answer(req.query)
 except Exception as e:
  raise HTTPException(500, f'{type(e).__name__}: {e}')
