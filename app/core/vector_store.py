from pathlib import Path
import numpy as np, faiss
class VectorStore:
 def __init__(self,d):
  p=Path(d)/'semantic.faiss'
  if not p.exists(): raise RuntimeError('Semantic index missing. Run: python -m app.ingest')
  self.index=faiss.read_index(str(p))
 @property
 def size(self): return self.index.ntotal
 @property
 def dimension(self): return self.index.d
 @staticmethod
 def build(x,d):
  d=Path(d); d.mkdir(parents=True,exist_ok=True); x=np.asarray(x,dtype='float32'); faiss.normalize_L2(x); i=faiss.IndexFlatIP(x.shape[1]); i.add(x); faiss.write_index(i,str(d/'semantic.faiss'))
 def search(self,q,k):
  q=np.asarray(q,dtype='float32'); faiss.normalize_L2(q); _,ix=self.index.search(q,min(k,self.index.ntotal)); return [int(i) for i in ix[0] if int(i)>=0]
