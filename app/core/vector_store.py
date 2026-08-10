from pathlib import Path
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

class VectorStore:
    def __init__(self, index_dir: Path, dim=None):
        self.index_dir=index_dir
        self.faiss_path=index_dir/'semantic.faiss'
        self.npy_path=index_dir/'embeddings.npy'
        self.index=None
        self.emb=None
        if faiss is not None and self.faiss_path.exists():
            self.index=faiss.read_index(str(self.faiss_path))
        elif self.npy_path.exists():
            self.emb=np.load(self.npy_path).astype('float32')

    @staticmethod
    def build(embeddings, index_dir):
        index_dir.mkdir(parents=True,exist_ok=True)
        embeddings=np.asarray(embeddings,dtype='float32')
        np.save(index_dir/'embeddings.npy',embeddings)
        if faiss is not None:
            idx=faiss.IndexFlatIP(embeddings.shape[1]); idx.add(embeddings)
            faiss.write_index(idx,str(index_dir/'semantic.faiss'))

    def search(self,q,k):
        q=np.asarray(q,dtype='float32')
        if self.index is not None:
            _, ids=self.index.search(q,k); return ids[0].tolist()
        scores=self.emb @ q[0]
        return np.argsort(scores)[::-1][:k].tolist()
