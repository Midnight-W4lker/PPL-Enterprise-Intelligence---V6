import math, re
from collections import Counter
import numpy as np

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs=docs; self.k1=k1; self.b=b
        self.tf=[Counter(d) for d in docs]
        self.df=Counter()
        for d in docs: self.df.update(set(d))
        self.n=len(docs); self.avgdl=sum(len(d) for d in docs)/max(1,self.n)
    def get_scores(self, query):
        scores=np.zeros(self.n,dtype='float32'); q=set(query)
        for i,d in enumerate(self.docs):
            dl=len(d)
            for term in q:
                if term not in self.tf[i]: continue
                df=self.df.get(term,0)
                idf=math.log(1+(self.n-df+0.5)/(df+0.5))
                tf=self.tf[i][term]
                scores[i]+=idf*(tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*dl/max(1,self.avgdl)))
        return scores
