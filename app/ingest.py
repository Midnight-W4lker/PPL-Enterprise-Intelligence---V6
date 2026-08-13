import hashlib,json,pickle,re
from pathlib import Path
import fitz
from sentence_transformers import SentenceTransformer
from .config import *
from .core.bm25 import build_bm25
from .core.vector_store import VectorStore

SCHEMA_VERSION=7

def clean(x): return re.sub(r'\s+',' ',x or '').strip()

NUMERIC_RE=re.compile(r'^[\d,()\.\-\u2013\u2014%\s]+$')
def is_numeric(s):
 s=(s or '').strip()
 return s=='' or bool(NUMERIC_RE.match(s))

def load_manifest(raw_dir):
 p=Path(raw_dir)/'manifest.json'
 if not p.exists(): return {}
 try: return json.loads(p.read_text(encoding='utf8'))
 except Exception: return {}

def meta(path,manifest=None):
 manifest=manifest or {}
 ov=manifest.get(path.name,{})
 s=path.stem; lo=s.lower(); ym=re.search(r'(20\d{2})',lo); qm=re.search(r'\bq([1-4])\b',lo)
 y=ov.get('fiscal_year') or (int(ym.group(1)) if ym else None)
 q=ov.get('reporting_period') or (f'Q{qm.group(1)}' if qm else None)
 typ=ov.get('document_type') or ('annual_report' if ('annual' in lo or re.search(r'\bar\b',lo)) else ('quarterly_report' if q else ('financial_statement' if 'financial statement' in lo else ('investor_presentation' if 'presentation' in lo or 'investor' in lo else 'report'))))
 if ov.get('fiscal_year') or ov.get('reporting_period'): pass
 return {'document_id':hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16],'source_file':path.name,'document_name':s,'document_type':typ,'fiscal_year':y,'reporting_period':q or (f'FY{y}' if y else None)}

def page_lines(page):
 """All text lines on a page as (bbox, text), in the order PyMuPDF's layout engine emits them."""
 d=page.get_text('dict'); out=[]
 for block in d.get('blocks',[]):
  for line in block.get('lines',[]) or []:
   text=''.join(sp.get('text','') for sp in line.get('spans',[])).strip()
   if text: out.append((line['bbox'],text))
 return out

def x_overlaps(a,b): return not (a[2]<b[0] or a[0]>b[2])

def find_caption(lines,table_bbox,max_gap=28):
 """Text line directly above the table (same x-range, small y-gap) that isn't itself numeric —
 usually the table caption or a unit annotation like 'Rupees in thousand'."""
 tx0,ty0,tx1,ty1=table_bbox; best=None; best_dist=1e9
 for bbox,text in lines:
  bx0,by0,bx1,by1=bbox
  if by1<=ty0+1 and (ty0-by1)<max_gap and x_overlaps(bbox,table_bbox) and not is_numeric(text):
   dist=ty0-by1
   if dist<best_dist: best_dist=dist; best=text
 return best

def backfill_label(lines,table_bbox,row_bbox,y_tol=18):
 """When a table row's own cells are all numeric (label lives in a separate text block to the
 left, common in borderless financial-analysis layouts), find the nearest non-numeric line at the
 same vertical position outside the table's x-range and use it as the row label."""
 if not row_bbox: return None
 tx0=table_bbox[0]; ry0,ry1=row_bbox[1],row_bbox[3]; ycenter=(ry0+ry1)/2
 # Only trust short, label-like lines (e.g. "Trade debts") — long lines are almost always
 # wrapped narrative prose that happens to sit at the same height, not a real row label.
 candidates=[(bbox,txt) for bbox,txt in lines if bbox[2]<=tx0+2 and not is_numeric(txt) and len(txt.split())<=8 and not txt.rstrip().endswith(('.',',',';'))]
 if not candidates: return None
 best=min(candidates,key=lambda c:abs((c[0][1]+c[0][3])/2-ycenter))
 dist=abs((best[0][1]+best[0][3])/2-ycenter)
 return best[1] if dist<=y_tol else None

def chunk_words(text,size=180,overlap=30):
 words=text.split()
 if len(words)<=size: return [text] if words else []
 chunks=[]; i=0
 while i<len(words):
  chunks.append(' '.join(words[i:i+size]))
  if i+size>=len(words): break
  i+=max(1,size-overlap)
 return chunks

def try_ocr(page):
 try:
  import pytesseract
  from PIL import Image
  import io
  pix=page.get_pixmap(matrix=fitz.Matrix(2.2,2.2))
  img=Image.open(io.BytesIO(pix.tobytes('png')))
  return pytesseract.image_to_string(img)
 except Exception:
  return None

def extract_pdf(path,manifest=None):
 m=meta(path,manifest); out=[]
 with fitz.open(path) as doc:
  for pno,page in enumerate(doc,1):
   lines=page_lines(page)
   try: tables=page.find_tables().tables
   except Exception: tables=[]
   table_bboxes=[t.bbox for t in tables]

   # ---------- narrative text (everything not inside a detected table region) ----------
   narrative_lines=[txt for bbox,txt in lines if not any(
    bbox[1]>=tb[1]-2 and bbox[3]<=tb[3]+2 and x_overlaps(bbox,tb) for tb in table_bboxes)]
   page_text=clean(' '.join(narrative_lines))
   has_real_text=len(page_text.split())>=8

   if has_real_text:
    for ci,chunk in enumerate(chunk_words(page_text)):
     out.append({**m,'id':f"{m['document_id']}:p{pno}:n{ci}",'page':pno,'pdf_page':pno,
      'type':'narrative','section':'','table_title':'','field':'','row_label':'','headers':[],'values':[],
      'layout_confidence':'high','content':chunk})
   else:
    imgs=page.get_images()
    if len(page_text.strip())<5 and not imgs:
     pass  # genuinely blank page — nothing to index, nothing to falsely claim is "missing"
    else:
     ocr_text=try_ocr(page) if imgs else None
     if ocr_text and len(ocr_text.split())>=8:
      for ci,chunk in enumerate(chunk_words(clean(ocr_text))):
       out.append({**m,'id':f"{m['document_id']}:p{pno}:ocr{ci}",'page':pno,'pdf_page':pno,
        'type':'narrative_ocr','section':'','table_title':'','field':'','row_label':'','headers':[],'values':[],
        'layout_confidence':'medium','content':chunk})
     else:
      out.append({**m,'id':f"{m['document_id']}:p{pno}:img",'page':pno,'pdf_page':pno,
       'type':'unindexed_image_page','section':'','table_title':'','field':'','row_label':'','headers':[],'values':[],
       'layout_confidence':'none',
       'content':f"Page {pno} of {m['document_name']} is an image/graphic page with little or no extractable "
                 f"text (e.g. a photo, divider, or design element). Its content is not indexed for text search; "
                 f"do not assume it is blank or that it confirms the absence of information."})

   # ---------- tables ----------
   for ti,t in enumerate(tables,1):
    try: rows=t.extract()
    except Exception: continue
    if not rows: continue
    heads=[clean(x) for x in rows[0]]
    caption=find_caption(lines,t.bbox)
    title=f'Table on page {pno}'+(f' ({caption})' if caption else '')

    # Pathological case: PyMuPDF merged multiple stacked numeric lines into single cells.
    # Row/column alignment here cannot be trusted — do not fabricate row labels for it.
    compressed=any((cell or '').count('\n')>=2 for row in rows for cell in row)
    if compressed:
     raw=' || '.join(clean(c) for row in rows for c in row if c)
     out.append({**m,'id':f"{m['document_id']}:p{pno}:t{ti}:lowconf",'page':pno,'pdf_page':pno,
      'type':'table_low_confidence','section':'','table_title':title,'field':'','row_label':'',
      'headers':heads,'values':[],'layout_confidence':'low',
      'content':clean(f"{title} — LAYOUT WARNING: this table uses a multi-column numeric layout (common in "
        f"vertical/horizontal analysis pages) that could not be reliably parsed into row/column pairs by "
        f"automated extraction. Raw extracted values, order not guaranteed to match row labels: {raw}. "
        f"Do NOT state a specific figure from this evidence as fact. Instead tell the user which page to "
        f"check (page {pno}) and that the exact value should be visually confirmed there.")})
     continue

    for ri,row in enumerate(rows[1:],1):
     vals=[clean(x) for x in row]
     non_numeric=[v for v in vals if v and not is_numeric(v)]
     label=non_numeric[0] if non_numeric else None
     conf='high'
     if not label:
      row_bbox=t.rows[ri].bbox if ri<len(t.rows) else None
      label=backfill_label(lines,t.bbox,row_bbox)
      conf='medium' if label else 'low'
      label=label or 'Table row (label not confidently identified — verify on source page image)'
     pairs=[]
     for i,v in enumerate(vals):
      if v: pairs.append(f"{heads[i] if i<len(heads) and heads[i] else f'Column {i+1}'}: {v}")
     content=f'{title} — {label}: '+' | '.join(pairs)
     if caption: content=f'[{caption}] '+content
     if conf!='high':
      content+=' (NOTE: row label inferred from page position, not guaranteed accurate — verify against the source page image before quoting a figure.)'
     out.append({**m,'id':f"{m['document_id']}:p{pno}:t{ti}:r{ri}",'page':pno,'pdf_page':pno,
      'type':'table_row','section':'','table_title':title,'field':label,'row_label':label,
      'headers':heads,'values':vals,'layout_confidence':conf,'content':clean(content)})
 return out

def main():
 paths=sorted(RAW_DIR.rglob('*.pdf'))
 if not paths: raise FileNotFoundError(f'No PDFs in {RAW_DIR}')
 manifest=load_manifest(RAW_DIR)
 records=[]
 for p in paths: records+=extract_pdf(p,manifest)
 low=sum(1 for r in records if r.get('layout_confidence') in ('low','none'))
 med=sum(1 for r in records if r.get('layout_confidence')=='medium')
 print(f'Extracted {len(records)} evidence units from {len(paths)} PDF(s)  '
       f'[{low} low/no-confidence, {med} medium-confidence — these are hedged, not discarded]')
 model=SentenceTransformer(EMBED_MODEL,device='cuda',local_files_only=True)
 x=model.encode([r['content'] for r in records],normalize_embeddings=True,batch_size=16,show_progress_bar=True,convert_to_numpy=True).astype('float32')
 VectorStore.build(x,INDEX_DIR)
 with open(INDEX_DIR/'bm25.pkl','wb') as f: pickle.dump(build_bm25(records),f)
 (INDEX_DIR/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 (INDEX_DIR/'meta.json').write_text(json.dumps({'schema_version':SCHEMA_VERSION,'record_count':len(records),'embedding_model':str(EMBED_MODEL),'embedding_dimension':int(x.shape[1]),'documents':sorted({r['document_id'] for r in records})},indent=2),encoding='utf8')
 print(f'Index written to {INDEX_DIR}')
if __name__=='__main__': main()
