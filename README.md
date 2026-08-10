# PPL Corporate Intelligence RAG v5

A local-first corporate intelligence assistant for the **Pakistan Petroleum Limited (PPL) Annual Report 2025**.

v5 turns the v4 prototype into a more auditable document-intelligence application: retrieval is intent-aware, financial fields are first-class evidence, source cards expose exact values and quotes, citations are clickable, the embedded PDF viewer is rebuilt for every page jump, and the API exposes real health diagnostics.

## What changed in v5

### Retrieval and answer grounding

- **Intent-aware retrieval** for financial lookup, cause/explanation, risk and general questions.
- BGE-M3 dense retrieval + BM25 exact-term retrieval + RRF fusion + BGE Reranker.
- Larger candidate pool before reranking.
- Financial/table-row evidence receives a boost for exact-value questions.
- Narrative evidence receives a boost for cause/explanation questions.
- Stronger prompt rules prevent the model from turning correlation into causation.
- The model must distinguish reported facts, report-stated explanations and calculations/inferences.
- If the supplied evidence is insufficient, the answer must say so rather than guessing.

### Exact financial evidence

Financial rows now expose:

- Table title.
- Exact field / row label.
- Column headers.
- Values for each column.
- Printed Annual Report page.
- Physical PDF page.
- PDF bounding box when available.
- Exact quoted evidence.

### PDF viewer

- Citations such as `[E1]` are clickable.
- Evidence cards have a **View source** button.
- The viewer displays both `Report p.X` and `PDF p.Y`.
- Page navigation operates on the **physical PDF page**, not the printed report page.
- The PDF iframe is recreated for each jump and receives a clean `#page=N` fragment, avoiding the v4 stale-page problem.
- Manual PDF-page navigation is available.
- `Open PDF` remains available as a fallback.

### Diagnostics

The System Information panel and `/api/status` expose:

- Index/schema state.
- Evidence-unit count.
- PDF availability.
- Embedding model/device.
- Reranker model/device.
- Ollama availability.
- Configured Ollama model availability.
- Hugging Face offline state.

### GPU usage

v4 used CUDA for BGE-M3 but deliberately ran the reranker on CPU, which could push CPU utilization close to 100%.

v5 defaults the reranker to CUDA as well:

```text
BGE-M3 query embedding       -> RTX 5070
BGE Reranker v2 M3            -> RTX 5070
Qwen via Ollama               -> RTX 5070
BM25 / RRF / orchestration    -> CPU
```

Your RTX 5070 has 12 GB VRAM. If Ollama becomes VRAM-constrained or begins unloading/reloading, switch the reranker back to CPU:

```powershell
$env:RERANKER_DEVICE="cpu"
```

The default is:

```text
RERANKER_DEVICE=cuda
```

## Architecture

```text
                    PPL Annual Report 2025
                              |
                     PDF + OCR + extraction
                              |
             +----------------+----------------+
             |                                 |
         Narrative                         Tables
             |                                 |
             |                      +----------+----------+
             |                      |                     |
             |                  Full table            Table row
             |                                            |
             +----------------------+---------------------+
                                    |
                           Local evidence index
                                    |
                    +---------------+---------------+
                    |                               |
                  BGE-M3                          BM25
                 RTX 5070                         CPU
                    |                               |
                    +---------------+---------------+
                                    |
                                  RRF
                                    |
                          Candidate reranking
                              BGE Reranker
                               RTX 5070
                                    |
                         Intent-aware evidence
                                    |
                              Local Ollama
                                    |
                         Grounded answer [E#]
                                    |
                 +------------------+------------------+
                 |                                     |
          Exact evidence cards                 Embedded PDF
       field/table/value/quote/page          cited physical page
```

## Important: v5 requires re-ingestion

v5 changes the evidence schema from v4 to **schema 5**. The new index stores PDF coordinate metadata and enhanced field information.

Run this once after installing v5:

```powershell
python -m app.ingest
```

Expected output ends with something similar to:

```text
Extracted ... evidence units from PPL AR 2025.pdf
...
Index written to ...\data\index
```

Then check:

```powershell
Get-Content .\data\index\meta.json
```

and confirm:

```json
"schema_version": 5
```

### Important correction carried into v5

The financial-statement geometry function must receive the **PyMuPDF page object**, not the flattened page text. v5 uses:

```python
financial_statement_rows(page, statement_field)
```

not:

```python
financial_statement_rows(text, statement_field)
```

This avoids the v4 error:

```text
AttributeError: 'str' object has no attribute 'get_text'
```

## Local models / Hugging Face downloading

Place the already-downloaded models here:

```text
models/
├── bge-m3/
└── bge-reranker-v2-m3/
```

`app/config.py` enables:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

and the application loads transformer models using:

```python
local_files_only=True
```

Therefore v5 does **not** intentionally download the embedding or reranker models from Hugging Face when the local model directories are complete.

The configuration is imported before `sentence_transformers` in both ingestion and retrieval so the offline flags are active before the transformer library initializes.

## CUDA setup

Your tested configuration is:

```text
PyTorch 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5070
```

Verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

If CUDA is unavailable, reinstall the tested CUDA wheel in the active `.venv`.

## Ollama

Check:

```powershell
ollama list
```

The default v5 model is:

```text
qwen2.5:7b
```

The default endpoint is:

```text
http://127.0.0.1:11434
```

Test the API:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:11434/api/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"qwen2.5:7b","prompt":"Say hello","stream":false}'
```

If `ollama serve` reports:

```text
bind: Only one usage of each socket address...
```

do **not** start another server. It means an Ollama server is already listening on port 11434.

## Windows setup

From the project root:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Then verify CUDA and Ollama, ingest the report, and start the API.

## Start v5

```powershell
python -m app.ingest
```

then:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Using `python -m uvicorn` is intentional. It avoids the Windows virtual-environment launcher problem where an old `uvicorn.exe` can still reference a previous v3 `.venv` path.

Open:

```text
http://127.0.0.1:8000
```

## Health checks

Browser/API:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/status
http://127.0.0.1:8000/api/about
```

The status endpoint checks:

- API loaded.
- v5 index loaded.
- PDF exists.
- Embedding device.
- Reranker device.
- Ollama reachable.
- Configured Ollama model available.

## What the system can answer

Questions should be answerable when the requested information is present in the indexed Annual Report, including:

- Financial statement figures.
- Year-over-year comparisons.
- Profitability and revenue changes.
- Financial ratios.
- Cash flows.
- Assets, liabilities and equity.
- Business and operating performance.
- Strategy and resource allocation.
- Risk management.
- Corporate governance.
- Directors' report and management commentary.
- Exact table/row/field lookups.

It is **not** a general web-search assistant. If a fact is outside the supplied Annual Report evidence, the correct behavior is to say that the report evidence is insufficient.

## Recommended tests

### Exact financial field

```text
What was revenue from contracts with customers in 2025 and 2024?
```

Expected behavior:

- Exact table-row evidence.
- Exact field name.
- 2025 and 2024 values.
- Printed report page.
- Physical PDF page.
- Clickable citation.

### Cause / explanation

```text
Why did profit after tax decline?
```

Expected behavior:

- Narrative explanation is prioritized.
- Financial figures support the explanation.
- Correlation is not presented as a report-stated cause.

### Risk

```text
What were the major financial risks?
```

Expected behavior:

- Risk-related narrative receives a retrieval boost.
- Evidence is traceable to the report.

### Unsupported question

```text
What was PPL's share price on a date not covered by the report?
```

Expected behavior:

- The assistant should not invent an answer.
- It should state that the supplied report evidence is insufficient.

## Files

```text
ppl_rag_v5/
├── app/
│   ├── config.py
│   ├── ingest.py
│   ├── main.py
│   ├── rag.py
│   ├── core/
│   └── static/
├── data/
│   ├── raw/
│   │   └── PPL AR 2025.pdf
│   └── index/
├── models/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
└── run_windows.bat
```
#   P P L - E n t e r p r i s e - I n t e l l i g e n c e - - - V 6  
 #   P P L - E n t e r p r i s e - I n t e l l i g e n c e - - - V 6  
 