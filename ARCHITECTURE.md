# PPL Corporate Intelligence RAG v5 Architecture

## Design goal

Provide a private, local document-intelligence assistant where a generated answer can be audited back to the exact evidence unit and physical PDF page used by retrieval.

## Evidence hierarchy

```text
Annual Report PDF
    |
    +-- Physical PDF page
         |
         +-- Printed Annual Report page(s)
              |
              +-- Section / narrative
              |
              +-- Table
                   |
                   +-- Table row / exact field
                        |
                        +-- Column header + value
                        |
                        +-- PDF bounding box
```

The row/field layer is the key financial-intelligence abstraction. A metric such as `Profit after taxation` is independently retrievable with its table title, exact field, headers, values and source page.

## Ingestion

1. Open the local PPL Annual Report with PyMuPDF.
2. Extract normal page text.
3. Use Tesseract only when a page has insufficient text and OCR is available.
4. Detect report sections and printed page numbers.
5. Extract native PDF tables when available.
6. Extract known financial-statement rows using PDF geometry when the PDF does not expose them as tables.
7. Store row label, headers, values and bounding boxes.
8. Build BGE-M3 embeddings.
9. Build BM25 exact-term index.
10. Persist schema 5 metadata.

### Financial-statement geometry correction

The geometry extractor accepts the PyMuPDF `page` object because it calls:

```python
page.get_text("dict")
```

The v4 failure came from passing flattened `text` into this function. v5 passes the page object correctly.

## Retrieval

```text
Query
 |
 +--> BGE-M3 semantic search
 |
 +--> BM25 exact-term search
 |
 +--> Reciprocal Rank Fusion
 |
 +--> Candidate pool
 |
 +--> BGE Reranker
 |
 +--> Intent-aware type weighting
 |
 +--> Diversity filtering
 |
 +--> Final evidence
```

### Intent-aware weighting

- `financial_lookup`: table rows and tables receive a positive retrieval bonus.
- `cause`: narrative evidence receives a positive retrieval bonus.
- `risk`: risk-management narrative receives a positive section bonus.
- `general`: balanced retrieval.

This does not replace semantic retrieval; it resolves a common failure mode where a numerically similar table outranks the narrative passage that explicitly explains why a number changed.

## Generation

The local Ollama model receives only the selected report evidence.

The prompt requires:

- Every material factual statement to cite `[E#]`.
- No invented citations.
- No outside knowledge.
- No causal claim unless the report evidence supports it.
- Explicit separation of reported fact, report-stated explanation and calculation/inference.
- An insufficiency statement when the supplied evidence cannot establish the answer.

## Citation object

Each source returned by `/api/chat` contains:

- `evidence_id`
- `page` — printed Annual Report page.
- `pdf_page` — physical PDF page.
- `report_pages` — printed pages on that physical spread.
- `type`
- `section`
- `table`
- `field`
- `row_label`
- `headers`
- `values`
- `quote`
- `bbox`
- `value_bboxes`
- retrieval/reranker scores.

## PDF viewer

The frontend does not rely on mutating the hash of one long-lived iframe. For every citation jump it creates a fresh iframe and loads:

```text
/report?viewer_page=N&t=<timestamp>#page=N
```

The query parameter forces a new document load while the clean PDF fragment tells the browser's native PDF viewer which physical page to display.

The UI separately tracks printed report page and physical PDF page because PPL's report is presented as two-page spreads.

## GPU layout

Default v5:

```text
RTX 5070
├── BGE-M3 query embedding
├── BGE Reranker v2 M3
└── Ollama generation

CPU
├── BM25
├── RRF
├── filtering / orchestration
└── FastAPI
```

If Ollama needs more VRAM, set:

```powershell
$env:RERANKER_DEVICE="cpu"
```

## Locality

- PDF is local.
- Index is local.
- Embedding/reranker models are local.
- Hugging Face offline mode is enabled.
- Transformer loading uses `local_files_only=True`.
- Ollama is local.
- No external LLM API is required.
