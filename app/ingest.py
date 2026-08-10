import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pymupdf

from .config import PDF_PATH, INDEX_DIR, EMBED_MODEL
from sentence_transformers import SentenceTransformer

try:
    import pytesseract
    from PIL import Image
    OCR_OK = True
    TESSERACT = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if Path(TESSERACT).exists():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT
except Exception:
    OCR_OK = False

from .core.bm25 import BM25
from .core.vector_store import VectorStore
INDEX_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_FIELDS = [
    "Consolidated Statement of Profit or Loss",
    "Unconsolidated Statement of Profit or Loss",
    "Consolidated Statement of Financial Position",
    "Unconsolidated Statement of Financial Position",
    "Consolidated Statement of Comprehensive Income",
    "Unconsolidated Statement of Comprehensive Income",
    "Consolidated Statement of Cash Flows",
    "Unconsolidated Statement of Cash Flows",
    "Summary of Statement of Profit or Loss",
    "Financial Performance",
    "Operating Performance / Liquidity",
    "Capital Market / Capital Structure Analysis",
    "Employee Productivity Ratios",
    "Business Overview",
    "Directors' Report",
    "Directors Report",
    "Strategy & Resource Allocation",
    "Risk Management",
    "Corporate Governance",
]

HEADING_PATTERNS = [
    r"CONSOLIDATED STATEMENT OF PROFIT OR LOSS",
    r"UNCONSOLIDATED STATEMENT OF PROFIT OR LOSS",
    r"CONSOLIDATED STATEMENT OF FINANCIAL POSITION",
    r"UNCONSOLIDATED STATEMENT OF FINANCIAL POSITION",
    r"CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
    r"UNCONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME",
    r"CONSOLIDATED STATEMENT OF CASH FLOWS",
    r"UNCONSOLIDATED STATEMENT OF CASH FLOWS",
    r"SUMMARY OF STATEMENT OF PROFIT OR LOSS",
    r"FINANCIAL PERFORMANCE",
    r"OPERATING PERFORMANCE",
    r"CAPITAL MARKET",
    r"EMPLOYEE PRODUCTIVITY",
    r"BUSINESS OVERVIEW",
    r"DIRECTORS'? REPORT",
    r"STRATEGY(?:\s+&\s+| AND )RESOURCE ALLOCATION",
    r"RISK MANAGEMENT",
    r"CORPORATE GOVERNANCE",
]


def clean(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_cell(value):
    return clean(value).replace("|", "/")


def detect_field(text, fallback="Annual Report Narrative"):
    up = clean(text).upper()
    for field in KNOWN_FIELDS:
        if field.upper() in up:
            return field
    if "DIRECTORS" in up and "REPORT" in up:
        return "Directors' Report"
    return fallback


def extract_report_pages(text, pdf_page):
    matches = re.findall(r"Annual Report.?2025\s*(\d+)(?:\s+(\d+))?", text, flags=re.I)
    if matches:
        a, b = matches[-1]
        vals = [int(a)]
        if b:
            vals.append(int(b))
        return vals
    return [pdf_page]


def statement_report_page(statement_field, text, report_pages):
    # The supplied PPL PDF is a two-page spread: one PDF page can contain two
    # printed Annual Report pages. Map a statement to its title order on the spread.
    clean_text = clean(text).lower()
    fields = [
        f for f in [
            "Consolidated Statement of Financial Position",
            "Consolidated Statement of Profit or Loss",
            "Consolidated Statement of Comprehensive Income",
            "Consolidated Statement of Cash Flows",
            "Unconsolidated Statement of Financial Position",
            "Unconsolidated Statement of Profit or Loss",
        ] if clean(f).lower() in clean_text
    ]
    positions = sorted((clean_text.find(clean(f).lower()), f) for f in fields)
    for rank, (_, field) in enumerate(positions):
        if field == statement_field:
            return report_pages[min(rank, len(report_pages) - 1)]
    return report_pages[-1] if report_pages else None


def detect_page_section(text):
    up = clean(text).upper()
    # Prefer the longest/most specific known field occurring on the page.
    candidates = []
    for field in KNOWN_FIELDS:
        pos = up.find(field.upper())
        if pos >= 0:
            candidates.append((pos, len(field), field))
    if candidates:
        return sorted(candidates, key=lambda x: (-x[1], x[0]))[0][2]
    for pat in HEADING_PATTERNS:
        m = re.search(pat, up)
        if m:
            return clean(m.group(0).title())
    return "Annual Report Narrative"


def table_to_rows(rows):
    cleaned = []
    for row in rows:
        vals = [normalize_cell(c) for c in row]
        if any(vals):
            cleaned.append(vals)
    if not cleaned:
        return [], []
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    return cleaned[0], cleaned[1:]


def table_title_from_page(page_text, page_field, rows):
    # Use explicit financial-statement names first; otherwise use the page field.
    detected = detect_field(page_text, "")
    if detected and detected != "Annual Report Narrative":
        return detected
    first = clean(rows[0][0] if rows and rows[0] else "")
    return first[:120] if first else page_field


def row_label(row, headers):
    if not row:
        return "Table row"
    # In annual-report financial tables, the first populated cell is normally the metric label.
    for idx, cell in enumerate(row):
        if cell:
            # Avoid treating a lone note number as the field when a later textual label exists.
            if re.fullmatch(r"\d+(?:\.\d+)?", cell) and idx + 1 < len(row):
                continue
            return cell[:180]
    return "Table row"




def numeric_lines_after_marker(text, marker, after_phrase=None):
    lines = [clean(x) for x in text.splitlines() if clean(x)]
    start_idx = 0
    if after_phrase:
        phrase = clean(after_phrase).lower()
        matches = []
        for i in range(len(lines)):
            window = clean(" ".join(lines[i:i + 4])).lower()
            if phrase in window:
                matches.append(i)
        if matches:
            # The report can place the title on two lines and can contain multiple
            # statements on one PDF page. Select the title occurrence followed by a marker.
            candidates = [i for i in matches if any(marker.lower() == lines[j].lower() for j in range(i + 1, min(i + 120, len(lines))))]
            start_idx = candidates[-1] if candidates else matches[-1]
    try:
        idx = next(i for i in range(start_idx, len(lines)) if lines[i].lower() == marker.lower())
    except StopIteration:
        return []
    out = []
    for line in lines[idx + 1:]:
        if re.fullmatch(r"[-+]?\(?[\d,]+(?:\.\d+)?\)?", line) or line == "-":
            out.append(line)
        elif re.fullmatch(r"[-+]?\(?[\d,]+(?:\.\d+)?\)?\s+[-+]?\(?[\d,]+(?:\.\d+)?\)?", line):
            out.extend(re.findall(r"[-+]?\(?[\d,]+(?:\.\d+)?\)?|\-", line))
        if len(out) >= 80:
            break
    return out

def financial_statement_rows(page, field):
    """Recover exact financial-statement rows using PDF geometry.

    Returns row metadata including the label/value bounding boxes so v5 can
    trace a citation back to the relevant area of the PDF page.
    """
    specs = {
        "Consolidated Statement of Profit or Loss": [
            "Revenue from contracts with customers", "Operating expenses", "Royalties and other levies",
            "Gross profit", "Exploration expenses", "Administrative expenses", "Finance costs",
            "Share of loss of associates - net", "Other charges", "Other income", "Profit before taxation",
            "Taxation", "Profit after taxation", "Basic and diluted earnings per share (Rs.)"
        ],
        "Unconsolidated Statement of Profit or Loss": [
            "Revenue from contracts with customers", "Operating expenses", "Royalties and other levies",
            "Gross profit", "Exploration expenses", "Administrative expenses", "Finance costs",
            "Share of loss of associates - net", "Other charges", "Other income", "Profit before taxation",
            "Taxation", "Profit after taxation", "Basic and diluted earnings per share (Rs.)"
        ],
        "Consolidated Statement of Financial Position": [
            "Property, plant and equipment", "Intangible assets", "Long - term investments", "Long - term loans", "Long - term deposits",
            "Stores and spares", "Trade debts", "Loans and advances", "Trade deposits and short - term prepayments", "Interest accrued",
            "Current maturity of long - term loans", "Current maturity of long - term deposits", "Other receivables", "Short - term investments",
            "Cash and bank balances", "TOTAL ASSETS", "Share capital", "Reserves", "Provision for decommissioning obligation",
            "Long - term financing", "Deferred liabilities", "Deferred taxation - net", "Trade and other payables", "Unclaimed dividends",
            "Current maturity of long - term financing", "Taxation - net", "TOTAL LIABILITIES", "TOTAL EQUITY AND LIABILITIES", "CONTINGENCIES AND COMMITMENTS"
        ],
        "Unconsolidated Statement of Financial Position": [
            "Property, plant and equipment", "Intangible assets", "Long - term investments", "Long - term loans", "Long - term deposits",
            "Stores and spares", "Trade debts", "Loans and advances", "Trade deposits and short - term prepayments", "Interest accrued",
            "Current maturity of long - term loans", "Current maturity of long - term deposits", "Other receivables", "Short - term investments",
            "Cash and bank balances", "TOTAL ASSETS", "Share capital", "Reserves", "Provision for decommissioning obligation",
            "Long - term financing", "Deferred liabilities", "Deferred taxation - net", "Trade and other payables", "Unclaimed dividends",
            "Current maturity of long - term financing", "Taxation - net", "TOTAL LIABILITIES", "TOTAL EQUITY AND LIABILITIES", "CONTINGENCIES AND COMMITMENTS"
        ],
        "Consolidated Statement of Comprehensive Income": [
            "Profit after taxation", "Remeasurement loss on defined benefit plans - net",
            "Exchange differences on translation of subsidiaries & foreign associate (Pakistan International Oil Limited) - net",
            "Share of exchange differences on translation of foreign operation of the associate {Pakistan Minerals (Private) Limited}",
            "Other comprehensive income - loss", "Total comprehensive income for the year"
        ],
        "Consolidated Statement of Cash Flows": [
            "Receipts from customers", "Receipts of other income", "Payment to suppliers / service providers and employees",
            "Payment of indirect taxes and Government levies including royalties", "Income tax paid", "Payment of decommissioning obligation",
            "Finance costs paid", "Long-term loans - net", "Net cash generated from operating activities", "Capital expenditure",
            "Proceeds from disposal of property, plant and equipment", "Acquisition of short - term investments", "Proceeds from sale of short - term investments",
            "Equity investment in PIOL", "Equity investment in PMPL", "Finance income received", "Net cash used in investing activities",
            "Proceeds from long - term financing", "Repayments of long - term financing", "Payment of lease liabilities", "Dividends paid",
            "Net cash used in financing activities", "Net (decrease) / increase in cash and cash equivalents",
            "Cash and cash equivalents at the beginning of the year", "Effect of exchange rate changes on cash and cash equivalents",
            "Cash and cash equivalents at the end of the year"
        ],
    }
    labels = specs.get(field, [])
    if not labels:
        return []

    lines = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            txt = clean("".join(span["text"] for span in line["spans"]))
            if txt:
                lines.append((line["bbox"], txt))

    def norm(s):
        return re.sub(r"[^a-z0-9]+", " ", clean(s).lower()).strip()

    numeric_re = re.compile(r"[-+]?\(?[\d,]+(?:\.\d+)?\)?$")
    rows = []
    for label in labels:
        target = norm(label)
        matches = [(bbox, txt) for bbox, txt in lines if target in norm(txt)]
        if not matches:
            continue
        bbox, _ = matches[0]
        y = (bbox[1] + bbox[3]) / 2
        candidates = []
        # Financial position is the left half of the spread; profit/loss and related
        # statements are on the right half. Restrict the numeric search accordingly.
        if bbox[0] < 350:
            x_ok = lambda x: 350 <= x <= 600
        else:
            x_ok = lambda x: x >= 900
        for vb, vt in lines:
            vy = (vb[1] + vb[3]) / 2
            if abs(vy - y) <= 4.5 and (numeric_re.fullmatch(vt) or vt == "-") and vb[0] > bbox[2] + 20 and x_ok(vb[0]):
                candidates.append((vb, vt))
        candidates = sorted(candidates, key=lambda x: x[0][0])[-2:]
        if len(candidates) < 2:
            continue
        rows.append({
            "label": label,
            "values": [candidates[0][1], candidates[1][1]],
            "headers": ["2025", "2024"],
            "bbox": [float(x) for x in bbox],
            "value_bboxes": [
                [float(x) for x in candidates[0][0]],
                [float(x) for x in candidates[1][0]],
            ],
        })
    return rows

def extract():
    doc = pymupdf.open(PDF_PATH)
    records = []
    for pno, page in enumerate(doc, start=1):
        raw_text = page.get_text("text") or ""
        text = clean(raw_text)
        if len(text) < 80 and OCR_OK:
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = clean(pytesseract.image_to_string(img))
            except Exception as exc:
                print(f"OCR warning on page {pno}: {exc}")

        page_field = detect_page_section(text)
        report_pages = extract_report_pages(text, pno)
        if text:
            # Keep moderately sized overlapping narrative evidence units.
            paras = [clean(x) for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(clean(x)) > 35]
            step = 3
            for i in range(0, len(paras), step):
                chunk = " ".join(paras[i:i + 4])
                if len(chunk) < 120:
                    continue
                records.append({
                    "id": f"p{pno}_n{i}",
                    "page": report_pages[-1] if report_pages else pno,
                    "pdf_page": pno,
                    "report_pages": report_pages,
                    "type": "narrative",
                    "section": page_field,
                    "field": page_field,
                    "content": chunk,
                })

        # Financial statements in this report may expose labels and numeric columns
        # as separate text blocks, so PyMuPDF's geometric table detector can return
        # zero tables. v4 creates explicit row evidence from those known statement layouts.
        statement_fields = [
            f for f in KNOWN_FIELDS
            if f in {
                "Consolidated Statement of Financial Position",
                "Unconsolidated Statement of Financial Position",
                "Consolidated Statement of Profit or Loss",
                "Unconsolidated Statement of Profit or Loss",
                "Consolidated Statement of Comprehensive Income",
                "Consolidated Statement of Cash Flows",
            } and f.upper() in text.upper()
        ]
        for statement_field in statement_fields:
            fallback_rows = financial_statement_rows(page, statement_field)
            for fri, row in enumerate(fallback_rows):
                label, values, headers = row["label"], row["values"], row["headers"]
                content = f"Table: {statement_field}\nField: {label}\n" + " | ".join(f"{headers[i]}: {values[i]}" for i in range(len(values)))
                statement_page = statement_report_page(statement_field, text, report_pages)
                records.append({
                    "id": f"p{pno}_{statement_field.lower().replace(' ', '_')}_row{fri}",
                    "page": statement_page or pno,
                    "pdf_page": pno,
                    "report_pages": report_pages,
                    "type": "table_row",
                    "section": statement_field,
                    "field": label,
                    "table_title": statement_field,
                    "row_label": label,
                    "headers": headers,
                    "values": values,
                    "content": content,
                    "table_index": "statement",
                    "row_index": fri,
                    "bbox": row.get("bbox"),
                    "value_bboxes": row.get("value_bboxes", []),
                })

        try:
            finder = page.find_tables()
            for ti, table in enumerate(finder.tables):
                raw_rows = table.extract()
                headers, data_rows = table_to_rows(raw_rows)
                if not headers and not data_rows:
                    continue
                title = table_title_from_page(text, page_field, raw_rows)

                # Preserve the complete table as one evidence unit for context.
                table_lines = []
                if headers:
                    table_lines.append(" | ".join(headers))
                table_lines.extend(" | ".join(r) for r in data_rows)
                table_content = "\n".join(table_lines)
                if len(table_content) > 40:
                    records.append({
                        "id": f"p{pno}_table{ti}",
                        "page": report_pages[-1] if report_pages else pno,
                        "pdf_page": pno,
                        "report_pages": report_pages,
                        "type": "table",
                        "section": page_field,
                        "field": title,
                        "table_title": title,
                        "headers": headers,
                        "content": table_content,
                        "table_index": ti,
                        "bbox": [float(x) for x in getattr(table, "bbox", (0, 0, 0, 0))],
                    })

                # Row-level evidence is the key v4 change: individual financial metrics
                # become addressable fields instead of being buried in a page-sized table chunk.
                for ri, row in enumerate(data_rows):
                    if not any(row):
                        continue
                    label = row_label(row, headers)
                    pairs = []
                    for ci, value in enumerate(row):
                        if not value:
                            continue
                        header = headers[ci] if ci < len(headers) and headers[ci] else f"Column {ci + 1}"
                        pairs.append(f"{header}: {value}")
                    content = f"Table: {title}\nField: {label}\n" + " | ".join(pairs)
                    records.append({
                        "id": f"p{pno}_table{ti}_row{ri}",
                        "page": report_pages[-1] if report_pages else pno,
                        "pdf_page": pno,
                        "report_pages": report_pages,
                        "type": "table_row",
                        "section": page_field,
                        "field": label,
                        "table_title": title,
                        "row_label": label,
                        "headers": headers,
                        "values": row,
                        "content": content,
                        "table_index": ti,
                        "row_index": ri,
                        "bbox": [float(x) for x in getattr(table, "bbox", (0, 0, 0, 0))],
                    })
        except Exception as exc:
            print(f"Table detection warning on page {pno}: {exc}")

    return records


def main():
    records = extract()
    texts = [r["content"] for r in records]
    print(f"Extracted {len(records)} evidence units from {PDF_PATH.name}")

    model = SentenceTransformer(EMBED_MODEL, device="cuda", local_files_only=True)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    tokens = [re.findall(r"(?u)\b\w+\b", t.lower()) for t in texts]
    bm25 = BM25(tokens)

    with open(INDEX_DIR / "records.json", "w", encoding="utf8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    VectorStore.build(emb, INDEX_DIR)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(INDEX_DIR / "meta.json", "w", encoding="utf8") as f:
        json.dump({
            "schema_version": 5,
            "pdf": PDF_PATH.name,
            "records": len(records),
            "embedding_model": EMBED_MODEL,
            "evidence_types": sorted({r["type"] for r in records}),
            "features": ["table_rows", "field_metadata", "pdf_bboxes", "report_page_mapping"],
        }, f, indent=2)
    print("Index written to", INDEX_DIR)


if __name__ == "__main__":
    main()
