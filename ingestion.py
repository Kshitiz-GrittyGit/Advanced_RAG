from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import Counter
from datetime import datetime
import json
import os
import re
import psycopg2
import pdfplumber
import multiprocessing
import traceback

base_dir = Path(__file__).resolve().parent

DOCUMENTS_DIR = base_dir / 'documents'
OUTPUT_DIR    = base_dir / 'output'

DB_CONFIG = dict(
    host     = "localhost",
    database = "pdf_tables",
    user     = os.getenv("DB_USER", os.getenv("USER", "postgres")),
    password = os.getenv("DB_PASSWORD", ""),
)

CHUNK_SIZE    = 1500  # characters
CHUNK_OVERLAP = 200   # characters
N_WORKERS     = max(1, multiprocessing.cpu_count() - 2)

# Heading classification: font size ratio relative to body text
HEADING_L1_RATIO = 1.4   # 40%+ larger = major heading
HEADING_L2_RATIO = 1.1   # 10%+ larger = sub-heading


#############################################
# 1. FONT-SIZE ANALYSIS
#    Replaces partition_text heading detection.
#    Body text = most common font size in the document.
#    Anything larger is a heading, with level based on how much larger.
#############################################

def compute_dominant_font_size(pdf, sample_pages=30):
    """Most common font size across the document = body text size."""
    sizes = []
    for page in pdf.pages[:sample_pages]:
        for c in page.chars:
            if c.get("size"):
                sizes.append(round(c["size"], 1))
    if not sizes:
        return 12.0
    return Counter(sizes).most_common(1)[0][0]


def classify_heading(text, font_size, dominant_size, font_name=None):
    """
    Returns 1 (major), 2 (sub), or None (body) based on font size vs body text.
    Also checks bold fontname and ALL CAPS as heading signals at body font size.
    """
    text = text.strip()
    if not text or text.endswith(('.', ',', ';')):
        return None
    if len(text.split()) > 15:
        return None

    ratio = font_size / dominant_size if dominant_size > 0 and font_size > 0 else 1.0
    is_bold = font_name and ("Bold" in font_name or "bold" in font_name)

    if ratio >= HEADING_L1_RATIO:
        return 1
    if ratio >= HEADING_L2_RATIO:
        return 2
    if is_bold and len(text.split()) <= 10:
        return 2
    if text.isupper() and len(text.split()) <= 8:
        return 1
    return None


def group_chars_to_lines(chars, page_num):
    """Group pdfplumber chars into text lines by y-proximity (~3pt buckets)."""
    if not chars:
        return []

    sorted_chars = sorted(chars, key=lambda c: (round(c["top"] / 3) * 3, c["x0"]))
    lines       = []
    current_row = [sorted_chars[0]]
    current_y   = round(sorted_chars[0]["top"] / 3) * 3

    for c in sorted_chars[1:]:
        y = round(c["top"] / 3) * 3
        if y == current_y:
            current_row.append(c)
        else:
            _emit_line(current_row, page_num, lines)
            current_row = [c]
            current_y   = y

    _emit_line(current_row, page_num, lines)
    return lines


def _emit_line(row_chars, page_num, lines):
    text = "".join(ch["text"] for ch in row_chars).strip()
    if not text:
        return
    sizes = [ch["size"] for ch in row_chars if ch.get("size")]
    fonts = [ch.get("fontname", "") for ch in row_chars if ch.get("fontname")]
    lines.append({
        "text":      text,
        "font_size": sum(sizes) / len(sizes) if sizes else 0,
        "font_name": Counter(fonts).most_common(1)[0][0] if fonts else None,
        "top":       row_chars[0]["top"],
        "page":      page_num,
    })


def merge_lines_to_paragraphs(lines):
    """
    Merge consecutive body-text lines into full sentences/paragraphs.
    Headings stay as individual items. Body lines not ending with sentence
    punctuation are joined to the next line (they are PDF line-break continuations).
    """
    merged = []
    buffer = []

    for line in lines:
        if line.get("heading_level") is not None:
            if buffer:
                merged.append({
                    "text": " ".join(b["text"] for b in buffer),
                    "page": buffer[0]["page"],
                    "heading_level": None,
                })
                buffer = []
            merged.append(line)
        else:
            buffer.append(line)
            if line["text"].rstrip().endswith(('.', '?', '!', ':', ';')):
                merged.append({
                    "text": " ".join(b["text"] for b in buffer),
                    "page": buffer[0]["page"],
                    "heading_level": None,
                })
                buffer = []

    if buffer:
        merged.append({
            "text": " ".join(b["text"] for b in buffer),
            "page": buffer[0]["page"],
            "heading_level": None,
        })

    return merged


#############################################
# 2. ADAPTIVE GAP THRESHOLDS
#    Computed from each page's actual char heights instead of fixed magic numbers.
#############################################

def adaptive_gap_threshold(page):
    """Gap between row-groups = 1.8x median char height on this page."""
    heights = [c["bottom"] - c["top"] for c in page.chars
               if c.get("bottom") and c.get("top") and c["bottom"] > c["top"]]
    if not heights:
        return 12
    return sorted(heights)[len(heights) // 2] * 1.8


def adaptive_merge_threshold(page):
    """Merge nearby groups = 2.5x median char height."""
    heights = [c["bottom"] - c["top"] for c in page.chars
               if c.get("bottom") and c.get("top") and c["bottom"] > c["top"]]
    if not heights:
        return 30
    return sorted(heights)[len(heights) // 2] * 2.5


#############################################
# 3. TABLE DETECTION
#    PRIMARY:  pdfplumber find_tables() with false-positive validation
#    FALLBACK: column_cluster + row_spacing for borderless tables
#############################################

def _is_toc_table(rows):
    """
    Return True if this table is a table-of-contents fragment.
    TOC rows end with a short page-number string (e.g. '1', '53').
    If 60%+ of non-empty rows match this pattern, it's a TOC — skip it.
    """
    numeric_last = 0
    non_empty    = 0
    for row in rows:
        cells = [c for c in row if c is not None and str(c).strip()]
        if not cells:
            continue
        non_empty += 1
        if re.match(r'^\d{1,3}$', str(cells[-1]).strip()):
            numeric_last += 1
    return non_empty > 0 and (numeric_last / non_empty) >= 0.6


def find_valid_tables(page):
    """
    Use pdfplumber's find_tables() and reject false positives.
    A single-column "table" with long cell text is a paragraph in a box, not a table.
    TOC fragments (rows ending in page numbers) are also rejected.
    """
    valid = []
    for t in page.find_tables():
        rows = t.extract()
        if not rows or len(rows) < 2:
            continue
        col_counts = [sum(1 for c in r if c is not None) for r in rows]
        avg_cols   = sum(col_counts) / len(col_counts) if col_counts else 0

        # Reject false positives: "tables" that are really paragraphs in text boxes.
        total_cells  = sum(len(r) for r in rows)
        total_text   = sum(len(str(c or '')) for r in rows for c in r)
        avg_cell_len = total_text / max(total_cells, 1)
        if avg_cell_len > 50 and avg_cols <= 2:
            continue

        # Reject table-of-contents fragments
        if _is_toc_table(rows):
            continue

        valid.append({"rows": rows, "bbox": t.bbox})
    return valid


def column_cluster_score(words):
    """Borderless fallback: 3+ x-aligned columns."""
    if len(words) < 5:
        return 0
    xs     = [[w["x0"]] for w in words]
    labels = DBSCAN(eps=10, min_samples=3).fit(xs).labels_
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return 2 if clusters >= 3 else 0


def row_spacing_score(words):
    """Borderless fallback: consistent vertical row gaps."""
    if len(words) < 5:
        return 0
    ys = sorted(set(round(w["top"]) for w in words))
    if len(ys) < 3:
        return 0
    diffs      = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    mean_diff  = sum(diffs) / len(diffs)
    consistent = sum(1 for d in diffs if abs(d - mean_diff) < 3)
    return 1 if consistent / len(diffs) > 0.6 else 0


def is_borderless_table(cropped):
    """Check if a text region is a borderless table (no ruling lines)."""
    words = cropped.extract_words()
    return column_cluster_score(words) + row_spacing_score(words) >= 3


#############################################
# 4. BOILERPLATE DETECTION
#    Finds repeated text in top/bottom margins across 3+ pages.
#############################################

def detect_boilerplate(pdf, top_pt=50, bot_pt=50, min_repeats=3):
    """Return set of header/footer strings that repeat across pages."""
    top_texts, bot_texts = [], []

    for page in pdf.pages:
        chars = page.chars
        if not chars:
            top_texts.append("")
            bot_texts.append("")
            continue

        top_chars = sorted([c for c in chars if c["top"] < top_pt],
                           key=lambda c: (c["top"], c["x0"]))
        top_texts.append("".join(c["text"] for c in top_chars).strip())

        bot_chars = sorted([c for c in chars if c["top"] > page.height - bot_pt],
                           key=lambda c: (c["top"], c["x0"]))
        bot_texts.append("".join(c["text"] for c in bot_chars).strip())

    boilerplate = set()
    for text, count in Counter(top_texts).items():
        if count >= min_repeats and text:
            boilerplate.add(text)
    for text, count in Counter(bot_texts).items():
        if count >= min_repeats and text:
            boilerplate.add(text)
    return boilerplate


def strip_boilerplate(text, boilerplate):
    for bp in boilerplate:
        text = text.replace(bp, "")
    return text.strip()


#############################################
# 5. PAGE REGION EXTRACTION
#    Splits page into table bboxes + text slices between them.
#############################################

def extract_text_regions(page, table_bboxes, page_num, boilerplate=None):
    """Extract text from areas of the page not covered by table bounding boxes."""
    pw, ph = page.width, page.height
    sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])

    y_edges = [0]
    for bbox in sorted_bboxes:
        y_edges.append(bbox[1])
        y_edges.append(bbox[3])
    y_edges.append(ph)

    regions = []
    for i in range(0, len(y_edges) - 1, 2):
        y0, y1 = y_edges[i], y_edges[i + 1]
        if y1 - y0 < 10:
            continue
        cropped = page.crop((0, max(0, y0), pw, min(ph, y1)))
        text = cropped.extract_text()
        if text and text.strip():
            cleaned = strip_boilerplate(text.strip(), boilerplate) if boilerplate else text.strip()
            if cleaned:
                regions.append({
                    "text":  cleaned,
                    "chars": cropped.chars,
                    "page":  page_num,
                    "y0":    y0,
                    "y1":    y1,
                })
    return regions


#############################################
# 6. HIERARCHICAL JSON
#############################################

def build_hierarchical_json(paragraphs):
    """Build nested document tree from font-size-classified paragraphs."""
    root  = {"title": "ROOT", "level": 0, "page": None, "content": [], "children": []}
    stack = [root]

    for para in paragraphs:
        text  = para["text"]
        page  = para["page"]
        level = para.get("heading_level")

        if not text:
            continue

        if level is not None:
            section = {"title": text, "level": level, "page": page, "content": [], "children": []}
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["children"].append(section)
            stack.append(section)
        else:
            line_type = "ListItem" if re.match(r'^[*\u2022\-\u2013\u2014]\s', text) or re.match(r'^\(\w+\)\s', text) else "NarrativeText"
            stack[-1]["content"].append({"text": text, "type": line_type, "page": page})

    return root


#############################################
# 7. CHUNKING
#############################################

SENTENCE_END = {".", "?", "!", ":"}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _join_content_items(items):
    """Join content items respecting sentence continuity."""
    if not items:
        return ""
    parts = [items[0]["text"].strip()]
    for i in range(1, len(items)):
        prev = items[i - 1]["text"].strip()
        curr = items[i]["text"].strip()
        sep  = "\n\n" if prev and prev[-1] in SENTENCE_END else " "
        if sep == " ":
            parts[-1] = parts[-1] + " " + curr
        else:
            parts.append(curr)
    return "\n\n".join(parts)


def chunk_section(section, parent_breadcrumb=""):
    breadcrumb = f"{parent_breadcrumb} > {section['title']}".strip(" >")
    chunks = []

    if section["content"]:
        full_text = _join_content_items(section["content"])
        page      = section["content"][0]["page"]
        for piece in splitter.split_text(full_text):
            if piece.strip():
                chunks.append({"breadcrumb": breadcrumb, "page": page, "text": piece.strip()})

    for child in section.get("children", []):
        chunks.extend(chunk_section(child, parent_breadcrumb=breadcrumb))

    return chunks


#############################################
# 8. TABLE → TEXT FLATTENING
#    Converts extracted table rows into readable text chunks for embedding.
#############################################

# Patterns that indicate a cell is a column header label, not data
_YEAR_RE    = re.compile(r'\b(20\d{2})\b')
_QUARTER_RE = re.compile(r'\b(Q[1-4])\b', re.IGNORECASE)


def _looks_like_header_row(row):
    """
    Strategy 1: Return True if this row looks like column headers.
    Signals: contains fiscal years or quarters, no $ values, short cells.
    """
    cells = [str(c).strip() if c else "" for c in row if c is not None]
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    has_year    = any(_YEAR_RE.search(c) for c in non_empty)
    has_quarter = any(_QUARTER_RE.search(c) for c in non_empty)
    has_dollar  = any("$" in c or (c.replace(",", "").isdigit() and len(c) > 3) for c in non_empty)
    avg_len     = sum(len(c) for c in non_empty) / len(non_empty)
    return (has_year or has_quarter) and not has_dollar and avg_len < 40


def _extract_header_from_above(page, bbox):
    """
    Strategy 2: Crop the 60pt strip above the table bounding box and pull
    fiscal year / quarter labels from the text there.
    Returns a list of column label strings, or [] if nothing found.
    """
    x0, y0, x1, _ = bbox
    strip_y0 = max(0, y0 - 60)
    if strip_y0 >= y0:
        return []

    cropped = page.crop((x0, strip_y0, x1, y0))
    text    = cropped.extract_text() or ""

    years    = _YEAR_RE.findall(text)
    quarters = _QUARTER_RE.findall(text.upper())

    labels = []
    if years:
        # deduplicate while preserving order
        seen = set()
        for y in years:
            if y not in seen:
                labels.append(y)
                seen.add(y)
    elif quarters:
        seen = set()
        for q in quarters:
            if q not in seen:
                labels.append(q)
                seen.add(q)

    return labels


def _detect_header(rows, page, bbox):
    """
    Combine Strategy 1 and 2.
    Returns (header_labels, data_rows).
    header_labels is a list of column name strings (may be empty if not found).
    data_rows is the rows to actually flatten (first row removed if it was a header).
    """
    # Strategy 1: first row looks like a header?
    if rows and _looks_like_header_row(rows[0]):
        header_cells = [str(c).strip() if c else "" for c in rows[0] if c is not None]
        labels = [c for c in header_cells if c]
        return labels, rows[1:]

    # Strategy 2: text above the table
    if page is not None and bbox is not None:
        labels = _extract_header_from_above(page, bbox)
        if labels:
            return labels, rows

    return [], rows


def _flatten_row(cells):
    """Join non-empty cells, collapsing '$' with the next value and '%' with the previous."""
    parts = []
    skip = False
    for i, c in enumerate(cells):
        if skip:
            skip = False
            continue
        c = str(c).strip() if c else ""
        if not c:
            continue
        if c == "$" and i + 1 < len(cells) and cells[i + 1]:
            parts.append(f"${str(cells[i + 1]).strip()}")
            skip = True
        elif c == "%":
            if parts:
                parts[-1] = parts[-1] + "%"
        else:
            parts.append(c)
    return " | ".join(parts)


def flatten_table(rows, page_num, source, page=None, bbox=None):
    """
    Convert a table into a single text block suitable for embedding.
    Detects column headers via Strategy 1 (header row) or Strategy 2 (text above).
    Each data row becomes pipe-delimited values. Block is attributed with source and page.
    """
    header_labels, data_rows = _detect_header(rows, page, bbox)

    lines = []
    for row in data_rows:
        cells = [str(c).strip() if c else "" for c in row]
        line = _flatten_row(cells)
        if line:
            lines.append(line)

    if not lines:
        return None

    attribution = f"[Table from {source}, page {page_num}]"
    if header_labels:
        col_header = "Columns: " + " | ".join(header_labels)
        return attribution + "\n" + col_header + "\n" + "\n".join(lines)
    return attribution + "\n" + "\n".join(lines)


#############################################
# 9. WORKER — one PDF per call, with error handling
#############################################

def process_pdf(pdf_path):
    """Process a single PDF. Returns (name, chunks, tables, status, error)."""
    pdf_path = Path(pdf_path)
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        table_count   = 0
        table_chunks  = []   # flattened table text for embedding

        with pdfplumber.open(pdf_path) as pdf:
            dominant_size = compute_dominant_font_size(pdf)
            boilerplate   = detect_boilerplate(pdf)
            all_lines     = []

            for page_num, page in enumerate(pdf.pages, start=1):
                # PRIMARY table detection
                valid_tables = find_valid_tables(page)
                table_bboxes = [t["bbox"] for t in valid_tables]

                for t in valid_tables:
                    cursor.execute(
                        "INSERT INTO extracted_tables (source, page_number, table_json) VALUES (%s, %s, %s)",
                        (pdf_path.name, page_num, json.dumps(t["rows"]))
                    )
                    table_count += 1

                    # Flatten table to text chunk for vector embedding
                    flat = flatten_table(t["rows"], page_num, pdf_path.name,
                                        page=page, bbox=t["bbox"])
                    if flat:
                        table_chunks.append({
                            "breadcrumb": f"TABLE > page {page_num}",
                            "page": page_num,
                            "text": flat,
                            "type": "table",
                        })

                # Text regions between tables
                text_regions = extract_text_regions(page, table_bboxes, page_num, boilerplate)

                for region in text_regions:
                    cropped = page.crop((0, region["y0"], page.width, region["y1"]))

                    # FALLBACK: borderless table check
                    if is_borderless_table(cropped):
                        for table in cropped.extract_tables():
                            if table:
                                cursor.execute(
                                    "INSERT INTO extracted_tables (source, page_number, table_json) VALUES (%s, %s, %s)",
                                    (pdf_path.name, page_num, json.dumps(table))
                                )
                                table_count += 1

                                flat = flatten_table(table, page_num, pdf_path.name)
                                if flat:
                                    table_chunks.append({
                                        "breadcrumb": f"TABLE > page {page_num}",
                                        "page": page_num,
                                        "text": flat,
                                        "type": "table",
                                    })
                        continue

                    # Extract lines with font sizes for heading detection
                    lines = group_chars_to_lines(region["chars"], page_num)
                    for line in lines:
                        line["heading_level"] = classify_heading(
                            line["text"], line["font_size"], dominant_size, line.get("font_name")
                        )
                    all_lines.extend(lines)

        conn.commit()
        cursor.close()
        conn.close()

        # Merge body lines into paragraphs, build hierarchy, chunk
        paragraphs = merge_lines_to_paragraphs(all_lines)
        doc_json   = build_hierarchical_json(paragraphs)
        narrative_chunks = chunk_section(doc_json)

        # Combine narrative + table chunks
        all_chunks = narrative_chunks + table_chunks

        output = {
            "metadata": {
                "source":             pdf_path.name,
                "dominant_font_size": dominant_size,
                "boilerplate_count":  len(boilerplate),
                "tables_extracted":   table_count,
                "table_chunks":       len(table_chunks),
                "narrative_chunks":   len(narrative_chunks),
                "rag_chunks":         len(all_chunks),
            },
            "hierarchy": doc_json,
            "chunks":    all_chunks,
        }

        out_path = OUTPUT_DIR / (pdf_path.stem + ".json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        return pdf_path.name, len(all_chunks), table_count, "ok", ""

    except Exception as e:
        return pdf_path.name, 0, 0, "error", traceback.format_exc()


#############################################
# 9. MAIN — with manifest logging
#############################################

def ensure_database():
    target_db = DB_CONFIG["database"]
    for fallback_db in ("postgres", DB_CONFIG["user"]):
        try:
            admin_conn = psycopg2.connect(**{**DB_CONFIG, "database": fallback_db})
            admin_conn.autocommit = True
            cur = admin_conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{target_db}"')
                print(f"Created database '{target_db}'")
            cur.close()
            admin_conn.close()
            return
        except psycopg2.OperationalError:
            continue
    raise RuntimeError("Could not connect to any maintenance database.")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_database()

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_tables (
            id          SERIAL PRIMARY KEY,
            source      TEXT,
            page_number INT,
            table_json  JSONB
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs  |  Workers: {N_WORKERS}")

    manifest  = []
    ok_count  = 0
    err_count = 0

    with multiprocessing.Pool(N_WORKERS) as pool:
        for name, chunks, tables, status, error in pool.imap_unordered(process_pdf, pdf_files):
            manifest.append({
                "source":    name,
                "chunks":    chunks,
                "tables":    tables,
                "status":    status,
                "error":     error if error else None,
                "timestamp": datetime.now().isoformat(),
            })
            if status == "ok":
                ok_count += 1
                print(f"  OK  {name}: {chunks} chunks, {tables} tables")
            else:
                err_count += 1
                last_line = error.strip().splitlines()[-1] if error else "unknown"
                print(f"  ERR {name}: {last_line}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {ok_count} succeeded, {err_count} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
