from unstructured.partition.text import partition_text
from pathlib import Path
from sklearn.cluster import DBSCAN
import json
import os
import psycopg2
import pdfplumber
import multiprocessing

base_dir = Path(__file__).resolve().parent

DOCUMENTS_DIR = base_dir / 'documents'
OUTPUT_DIR    = base_dir / 'output'

# Override DB_USER with env var if postgres role doesn't exist on your system
DB_CONFIG = dict(
    host     = "localhost",
    database = "pdf_tables",
    user     = os.getenv("DB_USER", os.getenv("USER", "postgres")),
    password = os.getenv("DB_PASSWORD", ""),
)

CHUNK_SIZE      = 500   # approximate tokens (chars / 4)
CHUNK_OVERLAP   = 50
ROW_GAP_THRESH  = 12    # pt — vertical gap that signals a new region
MERGE_THRESH    = 30    # pt — merge bands that are closer than this
N_WORKERS       = max(1, multiprocessing.cpu_count() - 2)


#############################################
# LAYOUT SIGNALS  (applied per region, not per page)
#############################################

def ruling_line_score(region):
    """Primary: explicit horizontal/vertical lines drawn in the PDF."""
    h_lines = [l for l in region.lines if l.get("height", 0) < 2  and l.get("width", 0) > 20]
    v_lines = [l for l in region.lines if l.get("width",  0) < 2  and l.get("height", 0) > 10]
    rects   = [r for r in region.rects if r.get("width",  0) > 20 and r.get("height", 0) > 5]
    if len(h_lines) >= 3 or len(v_lines) >= 2 or len(rects) >= 2:
        return 3
    return 0


def column_cluster_score(words):
    """Secondary: 3+ distinct x-columns indicate tabular layout."""
    if len(words) < 5:
        return 0
    xs     = [[w["x0"]] for w in words]
    labels = DBSCAN(eps=10, min_samples=3).fit(xs).labels_
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return 2 if clusters >= 3 else 0


def row_spacing_score(words):
    """Secondary: consistent vertical gaps between rows."""
    if len(words) < 5:
        return 0
    ys = sorted(set(round(w["top"]) for w in words))
    if len(ys) < 3:
        return 0
    diffs      = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    mean_diff  = sum(diffs) / len(diffs)
    consistent = sum(1 for d in diffs if abs(d - mean_diff) < 3)
    return 1 if consistent / len(diffs) > 0.6 else 0


def is_table_region(region):
    words = region.extract_words()
    score = (
        ruling_line_score(region)
        + column_cluster_score(words)
        + row_spacing_score(words)
    )
    return score >= 3


#############################################
# REGION SPLITTING
# Detects horizontal bands by finding significant vertical gaps
# between word rows, then merges bands that are close together.
# This lets us handle pages like: para / table / para correctly.
#############################################

def _word_row_groups(page):
    """Return sorted list of (y_top, y_bottom) for each row cluster on the page."""
    words = page.extract_words()
    if not words:
        return []

    # Bucket words into rows (5pt quantization covers minor baseline shifts)
    rows = {}
    for w in words:
        key = round(w["top"] / 5) * 5
        rows.setdefault(key, []).append(w)

    sorted_ys = sorted(rows.keys())
    groups, g_start, prev_y = [], sorted_ys[0], sorted_ys[0]

    for y in sorted_ys[1:]:
        if y - prev_y > ROW_GAP_THRESH:
            groups.append((g_start, prev_y + 12))
            g_start = y
        prev_y = y

    groups.append((g_start, prev_y + 12))
    return groups


def _merge_groups(groups):
    """Merge bands that are within MERGE_THRESH of each other."""
    if not groups:
        return []
    merged = [list(groups[0])]
    for start, end in groups[1:]:
        if start - merged[-1][1] <= MERGE_THRESH:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(g) for g in merged]


def detect_regions(page):
    """
    Split page into horizontal bands and classify each as 'table' or 'text'.
    A single page like  [para] [table] [para]  produces three regions.
    Returns list of {'type', 'cropped': pdfplumber crop}.
    """
    bands   = _merge_groups(_word_row_groups(page))
    pw      = page.width
    ph      = page.height
    regions = []

    for y0, y1 in bands:
        y0c = max(0, y0 - 5)
        y1c = min(ph, y1 + 5)
        cropped     = page.crop((0, y0c, pw, y1c))
        region_type = "table" if is_table_region(cropped) else "text"
        regions.append({"type": region_type, "cropped": cropped})

    return regions


#############################################
# TABLE STORAGE  (pdfplumber)
#############################################

def store_table_region(cropped, page_num, source_name, cursor):
    for table in cropped.extract_tables():
        if table:
            cursor.execute(
                "INSERT INTO extracted_tables (source, page_number, table_json) VALUES (%s, %s, %s)",
                (source_name, page_num, json.dumps(table))
            )


#############################################
# TEXT CLASSIFICATION  (unstructured partition_text)
# partition_text is fast (no ML/OCR), applies unstructured's
# semantic heuristics (Title / NarrativeText / ListItem) on raw strings.
#############################################

def classify_text_regions(text_regions):
    """
    Run unstructured's partition_text on each text region.
    Attach page number since partition_text has no PDF context.
    """
    elements = []
    for region in text_regions:
        for elem in partition_text(text=region["text"]):
            elem.metadata.page_number = region["page"]
            elements.append(elem)
    return elements


#############################################
# HIERARCHICAL JSON
#############################################

def get_heading_level(element):
    text = element.text.strip()
    if len(text) < 4 or text.endswith('.'):
        return None
    font_size = getattr(element.metadata, "font_size", None)
    if font_size:
        if font_size >= 14: return 1
        if font_size >= 11: return 2
    if text.isupper() and len(text.split()) <= 8:
        return 1
    if text.istitle() and len(text.split()) <= 10:
        return 2
    return None


def build_hierarchical_json(elements):
    root  = {"title": "ROOT", "level": 0, "page": None, "content": [], "children": []}
    stack = [root]

    for elem in elements:
        text = elem.text.strip()
        cat  = elem.category
        page = elem.metadata.page_number
        if not text:
            continue

        if cat == "Title":
            level   = get_heading_level(elem) or 1
            section = {"title": text, "level": level, "page": page, "content": [], "children": []}
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["children"].append(section)
            stack.append(section)

        elif cat in ("NarrativeText", "ListItem", "Text"):
            stack[-1]["content"].append({"text": text, "type": cat, "page": page})

    return root


#############################################
# CHUNKING
#############################################

def _tokens(text):
    return len(text) // 4


def chunk_section(section, parent_breadcrumb=""):
    breadcrumb      = f"{parent_breadcrumb} > {section['title']}".strip(" >")
    buffer, buf_tok = [], 0
    chunks          = []

    for item in section["content"]:
        text, tok = item["text"], _tokens(item["text"])

        if buf_tok + tok > CHUNK_SIZE and buffer:
            chunks.append({"breadcrumb": breadcrumb, "page": item["page"], "text": " ".join(buffer)})
            overlap, ov_tok = [], 0
            for t in reversed(buffer):
                ov_tok += _tokens(t)
                if ov_tok >= CHUNK_OVERLAP:
                    break
                overlap.insert(0, t)
            buffer, buf_tok = overlap, sum(_tokens(t) for t in overlap)

        buffer.append(text)
        buf_tok += tok

    if buffer:
        last_page = section["content"][-1]["page"] if section["content"] else section.get("page")
        chunks.append({"breadcrumb": breadcrumb, "page": last_page, "text": " ".join(buffer)})

    for child in section.get("children", []):
        chunks.extend(chunk_section(child, parent_breadcrumb=breadcrumb))

    return chunks


#############################################
# WORKER — one PDF per call
#############################################

def process_pdf(pdf_path):
    pdf_path     = Path(pdf_path)
    text_regions = []

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for region in detect_regions(page):
                if region["type"] == "table":
                    store_table_region(region["cropped"], page_num, pdf_path.name, cursor)
                else:
                    text = region["cropped"].extract_text()
                    if text and text.strip():
                        text_regions.append({"text": text.strip(), "page": page_num})

    conn.commit()
    cursor.close()
    conn.close()

    elements   = classify_text_regions(text_regions)
    doc_json   = build_hierarchical_json(elements)
    all_chunks = [c for s in doc_json["children"] for c in chunk_section(s)]

    output = {
        "metadata": {
            "source":     pdf_path.name,
            "rag_chunks": len(all_chunks)
        },
        "hierarchy": doc_json,
        "chunks":    all_chunks
    }

    out_path = OUTPUT_DIR / (pdf_path.stem + ".json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return pdf_path.name, len(all_chunks)


#############################################
# MAIN
#############################################

def ensure_database():
    """
    Create the target database if it doesn't exist.
    Must use autocommit=True — CREATE DATABASE cannot run inside a transaction.
    Connects to the 'postgres' maintenance DB first; falls back to user's own DB.
    """
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
    raise RuntimeError("Could not connect to any maintenance database to create pdf_tables.")


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

    with multiprocessing.Pool(N_WORKERS) as pool:
        for name, chunks in pool.imap_unordered(process_pdf, pdf_files):
            print(f"  {name}: {chunks} chunks")

    print("Done.")


if __name__ == "__main__":
    main()
