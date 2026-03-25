from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import json
import psycopg2

base_dir = Path(__file__).resolve().parent

document_path = base_dir / 'documents' / 'apple_10-k.pdf'

OUTPUT_JSON = "parsed_document.json"

#############################################
# DATABASE CONNECTION
#############################################

conn = psycopg2.connect(
    host="localhost",
    database="pdf_tables",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS extracted_tables(
    id SERIAL PRIMARY KEY,
    page_number INT,
    table_html TEXT,
    table_json JSONB
)
""")

conn.commit()

#############################################
# PARSE ENTIRE DOCUMENT ONCE
# strategy="auto"  — uses hi_res when detectron2 is installed, fast otherwise
# infer_table_structure=True — lets unstructured detect & structure tables
#############################################

print("Parsing PDF (this may take a minute)...")

elements = partition_pdf(
    str(document_path),
    strategy="auto",
    infer_table_structure=True,
    include_page_breaks=False,
)

print(f"Extracted {len(elements)} elements")

table_elements = [e for e in elements if e.category == "Table"]
text_elements  = [e for e in elements if e.category != "Table"]

print(f"Tables: {len(table_elements)} | Text blocks: {len(text_elements)}")

#############################################
# STORE TABLES TO POSTGRES
#############################################

for elem in table_elements:
    page_num = elem.metadata.page_number
    html     = getattr(elem.metadata, "text_as_html", None)
    cursor.execute(
        "INSERT INTO extracted_tables (page_number, table_html, table_json) VALUES (%s, %s, %s)",
        (page_num, html, json.dumps({"raw_text": elem.text}))
    )

conn.commit()
print(f"Stored {len(table_elements)} tables to PostgreSQL")

#############################################
# HEADING LEVEL DETECTION
#############################################

def get_heading_level(element):
    """Return 1 for top-level section, 2 for sub-heading, None if not a real heading."""
    text = element.text.strip()

    if len(text) < 4 or text.endswith('.'):
        return None

    # Prefer font size metadata when available (unstructured hi_res mode provides it)
    font_size = getattr(element.metadata, "font_size", None)
    if font_size:
        if font_size >= 14:
            return 1
        if font_size >= 11:
            return 2

    # Fallback heuristics
    if text.isupper() and len(text.split()) <= 8:
        return 1
    if text.istitle() and len(text.split()) <= 10:
        return 2

    return None

#############################################
# BUILD HIERARCHICAL JSON
# Uses a stack so headings nest naturally at any depth
#############################################

def build_hierarchical_json(elements):
    root  = {"title": "ROOT", "level": 0, "page": None, "content": [], "children": []}
    stack = [root]

    for element in elements:
        text = element.text.strip()
        cat  = element.category
        page = element.metadata.page_number

        if not text:
            continue

        if cat == "Title":
            level = get_heading_level(element) or 1

            section = {
                "title":    text,
                "level":    level,
                "page":     page,
                "content":  [],
                "children": []
            }

            # Pop stack until we find a parent with a lower level number
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            stack[-1]["children"].append(section)
            stack.append(section)

        elif cat in ("NarrativeText", "ListItem", "Text"):
            stack[-1]["content"].append({
                "text": text,
                "type": cat,
                "page": page
            })

    return root

#############################################
# TOKEN-AWARE CHUNKING WITH OVERLAP
# Produces flat chunks with breadcrumb context for RAG retrieval
#############################################

CHUNK_SIZE    = 500   # approximate tokens (chars / 4)
CHUNK_OVERLAP = 50

def _approx_tokens(text):
    return len(text) // 4

def chunk_section(section, parent_breadcrumb=""):
    breadcrumb = f"{parent_breadcrumb} > {section['title']}".strip(" >")
    buffer     = []
    buf_tokens = 0
    chunks     = []

    for item in section["content"]:
        text   = item["text"]
        tokens = _approx_tokens(text)

        if buf_tokens + tokens > CHUNK_SIZE and buffer:
            chunks.append({
                "breadcrumb": breadcrumb,
                "page":       item["page"],
                "text":       " ".join(buffer)
            })
            # Carry over tail for overlap
            overlap, overlap_tok = [], 0
            for t in reversed(buffer):
                overlap_tok += _approx_tokens(t)
                if overlap_tok >= CHUNK_OVERLAP:
                    break
                overlap.insert(0, t)
            buffer, buf_tokens = overlap, sum(_approx_tokens(t) for t in overlap)

        buffer.append(text)
        buf_tokens += tokens

    if buffer:
        last_page = section["content"][-1]["page"] if section["content"] else section.get("page")
        chunks.append({"breadcrumb": breadcrumb, "page": last_page, "text": " ".join(buffer)})

    for child in section.get("children", []):
        chunks.extend(chunk_section(child, parent_breadcrumb=breadcrumb))

    return chunks

#############################################
# MAIN PIPELINE
#############################################

document_json = build_hierarchical_json(text_elements)

all_chunks = []
for section in document_json["children"]:
    all_chunks.extend(chunk_section(section))

output = {
    "metadata": {
        "source":            document_path.name,
        "total_elements":    len(elements),
        "tables_extracted":  len(table_elements),
        "rag_chunks":        len(all_chunks)
    },
    "hierarchy": document_json,
    "chunks":    all_chunks
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

cursor.close()
conn.close()

print(f"Done. {len(all_chunks)} RAG chunks written to {OUTPUT_JSON}")
