import pdfplumber
from pathlib import Path

base_dir = Path(__file__).resolve().parent

document_path = base_dir/'documents'/'apple_10-k.pdf'

with pdfplumber.open(document_path) as pdf:
    page = pdf.pages[21]   # choose the page you want

    words = page.extract_words()

    # Convert page to image
    im = page.to_image(resolution=150)

    # Draw rectangles around each word
    im.draw_rects(words)

    # Save debug image
    im.save("debug_layout.png")

    # Display
    im.show()

for w in words:
    print(
        f"{w['text']:20} | x0={w['x0']:.1f} | top={w['top']:.1f}"
    )