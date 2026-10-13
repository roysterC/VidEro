"""Render ai-influencer-guide.html to PDF with headless Chromium.

Two passes: the first render finds the page each section starts on, the second
fills those numbers into the contents page.

    python docs/guide/build.py [--chrome /path/to/chrome]
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
SRC = HERE / "ai-influencer-guide.html"
OUT = HERE / "AI-Influencer-Guide.pdf"
CHROME_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "chromium", "chromium-browser", "google-chrome", "chrome",
]


def find_chrome(explicit):
    for c in [explicit] + CHROME_CANDIDATES if explicit else CHROME_CANDIDATES:
        path = c if os.path.isfile(c) else shutil.which(c)
        if path:
            return path
    raise SystemExit("Chrome/Chromium not found; pass --chrome")


def render(chrome, html_path, pdf_path):
    subprocess.run(
        [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=5000", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )


def section_pages(pdf_path):
    """Map section id -> 1-based page number using each anchor's named destination."""
    reader = PdfReader(pdf_path)
    page_ids = {p.indirect_reference.idnum: i + 1 for i, p in enumerate(reader.pages)}
    pages = {}
    for name, dest in reader.named_destinations.items():
        pages[name.lstrip("/")] = page_ids.get(dest.page.idnum if hasattr(dest.page, "idnum") else dest.page)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome")
    args = ap.parse_args()
    chrome = find_chrome(args.chrome)

    html = SRC.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        draft = Path(tmp) / "draft.pdf"
        render(chrome, SRC, draft)
        pages = section_pages(draft)

        def fill(m):
            sid = m.group(1)
            return f'<span class="pg" data-for="{sid}">{pages.get(sid) or ""}</span>'

        filled = re.sub(r'<span class="pg" data-for="([^"]+)">[^<]*</span>', fill, html)
        missing = sorted(set(re.findall(r'data-for="([^"]+)"', html)) - {k for k, v in pages.items() if v})
        if missing:
            print("warning: no page found for", ", ".join(missing))
        tmp_html = HERE / ".build.html"  # same folder so relative paths keep working
        tmp_html.write_text(filled, encoding="utf-8")
        try:
            render(chrome, tmp_html, OUT)
        finally:
            tmp_html.unlink()
    print(f"wrote {OUT} ({len(PdfReader(OUT).pages)} pages)")


if __name__ == "__main__":
    main()
