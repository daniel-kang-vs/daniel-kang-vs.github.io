#!/usr/bin/env python3
"""Extract plain text from a PDF for ATS parse testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_with_pdfminer(pdf_path: Path) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(str(pdf_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract PDF text for ATS parse test")
    parser.add_argument("pdf_file")
    parser.add_argument("-o", "--output", help="Output .parse.txt path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file)
    if not pdf_path.is_file():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return 1

    text = ""
    for extractor in (extract_with_pypdf, extract_with_pdfminer):
        try:
            text = extractor(pdf_path)
            break
        except ImportError:
            continue
        except Exception as exc:
            print(f"Extractor failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "Error: install pypdf or pdfminer.six: pip install pypdf",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.output) if args.output else pdf_path.with_suffix(".parse.txt")
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
