"""PDF text extraction and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader

MAX_PAGES = 300


@dataclass
class Chunk:
    page: int  # 1-based page number, used for citations
    text: str


def extract_pages(file: BinaryIO | str) -> list[str]:
    reader = PdfReader(file)
    if reader.is_encrypted:
        raise ValueError("The PDF is password protected.")
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"The PDF has {len(reader.pages)} pages; the limit is {MAX_PAGES}.")
    pages = [re.sub(r"[ \t]+", " ", page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("No text found. The PDF may be scanned images; run OCR first (for example Amazon Textract).")
    return pages


def chunk_pages(pages: list[str], max_chars: int = 2000, overlap: int = 200) -> list[Chunk]:
    """Split each page into overlapping chunks. Chunks never cross a page, so citations stay exact."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    for number, text in enumerate(pages, start=1):
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                # Prefer to cut at the end of a sentence or line.
                cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if cut > start + max_chars // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(number, piece))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
