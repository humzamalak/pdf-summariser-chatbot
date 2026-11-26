"""Small BM25 keyword search over chunks. No vector database needed for one document."""

from __future__ import annotations

import math
import re
from collections import Counter

from pdfbot.document import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were will with".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class Bm25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        q = tokens(query)
        scores = []
        for i, d in enumerate(self.docs):
            score = 0.0
            for t in q:
                if t not in d:
                    continue
                tf = d[t]
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1))
                score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]
