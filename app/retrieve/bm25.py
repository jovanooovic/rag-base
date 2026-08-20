from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately small. Aggressive stoplists hurt on technical corpora where
# "not", "in", "no" carry meaning (policy documents, medical notes, contracts).
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "as", "that", "this", "it", "be", "from",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25.

    Why keyword search still matters in 2026: embeddings are bad at exact
    identifiers. Order numbers, SKUs, error codes, statute references, and
    person names are exactly what users paste into a support search box, and a
    pure-vector system misses them. Every RAG posting that says "it can't find
    the right document" is describing this failure.
    """

    def __init__(self, corpus: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        # +0.5 smoothing keeps the idf of a term present in every document at a
        # small positive value instead of negative, which would invert ranking.
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        if not self.n:
            return out
        for i, freq in enumerate(self.freqs):
            dl = self.lengths[i] or 1
            total = 0.0
            for term in q:
                f = freq.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out[i] = total
        return out
