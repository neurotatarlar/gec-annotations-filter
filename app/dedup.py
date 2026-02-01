"""Near-duplicate detection helpers using SimHash."""

import regex as reg
from simhash import Simhash, SimhashIndex
from typing import Generator, Iterable, Set

from .processing import tokenize_words


def normalize_for_hash(text: str) -> str:
    """Normalize text to stabilize hashing for deduplication."""
    normalized = reg.sub(r"\s+", " ", text.strip().lower())
    return normalized


def simhash_from_text(text: str) -> Simhash:
    """Compute a 64-bit SimHash over tokenized text."""
    return Simhash(tokenize_words(text), f=64)


def near_deduplicate_stream(
    rows: Iterable, distance: int = 3
) -> Generator:
    """
    Yield rows that are not near-duplicates according to SimHash.
    """
    index = SimhashIndex([], k=distance)
    seen_norm: Set[str] = set()
    for row in rows:
        norm_hash = row["norm_hash"]
        if norm_hash in seen_norm:
            continue
        sh = simhash_from_text(row["text"])
        near = index.get_near_dups(sh)
        if near:
            continue
        index.add(row["id"], sh)
        seen_norm.add(norm_hash)
        yield row
