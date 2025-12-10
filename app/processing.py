import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import regex as reg
import xxhash

from .config import ProcessingConfig, DEFAULT_CONFIG

try:
    from razdel import sentenize
except ImportError:  # pragma: no cover - optional dependency
    sentenize = None

# Basic patterns for contacts and unwanted structures.
URL_RE = reg.compile(r"(?i)\b(?:https?://|http://|www\.)\S+")
EMAIL_RE = reg.compile(r"(?i)[\p{L}0-9._%+-]+@[\p{L}0-9.-]+\.[a-z]{2,}")
PHONE_RE = reg.compile(
    r"(?x)(?:\+?7|8)?[\s\-]?(?:\(\d{3}\)|\d{3})[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)
CONTACT_RE = reg.compile(f"({URL_RE.pattern})|({EMAIL_RE.pattern})|({PHONE_RE.pattern})", reg.IGNORECASE)
FAKE_NAME_RE = reg.compile(r"\[\[([^\[\]]+)\]\]")
SINGLE_BRACKET_RE = reg.compile(r"(?<!\[)\[[^\[\]]+\](?!\])")
LETTER_RE = reg.compile(r"\p{L}", reg.UNICODE)
CYRILLIC_LETTER_RE = reg.compile(r"\p{IsCyrillic}", reg.UNICODE)
LATIN_LETTER_RE = reg.compile(r"\p{Latin}", reg.UNICODE)
ZERO_WIDTH_RE = reg.compile(r"[\u200b\ufeff\u2060]")
MULTISPACE_RE = reg.compile(r"[ \t\f\v\u00A0]+")
WORD_RE = reg.compile(r"[\p{L}\d]+(?:'[\p{L}\d]+)?", reg.UNICODE)


@dataclass
class ProcessedSegment:
    """Lightweight container for cleaned text and derived ids."""

    segment_id: str
    parent_digest: str
    text: str


def load_name_list(path: Path = DEFAULT_CONFIG.name_list_path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deterministic_hash(seed: str) -> int:
    """Stable 32-bit hash to drive deterministic replacements."""
    return xxhash.xxh32(seed).intdigest()


def normalize_whitespace(text: str) -> str:
    """
    Collapse excessive whitespace while preserving line breaks.

    Tabs and non-breaking spaces are converted to a single space.
    """
    text = text.replace("\r", "")
    text = ZERO_WIDTH_RE.sub("", text)
    lines = []
    for raw_line in text.splitlines():
        normalized = MULTISPACE_RE.sub(" ", raw_line)
        normalized = normalized.strip()
        lines.append(normalized)
    return "\n".join(lines)


def is_contact_only(text: str) -> bool:
    """
    Detect messages that contain only contact info (URLs/emails/phones).
    """
    cleaned = CONTACT_RE.sub(" ", text)
    cleaned = reg.sub(r"[\p{P}\p{S}\s]+", " ", cleaned)
    return cleaned.strip() == ""


def _fake_email(seed: str) -> str:
    h = deterministic_hash(seed)
    return f"mail{h % 100000:05d}@example.com"


def _fake_phone(seed: str) -> str:
    h = deterministic_hash(seed)
    prefix = 900 + (h % 90)
    middle = (h // 100) % 1000
    tail = h % 10000
    return f"+7 {prefix:03d} {middle:03d}-{tail // 100:02d}-{tail % 100:02d}"


def _fake_url(seed: str) -> str:
    h = deterministic_hash(seed)
    return f"https://{h % 1_000_000:06d}.com"


def replace_contacts(text: str) -> str:
    """
    Replace URLs, emails, and phone numbers with deterministic but fake values.
    """
    text = EMAIL_RE.sub(lambda m: _fake_email(m.group(0)), text)
    text = PHONE_RE.sub(lambda m: _fake_phone(m.group(0)), text)
    text = URL_RE.sub(lambda m: _fake_url(m.group(0)), text)
    return text


def replace_fake_names(text: str, names: Sequence[str]) -> str:
    """
    Replace placeholders like [[John]] with deterministic Tatar names.
    """
    if not names:
        return FAKE_NAME_RE.sub("", text)

    def _replacer(match: reg.Match) -> str:
        raw = match.group(1).strip()
        idx = deterministic_hash(raw) % len(names)
        return names[idx]

    return FAKE_NAME_RE.sub(_replacer, text)


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    if sentenize:
        return [s.text.strip() for s in sentenize(text)]
    # Fallback: split on sentence-ending punctuation.
    parts = reg.split(r"(?<=[\.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentence(sentence: str, cfg: ProcessingConfig) -> List[str]:
    """
    If a single sentence is too long, fall back to word-based chunking.
    """
    tokens = tokenize_words(sentence)
    chunks: List[str] = []
    current: List[str] = []
    for token in tokens:
        current.append(token)
        candidate = " ".join(current)
        if len(current) >= cfg.max_words or len(candidate) >= cfg.max_chars:
            chunks.append(candidate)
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_text(text: str, cfg: ProcessingConfig) -> List[str]:
    """
    Split text into segments that respect max word/char constraints.
    """
    words = tokenize_words(text)
    if len(words) <= cfg.max_words and len(text) <= cfg.max_chars:
        return [text.strip()]

    sentences = split_sentences(text)
    segments: List[str] = []
    current: List[str] = []

    def _flush_current():
        if current:
            segments.append(" ".join(current).strip())

    for sent in sentences:
        candidate = (" ".join(current + [sent])).strip() if current else sent
        if (
            len(tokenize_words(candidate)) <= cfg.max_words
            and len(candidate) <= cfg.max_chars
        ):
            current.append(sent)
            continue

        _flush_current()
        current = []

        if len(tokenize_words(sent)) > cfg.max_words or len(sent) > cfg.max_chars:
            segments.extend(chunk_sentence(sent, cfg))
        else:
            current.append(sent)

    _flush_current()
    return [seg for seg in segments if seg]


def count_tatar_letters(text: str, cfg: ProcessingConfig) -> int:
    tatar_set = set(cfg.tatar_letters)
    return sum(1 for ch in text if ch in tatar_set)


def passes_filters(text: str, cfg: ProcessingConfig) -> bool:
    ok, _ = _passes_filters_with_reason(text, cfg)
    return ok


def _passes_filters_with_reason(text: str, cfg: ProcessingConfig) -> tuple[bool, str]:
    letters = LETTER_RE.findall(text)
    if cfg.cyrillic_only and LATIN_LETTER_RE.search(text):
        return False, "latin_present"

    cyr_letters = CYRILLIC_LETTER_RE.findall(text)
    if cfg.cyrillic_only and len(cyr_letters) != len(letters):
        return False, "non_cyrillic_letters"

    if len(cyr_letters) < cfg.min_letters:
        return False, "min_letters"

    if count_tatar_letters(text, cfg) < cfg.min_tatar_letters:
        return False, "min_tatar_letters"

    word_count = len(tokenize_words(text))
    if word_count < cfg.min_words:
        return False, "min_words"
    if word_count > cfg.max_words:
        return False, "max_words"

    char_count = len(text)
    if char_count < cfg.min_chars:
        return False, "min_chars"
    if char_count > cfg.max_chars:
        return False, "max_chars"

    if SINGLE_BRACKET_RE.search(text):
        return False, "single_bracket_placeholder"

    return True, "pass"


def digest_from_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def process_text(
    raw_text: str,
    source_url: str,
    cfg: ProcessingConfig = DEFAULT_CONFIG,
    name_list: Sequence[str] | None = None,
    stats: Counter | None = None,
) -> List[ProcessedSegment]:
    """
    Clean a raw message and return zero or more valid segments.
    """
    def bump(key: str) -> None:
        if stats is not None:
            stats[key] += 1

    bump("rows_total")

    if not raw_text:
        bump("reject_empty")
        return []
    name_list = name_list or []
    text = raw_text.strip()
    if not text or text in cfg.system_messages:
        bump("reject_system_message")
        return []

    if is_contact_only(text):
        bump("reject_contact_only_raw")
        return []

    text = normalize_whitespace(text)

    text = replace_fake_names(text, name_list)
    text = replace_contacts(text)

    if is_contact_only(text):
        bump("reject_contact_only_clean")
        return []

    segments = split_text(text, cfg)
    if stats is not None:
        stats["segments_generated"] += len(segments)
    parent_digest = digest_from_url(source_url)
    processed: List[ProcessedSegment] = []
    for idx, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue
        ok, reason = _passes_filters_with_reason(segment, cfg)
        if not ok:
            bump(f"segment_reject_{reason}")
            continue
        segment_id = f"{parent_digest}#{idx}"
        processed.append(ProcessedSegment(segment_id=segment_id, parent_digest=parent_digest, text=segment))
        bump("segments_kept")
    return processed
