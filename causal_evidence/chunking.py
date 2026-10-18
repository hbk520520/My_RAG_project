"""Token windows with exact source offsets and valid Unicode boundaries."""
from __future__ import annotations

import calendar
import re
from datetime import date

from .models import EvidenceRecord, TextChunk


def parse_timestamp(text: str) -> date | None:
    """Only unambiguous explicit dates; month precision uses its latest possible day."""
    dates = set()
    for match in re.finditer(r"(\d{4})[-年](\d{1,2})(?:[-月](\d{1,2})日?|月)", text):
        year, month = int(match[1]), int(match[2])
        try:
            day = int(match[3]) if match[3] else calendar.monthrange(year, month)[1]
            dates.add(date(year, month, day))
        except ValueError:
            return None
    return dates.pop() if len(dates) == 1 else None


def admissible(record: EvidenceRecord | TextChunk, cutoff: date) -> bool:
    if record.timestamp is not None and record.timestamp >= cutoff:
        return False
    if record.available_at is not None and record.available_at >= cutoff:
        return False
    if record.timestamp is None:
        return (record.availability_verified and record.available_at is not None
                and record.available_at < cutoff)
    return True


def chunk_record(record: EvidenceRecord, *, chunk_size: int = 512,
                 overlap: int = 64, encoding=None) -> list[TextChunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if encoding is None:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(record.text, disallowed_special=())
    # A token can contain only part of a Chinese character. Only split at boundaries
    # which are also UTF-8 character boundaries; do not decode replacement characters.
    byte_to_char = {0: 0}
    offset = 0
    for index, char in enumerate(record.text):
        offset += len(char.encode("utf-8"))
        byte_to_char[offset] = index + 1
    boundaries = {0: 0}
    offset = 0
    for index, token in enumerate(tokens):
        offset += len(encoding.decode_single_token_bytes(token))
        if offset in byte_to_char:
            boundaries[index + 1] = byte_to_char[offset]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        while end not in boundaries and end > start:
            end -= 1
        if end == start:
            raise ValueError("chunk_size cannot hold one complete Unicode character")
        left, right = boundaries[start], boundaries[end]
        text = record.text[left:right]
        chunks.append(TextChunk(
            chunk_id=f"{record.record_id}:{start}:{end}", record_id=record.record_id,
            source_id=record.source_id,
            document_id=record.document_id, source_file=record.source_file,
            page_start=record.page_start, page_end=record.page_end,
            text=text, char_start=left, char_end=right, token_start=start, token_end=end,
            timestamp=record.timestamp or parse_timestamp(text),
            available_at=record.available_at, availability_verified=record.availability_verified,
            similarity=record.similarity,
        ))
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
        while start not in boundaries:
            start += 1
    return chunks
