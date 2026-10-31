"""Split a note body into chunks, one per heading section.

Short sections merge into the next one, long sections split on paragraphs with overlap.
Sizes are in characters: simpler to reason about than tokens, and close enough here.
"""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_MARKERS = ("```", "~~~")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    note: str
    index: int
    heading_path: list[str]
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.note}#{self.index}"


def sections(body: str) -> list[tuple[list[str], str]]:
    """(heading path, text) per heading; the text keeps its heading line. Code is not parsed."""
    stack: list[tuple[int, str]] = []
    result: list[tuple[list[str], list[str]]] = [([], [])]
    in_fence = False
    for line in HTML_COMMENT.sub("", body).splitlines():
        if line.lstrip().startswith(FENCE_MARKERS):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [(lvl, title) for lvl, title in stack if lvl < level] + [(level, match.group(2))]
            result.append(([title for _, title in stack], []))
        result[-1][1].append(line)
    return [(path, "\n".join(lines).strip()) for path, lines in result if "\n".join(lines).strip()]


def merge_short(parts: list[tuple[list[str], str]], min_chars: int) -> list[tuple[list[str], str]]:
    """Prepend each section shorter than min_chars to the next; a short last one joins the previous."""
    merged: list[tuple[list[str], str]] = []
    carry = ""
    for path, text in parts:
        text = f"{carry}\n\n{text}" if carry else text
        carry = ""
        if len(text) < min_chars:
            carry = text
            continue
        merged.append((path, text))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], f"{merged[-1][1]}\n\n{carry}")
        else:
            merged.append((parts[-1][0], carry))
    return merged


def tail(text: str, size: int) -> str:
    """The last `size` characters, starting at a word boundary."""
    end = text[-size:]
    if len(text) > size and " " in end:
        end = end[end.index(" ") + 1:]
    return end


def split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most max_chars, each starting with the previous one's tail."""
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap_chars
    paragraphs = []
    for para in PARAGRAPH.split(text):
        paragraphs += [para[i:i + step] for i in range(0, len(para), step)] if len(para) > max_chars else [para]
    pieces, current = [], ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        overlapped = f"{tail(current, overlap_chars)}\n\n{para}"
        current = overlapped if len(overlapped) <= max_chars else para
    pieces.append(current)
    return pieces


def chunk_note(note: str, body: str, min_chars: int, max_chars: int,
               overlap_chars: int) -> list[Chunk]:
    chunks = []
    for path, text in merge_short(sections(body), min_chars):
        for piece in split_long(text, max_chars, overlap_chars):
            chunks.append(Chunk(note, len(chunks), path, piece))
    return chunks
