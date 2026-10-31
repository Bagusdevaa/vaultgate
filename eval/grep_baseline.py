"""Baseline that emulates what a coding agent does without this system: grep the vault for
words from the question, then read the best-matching files whole until the budget is full.

No embeddings and no LLM: keywords are the question's words minus stopwords, and a file
scores the summed IDF of the distinct keywords it contains.
"""

import math
import re

from vault_rag.trace import Candidate

STOPWORDS = set("""
yang dan di ke dari itu ini apa apakah kenapa mengapa bagaimana gimana berapa kapan siapa
dimana mana untuk dengan atau juga sudah udah belum bisa ada tidak gak nggak enggak kalau
kalo terus lalu pas saat waktu jadi buat sama aku saya kamu dia kita kami mereka nya sih
kah lagi masih harus perlu pakai pake pernah punya seperti kayak karena soal tentang cara
biar supaya agar dalam pada oleh akan tadi bilang apa aja saja sebenarnya emang memang
the and are was were what why how when which who does did for with that this from into
its has have had not but you your can should would will about after before then than
""".split())

TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+-]*[a-z0-9+]|[a-z0-9]")


def keywords(question: str) -> list[str]:
    """Distinct lowercase tokens of 3+ characters that are not stopwords, in question order."""
    tokens = TOKEN.findall(question.lower())
    return list(dict.fromkeys(t for t in tokens if len(t) >= 3 and t not in STOPWORDS))


def rank_files(notes: dict[str, str], question: str) -> list[tuple[str, float]]:
    """Files containing at least one keyword, best first (ties broken by path)."""
    words = keywords(question)
    lowered = {path: text.lower() for path, text in notes.items()}
    df = {w: sum(w in text for text in lowered.values()) for w in words}
    idf = {w: math.log((len(notes) + 1) / (df[w] + 0.5)) for w in words if df[w]}
    scores = {path: sum(v for w, v in idf.items() if w in text) for path, text in lowered.items()}
    return sorted(((p, s) for p, s in scores.items() if s > 0), key=lambda ps: (-ps[1], ps[0]))


def read_files(notes: dict[str, str], ranking: list[tuple[str, float]],
               max_chars: int) -> list[Candidate]:
    """Whole files in ranking order until max_chars; the file that crosses it is cut."""
    context, used = [], 0
    for path, score in ranking:
        if used >= max_chars:
            break
        text = notes[path][: max_chars - used]
        used += len(text)
        context.append(Candidate(chunk_id=f"{path}#file", note=path, heading_path=[],
                                 text=text, source="grep", rrf_score=score))
    return context
