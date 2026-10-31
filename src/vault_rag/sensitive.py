"""Regex scan for data that must not leave the machine. Flags only, never redacts."""

import ipaddress
import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"(?<![\w.])(?:\+62|62|0)8\d{2}[-\s]?\d{3,4}[-\s]?\d{3,5}(?!\d)"
        r"|(?<![\w.])\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"
    ),
    "api_key": re.compile(
        r"\b(?:sk-or-v1-[A-Za-z0-9]{20,}|sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"
        r"|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|xox[abposr]-[A-Za-z0-9-]{10,}"
        r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}"
        r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
        r"|Bearer\s+[A-Za-z0-9._~+/-]{16,}"
    ),
    "credential": re.compile(
        # Letter boundaries (not \b) so env names like JEV_API_KEY= match but "tokens:" does not.
        r"(?i)(?<![a-z])(?:api[_-]?key|token|secret|password|passwd|pwd)(?![a-z])"
        r"\s*[:=]\s*['\"]?[^\s'\"`]{6,}"
    ),
    "ip": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "internal_url": re.compile(
        r"https?://(?:localhost|[\w.-]+\.(?:local|internal|lan|home|ts\.net)"
        r"|(?:\d{1,3}\.){3}\d{1,3})(?::\d+)?[^\s)\]>`]*"
        r"|\bssh\s+(?:-\S+\s+)*[\w.-]+@[\w.-]+"
    ),
    "mac_address": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "home_path": re.compile(r"(?:/Users|/home)/[A-Za-z][\w.-]*"),
    "currency": re.compile(
        r"(?i)(?:\bRp\.?\s?\d[\d.,]*|\b(?:IDR|USD)\s?\d[\d.,]*|[$€]\s?\d[\d.,]*(?:\s?[kmb]\b)?"
        r"|\b\d[\d.,]*\s?(?:juta|jt|miliar|milyar|ribu|rb)\b)"
    ),
}
SECRET_CATEGORIES = {"api_key", "credential"}
# Shell code is full of `$1`/`$0`, so amounts are only looked for outside code.
PROSE_ONLY = {"currency"}
FENCE_MARKERS = ("```", "~~~")
INLINE_CODE = re.compile(r"`[^`]*`")


@dataclass
class Finding:
    path: str
    line: int
    category: str
    match: str
    snippet: str


def mask(secret: str) -> str:
    return secret[:4] + "…"


def ip_kind(text: str) -> str | None:
    """'private'/'public' for a valid IPv4 address, None for things like version numbers."""
    try:
        address = ipaddress.IPv4Address(text)
    except ValueError:
        return None
    shared = ipaddress.IPv4Network("100.64.0.0/10")  # CGNAT, used by Tailscale
    return "private" if address.is_private or address in shared else "public"


def safe_snippet(line: str, width: int = 160) -> str:
    """The line with every secret-looking match masked, trimmed for the report."""
    for category in SECRET_CATEGORIES:
        line = PATTERNS[category].sub(lambda m: mask(m.group(0)), line)
    line = line.strip().replace("|", "\\|")
    return line if len(line) <= width else line[: width - 1] + "…"


def deny_pattern(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?i)(?<!\w)(?:{alternatives})(?!\w)")


def scan_text(path: str, text: str, deny: re.Pattern | None = None) -> list[Finding]:
    patterns = dict(PATTERNS, **({"deny_term": deny} if deny else {}))
    findings = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(FENCE_MARKERS):
            in_fence = not in_fence
        for category, pattern in patterns.items():
            searched = line
            if category in PROSE_ONLY:
                if in_fence:
                    continue
                searched = INLINE_CODE.sub("", line)
            for match in pattern.finditer(searched):
                found, label = match.group(0), category
                if category == "ip":
                    kind = ip_kind(found)
                    if kind is None:
                        continue
                    label = f"ip_{kind}"
                if category in SECRET_CATEGORIES:
                    found = mask(found)
                findings.append(Finding(path, number, label, found, safe_snippet(line)))
    return findings


def load_terms(text: str) -> list[str]:
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]
