"""Parse Obsidian markdown: wikilinks and how they resolve to notes."""

import re

FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
# Optional `!` marks an embed. Target stops at `|` (alias) or `#` (heading/block).
WIKILINK = re.compile(r"!?\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def wikilinks(text: str) -> list[str]:
    """Link targets in order of appearance, without alias/heading, code excluded."""
    text = INLINE_CODE.sub("", FENCE.sub("", text))
    targets = []
    for match in WIKILINK.finditer(text):
        target = match.group(1).strip().removesuffix(".md")
        if target:  # [[#heading]] points inside the same note
            targets.append(target)
    return targets


def resolve_link(target: str, source: str, note_ids: set[str]) -> str | None:
    """Map a link target to a note id (vault-relative path without `.md`).

    PersonalBrain keeps one folder per device (`bagusdeva/`, `deva/`) with the same
    subfolders, and links are written relative to the device folder
    (`[[Projects/pbpipe]]`). So try the source's own device first, then any other top
    folder, then a vault-wide path-suffix match if it is unique.
    """
    device = source.split("/", 1)[0]
    if f"{device}/{target}" in note_ids:
        return f"{device}/{target}"
    if target in note_ids:
        return target
    tops = sorted({n.split("/", 1)[0] for n in note_ids} - {device})
    for top in tops:
        if f"{top}/{target}" in note_ids:
            return f"{top}/{target}"
    matches = [n for n in note_ids if n.endswith("/" + target)]
    return matches[0] if len(matches) == 1 else None
