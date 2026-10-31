"""Validate a golden question file (jsonl) against the demo vault.

    uv run python scripts/validate_golden.py data/golden/draft.jsonl

Checks structure, that gold notes exist, that every evidence quote appears verbatim in its
note, that each hop_path step is a real wikilink or backlink, and that no text field holds
anything the sensitive-data scanner would flag. Exits 1 on any error. Reports counts per
category and per tag.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from vault_rag.config import REPO_ROOT, load_config
from vault_rag.ingest.parse import resolve_link, wikilinks
from vault_rag.sensitive import deny_pattern, load_terms, scan_text

CATEGORIES = ("keyword", "semantic", "multihop", "unanswerable")
TAGS = ("weak_hop", "general_knowledge_risk", "conflicting_notes", "answer_is_unknown")
REQUIRED = {"id": str, "category": str, "question": str, "gold_notes": list, "evidence": list,
            "hop_path": list, "reference_answer": str, "draft_notes": str, "tags": list}
# A note that is gold for many questions skews the set; listing every note that holds an
# answer can push past this, so it warns instead of failing.
MAX_QUESTIONS_PER_NOTE = 3


def load_vault(vault: Path) -> dict[str, str]:
    return {p.relative_to(vault).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(vault.rglob("*.md"))}


def link_edges(notes: dict[str, str]) -> set[tuple[str, str]]:
    """Directed (source, target) pairs between note paths, resolved like the ingest step."""
    ids = {path.removesuffix(".md") for path in notes}
    edges = set()
    for path, text in notes.items():
        source = path.removesuffix(".md")
        for target in wikilinks(text):
            resolved = resolve_link(target, source, ids)
            if resolved and resolved != source:
                edges.add((path, resolved + ".md"))
    return edges


def check_links(item: dict, edges: set[tuple[str, str]]) -> list[str]:
    path = item["hop_path"]
    if item["category"] != "multihop":
        return ["hop_path is only for multihop items"] if path else []
    if len(path) < 2:
        return ["multihop item needs a hop_path of at least 2 notes"]
    errors = [f"no wikilink or backlink between {a} and {b}"
              for a, b in zip(path, path[1:]) if (a, b) not in edges and (b, a) not in edges]
    if path[-1] not in item["gold_notes"]:
        errors.append(f"hop_path ends at {path[-1]}, which is not a gold note")
    return errors


def check_evidence(item: dict, notes: dict[str, str]) -> list[str]:
    errors = [f"gold note not in vault: {note}" for note in item["gold_notes"] if note not in notes]
    errors += [f"hop_path note not in vault: {note}" for note in item["hop_path"] if note not in notes]
    for ev in item["evidence"]:
        note, quote = ev.get("note"), ev.get("quote")
        if note not in notes:
            errors.append(f"evidence note not in vault: {note}")
        elif not quote or quote not in notes[note]:
            errors.append(f"quote not found verbatim in {note}: {quote!r}")
    if item["category"] == "unanswerable":
        if item["gold_notes"] or item["evidence"]:
            errors.append("unanswerable item must have empty gold_notes and evidence")
        return errors
    if not item["gold_notes"]:
        errors.append("answerable item has no gold_notes")
    cited = {ev.get("note") for ev in item["evidence"]}
    errors += [f"gold note has no evidence quote: {n}" for n in item["gold_notes"] if n not in cited]
    return errors


def check_sensitive(item: dict, deny) -> list[str]:
    texts = [item["question"], item["reference_answer"], item["draft_notes"]]
    texts += [ev.get("quote") or "" for ev in item["evidence"]]
    return [f"sensitive {f.category}: {f.match!r}"
            for text in texts for f in scan_text(item["id"], text, deny)]


def check_item(item: dict, notes: dict[str, str], edges: set, deny) -> list[str]:
    errors = [f"field {name!r} missing or not a {kind.__name__}"
              for name, kind in REQUIRED.items() if not isinstance(item.get(name), kind)]
    if errors:
        return errors
    if item["category"] not in CATEGORIES:
        return [f"unknown category {item['category']!r}"]
    errors = [f"unknown tag {tag!r}" for tag in item["tags"] if tag not in TAGS]
    errors += [f"duplicate tag {tag!r}" for tag, n in Counter(item["tags"]).items() if n > 1]
    if errors:
        return errors
    return check_evidence(item, notes) + check_links(item, edges) + check_sensitive(item, deny)


def check_duplicates(items: list[dict]) -> list[str]:
    errors = []
    for key in ("id", "question"):
        counts = Counter(item.get(key) for item in items)
        errors += [f"duplicate {key}: {value!r}" for value, n in counts.items() if n > 1]
    return errors


def note_warnings(items: list[dict]) -> list[str]:
    per_note = Counter(note for item in items for note in item.get("gold_notes") or [])
    return [f"{note} is gold for {n} questions (more than {MAX_QUESTIONS_PER_NOTE})"
            for note, n in sorted(per_note.items()) if n > MAX_QUESTIONS_PER_NOTE]


def print_counts(items: list[dict], vault_size: int) -> None:
    categories = Counter(item.get("category") for item in items)
    tags = Counter(tag for item in items for tag in item.get("tags") or [])
    untagged = sum(1 for item in items if not item.get("tags"))
    covered = {note for item in items for note in item.get("gold_notes") or []}
    print(f"{len(items)} questions: " + ", ".join(f"{c} {categories[c]}" for c in CATEGORIES))
    print("tags: " + ", ".join(f"{t} {tags[t]}" for t in TAGS) + f", untagged {untagged}")
    print(f"gold notes covered: {len(covered)} of {vault_size}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("golden", type=Path, help="jsonl file to validate")
    args = parser.parse_args()

    cfg = load_config()
    notes = load_vault(REPO_ROOT / cfg["paths"]["demo_vault"])
    if not notes:
        print("ERROR: demo vault is empty, run scripts/build_demo_vault.py build first")
        return 1
    terms_file = REPO_ROOT / cfg["demo_vault"]["deny_terms_file"]
    terms = load_terms(terms_file.read_text()) if terms_file.exists() else []
    if not terms:
        print(f"WARNING: no deny terms loaded from {terms_file.relative_to(REPO_ROOT)}")
    deny, edges = deny_pattern(terms), link_edges(notes)

    items, errors = [], []
    for number, line in enumerate(args.golden.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        items.append(item)
        errors += [f"line {number} ({item.get('id')}): {e}" for e in check_item(item, notes, edges, deny)]
    errors += check_duplicates(items)

    print_counts(items, len(notes))
    for warning in note_warnings(items):
        print(f"WARNING {warning}")
    for error in errors:
        print(f"ERROR {error}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
