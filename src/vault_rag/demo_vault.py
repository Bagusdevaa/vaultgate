"""Select a seed subgraph of the source vault and copy it into the demo vault.

The source vault is only ever read. Writes go to the demo vault directory alone.
"""

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vault_rag.ingest.parse import resolve_link, wikilinks


@dataclass
class VaultGraph:
    notes: dict[str, str]                     # note id (relative path, no .md) -> text
    out: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    back: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    link_count: int = 0

    def degree(self, note: str) -> int:
        return len(self.out[note] | self.back[note])


def load_notes(root: Path) -> dict[str, str]:
    notes = {}
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue  # .obsidian, .git, .claude, ...
        notes[rel.with_suffix("").as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return notes


def build_graph(notes: dict[str, str]) -> VaultGraph:
    graph = VaultGraph(notes)
    ids = set(notes)
    for note, text in notes.items():
        for target in wikilinks(text):
            graph.link_count += 1
            resolved = resolve_link(target, note, ids)
            if resolved is None:
                graph.unresolved.append((note, target))
            elif resolved != note:
                graph.out[note].add(resolved)
                graph.back[resolved].add(note)
    return graph


def is_skipped(note: str, skip_folders: set[str]) -> bool:
    return bool(set(note.split("/")[:-1]) & skip_folders)


def neighbors(graph: VaultGraph, note: str, cap: int, skip_folders: set[str]) -> list[str]:
    """Links and backlinks of a note, most connected first, name as tie-break."""
    candidates = [n for n in graph.out[note] | graph.back[note] if not is_skipped(n, skip_folders)]
    return sorted(candidates, key=lambda n: (-graph.degree(n), n))[:cap]


def select_subgraph(graph: VaultGraph, seeds: list[str], cap: int,
                    skip_folders: set[str]) -> set[str]:
    missing = [s for s in seeds if s not in graph.notes]
    if missing:
        raise ValueError(f"seed notes not found: {missing}")
    selected = set(seeds)
    for seed in seeds:
        selected.update(neighbors(graph, seed, cap, skip_folders))
    return selected


def check_destination(source: Path, dest: Path, allowed: Path) -> None:
    source, dest, allowed = source.resolve(), dest.resolve(), allowed.resolve()
    if dest != allowed:
        raise ValueError(f"refusing to write outside {allowed}: {dest}")
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        raise ValueError("source vault and demo vault must not overlap")


def copy_subgraph(source: Path, dest: Path, allowed: Path, selected: set[str]) -> None:
    check_destination(source, dest, allowed)
    dest.mkdir(parents=True, exist_ok=True)
    for child in dest.iterdir():  # clear previous build, keep the tracked placeholder
        if child.name == ".gitkeep":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for note in sorted(selected):
        target = dest / f"{note}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / f"{note}.md", target)


def broken_links(graph: VaultGraph, selected: set[str]) -> list[tuple[str, str, str]]:
    """Links from selected notes whose target is not in the demo vault."""
    ids = set(graph.notes)
    broken = []
    for note in sorted(selected):
        for target in wikilinks(graph.notes[note]):
            resolved = resolve_link(target, note, ids)
            if resolved is None:
                broken.append((note, target, "unresolved in source vault"))
            elif resolved not in selected:
                broken.append((note, target, f"outside demo vault ({resolved})"))
    return broken
