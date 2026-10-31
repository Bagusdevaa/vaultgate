"""Survey the source vault, or copy the configured seed subgraph into the demo vault.

    uv run python scripts/build_demo_vault.py survey
    uv run python scripts/build_demo_vault.py build

Prints aggregates only. The source vault is read, never written.
"""

import argparse
import sys
from collections import Counter

from vault_rag.config import REPO_ROOT, load_config
from vault_rag.demo_vault import (
    VaultGraph, broken_links, build_graph, copy_subgraph, is_skipped, load_notes,
    neighbors, select_subgraph,
)


def internal_links(graph: VaultGraph, selected: set[str]) -> int:
    return sum(len(graph.out[n] & selected) for n in selected)


def survey(graph: VaultGraph, demo: dict) -> None:
    skip = set(demo["skip_folders"])
    notes = graph.notes
    print(f"notes: {len(notes)}   wikilinks: {graph.link_count}   "
          f"unresolved: {len(graph.unresolved)}")
    linked = sum(1 for n in notes if graph.degree(n))
    print(f"notes with any link: {linked}   mean degree: "
          f"{sum(graph.degree(n) for n in notes) / len(notes):.1f}")

    print("\nnotes per folder:")
    folders = Counter("/".join(n.split("/")[:-1]) or "(root)" for n in notes)
    for folder, count in sorted(folders.items()):
        print(f"  {count:4d}  {folder}")

    print(f"\nlargest 1-hop neighborhoods among Projects (excluding {sorted(skip)}):")
    projects = [n for n in notes if "/Projects/" in f"/{n}"]
    sizes = {p: len(neighbors(graph, p, 10_000, skip)) for p in projects}
    for project, size in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        print(f"  {size:3d}  {project}")

    selected = select_subgraph(graph, demo["seeds"], demo["max_neighbors_per_note"], skip)
    print(f"\nconfigured seeds: {len(demo['seeds'])}   selected notes: {len(selected)}   "
          f"internal links: {internal_links(graph, selected)}")


def write_build_report(graph: VaultGraph, selected: set[str], path) -> int:
    broken = broken_links(graph, selected)
    folders = Counter(n.split("/")[-2] if "/" in n else "(root)" for n in selected)
    lines = ["# Demo vault build report", "",
             f"- Notes copied: {len(selected)}",
             f"- Internal links: {internal_links(graph, selected)}",
             f"- Broken links: {len(broken)}", "",
             "## Notes per folder", "", "| Folder | Notes |", "|---|---|"]
    lines += [f"| {f} | {c} |" for f, c in sorted(folders.items())]
    lines += ["", "## Broken links", "", "| Note | Link target | Why |", "|---|---|---|"]
    lines += [f"| {src} | `{target}` | {why} |" for src, target, why in broken]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(broken)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["survey", "build"])
    args = parser.parse_args()

    cfg = load_config()
    if cfg.vault_source_path is None:
        print("ERROR: set VAULT_SOURCE_PATH in .env")
        return 1
    demo = cfg["demo_vault"]
    graph = build_graph(load_notes(cfg.vault_source_path))

    if args.command == "survey":
        survey(graph, demo)
        return 0

    skip = set(demo["skip_folders"])
    selected = select_subgraph(graph, demo["seeds"], demo["max_neighbors_per_note"], skip)
    assert not any(is_skipped(n, skip) for n in selected)
    dest = REPO_ROOT / cfg["paths"]["demo_vault"]
    copy_subgraph(cfg.vault_source_path, dest, REPO_ROOT / "data" / "demo-vault", selected)
    report = REPO_ROOT / cfg["paths"]["reports_dir"] / "build_report.md"
    broken = write_build_report(graph, selected, report)
    print(f"copied {len(selected)} notes to {dest.relative_to(REPO_ROOT)}")
    print(f"broken links: {broken}   report: {report.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
