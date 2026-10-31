"""Index the demo vault into Qdrant (incremental) and rebuild the link graph.

    uv run python scripts/index_vault.py
    uv run python scripts/index_vault.py --vault /tmp/copy --index-dir /tmp/index

Only notes whose content changed since the last run are embedded again.
"""

import argparse
import sys
from pathlib import Path

from vault_rag import store
from vault_rag.config import REPO_ROOT, load_config
from vault_rag.graph.build import build_graph, save_graph
from vault_rag.ingest.index import index_vault, read_vault
from vault_rag.retrieve.embed import from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, help="vault to index (default: the demo vault)")
    parser.add_argument("--index-dir", type=Path, help="where the index lives (default: config)")
    args = parser.parse_args()

    cfg = load_config()
    vault = args.vault or REPO_ROOT / cfg["paths"]["demo_vault"]
    index_dir = args.index_dir or REPO_ROOT / cfg["paths"]["index_dir"]
    notes = read_vault(vault)
    if not notes:
        print(f"ERROR: no notes in {vault}")
        return 1
    index_dir.mkdir(parents=True, exist_ok=True)

    client = store.open_client(cfg["qdrant"], index_dir)
    stats = index_vault(notes, client, cfg["qdrant"]["collection"], from_config(cfg["embedding"]),
                        index_dir / "manifest.json", cfg["embedding"]["model"], cfg["chunking"])
    graph = build_graph({path: file.text for path, file in notes.items()})
    save_graph(graph, index_dir / "graph.json")
    client.close()

    print(f"notes: {len(notes)}  added {stats.added}, changed {stats.changed}, "
          f"removed {stats.removed}, unchanged {stats.unchanged}"
          + ("  (full rebuild)" if stats.rebuilt else ""))
    print(f"chunks embedded: {stats.chunks_embedded}   chunks in index: {stats.chunks_total}")
    print(f"graph: {graph.number_of_nodes()} notes, {graph.number_of_edges()} links")
    print(f"time: {stats.seconds:.1f}s   index: {index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
