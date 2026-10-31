"""Incremental indexing: only notes whose content hash changed are chunked and embedded again."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from vault_rag import store
from vault_rag.ingest.chunk import chunk_note
from vault_rag.ingest.parse import note_tags, resolve_link, split_frontmatter, wikilinks
from vault_rag.retrieve.embed import Embedder


@dataclass
class VaultFile:
    text: str
    mtime: float


@dataclass
class IndexStats:
    added: int
    changed: int
    removed: int
    unchanged: int
    chunks_embedded: int
    chunks_total: int
    rebuilt: bool
    seconds: float


def read_vault(root: Path) -> dict[str, VaultFile]:
    """Notes keyed by vault-relative path (with `.md`), hidden folders skipped."""
    notes = {}
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if not any(part.startswith(".") for part in rel.parts):
            notes[rel.as_posix()] = VaultFile(path.read_text(encoding="utf-8"), path.stat().st_mtime)
    return notes


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(model_id: str, chunking: dict) -> str:
    """Changes whenever stored vectors would differ for the same text."""
    return json.dumps({"model": model_id, "chunking": chunking}, sort_keys=True)


def note_chunks(path: str, file: VaultFile, note_ids: set[str],
                chunking: dict) -> list[tuple[dict, str]]:
    """(payload, text to embed) per chunk. The heading path is embedded with the text."""
    note_id = path.removesuffix(".md")
    meta, body = split_frontmatter(file.text)
    resolved = (resolve_link(target, note_id, note_ids) for target in wikilinks(body))
    # Resolved against the vault as it is now; a note added later does not refresh this
    # list until this note changes. graph.json is rebuilt on every run and stays current.
    links = sorted({f"{r}.md" for r in resolved if r and r != note_id})
    parts = note_id.split("/")
    common = {"note": path, "title": parts[-1], "device": parts[0],
              "folder": parts[-2] if len(parts) > 1 else "", "tags": note_tags(meta, body),
              "links": links, "mtime": file.mtime}
    items = []
    for chunk in chunk_note(path, body, **chunking):
        payload = dict(common, chunk_id=chunk.chunk_id, heading_path=chunk.heading_path,
                       text=chunk.text)
        items.append((payload, " > ".join(chunk.heading_path) + "\n\n" + chunk.text))
    return items


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(path: Path, fp: str, notes: dict[str, dict]) -> None:
    path.write_text(json.dumps({"fingerprint": fp, "notes": notes}, indent=1, sort_keys=True))


def index_vault(notes: dict[str, VaultFile], client: QdrantClient, collection: str,
                embedder: Embedder, manifest_path: Path, model_id: str,
                chunking: dict) -> IndexStats:
    start = time.perf_counter()
    fp = fingerprint(model_id, chunking)
    old = load_manifest(manifest_path)
    rebuilt = old.get("fingerprint") != fp or not client.collection_exists(collection)
    if rebuilt:
        store.reset_collection(client, collection)
    known = {} if rebuilt else old["notes"]

    hashes = {path: sha256(file.text) for path, file in notes.items()}
    removed = sorted(set(known) - set(notes))
    todo = [path for path in sorted(notes) if known.get(path, {}).get("sha256") != hashes[path]]
    # Forget the notes about to change first, so a crash mid-run makes the next run redo them.
    kept = {path: entry for path, entry in known.items() if path in notes and path not in todo}
    save_manifest(manifest_path, fp, kept)
    for path in removed + [p for p in todo if p in known]:
        store.delete_note(client, collection, path)

    note_ids = {path.removesuffix(".md") for path in notes}
    entries = dict(kept)
    payloads, texts = [], []
    for path in todo:
        items = note_chunks(path, notes[path], note_ids, chunking)
        entries[path] = {"sha256": hashes[path], "chunks": len(items)}
        payloads += [payload for payload, _ in items]
        texts += [text for _, text in items]
    store.upsert_chunks(client, collection, payloads, embedder.encode(texts))
    save_manifest(manifest_path, fp, entries)

    added = sum(1 for path in todo if path not in known)
    return IndexStats(added=added, changed=len(todo) - added, removed=len(removed),
                      unchanged=len(kept), chunks_embedded=len(texts),
                      chunks_total=store.count(client, collection), rebuilt=rebuilt,
                      seconds=time.perf_counter() - start)
