from qdrant_client import QdrantClient

from vault_rag.ingest.index import VaultFile, index_vault
from vault_rag.retrieve.embed import Embedding

CHUNKING = {"min_chars": 50, "max_chars": 400, "overlap_chars": 50}
BODY = "word " * 30


class CountingEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> list[Embedding]:
        self.texts += texts
        return [Embedding([1.0] + [0.0] * 1023, {len(t) % 50: 1.0}) for t in texts]


def vault() -> dict[str, VaultFile]:
    return {
        "dev/Projects/p.md": VaultFile(f"---\ntags: [project]\n---\n# p\n[[Bugs/p]]\n{BODY}\n## Next\n{BODY}", 1.0),
        "dev/Bugs/p.md": VaultFile(f"# p\n[[Projects/p]] #bug\n{BODY}", 2.0),
        "dev/Knowledge/k.md": VaultFile(f"# k\n{BODY}", 3.0),
    }


def run(notes, client, embedder, manifest, chunking=CHUNKING):
    return index_vault(notes, client, "vault", embedder, manifest, "fake-model", chunking)


def test_full_index_then_rerun_embeds_nothing(tmp_path):
    client, manifest = QdrantClient(":memory:"), tmp_path / "manifest.json"
    first = CountingEmbedder()
    stats = run(vault(), client, first, manifest)
    assert (stats.added, stats.rebuilt) == (3, True)
    assert stats.chunks_embedded == stats.chunks_total == len(first.texts) == 4

    again = CountingEmbedder()
    stats = run(vault(), client, again, manifest)
    assert again.texts == []
    assert (stats.unchanged, stats.chunks_embedded, stats.chunks_total) == (3, 0, 4)


def test_editing_one_note_reembeds_only_that_note(tmp_path):
    client, manifest = QdrantClient(":memory:"), tmp_path / "manifest.json"
    run(vault(), client, CountingEmbedder(), manifest)
    notes = vault()
    notes["dev/Knowledge/k.md"] = VaultFile(f"# k\n{BODY} edited", 4.0)
    embedder = CountingEmbedder()
    stats = run(notes, client, embedder, manifest)
    assert (stats.changed, stats.unchanged, stats.chunks_embedded, stats.chunks_total) == (1, 2, 1, 4)
    assert embedder.texts[0].startswith("k\n\n# k")  # heading path is embedded with the text


def test_removed_note_loses_its_points(tmp_path):
    client, manifest = QdrantClient(":memory:"), tmp_path / "manifest.json"
    run(vault(), client, CountingEmbedder(), manifest)
    notes = vault()
    del notes["dev/Projects/p.md"]
    stats = run(notes, client, CountingEmbedder(), manifest)
    assert (stats.removed, stats.chunks_total) == (1, 2)


def test_changed_chunk_settings_rebuild_everything(tmp_path):
    client, manifest = QdrantClient(":memory:"), tmp_path / "manifest.json"
    run(vault(), client, CountingEmbedder(), manifest)
    stats = run(vault(), client, CountingEmbedder(), manifest, dict(CHUNKING, max_chars=500))
    assert stats.rebuilt and stats.chunks_embedded == stats.chunks_total


def test_payload_carries_metadata(tmp_path):
    client = QdrantClient(":memory:")
    run(vault(), client, CountingEmbedder(), tmp_path / "manifest.json")
    points, _ = client.scroll("vault", limit=10)
    bug = next(p.payload for p in points if p.payload["note"] == "dev/Bugs/p.md")
    assert bug["chunk_id"] == "dev/Bugs/p.md#0"
    assert (bug["device"], bug["folder"], bug["title"], bug["mtime"]) == ("dev", "Bugs", "p", 2.0)
    assert bug["links"] == ["dev/Projects/p.md"] and bug["tags"] == ["bug"]
    assert bug["heading_path"] == ["p"]
