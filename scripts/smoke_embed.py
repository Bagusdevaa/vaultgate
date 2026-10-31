"""bge-m3 smoke test: encode 3 sentences, report dense dim, sparse tokens and RAM."""

import sys
import time

import psutil
import torch

from vault_rag.config import load_config
from vault_rag.retrieve.embed import from_config

SENTENCES = [
    "Hybrid search combines dense vectors with sparse lexical weights.",
    "Wikilinks in Obsidian form a graph between notes.",
    "Pencarian hibrida menggabungkan vektor padat dan bobot leksikal.",
]


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def main() -> int:
    emb = load_config()["embedding"]
    before = rss_gb()

    start = time.perf_counter()
    embedder = from_config(emb)
    embedder.model  # load now so load time and encode time are reported separately
    load_seconds = time.perf_counter() - start
    after_load = rss_gb()

    start = time.perf_counter()
    out = embedder.encode(SENTENCES)
    encode_seconds = time.perf_counter() - start

    print(f"model: {emb['model']}  device: {embedder.device}")
    print(f"load: {load_seconds:.1f}s   encode 3 sentences: {encode_seconds:.2f}s")
    print(f"dense dim: {len(out[0].dense)}")
    for sentence, embedding in zip(SENTENCES, out):
        print(f"  sparse tokens={len(embedding.sparse):3d}  {sentence}")
    print(f"RSS: before={before:.2f} GB  after load={after_load:.2f} GB  after encode={rss_gb():.2f} GB")
    # On Apple Silicon the weights sit in MPS memory, which RSS does not count.
    if torch.backends.mps.is_available():
        print(f"MPS driver memory: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
