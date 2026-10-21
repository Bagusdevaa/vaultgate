"""bge-m3 smoke test: encode 3 sentences, report dense dim, sparse tokens and RAM."""

import sys
import time

import psutil
from FlagEmbedding import BGEM3FlagModel

from vault_rag.config import load_config

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
    model = BGEM3FlagModel(emb["model"], use_fp16=emb["use_fp16"])
    load_seconds = time.perf_counter() - start
    after_load = rss_gb()

    start = time.perf_counter()
    out = model.encode(SENTENCES, return_dense=True, return_sparse=True, max_length=emb["max_length"])
    encode_seconds = time.perf_counter() - start

    print(f"model: {emb['model']}  device: {model.model.device}")
    print(f"load: {load_seconds:.1f}s   encode 3 sentences: {encode_seconds:.2f}s")
    print(f"dense dim: {out['dense_vecs'].shape[1]}")
    for sentence, weights in zip(SENTENCES, out["lexical_weights"]):
        print(f"  sparse tokens={len(weights):3d}  {sentence}")
    print(f"RSS: before={before:.2f} GB  after load={after_load:.2f} GB  after encode={rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
