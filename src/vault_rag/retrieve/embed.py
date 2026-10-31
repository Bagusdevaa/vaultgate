"""bge-m3 embeddings: one dense vector and one sparse (lexical) vector per text."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Embedding:
    dense: list[float]
    sparse: dict[int, float]  # token id -> weight


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[Embedding]: ...


class BgeM3Embedder:
    """Loads the model on first use, so an index run with nothing to embed costs nothing."""

    def __init__(self, model_id: str, use_fp16: bool, max_length: int, batch_size: int = 16):
        self.model_id, self.use_fp16 = model_id, use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Imported here so tests and tools that never embed skip loading torch.
            from FlagEmbedding import BGEM3FlagModel

            from vault_rag.hf import local_model_path

            self._model = BGEM3FlagModel(local_model_path(self.model_id), use_fp16=self.use_fp16)
        return self._model

    @property
    def device(self) -> str:
        return str(next(self.model.model.parameters()).device)

    def encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        out = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length,
                                return_dense=True, return_sparse=True)
        return [Embedding(dense.tolist(), {int(t): float(w) for t, w in weights.items()})
                for dense, weights in zip(out["dense_vecs"], out["lexical_weights"])]


def from_config(emb: dict) -> BgeM3Embedder:
    return BgeM3Embedder(emb["model"], emb["use_fp16"], emb["max_length"])
