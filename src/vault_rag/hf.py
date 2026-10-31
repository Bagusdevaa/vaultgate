"""Resolve Hugging Face models to a local folder before loading them."""

from huggingface_hub import snapshot_download

# FlagEmbedding downloads the whole repo when given a model id. For bge-m3 that
# includes a 2.3 GB ONNX export we never load, so fetch only what PyTorch needs.
SKIP_PATTERNS = ["onnx/*", "imgs/*", "*.onnx", "*.onnx_data"]


def local_model_path(model_id: str) -> str:
    return snapshot_download(model_id, ignore_patterns=SKIP_PATTERNS)


if __name__ == "__main__":
    # Fetch the configured models into HF_HOME (the Docker deploy runs this once).
    from vault_rag.config import load_config

    cfg = load_config()
    for model_id in (cfg["embedding"]["model"], cfg["gate"]["bge"]["model"]):
        print(model_id, "->", local_model_path(model_id))
