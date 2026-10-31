# Deploy to the home server

The whole system runs from `docker-compose.yml`: Qdrant, the API (bge-m3 and the lazy
bge reranker in-process), the Streamlit inspector, and an indexer that re-indexes the
mounted vault every hour. The vault is mounted read-only.

## Resources

Measured on the Mac (Docker Desktop, 8 GB VM, CPU only), the running stack uses about 2.6 GB:
API 2.0 GB (bge-m3, fp32), Qdrant 350 MB, indexer 220 MB between runs, inspector 50 MB. The bge
reranker adds ~2.3 GB only after the first `gate: bge` request. The indexer loads bge-m3
only while notes change, in a short-lived process. hp-server has 14 GiB with ~12 GiB
available, so both gates fit.

## Run it anywhere

```sh
docker compose build
docker compose run --rm -e HF_HUB_OFFLINE=0 indexer python -m vault_rag.hf   # models, once
docker compose up -d
curl http://127.0.0.1:8000/health          # API
open http://127.0.0.1:8501                 # inspector
```

Settings come from environment variables (or a `.env` next to the compose file):

| Variable | Default | Home server |
|---|---|---|
| `OPENROUTER_API_KEY` | – | required (Jev + generator) |
| `VAULTGATE_BIND` | `127.0.0.1` | `10.22.82.200` (ZeroTier) |
| `VAULTGATE_VAULT` | `./data/demo-vault` | `/data/vaultgate/demo-vault` |
| `VAULTGATE_MODELS` | `./.cache/hf` | `/data/vaultgate/models` |
| `VAULTGATE_INDEX` | `./data/docker/index` | `/data/vaultgate/index` |
| `VAULTGATE_QDRANT_DATA` | `./data/docker/qdrant` | `/data/vaultgate/qdrant` |
| `VAULTGATE_GOLDEN` | `./data/golden` | `/data/vaultgate/golden` (optional) |
| `VAULTGATE_INDEX_INTERVAL` | `3600` | `3600` |

## hp-server (follows `/data/homelab/AGENTS.md`)

The homelab rules shape the deploy:
- every service is a stack under `stacks/<name>/` in the homelab repo;
- state is bind-mounted under `/data/<service>/`;
- secrets live in a gitignored `.env` next to the stack;
- ports bind to the ZeroTier address only.

1. Code: clone the repo to `/data/vaultgate-src` (after the branches are pushed).
2. Data: copy the reviewed demo vault and, optionally, the golden files. Never mount
   `/data/vault`: only the sanitized demo vault was cleared for external APIs.
   ```sh
   ssh hp-server 'mkdir -p /data/vaultgate/{demo-vault,models,index,qdrant,golden} && sudo chown 10001 /data/vaultgate/index'
   rsync -a data/demo-vault/ hp-server:/data/vaultgate/demo-vault/
   rsync -a data/golden/{dev,test}.jsonl hp-server:/data/vaultgate/golden/
   ```
3. Models: copy them over the LAN instead of downloading them on the home connection:
   ```sh
   rsync -a ~/.cache/huggingface/hub/models--BAAI--bge-m3 ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3 hp-server:/data/vaultgate/models/hub/
   ```
4. Stack: `stacks/vaultgate/compose.yaml` in the homelab repo, through a PR:
   ```yaml
   # vaultgate: RAG over the demo vault. ZeroTier only (VAULTGATE_BIND in .env).
   include:
     - path: /data/vaultgate-src/docker-compose.yml
       env_file: .env
   ```
   and a gitignored `stacks/vaultgate/.env` (chmod 600), with this content:
   ```sh
   OPENROUTER_API_KEY=...
   VAULTGATE_BIND=10.22.82.200
   VAULTGATE_VAULT=/data/vaultgate/demo-vault
   VAULTGATE_MODELS=/data/vaultgate/models
   VAULTGATE_INDEX=/data/vaultgate/index
   VAULTGATE_QDRANT_DATA=/data/vaultgate/qdrant
   VAULTGATE_GOLDEN=/data/vaultgate/golden
   ```
5. Start it and run the homelab checklist:
   ```sh
   cd /data/homelab/stacks/vaultgate && docker compose up -d --build
   docker compose ps                        # all services running / healthy
   sudo ss -tlnp | grep -E ':(8000|8501)'   # 10.22.82.200 only, never 0.0.0.0
   ```
6. Laptop: point the MCP tool at the server by setting `VAULTGATE_API_URL=http://10.22.82.200:8000`
   in the repo's `.env`. The inspector is at `http://10.22.82.200:8501`.

## Known limits

- The API has no authentication. It is reachable only on ZeroTier, so every device on
  that network can query the demo vault.
- The API loads the link graph at startup. After the indexer adds or removes links, run
  `docker compose restart api` to pick up the new graph; new chunks are searchable
  right away.
- Embedding and the bge reranker run on CPU in the container. A full index of the demo
  vault took 277 s on the Mac's Docker VM (33 s natively on MPS); after that only
  changed notes are embedded. A Jev-gated `/ask` took 3.6 s end to end.
