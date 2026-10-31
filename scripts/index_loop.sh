#!/bin/sh
# Indexer container: re-index the mounted vault every INDEX_INTERVAL_SECONDS.
# Each run is a fresh process, so bge-m3 is only in memory while notes change.
set -u
while true; do
  python scripts/index_vault.py --vault /vault || echo "index run failed; retrying next interval"
  sleep "${INDEX_INTERVAL_SECONDS:-3600}"
done
