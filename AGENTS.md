# vault-rag — Agent Notes

mode: solo
base branch: main

## Test checklist (must pass before merging to main)

- `make test`
- `make smoke` (Jev and OpenRouter checks print `SKIPPED` without keys)

## Hotspots (conflict-prone files — check before touching)

| File | Owner | Resolution precedent |
|------|-------|----------------------|

## Tech debt

## Repo-specific notes

- `PLAN.md` is the phase-by-phase plan. Stop for review at the end of every phase.
- The source vault (`VAULT_SOURCE_PATH`, Deva's PersonalBrain) is read-only. Copies go to
  `data/demo-vault/` only.
- Nothing from the vault goes to Jev or OpenRouter until Deva has reviewed
  `flagged_report.md` (Phase 1).
- Model slugs in `config.yaml` are frozen once evaluation starts.
