"""Jev smoke test via OpenRouter's Decisions API: per-passage relevance, then sufficiency."""

import sys

from vault_rag.config import load_config
from vault_rag.gate.jev_gate import JevCall, JevClient

QUERY = "How long do I have to return an online order to ACME Shop?"
DOCUMENTS = [
    "ACME Shop accepts online returns within 30 days of delivery.",
    "ACME Shop in-store purchases can be returned within 14 days of purchase.",
    "ACME Shop products come with a one-year repair warranty.",
    "FooBar Shop accepts online returns within 60 days of delivery.",
    "Bread is made from flour, water, yeast and salt.",
]
# Related to the query but unable to answer it: a correct gate should abstain on these.
NO_ANSWER = [DOCUMENTS[1], DOCUMENTS[3]]


def describe(call: JevCall) -> str:
    u = call.usage
    return (f"{call.seconds:.2f}s, {call.model}, input={u.get('input_tokens')} "
            f"output={u.get('output_tokens')} cost=${u.get('cost')}")


def main() -> int:
    cfg = load_config()
    if not cfg.openrouter_api_key:
        print("SKIPPED: OPENROUTER_API_KEY not set")
        return 0
    jev = cfg["gate"]["jev"]
    client = JevClient(cfg.openrouter_api_key, jev["model"], jev["endpoint"])

    relevance = client.relevance(QUERY, DOCUMENTS)
    print(f"== relevance, threshold {jev['threshold']} ({describe(relevance)})")
    kept = []
    for text, score in zip(DOCUMENTS, relevance.scores):
        keep = score >= jev["threshold"]
        kept += [text] if keep else []
        print(f"  {score:.3f}  {'keep' if keep else 'drop'}  {text}")

    cutoff = jev["sufficiency_threshold"]
    on_kept = client.sufficiency(QUERY, kept)
    on_none = client.sufficiency(QUERY, NO_ANSWER)
    print(f"\n== sufficiency, cutoff {cutoff}")
    print(f"  kept passages:      P={on_kept.scores[0]:.3f}  ({describe(on_kept)})")
    print(f"  no-answer passages: P={on_none.scores[0]:.3f}  ({describe(on_none)})")
    return 0 if on_kept.scores[0] >= cutoff > on_none.scores[0] else 1


if __name__ == "__main__":
    sys.exit(main())
