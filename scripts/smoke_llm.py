"""OpenRouter smoke test: one call to the generator and one to the judge."""

import sys
import time

from openai import OpenAI

from vault_rag.config import load_config

PROMPT = "In one sentence: what does a reranker do in a RAG pipeline?"


def call(client: OpenAI, role: str, model: str, temperature: float) -> None:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        max_tokens=200,
        extra_body={"usage": {"include": True}},
    )
    seconds = time.perf_counter() - start
    usage = response.usage
    # OpenRouter adds `cost` (USD) to usage when usage accounting is requested.
    cost = getattr(usage, "cost", None)
    print(f"\n== {role}: {model} ({seconds:.2f}s)")
    print(f"  served by: {response.model}")
    print(f"  tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} cost={cost}")
    print(f"  reply: {response.choices[0].message.content.strip()}")


def main() -> int:
    cfg = load_config()
    if not cfg.openrouter_api_key:
        print("SKIPPED: OPENROUTER_API_KEY not set")
        return 0
    llm = cfg["llm"]
    if "qwen" in llm["judge"].lower():
        print("ERROR: judge must not be from the Qwen family")
        return 1
    client = OpenAI(base_url=llm["base_url"], api_key=cfg.openrouter_api_key)
    call(client, "generator", llm["generator"], llm["temperature"])
    call(client, "judge", llm["judge"], llm["temperature"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
