"""Split data/golden/draft.jsonl into dev.jsonl (threshold tuning) and test.jsonl (held out).

    uv run python scripts/split_golden.py

Stratified by category with a fixed seed, so the split is reproducible. Validate the draft
first with scripts/validate_golden.py.
"""

import json
import sys
from collections import Counter

from vault_rag.config import REPO_ROOT, load_config
from vault_rag.golden import split

DEV_SIZE = 15
SEED = 20261101


def main() -> int:
    golden = REPO_ROOT / load_config()["paths"]["golden_dir"]
    lines = (golden / "draft.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    dev, test = split(items, DEV_SIZE, SEED)
    for name, part in (("dev", dev), ("test", test)):
        rows = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in part)
        (golden / f"{name}.jsonl").write_text(rows, encoding="utf-8")
        counts = Counter(item["category"] for item in part)
        print(f"{name}.jsonl: {len(part)} questions  " +
              ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
