"""Scan the demo vault for sensitive data and write reports/flagged_report.md.

    uv run python scripts/scan_sensitive.py

Flags only. Review the report and edit or drop notes in data/demo-vault/ by hand.
"""

import sys
from collections import Counter, defaultdict

from vault_rag.config import REPO_ROOT, load_config
from vault_rag.sensitive import Finding, deny_pattern, load_terms, scan_text

COLUMNS = ["deny_term", "api_key", "credential", "email", "phone", "ip_public",
           "ip_private", "mac_address", "internal_url", "home_path", "currency"]


def report(findings: list[Finding], scanned: int) -> str:
    by_file: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_file[finding.path].append(finding)
    totals = Counter(f.category for f in findings)

    lines = ["# Flagged report", "",
             f"Scanned {scanned} notes; {len(by_file)} have findings, {len(findings)} in total.",
             "Secrets are masked to their first 4 characters. Nothing was redacted.", "",
             "## Totals", "", "| Category | Findings |", "|---|---|"]
    lines += [f"| {c} | {totals[c]} |" for c in COLUMNS if totals[c]]
    lines += ["", "## Per note", "", "| Note | " + " | ".join(COLUMNS) + " |",
              "|---|" + "---|" * len(COLUMNS)]
    for path in sorted(by_file, key=lambda p: (-len(by_file[p]), p)):
        counts = Counter(f.category for f in by_file[path])
        lines.append(f"| {path} | " + " | ".join(str(counts[c] or "") for c in COLUMNS) + " |")
    lines += ["", "## Details"]
    for path in sorted(by_file):
        lines += ["", f"### {path}", "", "| Line | Category | Match | Context |", "|---|---|---|---|"]
        for f in by_file[path]:
            match = f.match.replace("|", "\\|")
            lines.append(f"| {f.line} | {f.category} | `{match}` | {f.snippet} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config()
    vault = REPO_ROOT / cfg["paths"]["demo_vault"]
    terms_file = REPO_ROOT / cfg["demo_vault"]["deny_terms_file"]
    terms = load_terms(terms_file.read_text()) if terms_file.exists() else []
    if not terms:
        print(f"WARNING: no deny terms loaded from {terms_file.relative_to(REPO_ROOT)}")

    notes = sorted(vault.rglob("*.md"))
    if not notes:
        print("ERROR: demo vault is empty, run scripts/build_demo_vault.py build first")
        return 1
    deny = deny_pattern(terms)
    findings = []
    for path in notes:
        findings += scan_text(path.relative_to(vault).as_posix(), path.read_text(encoding="utf-8"), deny)

    out = REPO_ROOT / cfg["paths"]["reports_dir"] / "flagged_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report(findings, len(notes)))
    totals = Counter(f.category for f in findings)
    print(f"scanned {len(notes)} notes, {len(findings)} findings, deny terms: {len(terms)}")
    for category in COLUMNS:
        if totals[category]:
            print(f"  {totals[category]:4d}  {category}")
    print(f"report: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
