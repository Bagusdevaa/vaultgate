from vault_rag.sensitive import deny_pattern, ip_kind, load_terms, scan_text


def categories(text: str, deny=None) -> set[str]:
    return {f.category for f in scan_text("n.md", text, deny)}


def test_each_category_fires():
    assert "email" in categories("mail me at someone@example.com")
    assert "phone" in categories("WA 0812-3456-7890")
    assert "phone" in categories("call +62 812 3456 7890")
    assert "api_key" in categories("key sk-or-v1-" + "a" * 40)
    assert "api_key" in categories("token ghp_" + "b" * 36)
    assert "credential" in categories("password = hunter2hunter2")
    assert "credential" in categories("JEV_API_KEY=abcdef123456")
    assert "credential" not in categories("tokens: 1500000")
    assert "ip_private" in categories("box at 192.168.1.20")
    assert "ip_private" in categories("tailscale 100.101.5.6")
    assert "ip_public" in categories("server 8.8.8.8")
    assert "internal_url" in categories("open http://localhost:5173/admin")
    assert "internal_url" in categories("run ssh deva@hp-server")
    assert "currency" in categories("harga Rp 1.500.000")
    assert "currency" in categories("budget 2 juta")
    assert "currency" in categories("costs $8")
    assert "mac_address" in categories("matched on MAC b0:5c:da:34:11:2c")
    assert "home_path" in categories("files in /home/someone/app")


def test_shell_variables_in_code_are_not_currency():
    assert "currency" not in categories("the helper used `$0` to find itself")
    assert "currency" not in categories("```sh\nawk '{print $1}'\n```")
    assert "currency" in categories("```sh\necho hi\n```\nthat cost $8")


def test_clean_text_has_no_findings():
    assert categories("Hybrid search combines dense and sparse vectors.") == set()


def test_version_numbers_are_not_ips():
    assert ip_kind("1.2.300.4") is None
    assert "ip_private" not in categories("bump to 999.1.2.3")


def test_secrets_never_appear_in_full():
    secret = "sk-or-v1-" + "z" * 40
    findings = scan_text("n.md", f"OPENROUTER_API_KEY={secret}")
    assert findings
    for finding in findings:
        assert secret not in finding.match
        assert secret not in finding.snippet


def test_deny_terms_match_whole_words_case_insensitively():
    deny = deny_pattern(load_terms("# comment\nAcme Corp\n\nfoo\n"))
    assert "deny_term" in categories("working for ACME corp today", deny)
    assert "deny_term" not in categories("food is good", deny)
