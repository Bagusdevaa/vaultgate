from eval.grep_baseline import keywords, rank_files, read_files


def test_keywords_drop_stopwords_short_words_and_duplicates():
    q = "Kenapa launchd job di Mac itu gagal, terus launchd bilang 'state = not running'?"
    assert keywords(q) == ["launchd", "job", "mac", "gagal", "state", "running"]
    assert keywords("pakai qwen3.6-35b atau bge-m3?") == ["qwen3.6-35b", "bge-m3"]


def test_rare_keywords_outrank_common_ones():
    notes = {"a.md": "launchd PATH is empty", "b.md": "mac mac mac", "c.md": "mac launchd",
             "d.md": "mac notes", "e.md": "nothing here"}
    ranking = rank_files(notes, "launchd on mac")
    assert [p for p, _ in ranking] == ["c.md", "a.md", "b.md", "d.md"]
    assert "e.md" not in dict(ranking)


def test_read_files_fills_the_budget_and_cuts_the_last_file():
    notes = {"a.md": "x" * 60, "b.md": "y" * 60, "c.md": "z" * 60}
    context = read_files(notes, [("a.md", 2.0), ("b.md", 1.0), ("c.md", 0.5)], max_chars=100)
    assert [c.note for c in context] == ["a.md", "b.md"]
    assert [len(c.text) for c in context] == [60, 40]
    assert context[0].source == "grep"
