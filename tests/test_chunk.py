from vault_rag.ingest.chunk import chunk_note, merge_short, sections, split_long

LONG = "x" * 300


def test_heading_path_follows_nesting():
    body = f"# Note\n\n## Sessions\n\n#### Day one\n{LONG}\n\n## Other\nb"
    paths = [path for path, _ in sections(body)]
    assert paths == [["Note"], ["Note", "Sessions"], ["Note", "Sessions", "Day one"], ["Note", "Other"]]


def test_headings_inside_code_fences_are_text():
    body = "# Note\nintro\n```sh\n# install deps\nuv sync\n```\n## Next\nmore"
    parts = sections(body)
    assert [path for path, _ in parts] == [["Note"], ["Note", "Next"]]
    assert "# install deps" in parts[0][1]


def test_html_comments_are_removed():
    body = "<!-- claude-session:abc start -->\n# Note\nbody\n<!-- claude-session:abc end -->"
    assert sections(body) == [(["Note"], "# Note\nbody")]


def test_short_sections_merge_into_the_next():
    parts = [(["Note"], "# Note\nRelated: [[a]]"), (["Note", "Sessions"], "## Sessions"),
             (["Note", "Sessions", "Day"], f"#### Day\n{LONG}")]
    merged = merge_short(parts, 200)
    assert len(merged) == 1
    assert merged[0][0] == ["Note", "Sessions", "Day"]
    assert merged[0][1].startswith("# Note\nRelated: [[a]]\n\n## Sessions\n\n#### Day")


def test_short_last_section_joins_the_previous_and_tiny_note_survives():
    merged = merge_short([(["A"], LONG), (["A", "B"], "tail")], 200)
    assert merged == [(["A"], f"{LONG}\n\ntail")]
    assert merge_short([(["A"], "tiny")], 200) == [(["A"], "tiny")]


def test_long_section_splits_with_overlap_and_respects_max():
    paragraphs = [f"para{i} " + "word " * 60 for i in range(10)]  # ~300 chars each
    pieces = split_long("\n\n".join(paragraphs), max_chars=800, overlap_chars=100)
    assert len(pieces) > 1
    assert all(len(p) <= 800 for p in pieces)
    for previous, piece in zip(pieces, pieces[1:]):
        assert piece.split("\n\n")[0] in previous  # starts with the previous piece's tail
    assert all(p.split()[0] in "".join(pieces) for p in paragraphs)


def test_one_huge_paragraph_is_cut_by_characters():
    pieces = split_long("y" * 5000, max_chars=1000, overlap_chars=100)
    assert all(len(p) <= 1000 for p in pieces)
    assert sum(p.count("y") for p in pieces) >= 5000


def test_chunk_ids_are_sequential_per_note():
    body = f"# N\n{LONG}\n## A\n{LONG}\n## B\n{LONG}"
    chunks = chunk_note("dev/Projects/n", body, min_chars=200, max_chars=1800, overlap_chars=200)
    assert [c.chunk_id for c in chunks] == ["dev/Projects/n#0", "dev/Projects/n#1", "dev/Projects/n#2"]
