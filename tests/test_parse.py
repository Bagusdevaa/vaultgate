from vault_rag.ingest.parse import note_tags, resolve_link, split_frontmatter, wikilinks


def test_plain_alias_heading_and_embed():
    text = "See [[a]], [[Projects/b|B]], [[c#Setup]], [[d#Setup|alias]] and ![[e.png]]."
    assert wikilinks(text) == ["a", "Projects/b", "c", "d", "e.png"]


def test_md_suffix_and_whitespace_are_normalized():
    assert wikilinks("[[ notes/x.md ]]") == ["notes/x"]


def test_same_note_heading_link_is_skipped():
    assert wikilinks("jump to [[#Usage]]") == []


def test_links_in_code_are_ignored():
    text = "real [[a]]\n```\nfake [[b]]\n```\nalso `[[c]]` inline, then [[d]]"
    assert wikilinks(text) == ["a", "d"]


NOTES = {"bagusdeva/Projects/pbpipe", "deva/Projects/pbpipe", "deva/Knowledge/ssh",
         "bagusdeva/Bugs/x", "deva/Bugs/x"}


def test_resolves_inside_own_device_first():
    assert resolve_link("Projects/pbpipe", "deva/Daily/d", NOTES) == "deva/Projects/pbpipe"
    assert resolve_link("Projects/pbpipe", "bagusdeva/Bugs/x", NOTES) == "bagusdeva/Projects/pbpipe"


def test_falls_back_to_other_device():
    assert resolve_link("Knowledge/ssh", "bagusdeva/Projects/pbpipe", NOTES) == "deva/Knowledge/ssh"


def test_unique_basename_resolves_and_ambiguous_does_not():
    assert resolve_link("ssh", "bagusdeva/Bugs/x", NOTES) == "deva/Knowledge/ssh"
    assert resolve_link("x", "elsewhere/y", NOTES) is None
    assert resolve_link("missing", "deva/Bugs/x", NOTES) is None


def test_frontmatter_is_split_from_body():
    meta, body = split_frontmatter("---\ntype: project\ntags: [a, b]\n---\n# Title\n")
    assert meta == {"type": "project", "tags": ["a", "b"]}
    assert body == "# Title\n"
    assert split_frontmatter("# no frontmatter") == ({}, "# no frontmatter")
    assert split_frontmatter("---\n: : bad\n---\nbody")[1] == "body"


def test_tags_combine_frontmatter_and_inline_without_headings_or_code():
    meta = {"tags": ["project", "auto-ingest"], "tag": "pbpipe"}
    body = "# Heading\nwork on #mira and #news\n```sh\n# not a tag #x\n```\nissue #12 and `#y`"
    assert note_tags(meta, body) == ["project", "auto-ingest", "pbpipe", "mira", "news"]
    assert note_tags({"tags": "solo"}, "") == ["solo"]
