import pytest

from vault_rag.demo_vault import (
    broken_links, build_graph, check_destination, copy_subgraph, neighbors, select_subgraph,
)

SKIP = {"Daily", "Handoffs"}


def tiny_vault() -> dict[str, str]:
    return {
        "dev/Projects/p": "[[Bugs/p]] [[Decisions/p]] [[Knowledge/k1]] [[Knowledge/k2]]",
        "dev/Bugs/p": "[[Projects/p]] [[Knowledge/k1]]",
        "dev/Decisions/p": "[[Projects/p]]",
        "dev/Knowledge/k1": "[[Knowledge/far]]",
        "dev/Knowledge/k2": "",
        "dev/Knowledge/far": "",
        "dev/Daily/2026-10-21": "[[Projects/p]] [[Knowledge/k1]]",
    }


def test_neighbors_skip_daily_and_rank_by_degree_then_name():
    graph = build_graph(tiny_vault())
    # Degrees: k1=4 (the Daily backlink still counts), Bugs/p=2, Decisions/p=1, k2=1.
    assert neighbors(graph, "dev/Projects/p", 2, SKIP) == ["dev/Knowledge/k1", "dev/Bugs/p"]
    # Equal degree falls back to name order.
    assert neighbors(graph, "dev/Projects/p", 4, SKIP)[2:] == ["dev/Decisions/p", "dev/Knowledge/k2"]
    assert "dev/Daily/2026-10-21" not in neighbors(graph, "dev/Projects/p", 10, SKIP)


def test_select_is_deterministic_and_capped():
    graph = build_graph(tiny_vault())
    first = select_subgraph(graph, ["dev/Projects/p"], 3, SKIP)
    assert first == select_subgraph(graph, ["dev/Projects/p"], 3, SKIP)
    assert len(first) == 4  # seed + 3 neighbors


def test_unknown_seed_is_an_error():
    with pytest.raises(ValueError):
        select_subgraph(build_graph(tiny_vault()), ["dev/Projects/nope"], 5, SKIP)


def test_broken_links_point_outside_the_selection():
    graph = build_graph(tiny_vault())
    selected = {"dev/Projects/p", "dev/Knowledge/k1"}
    broken = {(src, target) for src, target, _ in broken_links(graph, selected)}
    assert ("dev/Knowledge/k1", "Knowledge/far") in broken
    assert ("dev/Projects/p", "Knowledge/k1") not in broken


def test_copy_preserves_folders_and_clears_old_build(tmp_path):
    source, dest = tmp_path / "vault", tmp_path / "demo"
    for note, text in tiny_vault().items():
        (source / f"{note}.md").parent.mkdir(parents=True, exist_ok=True)
        (source / f"{note}.md").write_text(text)
    dest.mkdir()
    (dest / ".gitkeep").touch()
    (dest / "stale.md").write_text("old")

    copy_subgraph(source, dest, dest, {"dev/Projects/p", "dev/Bugs/p"})

    assert (dest / "dev/Projects/p.md").read_text() == tiny_vault()["dev/Projects/p"]
    assert (dest / "dev/Bugs/p.md").exists()
    assert not (dest / "stale.md").exists()
    assert (dest / ".gitkeep").exists()


def test_refuses_unexpected_or_overlapping_destination(tmp_path):
    allowed = tmp_path / "repo/data/demo-vault"
    with pytest.raises(ValueError):
        check_destination(tmp_path / "vault", tmp_path / "elsewhere", allowed)
    with pytest.raises(ValueError):
        check_destination(tmp_path / "repo", allowed, allowed)  # demo vault inside source
