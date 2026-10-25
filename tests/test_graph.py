from vault_rag.graph.build import build_graph, load_graph, save_graph

NOTES = {
    "dev/Projects/p.md": "[[Bugs/p|bugs]] [[Knowledge/k]] [[Projects/p#Top]] [[missing]]",
    "dev/Bugs/p.md": "back to [[Projects/p]]",
    "dev/Knowledge/k.md": "```\n[[Bugs/p]]\n```",
}


def test_edges_follow_resolved_links_only():
    graph = build_graph(NOTES)
    assert set(graph.nodes) == set(NOTES)
    assert set(graph.edges) == {("dev/Projects/p.md", "dev/Bugs/p.md"),
                                ("dev/Projects/p.md", "dev/Knowledge/k.md"),
                                ("dev/Bugs/p.md", "dev/Projects/p.md")}
    # Backlinks are the reverse edges.
    assert set(graph.predecessors("dev/Knowledge/k.md")) == {"dev/Projects/p.md"}


def test_save_and_load_round_trip(tmp_path):
    graph = build_graph(NOTES)
    save_graph(graph, tmp_path / "graph.json")
    loaded = load_graph(tmp_path / "graph.json")
    assert loaded.is_directed()
    assert set(loaded.edges) == set(graph.edges) and set(loaded.nodes) == set(graph.nodes)
