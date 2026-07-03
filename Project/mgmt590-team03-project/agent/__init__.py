"""Phase-2 Agent Layer — LangGraph orchestration over the optimization engine."""

__all__ = ["build_graph"]


def __getattr__(name: str):
    if name == "build_graph":
        from agent.graph import build_graph

        return build_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
