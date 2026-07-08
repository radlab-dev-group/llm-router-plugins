"""
Core components of the anonymizer library.
"""

# Lazy re-exports to avoid circular imports (rules → core → masker → rules).
import typing


def __getattr__(name: str) -> typing.Any:
    if name == "FastMasker":
        from llm_router_plugins.maskers.fast_masker.core.masker import FastMasker

        return FastMasker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> typing.List[str]:
    return ["FastMasker"]
