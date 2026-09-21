"""
Shared routing target definition used by the routing plugins.

A :class:`RoutingTarget` is the smallest unit the embedding router operates on:
a named routing destination (``name`` → ``model_name``) described by a
``description`` and a set of representative ``examples`` that get embedded to
build the FAISS index.

Concrete plugins extend this contract as needed (for example
``AgenticRouting`` adds heuristic detection fields on top of it).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RoutingTarget:
    """
    Definition of a single routing target.

    Each target describes a semantic domain (e.g. ``code-generation``,
    ``creative-writing``) along with the model to route to when that
    domain is detected.

    Parameters
    ----------
    name : str
        Unique identifier for this target (used in ``target_name`` in results).
    model_name : str
        The model name to select when this target is the best match.
    description : str
        Human-readable description used for embedding.
    examples : Tuple[str, ...]
        Example user queries used for embedding.  These should be representative
        of the queries that should route to this target.
    """

    name: str
    model_name: str
    description: str
    examples: Tuple[str, ...]
