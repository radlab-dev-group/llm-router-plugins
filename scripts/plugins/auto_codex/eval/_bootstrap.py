"""
Shared setup for the Codex routing evaluation scripts.

The scripts run from a checkout without installing the package, so every
entry point calls :func:`add_repo_to_path` before importing
``llm_router_plugins``.  Paths default to the repository layout
(``agents-conversation/`` for the recorded traffic,
``llm_router_plugins/resources/`` for the production configuration).
"""

import re
import sys
from pathlib import Path

#: Repository root, computed from ``scripts/plugins/auto_codex/eval/``.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Recorded Codex traffic used as the evaluation corpus.
DEFAULT_LOG_ROOT = REPO_ROOT / "agents-conversation" / "codex"

#: Production routing configuration replayed by default.
DEFAULT_CONFIG = (
    REPO_ROOT
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "agentic_routing_codex.json"
)

#: Model the Codex CLI asks for; anything else must pass through untouched.
DEFAULT_TRIGGER = "auto_codex"


#: Router logs and their rotations: ``server.log``, ``llm-router.log.3``.
_LOG_NAME_RE = re.compile(r"\.log(?:\.\d+)?\Z", re.IGNORECASE)


def add_repo_to_path() -> None:
    """Make ``llm_router_plugins`` importable when running from a checkout."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _is_log_file(path):
    """
    Report whether *path* is a router log, ignoring anything else in the tree.

    The exported corpora and the per-request documents live under the same
    directory as the recorded logs, so a plain ``rglob("*")`` would walk the
    tools' own output.  Only ``*.log`` and its rotations (``*.log.1``,
    ``*.log.27``) are read; a file named explicitly is always accepted.

    Parameters
    ----------
    path : pathlib.Path
        Candidate file.

    Returns
    -------
    bool
        ``True`` for a log file or log rotation.
    """
    return path.is_file() and _LOG_NAME_RE.search(path.name) is not None


def log_files(paths):
    """
    Expand positional log arguments into concrete files.

    Directories are searched recursively for ``*.log`` and ``*.log.N``
    rotations; files are taken as given.  Results keep the caller's order and
    contain no duplicates.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Files and/or directories to expand.

    Returns
    -------
    list of pathlib.Path
        Existing files to read.

    Raises
    ------
    FileNotFoundError
        If an expanded path does not exist.
    """
    found = []
    for entry in paths:
        path = Path(entry)
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {path}")
        if path.is_dir():
            found.extend(
                sorted(item for item in path.rglob("*") if _is_log_file(item))
            )
        else:
            found.append(path)
    unique = []
    seen = set()
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
