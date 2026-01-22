"""
Utility helpers for file‑system operations.

The module currently provides a single helper that walks a directory tree,
collects the textual contents of files matching a set of extensions, and
returns them as a list of strings.  It is deliberately lightweight and
has no external dependencies, making it suitable for use in scripts,
CLI tools, or test fixtures.
"""

import sys

from typing import List
from pathlib import Path


def read_files_from_dir(base_path: Path, extensions: List[str]) -> List[str]:
    """
    Collect the textual content of all files matching *extensions*.

    Parameters
    ----------
    base_path : pathlib.Path
        Root directory from which to start the recursive search.
    extensions : List[str]
        List of file suffixes (including the leading dot, e.g. ``[".txt", ".md"]``)
        that should be considered.  Files whose suffix is not in this list are
        ignored.

    Returns
    -------
    List[str]
        A list containing the UTF‑8 decoded contents of each matching file,
        ordered by the underlying ``Path.rglob`` traversal.

    Raises
    ------
    SystemExit
        If no files matching the requested extensions are found.  The
        function writes an explanatory message to ``stderr`` before exiting.

    Notes
    -----
    * Files that cannot be read (permission errors, decoding issues, etc.) are
      skipped with a warning printed to ``stderr``; the processing continues
      with the remaining files.
    * The function exits the entire program on an empty result set because,
      in the typical indexing use‑case, an empty corpus is considered a
      configuration error.
    """
    texts: List[str] = []
    for path in base_path.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"Skipping unreadable file {path}: {exc}\n")
    if not texts:
        sys.stderr.write("No matching files were found to index.\n")
        sys.exit(1)
    return texts
