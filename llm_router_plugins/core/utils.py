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
