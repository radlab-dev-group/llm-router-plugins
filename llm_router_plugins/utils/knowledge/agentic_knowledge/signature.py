"""
Identity of a failure: the same error twice, whatever the clock says.

Recipes and dead ends are matched by the failure they belong to, and the loop
detector escalates only on a failure that repeats *identically*.  Neither works
on raw tool output, which differs on every run by line numbers, addresses,
durations and the absolute path of the checkout.

:func:`signature_of` reduces a failure report to the part that survives a
re-run: the exception type and message, the call chain that raised it, and the
test that failed.  Paths lose their directory and their line numbers, numbers
lose their value, and volatile identifiers lose themselves.  What is left is
the signature; its digest is what travels in a knowledge document.

The function recognizes the shapes an agent actually reports — a Python
traceback, a pytest summary, a ``go test`` failure, a non-zero exit of a
build or test command, a TypeScript or ESLint block, a Rust compiler error —
and returns ``None`` for anything that does not read like a failure at all,
so a green run never looks like a known problem.

Example
-------
::

    a = signature_of('File "/repo/x.py", line 12, in run\\nKeyError: "missing"')
    b = signature_of('File "/other/x.py", line 99, in run\\nKeyError: "missing"')
    a.signature_hash == b.signature_hash   # True
"""

import hashlib
import re

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["ErrorSignature", "signature_of", "looks_like_failure"]

#: Length cap of the signature text that reaches a knowledge document.
_MAX_SIGNATURE_CHARS = 400

_DIGEST_CHARS = 16

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")

# Identifiers and stamps that change on every run.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{4,}")
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{7,}\b")
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d+)?\b"
)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|us|ns|s|e-?\d+|seconds?|secs?|min(?:ute)?s?)\b"
)

# Source locations: ``file.py:12``, ``file.py", line 12``, ``a/b.ts(12,5)``.
_LINE_NUMBER_RES = (
    re.compile(
        r"(?P<head>[A-Za-z0-9_./\\-]+\.(?:py|pyi|js|jsx|ts|tsx|rs|go|java|"
        r"kt|rb|php|c|cc|cpp|h|hpp|cs|scala|swift|sh|yml|yaml|json|toml|md))"
        r"(?P<tail>:(?:line )?\d+(?::\d+)?)"
    ),
    re.compile(
        r"(?P<head>[A-Za-z0-9_./\-]+\.(?:py|js|jsx|ts|tsx|rs|go|java))"
        r"\((?P<tail>\d+)(?:,\d+)?\)"
    ),
)
_FRAME_RE = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line \d+(?:, in (?P<func>[^\s(]+))?'
)
# ``go test``: a ``--- FAIL: TestName`` header and a ``FAIL\tpkg\t0.005s`` tail.
_GO_FAIL_RE = re.compile(r"^(?:--+\s*FAIL\b|FAIL\t\S+)", re.IGNORECASE)
_PYTEST_LOCATION_RE = re.compile(
    r"^(?P<mark>FAILED|ERROR) (?P<node>[^\s:]+(?::：)?)(?P<rest>.*)$"
)
_SUMMARY_RES = (
    re.compile(r"=+\s*\d+ (?:failed|error)", re.IGNORECASE),
    re.compile(r"^\s*\d+ (?:failed|passed|error)", re.IGNORECASE),
)
#: A pytest summary says "N of these failed", which counts how often the run
#: was invoked, not what broke.  It is replaced whole, so a re-run with
#: ``--lf`` or a flaking sibling leaves the signature of the failure itself.
_SUMMARY_PLACEHOLDER = "<test-run-summary>"
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?![\w.])")

# A number that is the payload of an assertion is the failure: ``expected 3
# got 4`` and ``expected 4 got 3`` are two walls, and so are ``== 100`` and
# ``== 900``.  Those are kept while the surrounding path, line and address
# noise is stripped.  A bare ``<``/``>`` is deliberately absent: it would
# match the placeholder brackets the earlier passes insert and guard numbers
# that are only run noise.
_ASSERTION_VALUE_RE = re.compile(
    r"(?P<pre>"
    r"=={1,3}|!={1,2}|<=|>=|<>|=~"
    r"|\b(?:expected|expect|expects|wanted|want|wants|got|received|actual"
    r"|equals|equal to|to be|should be|should equal|returns|returned"
    r"|with value|value of|evaluated to|resolved to|but (?:got|received))\b"
    r"|exited with code|exit code"
    r")"
    r"(?P<gap>[\s:=,]*[+-]?)"
    r"(?P<num>\d+)(?![\w.])"
)
#: Stands in for one assertion payload while the number sweep runs.  No digit,
#: no hex, no path, so nothing else in :func:`_normalize` touches it.
_KEEP_TOKEN = "\x00\x01kept\x01\x00"

#: Lines that make a report a failure.  Checked against the whole report, so
#: a green run that merely mentions an ``error handler`` is not one.
_FAILURE_RES = (
    re.compile(
        r"^\s*(?:E\s+|error:|Error:|FAILED|ERROR)?\s*"
        r"(?:[A-Za-z_][\w.]*\.)?\w*(?:Error|Exception|Failure|Fault)\s*:",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:FAILED|ERROR)\s", re.MULTILINE),
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"^\s*(?:panic|fatal):\s", re.MULTILINE),
    re.compile(r"\b(?:panic|panicked)(?:\s+at)?:?\s"),
    re.compile(r"^\s*(?:npm ERR!|not ok \d+|make:\s*\*\*\*)", re.MULTILINE),
    re.compile(r"exit code:?\s*[1-9]\d*", re.IGNORECASE),
    re.compile(r"Process exited with code\s+[1-9]\d*", re.IGNORECASE),
    re.compile(r"^\s*error\[E\d+\]", re.MULTILINE),
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"\berror TS\d+\b"),
    re.compile(r"^\s*\S+:\d+:\d+:\s+error\b", re.MULTILINE),
    re.compile(r"^\s*--+\s*FAIL\b", re.MULTILINE),
    re.compile(r"\bFAIL\t\S+", re.MULTILINE),
    re.compile(r"^\s*\d+:\d+\s+error\b", re.MULTILINE),
)

#: Report lines that say nothing the signature needs.
_NOISE_RES = (
    re.compile(r"^\s*(?:INFO|DEBUG|WARN|WARNING|TRACE)\b"),
    re.compile(r"^\s*(?:Wall time|Chunk ID|Original token count|Output):"),
    re.compile(r"^\s*(?:\-|\+|~){3,}\s*$"),
    re.compile(r"^\s*(?:=|-|_){4,}\s*$"),
    re.compile(r"^\s*$"),
)


@dataclass(frozen=True)
class ErrorSignature:
    """
    The identity of one failure.

    Parameters
    ----------
    signature : str
        The normalized text two failures share.  Readable, so a recipe can
        quote it, and stable, so it can be compared.
    signature_hash : str
        Digest of :attr:`signature`; what a knowledge document filters on.
    paths : Tuple[str, ...]
        Source files the report names, in the order they appear.  Directory
        free, so they stay comparable across checkouts.
    """

    signature: str
    signature_hash: str
    paths: Tuple[str, ...] = ()


def looks_like_failure(text: str) -> bool:
    """
    Whether *text* reports something going wrong.

    Parameters
    ----------
    text : str
        A tool output, log tail or command result.

    Returns
    -------
    bool
        ``True`` when a recognized failure marker is present.
    """
    if not text:
        return False
    sample = _ANSI_RE.sub("", text)[:20000]
    return any(pattern.search(sample) for pattern in _FAILURE_RES)


def signature_of(text: str) -> Optional[ErrorSignature]:
    """
    Return the identity of the failure *text* reports, or ``None``.

    Parameters
    ----------
    text : str
        The tool output to read.

    Returns
    -------
    ErrorSignature or None
        ``None`` when *text* does not read like a failure, or leaves nothing
        behind once the run-specific noise is stripped.
    """
    if not looks_like_failure(text):
        return None
    cleaned = _ANSI_RE.sub("", text)
    paths = _collect_paths(cleaned)
    core = _core_lines(cleaned)
    if not core:
        return None
    signature = _normalize("\n".join(core))
    if not signature:
        return None
    return ErrorSignature(
        signature=signature[:_MAX_SIGNATURE_CHARS],
        signature_hash=_digest(signature),
        paths=paths,
    )


def _core_lines(text: str) -> Tuple[str, ...]:
    """Pick the lines that define the failure, oldest first."""
    lines = text.splitlines()
    traceback_at = [
        index for index, line in enumerate(lines) if _TRACEBACK_RE.search(line)
    ]
    if traceback_at:
        return _traceback_core(lines, traceback_at[-1])
    chosen = []
    for line in lines:
        if any(pattern.match(line) for pattern in _NOISE_RES):
            continue
        if not _is_failure_line(line):
            continue
        stripped = _strip_location_noise(line)
        if _is_summary_line(stripped):
            stripped = _SUMMARY_PLACEHOLDER
        chosen.append(stripped)
    if not chosen:
        return ()
    return tuple(_deduplicated(chosen))[:12]


def _traceback_core(lines: list, start: int) -> Tuple[str, ...]:
    """Reduce a Python traceback to its call chain and its final exception."""
    chain = []
    tail = []
    for line in lines[start + 1 :]:
        match = _FRAME_RE.match(line)
        if match:
            name = match.group("func")
            path = _basename(match.group("file"))
            chain.append(f"{path}:{name}" if name else path)
            continue
        if not line.strip() or line.startswith((" ", "\t")):
            if tail and line.startswith(("  ", "\t")):
                tail.append(line.strip())
            continue
        tail.append(line.strip())
    core = [f"chain: {', '.join(chain[-4:])}"] if chain else []
    core.extend(tail[:6])
    return tuple(core)[:12]


def _is_failure_line(line: str) -> bool:
    """Whether a single line carries the failure itself."""
    text = line.strip()
    if not text:
        return False
    if _TRACEBACK_RE.search(text):
        return False
    if text.startswith(('"File "', '"  File "', "--> ", "::", "note:")):
        return False
    if any(pattern.match(line) for pattern in _SUMMARY_RES):
        return True
    if _GO_FAIL_RE.match(text):
        return True
    if re.match(r"^\d+:\d+\s+error\b", text):
        return True
    if re.search(r"\bexit(?:ed)? (?:with )?code:?\s+[1-9]\d*", text):
        return True
    if text.startswith(
        (
            "E ",
            "E\t",
            "error:",
            "Error:",
            "ERROR",
            "FAILED",
            "panic:",
            "panicked",
            "fatal:",
            "npm ERR!",
            "not ok ",
        )
    ):
        return True
    if re.match(r"^error\[E\d+\]", text):
        return True
    if re.search(r"\b(?:panicked at|command not found|error TS\d+)\b", text):
        return True
    if re.search(r":\s*(?:fatal )?error\b", text):
        return True
    return bool(
        re.match(r"^(?:[A-Za-z_][\w.]*\.)?\w*(?:Error|Exception|Failure)\b", text)
    )


def _is_summary_line(line: str) -> bool:
    """Whether a line is a pytest summary of how the whole run went."""
    return any(pattern.match(line) for pattern in _SUMMARY_RES)


def _strip_location_noise(line: str) -> str:
    """Drop pytest's ``-A`` decorations and progress bars from a line."""
    text = line.strip()
    text = re.sub(r"^_+ .+ _+$", "", text).strip()
    text = re.sub(r"^E\s+", "E ", text)
    return text


def _collect_paths(text: str) -> Tuple[str, ...]:
    """Collect the source files a report mentions, without their directories."""
    found = []
    for match in re.finditer(
        r"(?:File \"|\bat )?([\w./\-\\-]+\.(?:py|js|jsx|ts|tsx|rs|go|java|rb|c|"
        r"cc|cpp|h|hpp|cs|php|swift|kt|scala))\b",
        text,
    ):
        name = _basename(match.group(1))
        if name and name not in found:
            found.append(name)
    return tuple(found[:12])


def _basename(path: str) -> str:
    """Return the file name of a path written with either separator."""
    text = (path or "").replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def _normalize(text: str) -> str:
    """Strip everything that differs between two runs of one failure.

    The payload of an assertion (``expected 3 got 4``, ``== 100``) is what the
    next run is compared on, so it is set aside before the number sweep and
    put back after; every other integer — line numbers, counts, byte sizes —
    still collapses to ``<n>``.
    """
    value, kept = _mask_assertion_values(text)
    value = _UUID_RE.sub("<id>", value)
    value = _ADDRESS_RE.sub("<addr>", value)
    value = _TIMESTAMP_RE.sub("<time>", value)
    value = _DURATION_RE.sub("<dur>", value)
    value = _DATE_RE.sub("<date>", value)
    for pattern in _LINE_NUMBER_RES:
        value = pattern.sub(lambda match: f"{match.group('head')}:<line>", value)
    value = _HASH_RE.sub("<hash>", value)
    value = _NUMBER_RE.sub("<n>", value)
    value = _restore_assertion_values(value, kept)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n+", "\n", value)
    return value.strip()


def _mask_assertion_values(text: str) -> Tuple[str, Tuple[str, ...]]:
    """Replace assertion payloads with a sentinel, returning the originals.

    The values are collected on the way through, so the i-th sentinel in the
    masked text is put back to the i-th value; a text with no assertion
    payload is returned untouched.
    """
    kept: list = []

    def _replace(match: "re.Match") -> str:
        kept.append(match.group("num"))
        return f"{match.group('pre')}{match.group('gap')}{_KEEP_TOKEN}"

    masked = _ASSERTION_VALUE_RE.sub(_replace, text)
    return masked, tuple(kept)


def _restore_assertion_values(value: str, kept: Tuple[str, ...]) -> str:
    """Put the set-aside assertion values back where their sentinels sit."""
    if not kept:
        return value
    remaining = list(kept)

    def _replace(_match: "re.Match") -> str:
        return remaining.pop(0) if remaining else _match.group(0)

    return re.sub(re.escape(_KEEP_TOKEN), _replace, value)


def _deduplicated(lines: Tuple[str, ...]) -> Tuple[str, ...]:
    """Keep the first of every equal line, preserving order."""
    seen = set()
    result = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return tuple(result)


def _digest(signature: str) -> str:
    """Hash a normalized signature to the identifier documents filter on."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
