"""
Opt-in capture and offline replay of the payloads this plugin is handed.

The layer mines what the router hands it, so when it mines nothing the only
question left is "what did the request actually carry".  A capture answers it
exactly: set the environment variable, one request is written to the directory
verbatim, and :func:`replay` (or the module's ``__main__``) re-runs that exact
payload through the real parser, the real phase pipeline and the real miner,
with no SSE and no Redis.  The captured file is therefore a faithful fixture
for a regression test.

Environment
-----------
LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_DIR
    Directory to write captures into.  **Empty (the default): nothing is
    written.**  The directory is created when first used.
LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_MAX
    Most capture files to keep; older ones are removed.  Default ``50``.

Captures contain full request payloads: prompts, tool outputs, and anything
else the agent sent.  They are a debugging aid for a machine the operator
controls, never a log line, and they are never emitted by the layer otherwise.
"""

import json
import os
import time

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "DUMP_DIR_ENV",
    "DUMP_MAX_ENV",
    "capture_payload",
    "replay",
    "main",
]

DUMP_DIR_ENV = "LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_DIR"
DUMP_MAX_ENV = "LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_MAX"

#: Default number of capture files kept per directory.
_DEFAULT_MAX_FILES = 50

#: A request this large is almost certainly not the interesting one, and one
#: would not want it on disk.  Skip the write, not the request.
_MAX_CAPTURE_CHARS = 20_000_000


def _dump_dir() -> Optional[Path]:
    """Return the capture directory, or ``None`` when capturing is off."""
    raw = os.getenv(DUMP_DIR_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _dump_max() -> int:
    try:
        return max(1, int(os.getenv(DUMP_MAX_ENV, str(_DEFAULT_MAX_FILES))))
    except ValueError:
        return _DEFAULT_MAX_FILES


def capture_payload(payload: Any, logger: Any = None) -> Optional[Path]:
    """
    Write one payload to the capture directory, when capturing is on.

    Never raises and never blocks the request path: a failure to write is a
    warning with no payload content.

    Parameters
    ----------
    payload : Any
        The request payload as the router handed it to the plugin.
    logger : logging.Logger, optional
        Where a skipped or failed capture is noted.

    Returns
    -------
    pathlib.Path or None
        The file written, or ``None`` when capturing is off or failed.
    """
    directory = _dump_dir()
    if directory is None:
        return None
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > _MAX_CAPTURE_CHARS:
            if logger is not None:
                logger.info(
                    "agentic knowledge: capture skipped, payload too large"
                    " (%d chars)",
                    len(text),
                )
            return None
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = (
            directory / f"kb-{stamp}-{os.getpid()}-{int(time.time() * 1000)}.json"
        )
        target.write_text(text, encoding="utf-8")
        _prune(directory, _dump_max())
        return target
    except Exception as exc:  # a capture must never break a request
        if logger is not None:
            logger.warning(
                "agentic knowledge: capture failed: %s", type(exc).__name__
            )
        return None


def _prune(directory: Path, keep: int) -> None:
    """Keep only the newest *keep* capture files."""
    files = sorted(directory.glob("kb-*.json"))
    for stale in files[: max(0, len(files) - keep)]:
        try:
            stale.unlink()
        except OSError:
            pass


def _item_shapes(payload: Any) -> Dict[str, int]:
    """Count the wire items of one payload, naming no content."""
    items = payload.get("input") if isinstance(payload, dict) else None
    shapes: Dict[str, int] = {}
    if not isinstance(items, list):
        return shapes
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or ("message" if item.get("role") else "?"))
        name = item.get("name")
        if name:
            kind = f"{kind}:{name}"
        shapes[kind] = shapes.get(kind, 0) + 1
    return shapes


def _evidence_shape(evidence: Any) -> str:
    """Summarize the phase evidence, naming no text."""
    parts = []
    for item in evidence or ():
        outcome = getattr(item, "succeeded", None)
        parts.append(
            f"{getattr(item, 'kind', '?')}"
            f":{getattr(item, 'mode', '') or '-'}"
            f":{'ok' if outcome else 'fail' if outcome is False else '-'}"
        )
    return "+".join(parts) or "-"


def replay(path: Path) -> str:
    """
    Re-run one captured payload through the real read-and-mine path.

    Returns
    -------
    str
        A multi-line report: the item types the request carried, the tool
        calls the parser saw, the phase evidence it produced, and the events
        the miner mined.  This is the offline answer to "why did nothing get
        written" for one exact request.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
        CodexAgentAdapter,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import (
        EpisodeMiner,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
        ScopeResolver,
    )

    adapter = CodexAgentAdapter(scope_resolver=ScopeResolver())
    lines = [
        f"payload: {path}",
        "phase_rules: "
        + (
            f"on (test_mode={adapter.phase_rules.test_mode})"
            if adapter.phase_rules
            else "OFF — nothing can be mined"
        ),
    ]
    shapes = _item_shapes(payload)
    lines.append(
        "items: "
        + (",".join(f"{k}:{v}" for k, v in sorted(shapes.items())) or "none")
    )
    turn = adapter.parse(payload)
    if turn is None:
        lines.append("parse: None (no work mode, no activity)")
        return "\n".join(lines)
    lines.append(
        f"parse: class={turn.request_class} scope={turn.scope or '-'}"
        f" mode={turn.mode or '-'}"
    )
    calls: Dict[str, int] = {}
    for item in turn.activity:
        if item.kind == "function_call":
            calls[item.name or "?"] = calls.get(item.name or "?", 0) + 1
    lines.append(
        f"activity={len(turn.activity)} tools=[{','.join(f'{k}:{v}' for k, v in sorted(calls.items())) or '-'}]"
    )
    lines.append(f"evidence={len(turn.evidence)}[{_evidence_shape(turn.evidence)}]")
    miner = EpisodeMiner(
        test_mode=getattr(adapter.phase_rules, "test_mode", "test") or "test"
    )
    result = miner.mine(turn)
    lines.append(
        f"mined: events={len(result.events)}"
        f" kinds={','.join(sorted({str(e.kind) for e in result.events})) or '-'}"
        f" loop={result.loop_count}"
    )
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    """Replay one or more captured payloads and print the report."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m " + __name__,
        description="Replay captured agentic-knowledge payloads offline.",
    )
    parser.add_argument("payloads", nargs="+", help="Captured payload JSON files")
    args = parser.parse_args(argv)
    report = "\n\n".join(replay(Path(name)) for name in args.payloads)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
