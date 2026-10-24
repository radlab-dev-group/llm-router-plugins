#!/usr/bin/env python3
"""
Recover Codex request payloads and their recorded decisions from router logs.

llm-router logs every prepared request as an indented JSON body and, for a
routed Codex request, one ``Codex routing: mode=... source=...`` line shortly
after it.  This script walks the logs once and writes one JSON line per
request: the payload as the router saw it plus the decision the server logged
for it.  Recorded decisions are kept only for later comparison; the replay in
``replay.py`` always re-derives the decision from the current code.

Output record::

    {"source", "line", "ts", "payload": {...}, "recorded": {...} | null}

Examples
--------
::

    # everything recorded, into the default corpus file
    python3 scripts/plugins/auto_codex/eval/extract_payloads.py

    # one session, a named corpus, keeping only trigger-model requests
    python3 scripts/plugins/auto_codex/eval/extract_payloads.py \\
        agents-conversation/codex/conv-03 -o /tmp/conv-03.jsonl --only-trigger
"""

import argparse
import json
import re
import sys
from pathlib import Path

from _bootstrap import DEFAULT_LOG_ROOT, DEFAULT_TRIGGER, log_files

#: ``2026-01-02 03:04:05,123 INFO logger.name: message``
LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (\w+) (\S+): (.*)$"
)

#: A routing decision emitted by the Codex routing plugin.
DECISION_RE = re.compile(
    r"^Codex routing: mode=(\S+) source=(\S+) model=(\S+) similarity=(\S+)"
    r"(?: class=(\S+))?(.*)$"
)


def parse_decision(message):
    """
    Read a ``Codex routing:`` line into a dict.

    Parameters
    ----------
    message : str
        The log message after the logger name.

    Returns
    -------
    dict or None
        Decision fields, or ``None`` when *message* is not a routing line.
    """
    match = DECISION_RE.match(message)
    if not match:
        return None
    extra = dict(
        field.split("=", 1)
        for field in (match.group(6) or "").strip().split()
        if "=" in field
    )
    return {
        "mode": match.group(1),
        "source": match.group(2),
        "model": match.group(3),
        "similarity": match.group(4),
        "class": match.group(5),
        **extra,
    }


def iter_requests(path):
    """
    Yield ``{"line", "ts", "payload", "recorded"}`` for one log file.

    A payload starts at a log line ending with ``{`` and ends at a lone ``}``.
    A block interrupted by another log line cannot be parsed and is counted in
    ``iter_requests.dropped``; a payload whose decision never arrived is yielded
    with ``recorded=None``.

    Parameters
    ----------
    path : pathlib.Path
        Router log file.

    Yields
    ------
    dict
        One recovered request.
    """
    buffer = []
    started_at = None
    for number, raw in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        line = raw.rstrip("\n")
        match = LINE_RE.match(line)
        if buffer:
            if line == "}":
                body = "\n".join(buffer) + "\n}"
                buffer, (start_line, start_ts) = [], started_at
                try:
                    payload = json.loads(body)
                except (ValueError, RecursionError):
                    iter_requests.dropped += 1
                    continue
                if isinstance(payload, dict):
                    yield {"line": start_line, "ts": start_ts, "payload": payload}
                continue
            if match:
                iter_requests.dropped += 1
                buffer, started_at = [], None
                continue
            buffer.append(line)
            continue
        if match and line.endswith("{"):
            started_at = (number, match.group(1))
            buffer = ["{"]
    iter_requests.truncated += 1 if buffer else 0


iter_requests.dropped = 0
iter_requests.truncated = 0


def pair_with_decisions(path, requests):
    """
    Attach each request the decision logged for it, in stream order.

    Parameters
    ----------
    path : pathlib.Path
        Log file the requests came from, used to read the decision lines.
    requests : iterable of dict
        Requests recovered from the same file, in order.

    Yields
    ------
    dict
        Requests with ``recorded`` filled in when a decision followed.
    """
    decisions = []
    for number, raw in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        match = LINE_RE.match(raw.rstrip("\n"))
        if not match:
            continue
        decision = parse_decision(match.group(4))
        if decision:
            decisions.append((number, decision))

    index = 0
    for request in requests:
        record = dict(request, recorded=None)
        while index < len(decisions) and decisions[index][0] < request["line"]:
            index += 1
        if index < len(decisions):
            record["recorded"] = decisions[index][1]
            index += 1
        yield record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "logs",
        nargs="*",
        default=[DEFAULT_LOG_ROOT],
        help="log files or directories (default: recorded Codex traffic)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("codex-corpus.jsonl"),
        help="corpus to write (default: ./codex-corpus.jsonl)",
    )
    parser.add_argument(
        "--trigger",
        default=DEFAULT_TRIGGER,
        help=f"trigger model name (default: {DEFAULT_TRIGGER})",
    )
    parser.add_argument(
        "--only-trigger",
        action="store_true",
        help="drop requests that do not carry the trigger model",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N requests")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress")
    return parser.parse_args(argv)


def select_logs(paths):
    """Expand *paths* into files, refusing an empty selection."""
    found = log_files(paths)
    if not found:
        raise SystemExit("No log files selected")
    return found


def main(argv=None):
    args = parse_args(argv)
    total = 0
    stop = False
    with open(args.output, "w", encoding="utf-8") as handle:
        for path in select_logs(args.logs):
            count = 0
            for record in pair_with_decisions(path, iter_requests(path)):
                if args.only_trigger and record["payload"].get("model") != (
                    args.trigger
                ):
                    continue
                handle.write(json.dumps({"source": str(path), **record}) + "\n")
                count += 1
                total += 1
                if args.limit and total >= args.limit:
                    stop = True
                    break
            if not args.quiet:
                print(f"{path}: {count} requests")
            if stop:
                break
    if not args.quiet:
        print(
            f"wrote {total} requests to {args.output} "
            f"(unparsable blocks dropped: {iter_requests.dropped})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
