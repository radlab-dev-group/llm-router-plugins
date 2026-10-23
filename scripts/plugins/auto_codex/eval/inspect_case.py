#!/usr/bin/env python3
"""
Show one request: what the router saw next to what it decided.

Numbers from ``replay.py`` say how often a layer answers; fixing a misroute
needs the case itself.  This prints the tail of the recovered conversation
(user text, assistant utterances, tool calls and their outputs) with the
current decision and the recorded one under it, which is how a disagreement is
attributed to intent text, activity evidence or a threshold.

Corpus and decisions are matched by ``source`` and ``line``, so any two files
produced by the same extraction can be combined — for example the deterministic
and the semantic replay of one corpus.

Examples
--------
::

    python3 scripts/plugins/auto_codex/eval/inspect_case.py \\
        /tmp/corpus.jsonl /tmp/decisions.jsonl --line 104387
    python3 scripts/plugins/auto_codex/eval/inspect_case.py \\
        /tmp/corpus.jsonl /tmp/decisions.jsonl --mode debug --limit 3
"""

import argparse
import json
import sys
from pathlib import Path

#: Fields of the decision worth reading next to the conversation.
DECISION_FIELDS = (
    ("mode", "mode"),
    ("layer", "layer"),
    ("class", "class"),
    ("similarity", "similarity"),
    ("reason", "reason"),
    ("evidence", "evidence"),
    ("semantic", "semantic"),
    ("memory", "memory"),
)


def load_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conversation(payload, limit):
    """
    Flatten request items into ``(label, text)`` pairs for display.

    Parameters
    ----------
    payload : dict
        A Codex request payload.
    limit : int
        Maximum characters per shown text.

    Returns
    -------
    list of tuple
        One entry per shown item, oldest first.
    """
    shown = []
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            role = item.get("role", "?")
            content = item.get("content")
            if isinstance(content, str):
                shown.append((role, content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        shown.append((f"{role}:{part.get('type')}", part["text"]))
        elif kind in ("function_call", "custom_tool_call"):
            name = item.get("name") or ""
            body = item.get("arguments") or item.get("input") or ""
            shown.append((f"call:{name}", str(body)))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            body = item.get("output") or ""
            text = body if isinstance(body, str) else json.dumps(body)
            shown.append(("output", text))
        elif kind in ("reasoning",):
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    shown.append(("reasoning", part["text"]))
    return [(label, " ".join(text.split())[:limit]) for label, text in shown]


def describe(record, decision, args):
    """Print one case."""
    recorded = record.get("recorded") or {}
    print("=" * 78)
    print(f"{record.get('source')}:{record.get('line')} {record.get('ts', '')}")
    current = " ".join(
        f"{label}={decision.get(key)}"
        for label, key in DECISION_FIELDS
        if label == "mode" or decision.get(key) not in (None, "")
    )
    print(f"  CURRENT  {current}")
    print(
        "  RECORDED "
        f"mode={recorded.get('mode')} source={recorded.get('source')} "
        f"similarity={recorded.get('similarity')}"
    )
    shown = conversation(record["payload"], args.chars)
    if args.items and len(shown) > args.items:
        shown = shown[-args.items :]
    for label, body in shown:
        print(f"    [{label}] {body}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="corpus from extract_payloads")
    parser.add_argument("decisions", type=Path, help="decisions from replay")
    parser.add_argument("--line", type=int, help="request line in the log file")
    parser.add_argument("--mode", help="first N requests routed to this mode")
    parser.add_argument("--layer", help="only requests decided by this layer")
    parser.add_argument(
        "--limit", type=int, default=1, metavar="N", help="cases to print"
    )
    parser.add_argument(
        "--items", type=int, default=6, help="conversation items to show"
    )
    parser.add_argument("--chars", type=int, default=240, help="chars per item")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    decisions = {
        (row["source"], row["line"]): row for row in load_jsonl(args.decisions)
    }
    if args.line is None and args.mode is None and args.layer is None:
        raise SystemExit("give --line, --mode or --layer")

    shown = 0
    for record in load_jsonl(args.corpus):
        key = (record["source"], record["line"])
        decision = decisions.get(key, {})
        if args.line is not None and record["line"] != args.line:
            continue
        if args.mode and decision.get("mode") != args.mode:
            continue
        if args.layer and decision.get("layer") != args.layer:
            continue
        describe(record, decision, args)
        shown += 1
        if shown >= args.limit:
            break
    if not shown:
        print("no matching request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
