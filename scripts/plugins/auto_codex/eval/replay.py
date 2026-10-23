#!/usr/bin/env python3
"""
Replay a recovered corpus through the current Codex routing plugin.

Every payload is classified by the code checked out here, never by the decision
recorded in the log, so a change in the cascade shows up as a change in the
numbers instead of being blended with the recorded past.  The recorded decision
is carried alongside each row for ``compare_history.py``.

Besides the distributions, three integrity invariants are checked on every
request, because they are what a routing change must never break:

* ``apply()`` raises nothing (routing fails open),
* a payload that does not carry the trigger model comes back as the very same
  object, byte for byte,
* a routed payload gains exactly ``agent_mode`` and ``routing``.

Examples
--------
::

    python3 scripts/plugins/auto_codex/eval/replay.py /tmp/corpus.jsonl
    python3 scripts/plugins/auto_codex/eval/replay.py /tmp/corpus.jsonl \\
        --semantic -o /tmp/decisions.jsonl
"""

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, DEFAULT_TRIGGER, add_repo_to_path

add_repo_to_path()

from llm_router_plugins.utils.routing.agentic_routing.codex import (  # noqa: E402
    config as codex_config,
    plugin as codex_plugin,
)

#: Environment switch for the embedding layer of the plugin.
SEMANTIC_ENV = "LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED"


def digest(payload):
    """
    Return a short, order-independent hash of a payload.

    Parameters
    ----------
    payload : Any
        JSON-compatible value.

    Returns
    -------
    str
        Twelve hex digits identifying the exact content.
    """
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def iter_corpus(path):
    """Yield decoded records from a JSONL corpus, skipping blank lines."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield number, json.loads(line)


def decide(plugin, record):
    """
    Route one corpus record and describe the outcome.

    Parameters
    ----------
    plugin : CodexRoutingPlugin
        Plugin built from the configuration under test.
    record : dict
        Corpus record with ``payload`` and, when known, ``recorded``.

    Returns
    -------
    dict
        Decision row; ``error`` is set instead of a mode when routing raised.
    """
    payload = record["payload"]
    row = {
        "source": record.get("source", ""),
        "line": record.get("line", 0),
        "ts": record.get("ts", ""),
        "in_model": payload.get("model"),
    }
    recorded = record.get("recorded") or {}
    row["recorded_mode"] = recorded.get("mode")
    row["recorded_layer"] = recorded.get("source")

    # ``annotate_routing`` mutates the payload in place, so the shape before
    # the call is the only reference an added-key check can be measured against.
    before_keys = set(payload)
    before_digest = digest(payload)

    try:
        result = plugin.apply(payload)
    except Exception as exc:  # routing must never raise; count if it does
        row.update({"error": type(exc).__name__, "triggered": False})
        return row

    routing = (result or {}).get("routing") or {}
    triggered = routing.get("plugin") == plugin.name
    row.update(
        {
            "triggered": triggered,
            "mode": routing.get("agent_mode") or (result or {}).get("agent_mode"),
            "layer": routing.get("source"),
            "similarity": routing.get("similarity"),
            "class": routing.get("codex_class"),
            "reason": routing.get("reason"),
            "evidence": routing.get("evidence"),
            "memory": routing.get("memory"),
            "semantic": routing.get("semantic"),
            "out_model": (result or {}).get("model"),
        }
    )
    if triggered:
        row["added_keys"] = sorted(set(result) - before_keys - {"model"})
    else:
        # Nothing was routed: the caller must get the very same object back.
        row["passthrough"] = {
            "same_object": result is payload,
            "unchanged": digest(result) == before_digest,
        }
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "corpus", type=Path, help="corpus written by extract_payloads"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="decisions JSONL to write (default: stdout, report only)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="routing config to replay",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="enable the embedding layer (default: deterministic cascade)",
    )
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER, help="trigger model")
    parser.add_argument("--limit", type=int, default=0, help="stop after N requests")
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="also replay backwards with a second plugin and compare",
    )
    return parser.parse_args(argv)


def report(rows):
    """Print the distribution and integrity summary of a replay."""
    routed = [row for row in rows if row.get("triggered")]
    errors = [row for row in rows if row.get("error")]
    print(f"replayed={len(rows)} routed={len(routed)} errors={len(errors)}")
    for title, key in (("layer", "layer"), ("mode", "mode")):
        counts = collections.Counter(row.get(key) for row in routed)
        print(f"\n--- {title} distribution ---")
        for name, count in counts.most_common():
            print(f"  {str(name):<18} {count}")

    passthrough = [row for row in rows if "passthrough" in row]
    broken = [row for row in passthrough if not row["passthrough"]["same_object"]]
    mutated = [row for row in passthrough if not row["passthrough"]["unchanged"]]
    print("\n--- integrity ---")
    print(f"  passthrough requests: {len(passthrough)}")
    print(f"  returned another object: {len(broken)}")
    print(f"  mutated anyway: {len(mutated)}")
    added = collections.Counter(tuple(row.get("added_keys") or ()) for row in routed)
    for keys, count in added.most_common():
        print(f"  keys added {list(keys)}: {count}")
    for row in errors[:3]:
        print(f"  ERROR {row['source']}:{row['line']} {row['error']}")


def main(argv=None):
    args = parse_args(argv)
    os.environ[SEMANTIC_ENV] = "true" if args.semantic else "false"
    config = codex_config.CodexRoutingConfig.from_file(args.config)
    plugin = codex_plugin.CodexRoutingPlugin(logger=None, config=config)

    records = list(iter_corpus(args.corpus))
    if args.limit:
        records = records[: args.limit]

    rows = [decide(plugin, record) for _, record in records]
    report(rows)

    if args.determinism:
        forward = [(row["source"], row["line"], row.get("mode")) for row in rows]
        second = codex_plugin.CodexRoutingPlugin(logger=None, config=config)
        backward = [decide(second, record) for _, record in reversed(records)]
        reverse = [
            (row["source"], row["line"], row.get("mode")) for row in backward
        ][::-1]
        print("\n--- determinism (order independence) ---")
        print("  identical:", forward == reverse)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(rows)} decisions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
