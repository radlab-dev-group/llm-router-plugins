#!/usr/bin/env python3
"""
Compare a replay against the decisions the recorded server made.

This is a diagnostic, not a score.  The recorded lines come from whatever build
and configuration produced the log, so disagreement is expected after any change
to the cascade; it answers "what moved and where", while correctness is measured
against labeled cases (``llm_router_plugins.utils.routing.agentic_routing.codex.
evaluation``).

Use it to (1) bound the blast radius of a change and (2) fish the disagreements
out for a human look, which is how the dead ``debug`` layer and the meta-text
contamination were found.

Examples
--------
::

    python3 scripts/plugins/auto_codex/eval/compare_history.py /tmp/decisions.jsonl
    python3 scripts/plugins/auto_codex/eval/compare_history.py \\
        /tmp/decisions.jsonl --show 8
"""

import argparse
import collections
import json
import sys
from pathlib import Path

#: Rows without a recorded decision cannot be compared.
UNRECORDED = "(none)"


def load_decisions(path):
    """Read decision rows written by ``replay.py``."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decisions", type=Path, help="decisions JSONL from replay")
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="print the first N disagreeing requests (default: 0)",
    )
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="also report the distribution of recorded layers",
    )
    parser.add_argument(
        "--list-unrouted",
        action="store_true",
        help="list requests routed by the recorded server but not now",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_decisions(args.decisions)
    recorded = [row for row in rows if row.get("recorded_mode")]
    # Only requests the current code routed can disagree about a mode; a
    # request left untouched now is a coverage change, not a wrong mode.
    comparable = [row for row in recorded if row.get("triggered")]
    unrouted = [row for row in recorded if not row.get("triggered")]
    agreed = [row for row in comparable if row.get("mode") == row["recorded_mode"]]

    print(f"rows={len(rows)} with_recorded_decision={len(recorded)}")
    print(f"comparable={len(comparable)} unrouted_now={len(unrouted)}")
    if comparable:
        share = 100.0 * len(agreed) / len(comparable)
        print(f"identical mode: {len(agreed)}/{len(comparable)} ({share:.1f}%)")

    print("\n--- mode: recorded -> current (disagreements) ---")
    confusion = collections.Counter(
        (row["recorded_mode"], row.get("mode"))
        for row in comparable
        if row.get("mode") != row["recorded_mode"]
    )
    if not confusion:
        print("  (none)")
    for (before, after), count in confusion.most_common(15):
        print(f"  {before:>16} -> {str(after):<16} {count}")

    print("\n--- agreement per source file ---")
    per_source = collections.defaultdict(lambda: [0, 0])
    for row in comparable:
        key = Path(row["source"]).name
        per_source[key][1] += 1
        if row.get("mode") == row["recorded_mode"]:
            per_source[key][0] += 1
    for key in sorted(per_source):
        hits, total = per_source[key]
        print(f"  {key:<40} {hits:>4}/{total:<4} {100.0 * hits / total:5.1f}%")

    layers = [("current", "layer")]
    if args.recorded:
        layers.append(("recorded", "recorded_layer"))
    for title, key in layers:
        print(f"\n--- {title} layer distribution ---")
        counts = collections.Counter(row.get(key) for row in comparable)
        for name, count in counts.most_common():
            print(f"  {str(name):<20} {count}")

    if args.list_unrouted:
        print("\n--- recorded but not routed now ---")
        for row in unrouted[:50]:
            print(
                f"  {row['source']}:{row['line']} "
                f"recorded={row['recorded_mode']}/{row['recorded_layer']} "
                f"in_model={row.get('in_model')}"
            )

    if args.show:
        print("\n--- first disagreements ---")
        shown = 0
        for row in comparable:
            if row.get("mode") == row["recorded_mode"]:
                continue
            print(
                f"  {row['source']}:{row['line']} "
                f"recorded={row['recorded_mode']}/{row['recorded_layer']} "
                f"current={row.get('mode')}/{row.get('layer')} "
                f"reason={row.get('reason')} evidence={row.get('evidence')}"
            )
            shown += 1
            if shown >= args.show:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
