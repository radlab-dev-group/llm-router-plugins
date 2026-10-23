#!/usr/bin/env python3
"""
Write every recorded Codex request to its own JSON, with the live decision.

The input is the recorded router traffic: request bodies and the
``Codex routing:`` line the server logged for each of them.  Parsing is shared
with the evaluation tooling (``scripts/plugins/auto_codex/eval``), so the same
recovery rules apply in both places.

For every recovered payload the plugin from *this checkout* decides the mode and
the model -- never the decision stored in the log -- and one file is written::

    agents-conversation/codex/payloads/codex_payload_001.json
    agents-conversation/codex/payloads/codex_payload_002.json
    ...

Document shape::

    {
      "id": 1, "source": "...log", "line": 282, "logged_at": "...",
      "trigger": "auto_codex",
      "routed": true,
      "decision": {                      # what the current code decided
        "mode": "git_review", "model": "qwen/...", "layer": "phase",
        "similarity": 1.0, "class": "main", "reason": "git log",
        "evidence": "command", "memory": null, "semantic": null, "error": null
      },
      "routing": { ...raw annotation the plugin wrote... } | null,
      "recorded": { ...decision from the log... } | null,
      "payload": { ...request exactly as the CLI sent it, untouched... }
    }

``payload`` is the request *before* routing: the plugin rewrites ``model`` and
adds ``routing``/``agent_mode`` in place, so a copy is routed and the original
is what gets stored.  Requests the plugin does not own (another model than the
trigger) are written with ``routed: false`` and the model they came with;
``--only-trigger`` drops them instead.

Files are numbered in recovery order, which is stable for a given set of logs
and filters, so a run can be resumed: an existing file is reused unless
``--overwrite`` is given, and its ``source``/``line`` are checked against the
request it would receive, because a changed filter would otherwise silently
relabel already written files.

Examples
--------
::

    # the whole recorded corpus, deterministic cascade
    python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py

    # one session, embedding layer on, into a scratch directory
    python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py \\
        agents-conversation/codex/conv-03 --semantic -o /tmp/payloads
    python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py --limit 20
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# The request parser and its helpers live in the sibling ``eval`` tool set;
# importing them keeps one recovery rule for both entry points.
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from _bootstrap import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_LOG_ROOT,
    DEFAULT_TRIGGER,
    REPO_ROOT,
    add_repo_to_path,
    log_files,
)

add_repo_to_path()

from extract_payloads import iter_requests, pair_with_decisions  # noqa: E402

from llm_router_plugins.utils.routing.agentic_routing.codex import (  # noqa: E402
    config as codex_config,
    plugin as codex_plugin,
)

#: Where the per-request documents are written by default.
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "agents-conversation" / "codex" / "payloads"

#: Environment switch for the embedding layer of the plugin.
SEMANTIC_ENV = "LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED"

#: Name of one exported document.
FILE_TEMPLATE = "codex_payload_{number:03d}.json"


def route(plugin, payload, trigger):
    """
    Decide a payload and describe the outcome.

    Parameters
    ----------
    plugin : CodexRoutingPlugin
        Plugin built from the configuration under test.
    payload : dict
        Request to route.  A copy is passed, the plugin mutates in place.
    trigger : str
        Model the plugin owns; anything else must pass through untouched.

    Returns
    -------
    dict
        ``routed`` flag, a ``decision`` block, the raw ``routing`` annotation,
        and ``original`` -- the untouched request.
    """
    original = copy.deepcopy(payload)
    annotated = copy.deepcopy(payload)
    decision = {
        "mode": None,
        "model": payload.get("model"),
        "layer": None,
        "similarity": None,
        "class": None,
        "reason": None,
        "evidence": None,
        "memory": None,
        "semantic": None,
        "error": None,
    }
    try:
        result = plugin.apply(annotated)
    except Exception as exc:  # routing fails open, but the corpus must record it
        decision["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "routed": False,
            "decision": decision,
            "routing": None,
            "original": original,
        }

    annotation = (result or {}).get("routing")
    routed = bool(annotation) and annotation.get("plugin") == plugin.name
    decision.update(
        {
            "mode": (annotation or {}).get("agent_mode")
            or (result or {}).get("agent_mode"),
            "model": (result or {}).get("model"),
            "layer": (annotation or {}).get("source"),
            "similarity": (annotation or {}).get("similarity"),
            "class": (annotation or {}).get("codex_class"),
            "reason": (annotation or {}).get("reason"),
            "evidence": (annotation or {}).get("evidence"),
            "memory": (annotation or {}).get("memory"),
            "semantic": (annotation or {}).get("semantic"),
        }
    )
    return {
        "routed": routed,
        "decision": decision,
        "routing": annotation if routed else None,
        "original": original,
    }


def resumable(path, expected_source, expected_line):
    """
    Report whether an existing document may be kept for this request.

    Parameters
    ----------
    path : pathlib.Path
        Document written by an earlier run.
    expected_source : str
        Log file the current request came from.
    expected_line : int
        Line the current request was recovered from.

    Returns
    -------
    bool
        ``True`` when the file describes the same request.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return False
    return (
        stored.get("source") == expected_source
        and stored.get("line") == expected_line
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "logs",
        nargs="*",
        help="log files or directories (default: recorded Codex traffic)",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"directory for the per-request JSON (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="routing config to use"
    )
    parser.add_argument(
        "--semantic", action="store_true", help="enable the embedding layer"
    )
    parser.add_argument(
        "--trigger",
        default=DEFAULT_TRIGGER,
        help=f"trigger model (default: {DEFAULT_TRIGGER})",
    )
    parser.add_argument(
        "--only-trigger",
        action="store_true",
        help="skip requests that do not carry the trigger model",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N requests")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-decide existing documents"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ[SEMANTIC_ENV] = "true" if args.semantic else "false"

    files = log_files(args.logs or [DEFAULT_LOG_ROOT])
    if not files:
        raise SystemExit("No log files selected")

    plugin = codex_plugin.CodexRoutingPlugin(
        logger=None, config=codex_config.CodexRoutingConfig.from_file(args.config)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    written = kept = routed = failed = skipped = 0
    stop = False
    for path in files:
        for record in pair_with_decisions(path, iter_requests(path)):
            payload = record["payload"]
            if args.only_trigger and payload.get("model") != args.trigger:
                skipped += 1
                continue

            number = written + kept + 1
            target = args.output_root / FILE_TEMPLATE.format(number=number)
            if target.exists() and not args.overwrite:
                if not resumable(target, str(path), record["line"]):
                    raise SystemExit(
                        f"{target} describes another request; pass --overwrite "
                        "or choose an empty --output-root"
                    )
                kept += 1
                if args.limit and written + kept >= args.limit:
                    stop = True
                    break
                continue

            outcome = route(plugin, payload, args.trigger)
            document = {
                "id": number,
                "source": str(path),
                "line": record["line"],
                "logged_at": record["ts"],
                "trigger": args.trigger,
                "routed": outcome["routed"],
                "decision": outcome["decision"],
                "routing": outcome["routing"],
                "recorded": record.get("recorded"),
                "payload": outcome["original"],
            }
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=1)
            written += 1
            routed += 1 if outcome["routed"] else 0
            failed += 1 if outcome["decision"]["error"] else 0
            if args.limit and written + kept >= args.limit:
                stop = True
                break
        if stop:
            break

    print(
        f"documents={written + kept} written={written} kept={kept} "
        f"skipped={skipped} routed={routed} errors={failed}"
    )
    print(f"output: {args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
