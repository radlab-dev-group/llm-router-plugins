#!/usr/bin/env python3
"""
Score the Codex cascade against the exported payload documents.

The corpus is the per-request documents written by
``payload_parser/export_payloads.py`` (``agents-conversation/codex/payloads/``);
the recorded logs are not read here.  Every payload is routed by the code
checked out here and compared with an expected mode produced by a separate
labeler, so a change in the cascade is measured instead of asserted.

The labeler is independent code, and it reads only what a human sees in the
same payload.  Confidence-ordered rules:

===========  ===========================================================  ========
rule         what decides                                                  label
===========  ===========================================================  ========
class        ``request_kind`` / the title prompt of a toolless call        high
mode         a ``<collaboration_mode>`` block declaring Plan Mode          high
activity     the ordered actions of the active turn, bound by ``call_id``  high
intent       a high-precision phrase of the user's own words                medium
===========  ===========================================================  ========

The activity rule reproduces the semantics the phase layer documents — the
latest signal of the turn decides, an interleaved step inside a run of
another phase does not flip it, a weak signal (a linter) decides only when
the turn has no strong one — but reads the payload with its own parser, so
the two implementations agree only when they are both right.  A failed test
is expected as ``debug`` at the moment of the call that failed: an output
that arrives later does not move the failure after the work it explains.

A payload no rule reaches confidently stays unlabeled and is counted
separately, so the accuracy is never inflated by guessed expectations.

Examples
--------
::

    python3 scripts/plugins/auto_codex/eval/score_payloads.py
    python3 scripts/plugins/auto_codex/eval/score_payloads.py --config my.json
    python3 scripts/plugins/auto_codex/eval/score_payloads.py --explain 12 40
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

from _bootstrap import DEFAULT_CONFIG, add_repo_to_path

add_repo_to_path()

from llm_router_plugins.utils.routing.agentic_routing.codex import (  # noqa: E402
    config as codex_config,
    plugin as codex_plugin,
)

TRIGGER = "auto_codex"
WINDOW = 8

_TITLE_PROMPT_RE = re.compile(
    r"^(?:generate|produce|create|write|draft|suggest)\b[^\n]{0,120}\btitle\b",
    re.IGNORECASE,
)
_COLLABORATION_RE = re.compile(
    r"<collaboration_mode>(.*?)(?:</collaboration_mode>|\Z)",
    re.DOTALL,
)
_BOILERPLATE_RE = re.compile(
    r"<(?:environment_context|codex_internal_context)\b[^>]*>.*?"
    r"</(?:environment_context|codex_internal_context)>",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_OUTPUT_MARKER_RE = re.compile(r"^Output:[ \t]*$", re.MULTILINE)
_EXIT_RE = re.compile(r"exited with code (\d+)", re.IGNORECASE)
_TEST_RUN_RE = re.compile(
    r"\b(?:pytest|python3?\s+-m\s+(?:pytest|unittest)|unittest|npm\s+(?:run\s+)?test)\b"
)
_GIT_RE = re.compile(
    r"\bgit\s+(?:log|blame|show|diff|add|commit|push|pull|fetch|rebase|merge|"
    r"cherry-pick|checkout|switch|branch|worktree|restore|stash|tag|mv|rm)\b"
)
_LINT_RE = re.compile(r"\b(?:ruff|flake8|mypy|pyright|pylint|isort|black|bandit)\b")
_PATCH_FILE_RE = re.compile(
    r"\*\*\* (?:Add|Update|Delete) File: (.+?)\s*$", re.MULTILINE
)
_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__)/|^(?:test_[^/]+\.py|\S+_test\.py"
    r"|\S+\.(?:test|spec)\.[\w]+)$"
)
_WRITE_RE = re.compile(
    r"(?:(?:^|[\s;&|])>>?|\btee\b(?:\s+-\w+)*\s+)\s*[\"']?(?P<path>[^\s\"'`|&<>]+)"
)

#: High-precision phrases of a user's own request, per expected mode.  Used
#: only when the turn carries no executed action.
_INTENT_PATTERNS = {
    "review": (
        r"\bco o (?:tym|nim|niej|tym kodzie) s[ąa]dzisz\b",
        r"\bczy .{0,60}\b(?:zgodn|spójn|spojn)\w*\b",
        r"\bzr[óo]b (?:przegl[áa]d|review|kod review)\b",
        r"\breview the (?:code|implementation|changes)\b",
    ),
    "debug": (
        r"\bdlaczego (?:to|ten|ten kod)? ?(?:nie dzia[łl]a|wywala si[ęe]|pada)\b",
        r"\bnapraw\b.{0,40}\b(?:b[łl]ąd|bład|error|test|awari[ęe]|ugrz?\b)",
        r"\bfix the (?:bug|error|failure|issue|regression)\b",
    ),
    "test": (
        r"\b(?:uruchom|uruchomi|przepu[śs]c| odpal)\w* (?:testy|test)\b",
        r"\brun the (?:tests|test suite|test suite)\b",
    ),
    "git_review": (
        r"\bzobacz na (?:commity|histori[ęe]|diff|branch|zmiany)\b",
        r"\b(?:podsumuj|opisz|przejrzyj) (?:commity|zmiany|histori[ęe]|diff)\b",
    ),
    "implement": (
        r"^\s*(?:dodaj|zmień|zmien|napisz|zaktualizuj|popraw|utwórz|stwórz|"
        r"usun|usuń)\b",
        r"^\s*(?:add|change|write|update|fix|implement|create|remove)\b",
    ),
}


def _payload_texts(payload):
    """Every text the payload carries: instructions and input items."""
    texts = [payload.get("instructions") or ""]
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def _user_texts(payload):
    """The user messages of a payload, oldest first."""
    out = []
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append(part["text"])
    return out


def _request_class(payload):
    """Read the request class straight from the wire format."""
    metadata = payload.get("client_metadata") or {}
    raw = metadata.get("x-codex-turn-metadata") or ""
    try:
        kind = json.loads(raw).get("request_kind", "")
    except (ValueError, TypeError, AttributeError):
        kind = ""
    if kind == "compaction":
        return "compaction"
    if payload.get("tools"):
        return "main"
    text = "\n".join(_user_texts(payload)).strip()
    if _TITLE_PROMPT_RE.match(text):
        return "aux_title"
    return "main"


def _collaboration_mode(payload):
    for text in _payload_texts(payload):
        match = _COLLABORATION_RE.search(text)
        if match and re.search(r"\bPlan Mode\b", match.group(1)):
            return "plan"
    return ""


def _activity(payload):
    """The assistant and tool events after the last user message."""
    items = payload.get("input") or []
    last_user = -1
    for index, item in enumerate(items):
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "user"
        ):
            last_user = index
    return [item for item in items[last_user + 1 :] if isinstance(item, dict)]


def _exit_status(output):
    """The execution status of a tool output, read from its envelope."""
    marker = _OUTPUT_MARKER_RE.search(output)
    envelope = output[: marker.start()] if marker else output
    codes = [int(match.group(1)) for match in _EXIT_RE.finditer(envelope)]
    if codes:
        return 0 not in codes
    return None


def _written_target(arguments):
    """The file a command writes through redirection, or ``None``."""
    match = _WRITE_RE.search(arguments)
    if match is None:
        return None
    target = match.group("path")
    if "/" not in target and "." not in target:
        return None
    if target.startswith(("/dev/", "$", "%")):
        return None
    return target


def _turn_signals(payload):
    """
    Return the chronological phase signals of the active turn.

    Strong signals (an executed test, a written file, a git operation, a test
    that failed) and weak ones (a linter) are separated: a weak signal is
    expected to decide only when the turn produced no strong one.
    """
    strong = []
    strong_reasons = []
    weak = []
    weak_reasons = []
    calls = {}
    for item in _activity(payload):
        kind = item.get("type")
        if kind == "function_call":
            arguments = str(item.get("arguments", ""))
            calls[item.get("call_id")] = arguments
            paths = _PATCH_FILE_RE.findall(arguments)
            if paths:
                if all(_TEST_PATH_RE.search(path) for path in paths):
                    strong.append("test")
                else:
                    strong.append("implement")
                strong_reasons.append("patched files")
                continue
            written = _written_target(arguments)
            if written is not None:
                if _TEST_PATH_RE.search(written):
                    strong.append("test")
                else:
                    strong.append("implement")
                strong_reasons.append("wrote a file")
                continue
            if _TEST_RUN_RE.search(arguments):
                strong.append("test")
                strong_reasons.append("test run")
            elif _GIT_RE.search(arguments):
                strong.append("git_review")
                strong_reasons.append("git")
            elif _LINT_RE.search(arguments):
                weak.append("review")
                weak_reasons.append("linter")
        elif kind == "function_call_output":
            arguments = calls.get(item.get("call_id"), "")
            if not _TEST_RUN_RE.search(arguments):
                continue
            if _exit_status(str(item.get("output", ""))) is True:
                strong.append("debug")
                strong_reasons.append("test run failed")
    if strong:
        return strong, strong_reasons
    return weak, weak_reasons


def _dominant(modes, window=WINDOW):
    """Latest signal wins; a rival dominating the window keeps its ground."""
    recent = modes[-window:]
    counts = collections.Counter(recent)
    newest = modes[-1]
    if counts[newest] > 1:
        return newest
    rival = max(counts, key=lambda mode: (counts[mode], mode == newest))
    if rival != newest and counts[rival] >= 3 and counts[rival] > len(recent) // 2:
        return rival
    return newest


def _clean_intent(text):
    """Mask the CLI boilerplate and pasted code out of the user's words."""
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text.strip()


def expected_label(payload):
    """
    Return the expected ``(mode, reason, confidence)`` for one payload.

    Parameters
    ----------
    payload : dict
        A raw request payload (the ``payload`` field of a document).

    Returns
    -------
    (str, str, str)
        Expected mode, the rule that produced it, and ``high``, ``medium``
        or ``none`` — ``none`` means no rule was confident, and the payload
        is then excluded from the accuracy.
    """
    request_class = _request_class(payload)
    if request_class == "compaction":
        return "compaction", "class:compaction", "high"
    if request_class == "aux_title":
        return "aux_title", "class:aux_title", "high"
    if _collaboration_mode(payload) == "plan":
        return "plan", "collaboration_mode", "high"

    modes, reasons = _turn_signals(payload)
    if modes:
        mode = _dominant(modes)
        index = max(i for i, item in enumerate(modes) if item == mode)
        return mode, f"activity:{reasons[index]}", "high"

    intent = _clean_intent("\n".join(_user_texts(payload)))
    for mode, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, intent, re.IGNORECASE):
                return mode, "intent", "medium"
    return "implement", "unlabeled", "none"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--payloads",
        type=Path,
        default=Path("agents-conversation/codex/payloads"),
        help="directory with codex_payload_NNN.json documents",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="routing config under test",
    )
    parser.add_argument(
        "--explain",
        type=int,
        nargs="*",
        default=[],
        help="payload ids to print in detail",
    )
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="list every payload whose mode differs from the expected one",
    )
    parser.add_argument(
        "--gold",
        action="store_true",
        help=(
            "compare against the decision.gold_label stored in every document "
            "instead of the labeler of this file"
        ),
    )
    return parser.parse_args(argv)


def route(plugin, payload):
    """Route one payload and report the outcome without raising."""
    try:
        result = plugin.apply(json.loads(json.dumps(payload)))
    except Exception as exc:  # routing must fail open; count it if it does
        return None, f"error:{type(exc).__name__}"
    routing = (result or {}).get("routing") or {}
    return routing.get("agent_mode"), routing.get("source")


def main(argv=None):
    args = parse_args(argv)
    os.environ["LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED"] = (
        "false"
    )

    config = codex_config.CodexRoutingConfig.from_file(args.config)
    plugin = codex_plugin.CodexRoutingPlugin(logger=None, config=config)

    documents = sorted(args.payloads.glob("codex_payload_*.json"))
    if not documents:
        sys.exit(f"no payloads under {args.payloads}")

    rows = []
    skipped = 0
    unlabeled = 0
    for path in documents:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        payload = document["payload"]
        if payload.get("model") != TRIGGER:
            skipped += 1
            continue
        if args.gold:
            gold = (document.get("decision") or {}).get("gold_label")
            if not gold:
                unlabeled += 1
                continue
            mode, reason, confidence = gold, "gold_label", "high"
        else:
            mode, reason, confidence = expected_label(payload)
        actual, layer = route(plugin, payload)
        rows.append(
            {
                "id": document["id"],
                "payload": payload,
                "expected": mode,
                "reason": reason,
                "confidence": confidence,
                "actual": actual,
                "layer": layer,
            }
        )

    labeled = [row for row in rows if row["confidence"] != "none"]
    correct = [row for row in labeled if row["actual"] == row["expected"]]
    errors = [row for row in rows if str(row["layer"]).startswith("error:")]

    print(f"payloads={len(rows)} not_triggered={skipped} labeled={len(labeled)}")
    if args.gold and unlabeled:
        print(f"without a gold label (excluded): {unlabeled}")
    print(
        f"accuracy on labeled: {len(correct)}/{len(labeled)} "
        f"({100.0 * len(correct) / max(1, len(labeled)):.1f}%)"
    )
    if errors:
        print(f"errors: {len(errors)}")

    print("\n--- per expected mode ---")
    by_expected = collections.defaultdict(list)
    for row in labeled:
        by_expected[row["expected"]].append(row)
    for mode in sorted(by_expected):
        group = by_expected[mode]
        hits = sum(1 for row in group if row["actual"] == mode)
        print(f"  {mode:<12} {hits:>3}/{len(group):<3}")
        confusion = collections.Counter(
            row["actual"] for row in group if row["actual"] != mode
        )
        if confusion:
            print(
                "      misrouted -> "
                + ", ".join(f"{m}: {c}" for m, c in confusion.most_common(4))
            )

    print("\n--- expected mode distribution ---")
    for mode, count in collections.Counter(
        row["expected"] for row in labeled
    ).most_common():
        print(f"  {mode:<12} {count}")

    if args.mismatch:
        print("\n--- mismatches ---")
        for row in labeled:
            if row["actual"] != row["expected"]:
                print(
                    f"  #{row['id']:>3} expected={row['expected']:<12} "
                    f"actual={str(row['actual']):<12} layer={str(row['layer'])}"
                )

    if args.explain:
        wanted = set(args.explain)
        print("\n--- cases ---")
        for row in rows:
            if row["id"] in wanted:
                print(
                    f"  #{row['id']:>3} expected={row['expected']:<12} "
                    f"actual={str(row['actual']):<12} layer={str(row['layer']):<14} "
                    f"why={row['reason']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
