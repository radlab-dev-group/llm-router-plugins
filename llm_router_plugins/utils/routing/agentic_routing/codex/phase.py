"""
Conservative, turn-local phase signals from observed Codex activity.

A *phase* is what the agent is doing right now — running tests, editing a file,
reading commit history — as distinct from what the user asked for two turns
ago.  This layer reads the activity that followed the newest genuine user
command and turns it into ordered, explainable :class:`PhaseEvidence`.

What counts as evidence
-----------------------
- an **announcement**: a short current sentence about the action the agent is
  taking now (``Teraz uruchomię testy.``, ``Now I will update the CHANGELOG.``),
- a **command**: an executed shell command recognized from the configured rule
  table.  Commands are tokenized, never run, and safe compound commands count:
  a neutral ``cd``, an environment assignment and a pipe into a configured
  read-only filter keep the phase of the command they wrap,
- a **patch**: a Codex patch envelope, classified by the paths it touches,
- a **test failure**: a failed exit status on the output *linked to a test
  command recognized in this turn*.

What never counts
-----------------
Contradictory segments, unsupported shell syntax, substitutions and redirects
abstain instead of guessing.  A quoted or hypothetical action, a code sample and
a line that merely mentions a tool are not evidence.  Neither is an
infrastructure error, nor the word "failed" inside the content of a file the
agent happened to read: a failure only counts as the execution status of a
linked test call, read from the result envelope rather than searched for in the
captured body.
"""

import json
import re
import shlex

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from .payload import CodexActivity
from .phase_config import CodexPhaseConfig

__all__ = [
    "EVIDENCE_ANNOUNCEMENT",
    "EVIDENCE_COMMAND",
    "EVIDENCE_PATCH",
    "EVIDENCE_TEST_FAILURE",
    "PhaseEvidence",
    "collect_phase_evidence",
    "detect_phase_evidence",
    "detect_phase",
]

#: An announcement of the current action by the assistant.
EVIDENCE_ANNOUNCEMENT = "announcement"

#: An executed command recognized from the configured rule table.
EVIDENCE_COMMAND = "command"

#: A Codex patch envelope, classified by the paths it touches.
EVIDENCE_PATCH = "patch"

#: A failed execution status on a linked test command.
EVIDENCE_TEST_FAILURE = "test_failure"

#: Reason of a command whose every segment is neutral (``cd``, ``git status``).
REASON_NEUTRAL = "neutral"

#: Reason of several segments that agree on one phase without naming one rule.
REASON_CONSISTENT = "consistent"

#: Separates the sequential segments of a compound command.
_SEQUENCE_OPERATORS = (";", "&&")

#: Marks the end of the execution envelope and the start of the captured body.
_OUTPUT_MARKER = re.compile(r"^Output:[ \t]*$", re.IGNORECASE)

_EXIT_CODE = re.compile(
    r"^\s*(?:(?:process\s+)?exited\s+with\s+code\s+|"
    r"exit[ _-]code\s*[:=]?\s*)(-?\d+)\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)


@dataclass(frozen=True)
class PhaseEvidence:
    """
    One explainable reason to believe the agent is in a given work phase.

    Parameters
    ----------
    mode : str
        The work mode this evidence points at.
    kind : str
        What produced it: :data:`EVIDENCE_ANNOUNCEMENT`,
        :data:`EVIDENCE_COMMAND`, :data:`EVIDENCE_PATCH` or
        :data:`EVIDENCE_TEST_FAILURE`.
    reason : str
        Short machine-readable cause, e.g. ``"git log"`` or
        ``"announcement"``.
    event_id : str
        Identifier of the payload item that produced the evidence, when the
        payload carried one.
    call_id : str
        Identifier linking a call to its output; empty for an announcement.
    completed : bool
        Whether a linked result has settled the action.
    succeeded : bool or None
        Execution status once a linked result is known; ``None`` before that,
        and for an action that carries no status at all.
    """

    mode: str
    kind: str
    reason: str
    event_id: str = ""
    call_id: str = ""
    completed: bool = False
    succeeded: Optional[bool] = None

    def settle(self, succeeded: Optional[bool]) -> "PhaseEvidence":
        """
        Return a copy settled by a linked result.

        Parameters
        ----------
        succeeded : bool or None
            The execution status read from the result, or ``None`` when the
            result carries no status.

        Returns
        -------
        PhaseEvidence
            This evidence with :attr:`completed` set and :attr:`succeeded`
            filled in.
        """
        return replace(self, completed=True, succeeded=succeeded)


def _announcement(text: str, rules: CodexPhaseConfig) -> Optional[str]:
    """
    Return the phase announced by *text*, or ``None``.

    A trailing newline and a short explanatory follow-up are tolerated.  Only
    the first line can announce: a second line that would announce too makes
    the utterance describe two actions at once, and a long body is a report
    rather than an announcement of what comes next.

    Parameters
    ----------
    text : str
        One assistant utterance.
    rules : CodexPhaseConfig
        Validated announcement rules.

    Returns
    -------
    str or None
        The single announced mode, or ``None`` when nothing is announced.
    """
    if "```" in text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first, rest = lines[0], lines[1:]
    budget = rules.announcement_followup_max_chars
    if rest and (budget <= 0 or sum(len(line) for line in rest) > budget):
        return None
    if any(_announces(line, rules) for line in rest):
        return None
    return _match_announcement(first, rules)


def _announces(text: str, rules: CodexPhaseConfig) -> bool:
    """
    Report whether *text* names an action, hedge or not.

    A follow-up line is checked this way rather than as an announcement: "but
    first I will diagnose the error" is no evidence of debugging, yet it does
    contradict the sentence before it, and a line that mentions two actions
    announces neither.
    """
    return _match_announcement(text, rules, ignore_uncertain=True) is not None


def _match_announcement(
    text: str, rules: CodexPhaseConfig, ignore_uncertain: bool = False
) -> Optional[str]:
    """Match one line against the configured announcement patterns."""
    if not ignore_uncertain and rules.uncertain.search(text):
        return None
    match = rules.announcement_prefix.match(text)
    if not match:
        return None
    action = text[match.end():].rstrip(".! ").replace("`", "")
    phases = [
        phase for phase, pattern in rules.announcements
        if pattern.fullmatch(action)
    ]
    return phases[0] if len(phases) == 1 else None


def _tokenize(text: str) -> Optional[List[str]]:
    """
    Read the ``cmd``/``command`` argument of a call into shell tokens.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.

    Returns
    -------
    Optional[List[str]]
        The tokens, or ``None`` when the arguments are malformed or use shell
        constructs this reader refuses to interpret — a substitution, a
        redirect, or quoting mixed with operators.
    """
    try:
        arguments = json.loads(text)
        if not isinstance(arguments, dict):
            return None
        commands = [
            arguments[key] for key in ("cmd", "command") if key in arguments
        ]
        if not commands or any(not isinstance(cmd, str) for cmd in commands):
            return None
        if len(set(commands)) != 1:
            return None
        command = commands[0]
        if any(char in command for char in ("$", "`", "\n", "<", ">")):
            return None
        if any(char in command for char in "\"'") and any(
            char in command for char in ";&|()"
        ):
            return None
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        return list(lexer)
    except (ValueError, TypeError):
        return None


def _split_command(
    text: str, rules: CodexPhaseConfig
) -> Optional[List[List[str]]]:
    """
    Reduce a command to the producers whose phases must agree.

    Sequential operators (``;``, ``&&``) split the command into pipelines that
    run one after another, so their phases must agree.  A pipeline's first
    segment is the work; the segments after a pipe are consumers, allowed only
    when each is a configured neutral filter such as ``head`` or ``tee`` — they
    observe the output without changing what the agent is doing, so ``git show
    X | head -80`` is still a commit inspection.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig
        Validated filter rules.

    Returns
    -------
    Optional[List[List[str]]]
        One token list per producer segment, or ``None`` when the command
        cannot be interpreted.
    """
    tokens = _tokenize(text)
    if not tokens:
        return None

    pipelines: List[List[List[str]]] = [[]]
    stage: List[str] = []
    for token in tokens:
        if token in _SEQUENCE_OPERATORS or token == "|":
            if not stage:
                return None
            pipelines[-1].append(stage)
            stage = []
            if token in _SEQUENCE_OPERATORS:
                pipelines.append([])
            continue
        if token and all(char in ";&|()" for char in token):
            return None
        stage.append(token)
    if not stage:
        return None
    pipelines[-1].append(stage)

    segments: List[List[str]] = []
    for pipeline in pipelines:
        if not pipeline:
            continue
        for consumer in pipeline[1:]:
            if not _is_filter(consumer, rules):
                return None
        segments.append(pipeline[0])
    return segments or None


def _is_filter(stage: List[str], rules: CodexPhaseConfig) -> bool:
    """
    Report whether a pipeline consumer is a configured read-only filter.

    Parameters
    ----------
    stage : List[str]
        The consumer's tokens, executable first.
    rules : CodexPhaseConfig
        Validated filter patterns.

    Returns
    -------
    bool
        ``True`` when the executable is listed among the neutral filters.
    """
    if not stage:
        return False
    executable = PurePosixPath(stage[0]).name
    return any(
        pattern.fullmatch(executable) for pattern in rules.neutral_filters
    )


def _command_phase(
    text: str, rules: CodexPhaseConfig
) -> Tuple[Optional[str], str]:
    """
    Return the phase of an executed command and the reason naming its rule.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig
        Validated command rules.

    Returns
    -------
    Tuple[Optional[str], str]
        ``(mode, reason)`` when the command is recognized, ``(None, reason)``
        when it is explicitly neutral and ``(None, "")`` when the reader
        abstains.
    """
    segments = _split_command(text, rules)
    if segments is None:
        return None, ""
    modes = set()
    reasons = set()
    for words in segments:
        while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
            words = words[1:]
        if not words:
            return None, ""
        executable = PurePosixPath(words[0]).name
        args = words[1:]
        if any(arg in ("--help", "-h", "--version") for arg in args):
            return None, ""
        if executable == "git":
            while args:
                if args[0] == "--no-pager":
                    args = args[1:]
                elif args[0] == "-C" and len(args) >= 2:
                    args = args[2:]
                else:
                    break
        matches = {
            rule for rule in rules.commands
            if rule.executable.fullmatch(executable)
            and tuple(args[:len(rule.args_prefix)]) == rule.args_prefix
        }
        if len(matches) != 1:
            return None, ""
        rule = next(iter(matches))
        reasons.add(f"{executable} {' '.join(rule.args_prefix)}".strip())
        modes.add(rule.mode)
    # A neutral segment (``cd``, ``git status``) says nothing and contradicts
    # nothing; two different phases in one command are a conflict, not a tie.
    signals = modes - {None}
    if not signals:
        return None, REASON_NEUTRAL
    if len(signals) != 1:
        return None, ""
    return signals.pop(), reasons.pop() if len(reasons) == 1 else REASON_CONSISTENT


def _test_path(path: str, rules: CodexPhaseConfig) -> bool:
    """Whether *path* names a test file under the configured test layout."""
    parts = PurePosixPath(path).parts
    name = parts[-1].lower()
    return (
        any(part.lower() in rules.test_directories for part in parts[:-1])
        or name.startswith(rules.test_filename_prefixes)
        or bool(rules.test_filename_pattern.search(name))
    )


def _patch_phase(
    text: str, rules: CodexPhaseConfig
) -> Tuple[Optional[str], str]:
    """
    Return the phase of a Codex patch envelope, by the paths it touches.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call, or the envelope itself.
    rules : CodexPhaseConfig
        Validated patch and test-path rules.

    Returns
    -------
    Tuple[Optional[str], str]
        ``(mode, reason)``, or ``(None, "")`` for anything that is not a
        well-formed envelope over safe relative paths.
    """
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        patches = [arguments[key] for key in ("patch", "input") if key in arguments]
        if not patches or any(not isinstance(patch, str) for patch in patches):
            return None, ""
        if len(set(patches)) != 1:
            return None, ""
        text = patches[0]
    lines = text.strip().splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        return None, ""
    paths = []
    operation = None
    for line in lines[1:-1]:
        match = re.fullmatch(r"\*\*\* (Add|Update|Delete) File: (.+)", line)
        if match:
            operation = match[1]
            paths.append(match[2])
        elif line.startswith("*** Move to: ") and operation == "Update":
            paths.append(line[len("*** Move to: "):])
        elif operation == "Add" and line.startswith("+"):
            continue
        elif operation == "Update" and (
            line.startswith((" ", "+", "-", "@@"))
            or line == "*** End of File"
        ):
            continue
        else:
            return None, ""
    if not paths:
        return None, ""
    for path in paths:
        if (
            not path.strip() or path != path.strip() or "\\" in path
            or not PurePosixPath(path).parts or ":" in path
            or any(ord(char) < 32 for char in path)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or path.endswith("/")
        ):
            return None, ""
    mode = (
        rules.test_mode if all(_test_path(path, rules) for path in paths)
        else rules.implement_mode
    )
    return mode, f"patch {len(paths)} file(s)"


def _execution_status(text: str) -> Optional[bool]:
    """
    Read the execution status of a tool result, ignoring its captured body.

    The Codex shell wrapper prefixes the captured output with an envelope
    (``Process exited with code N`` …) and separates it from the body with an
    ``Output:`` line.  Only that envelope is authoritative: an exit code, a
    traceback or the word "failed" inside the captured text is data the agent
    is looking at — a log file, a test's own report — and says nothing about
    whether *this* call succeeded.

    Parameters
    ----------
    text : str
        The tool output.

    Returns
    -------
    bool or None
        ``True``/``False`` when a status is known, ``None`` when the result
        carries none — an unknown status is never read as a failure.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "exit_code" in parsed:
        code = parsed["exit_code"]
        if type(code) is not int:
            return None
        return code == 0

    match = _OUTPUT_MARKER.search(text)
    envelope = text[:match.start()] if match else text
    codes = [int(found[1]) for found in _EXIT_CODE.finditer(envelope)]
    if codes:
        return not all(code != 0 for code in codes)
    if _TRACEBACK.search(envelope):
        return False
    return None


def collect_phase_evidence(
    activity: Tuple[CodexActivity, ...], rules: CodexPhaseConfig,
) -> Tuple[PhaseEvidence, ...]:
    """
    Return every phase evidence of the active turn, oldest first.

    Only explicit current announcements, recognized executed commands and Codex
    patch envelopes count; a test failure additionally requires a call linked
    in this turn.  Neutral or ambiguous activity simply produces no evidence,
    leaving the earlier signal in place.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The assistant and tool events after the newest genuine user command.
    rules : CodexPhaseConfig
        Validated phase rules.

    Returns
    -------
    Tuple[PhaseEvidence, ...]
        The evidence in the order it happened, empty when nothing was recognized.
    """
    if not rules.enabled:
        return ()
    evidence: List[PhaseEvidence] = []
    calls: Dict[str, Tuple[str, int]] = {}
    for item in activity:
        if item.kind == "assistant":
            mode = _announcement(item.text, rules)
            if mode is not None:
                evidence.append(PhaseEvidence(
                    mode=mode, kind=EVIDENCE_ANNOUNCEMENT, reason="announcement",
                    event_id=item.event_id, completed=True,
                ))
        elif item.kind == "function_call":
            if item.name in rules.command_tools:
                mode, reason = _command_phase(item.text, rules)
                kind = EVIDENCE_COMMAND
            elif item.name in rules.patch_tools:
                mode, reason = _patch_phase(item.text, rules)
                kind = EVIDENCE_PATCH
            else:
                continue
            if mode is None:
                continue
            if item.call_id:
                calls.setdefault(item.call_id, (item.name, len(evidence)))
            evidence.append(PhaseEvidence(
                mode=mode, kind=kind, reason=reason,
                event_id=item.event_id, call_id=item.call_id,
            ))
        elif item.kind == "function_call_output" and item.call_id and item.name:
            linked = calls.get(item.call_id)
            if linked is None or linked[0] != item.name:
                continue
            settled = evidence[linked[1]]
            status = _execution_status(item.text)
            evidence[linked[1]] = settled.settle(status)
            if (
                settled.mode == rules.test_mode and status is False
                and settled.kind == EVIDENCE_COMMAND
            ):
                evidence.append(PhaseEvidence(
                    mode=rules.failure_mode, kind=EVIDENCE_TEST_FAILURE,
                    reason="test command failed", event_id=item.event_id,
                    call_id=item.call_id, completed=True, succeeded=False,
                ))
    return tuple(evidence)


def detect_phase_evidence(
    activity: Tuple[CodexActivity, ...], rules: CodexPhaseConfig,
) -> Optional[PhaseEvidence]:
    """
    Return the latest phase evidence, or ``None`` when there is none.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The assistant and tool events after the newest genuine user command.
    rules : CodexPhaseConfig
        Validated phase rules.

    Returns
    -------
    Optional[PhaseEvidence]
        The most recent evidence, carrying its kind, reason and settlement.
    """
    evidence = collect_phase_evidence(activity, rules)
    return evidence[-1] if evidence else None


def detect_phase(
    activity: Tuple[CodexActivity, ...], rules: CodexPhaseConfig,
) -> Optional[str]:
    """
    Return the latest unambiguous phase, without retaining cross-turn state.

    Compatibility view of :func:`detect_phase_evidence` for callers that only
    need the mode name.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The assistant and tool events after the newest genuine user command.
    rules : CodexPhaseConfig
        Validated phase rules.

    Returns
    -------
    Optional[str]
        The most recent recognized phase, or ``None``.
    """
    evidence = detect_phase_evidence(activity, rules)
    return evidence.mode if evidence is not None else None


def describe_activity(
    activity: Tuple[CodexActivity, ...],
    rules: Optional[CodexPhaseConfig] = None,
    limit: int = 3,
) -> str:
    """
    Describe what the agent is *doing*, without repeating what it read.

    A raw tool output is the content of some file, the body of some page, the
    log of some build — topically about whatever the agent happened to look at,
    and about the work only through the action that produced it.  Embedding it
    lets one large read dominate the description of the current activity.  This
    renders the structure instead: which tool ran, which recognized action it
    is when a rule names it, whether it succeeded, how many files a patch
    touched.  The result is bounded, and it stays identical while the action
    stays the same even when the text the tools returned changes.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The events of the active user turn.
    rules : CodexPhaseConfig, optional
        Phase rules used to name recognized actions.  Without them the
        description still reports tools and statuses.
    limit : int
        How many of the most recent clauses to keep.

    Returns
    -------
    str
        One clause per event, oldest first; empty when nothing is describable.
    """
    named = {}
    if rules is not None and rules.enabled:
        for item in collect_phase_evidence(activity, rules):
            if item.call_id:
                # The action keeps its own name; a failure that settled it is
                # reported as its status, not as a second action.
                named.setdefault(item.call_id, item)
    clauses: List[str] = []
    for item in activity:
        if item.kind != "function_call":
            continue
        evidence = named.get(item.call_id)
        if item.name and rules is not None and item.name in rules.patch_tools:
            clause = (
                f"editing files ({evidence.reason})"
                if evidence is not None else "editing files"
            )
        elif evidence is not None and evidence.kind == EVIDENCE_COMMAND:
            clause = f"ran {evidence.reason}"
        else:
            clause = f"called {item.name or 'a tool'}"
        if evidence is not None and evidence.completed:
            clause += ": succeeded" if evidence.succeeded else ": failed"
        clauses.append(clause)
    if limit > 0:
        clauses = clauses[-limit:]
    return "\n".join(clause for clause in clauses if clause)
