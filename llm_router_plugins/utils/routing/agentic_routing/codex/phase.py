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
An executable the rule table does not name is no evidence in either direction:
it is skipped, and a command keeps no evidence only when none of its segments
is named.  Two different phases claimed by one segment contradict each other
and abstain.  Fragments that carry no phase of their own — a redirect, a
substitution, a label printed by ``echo``, a shell control word — are read as
noise and do not silence the recognizable command beside them.  A quoted or
hypothetical action, a code sample and a line that merely mentions a tool are
not evidence.  Neither is an infrastructure error, nor the word "failed" inside
the content of a file the agent happened to read: a failure only counts as the
execution status of a linked test call, read from the result envelope rather
than searched for in the captured body.

Two strengths
-------------
A ``strong`` rule names the work (running tests, inspecting history, editing
files); a ``weak`` rule only accompanies it (a linter, a type checker).  The
latest strong signal of a turn decides; a weak one is heard only while the turn
has produced no strong signal at all.
"""

import json
import re
import shlex

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Pattern, Tuple

from .payload import CodexActivity
from .phase_config import STRENGTH_STRONG, STRENGTH_WEAK, CodexPhaseConfig

__all__ = [
    "EVIDENCE_ANNOUNCEMENT",
    "EVIDENCE_COMMAND",
    "EVIDENCE_PATCH",
    "EVIDENCE_TEST_FAILURE",
    "STRENGTH_STRONG",
    "STRENGTH_WEAK",
    "PhaseEvidence",
    "collect_phase_evidence",
    "detect_phase_evidence",
    "detect_phase",
    "describe_activity",
    "patch_envelopes",
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

#: Reason of a phase recognized from a script the agent piped into an editor.
REASON_SCRIPTED_EDIT = "scripted edit"

#: Reason of an edit read from a shell command that writes a file directly —
#: a redirection, ``tee`` or an in-place edit — where no patch envelope is
#: present.  A weak signal: it says a file changed, not how, and it is counted
#: separately so its false positives can be measured.
REASON_EXEC_EDIT = "exec edit"

#: Separates the sequential segments of a compound command.
_SEQUENCE_OPERATORS = (";", "&&", "||")

#: Replaces a fragment whose value is only known when the shell expands it.
_SUBSTITUTION_PLACEHOLDER = "-"

#: File-descriptor duplication, e.g. ``2>&1``.
_FD_DUPLICATION = re.compile(r"\d*>&+\d+")

#: A command substitution, ``$(…)``, and its quoted forms.
_COMMAND_SUBSTITUTION = re.compile(r"[\"']?\$\([^)]*\)[\"']?")

#: A back-quoted substitution and the placeholder a substitution became.
_BACKTICK_SUBSTITUTION = re.compile(r"[\"']?`[^`]*`[\"']?")

#: A redirect, with or without an explicit descriptor: ``2>/dev/null``, ``> f``.
_REDIRECT = re.compile(r"\d*(?:>>|>)[^\s;&|]*")

#: A here-document introducer and its terminating word.
_HERE_DOCUMENT = re.compile(r"<<<?-?\s*(['\"]?)(\w+)\1")

#: The first and the last line of a Codex patch envelope.
_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"

#: Argument keys of a tool call that can carry a patch envelope, in the order
#: Codex sends them: an ``apply_patch`` call declares the patch, an
#: ``exec_command`` wraps that same envelope in the command applying it.
_PATCH_ARGUMENT_KEYS = ("patch", "input", "cmd", "command")

#: Words that structure a shell script without naming an action.
_SHELL_CONTROL_WORDS = frozenset(
    {
        "for",
        "while",
        "until",
        "do",
        "done",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "case",
        "esac",
        "in",
        "select",
        "function",
        "time",
        "!",
    }
)

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
    strength : str
        :data:`STRENGTH_STRONG` when the evidence names the work itself,
        :data:`STRENGTH_WEAK` when it only accompanies it.  A strong signal of
        the turn outranks a weak one however recent the weak one is.
    paths : Tuple[str, ...]
        Files the evidence names as written, when it names any: the targets of
        a shell edit, read straight off the command.  Empty for everything the
        rule table recognizes, which does not look at paths at all.
    """

    mode: str
    kind: str
    reason: str
    event_id: str = ""
    call_id: str = ""
    completed: bool = False
    succeeded: Optional[bool] = None
    strength: str = STRENGTH_STRONG
    paths: Tuple[str, ...] = ()

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
    action = text[match.end() :].rstrip(".! ").replace("`", "")
    phases = [
        phase for phase, pattern in rules.announcements if pattern.fullmatch(action)
    ]
    return phases[0] if len(phases) == 1 else None


def _without_here_document(command: str) -> Tuple[str, str]:
    """
    Split a command into its shell text and the bodies of its here-documents.

    A here-document is data, not shell syntax: ``python3 - <<'EOF'`` runs an
    interpreter, and everything until the closing word is that interpreter's
    program.  Reading it as shell would either fail or report the body's own
    words as commands, so the body is handed back separately.

    Parameters
    ----------
    command : str
        The raw command string.

    Returns
    -------
    Tuple[str, str]
        The shell text, with each here-document replaced by a placeholder
        argument, and the collected bodies joined in order.
    """
    if "<<" not in command:
        return command, ""
    shell: List[str] = []
    bodies: List[str] = []
    lines = command.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _HERE_DOCUMENT.search(line)
        if match is None:
            shell.append(line)
            index += 1
            continue
        shell.append(line[: match.start()] + f" {_SUBSTITUTION_PLACEHOLDER} ")
        closing = match.group(2)
        index += 1
        body: List[str] = []
        while index < len(lines):
            if lines[index].strip() == closing:
                index += 1
                break
            body.append(lines[index])
            index += 1
        bodies.append("\n".join(body))
    return "\n".join(shell), "\n".join(bodies)


def _mask_quotes(text: str) -> str:
    """
    Replace every quoted span with a placeholder argument.

    A quoted word is data, not syntax: ``echo ';' pytest`` prints a semicolon
    and then runs a test, it does not sequence two commands.  A lexer that has
    already dropped the quoting cannot tell the two apart, so the span is
    replaced while the operators around it keep the meaning the shell gives
    them.  A command hidden inside a quoted string is not a command the agent
    named, and is not read as one.

    Parameters
    ----------
    text : str
        The command, here-document bodies already removed.

    Returns
    -------
    str
        The command with each closed span replaced by one placeholder; a quote
        that never closes is dropped as a typo.
    """
    result: List[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char not in "\"'":
            result.append(char)
            index += 1
            continue
        scan = index + 1
        while scan < length:
            current = text[scan]
            if char == '"' and current == "\\":
                scan += 2
                continue
            if current == char:
                break
            scan += 1
        if scan < length:
            result.append(f" {_SUBSTITUTION_PLACEHOLDER} ")
            index = scan + 1
        else:
            index += 1
    return "".join(result)


def _without_noise(command: str) -> str:
    """
    Reduce a command to what its structure says, dropping value-only fragments.

    A redirect says where output went, a substitution says which file name was
    expanded, and a duplicated descriptor says nothing at all.  None of them
    contradicts the command they decorate, so they are replaced rather than
    refused: reading them as noise is what lets ``cd repo && echo "=== git ==="
    && git log --oneline 2>/dev/null | cat`` still report a commit inspection.

    Parameters
    ----------
    command : str
        The raw command string, here-documents already removed.

    Returns
    -------
    str
        The command with noise replaced by placeholders and its line breaks
        turned into explicit sequential separators.
    """
    text = _mask_quotes(command)
    text = _FD_DUPLICATION.sub(" ", text)
    text = _COMMAND_SUBSTITUTION.sub(f" {_SUBSTITUTION_PLACEHOLDER} ", text)
    text = _BACKTICK_SUBSTITUTION.sub(f" {_SUBSTITUTION_PLACEHOLDER} ", text)
    text = _REDIRECT.sub(" ", text)
    return _separate_lines(text)


def _separate_lines(text: str) -> str:
    """Turn the line breaks outside quotes into explicit sequential operators."""
    result: List[str] = []
    quote = ""
    for char in text:
        if quote:
            result.append(char)
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            result.append(char)
        else:
            result.append(" && " if char == "\n" else char)
    return "".join(result)


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
        The tokens, or ``None`` when the arguments are malformed: not an
        object, a command that is not a string, or two keys disagreeing about
        what was run.  Shell constructs are no longer refused — redirects,
        substitutions and here-documents are read as noise by
        :func:`_without_noise`, and quoting the lexer cannot close is dropped
        rather than hiding the whole command.
    """
    try:
        arguments = json.loads(text)
        if not isinstance(arguments, dict):
            return None
        commands = [arguments[key] for key in ("cmd", "command") if key in arguments]
        if not commands or any(not isinstance(cmd, str) for cmd in commands):
            return None
        if len(set(commands)) != 1:
            return None
    except (ValueError, TypeError):
        return None

    shell = _without_noise(_without_here_document(commands[0])[0])
    lexer = shlex.shlex(shell, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return []


def _split_command(text: str, rules: CodexPhaseConfig) -> Optional[List[List[str]]]:
    """
    Reduce a command to the producer segments that can name a phase.

    Sequential operators (``;``, ``&&``, ``||``) split the command into
    pipelines that run one after another, so the last one naming a phase is the
    work the agent ended on.  A pipeline's first segment is the work; the
    segments after a pipe consume its output.  A consumer the rule table does
    not name, or one configured as read-only, says nothing and is ignored; a
    consumer naming a phase of its own contradicts the producer it consumes,
    and such a pipeline names no phase at all.  Shell control words structure
    the script without acting in it, and a segment left empty by their removal
    is skipped.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig
        Validated filter rules.

    Returns
    -------
    Optional[List[List[str]]]
        One token list per producer segment, in the order they run, or ``None``
        when the arguments themselves cannot be read.
    """
    tokens = _tokenize(text)
    if not tokens:
        return None

    pipelines: List[List[List[str]]] = [[]]
    stage: List[str] = []
    for token in tokens:
        if token in _SEQUENCE_OPERATORS or token == "|":
            if stage:
                pipelines[-1].append(stage)
                stage = []
            if token in _SEQUENCE_OPERATORS:
                pipelines.append([])
            continue
        if token and all(char in ";&|()" for char in token):
            if stage:
                pipelines[-1].append(stage)
                stage = []
            pipelines.append([])
            continue
        stage.append(token)
    if stage:
        pipelines[-1].append(stage)

    segments: List[List[str]] = []
    for pipeline in pipelines:
        stages = [
            [word for word in stage if word not in _SHELL_CONTROL_WORDS]
            for stage in pipeline
        ]
        stages = [stage for stage in stages if stage]
        if not stages:
            continue
        producer = stages[0]
        produced = _segment_signal(producer, rules)
        contradicted = produced is not None and any(
            _names_other_phase(stage, rules, produced[0]) for stage in stages[1:]
        )
        if not contradicted:
            segments.append(producer)
    return segments or None


def _names_other_phase(words: List[str], rules: CodexPhaseConfig, mode: str) -> bool:
    """Whether one pipeline segment names a phase other than *mode*."""
    signal = _segment_signal(words, rules)
    return signal is not None and signal[0] != mode


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
    return _matches_executable(stage[0], rules.neutral_filters)


def _is_neutral_executable(word: str, rules: CodexPhaseConfig) -> bool:
    """Whether an executable is configured to carry no phase of its own."""
    return _matches_executable(word, rules.neutral_executables)


def _matches_executable(word: str, patterns: Tuple[Pattern[str], ...]) -> bool:
    """Match a bare or path-qualified executable against rule patterns."""
    if not word:
        return False
    executable = PurePosixPath(word).name
    return any(pattern.fullmatch(executable) for pattern in patterns)


def _segment_signal(
    words: List[str], rules: CodexPhaseConfig
) -> Optional[Tuple[str, str, str]]:
    """
    Return the ``(mode, reason, strength)`` of one producer segment.

    An executable the table does not name, a neutral one, a help screen and an
    environment assignment all answer ``None``: they carry no phase, which is
    not the same as denying the phases of the segments around them.  When
    several rules match, the one naming the longest argument prefix wins; a
    genuine disagreement between equally specific rules abstains.

    Parameters
    ----------
    words : List[str]
        The segment's tokens, executable first.
    rules : CodexPhaseConfig
        Validated command rules.

    Returns
    -------
    Optional[Tuple[str, str, str]]
        The phase this segment names, or ``None`` when it names none.
    """
    while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
        words = words[1:]
    if not words:
        return None
    executable = PurePosixPath(words[0]).name
    args = words[1:]
    if any(arg in ("--help", "-h", "--version") for arg in args):
        return None
    if executable == "git":
        while args:
            if args[0] == "--no-pager":
                args = args[1:]
            elif args[0] == "-C" and len(args) >= 2:
                args = args[2:]
            else:
                break
    matches = [
        rule
        for rule in rules.commands
        if rule.executable.fullmatch(executable)
        and tuple(args[: len(rule.args_prefix)]) == rule.args_prefix
        and rule.mode is not None
    ]
    if not matches:
        return None
    modes = {rule.mode for rule in matches}
    if len(modes) != 1:
        return None
    rule = next(match for match in matches if match.mode in modes)
    return (
        rule.mode,
        f"{executable} {' '.join(rule.args_prefix)}".strip(),
        rule.strength,
    )


def _command_signal(
    text: str, rules: CodexPhaseConfig
) -> Optional[Tuple[str, str, str]]:
    """
    Return the phase an executed command names, and how strongly it names it.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig
        Validated command rules.

    Returns
    -------
    Optional[Tuple[str, str, str]]
        ``(mode, reason, strength)``, or ``None`` when no segment names a phase.
    """
    segments = _split_command(text, rules)
    if segments is None:
        return None
    signals = [
        signal
        for signal in (_segment_signal(words, rules) for words in segments)
        if signal is not None
    ]
    if not signals:
        return None
    strong = [signal for signal in signals if signal[2] == STRENGTH_STRONG]
    chosen = strong[-1] if strong else signals[-1]
    if len({signal[0] for signal in signals}) == 1 and len(signals) > 1:
        reasons = {signal[1] for signal in signals}
        return (
            chosen[0],
            chosen[1] if len(reasons) == 1 else REASON_CONSISTENT,
            chosen[2],
        )
    return chosen


def _command_phase(text: str, rules: CodexPhaseConfig) -> Tuple[Optional[str], str]:
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
        when it is explicitly neutral and ``(None, "")`` when it names no
        phase at all.
    """
    signal = _command_signal(text, rules)
    if signal is not None:
        return signal[0], signal[1]
    segments = _split_command(text, rules)
    if segments and all(
        _is_neutral_executable(words[0], rules) or _is_filter(words, rules)
        for words in segments
        if words
    ):
        return None, REASON_NEUTRAL
    return None, ""


def _here_document_phase(
    text: str, rules: CodexPhaseConfig
) -> Tuple[Optional[str], str]:
    """
    Return the phase of a script the agent piped into an interpreter.

    An agent that edits through ``python3 - <<'EOF'`` never declares a patch,
    so without this the turn holds no sign of the files it changed.  Only an
    explicit write says so: a read, a print or a report is not an edit.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig
        Validated write patterns.

    Returns
    -------
    Tuple[Optional[str], str]
        ``(implement_mode, reason)`` when a write pattern matches the body,
        otherwise ``(None, "")``.
    """
    if not rules.implement_write_patterns:
        return None, ""
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        return None, ""
    if not isinstance(arguments, dict):
        return None, ""
    for key in ("cmd", "command"):
        command = arguments.get(key)
        if not isinstance(command, str):
            continue
        _shell, body = _without_here_document(command)
        if body and any(
            pattern.search(body) for pattern in rules.implement_write_patterns
        ):
            return rules.implement_mode, REASON_SCRIPTED_EDIT
    return None, ""


_SEQUENCE_OPERATORS_RE = re.compile(r"\s*(?:;|&&|\|\||\||\n)\s*")


def _command_text(text: str) -> str:
    """The shell text of one call's JSON arguments, or the text itself."""
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        return str(text or "")
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for key in ("cmd", "command"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value
    return ""


def _shell_words(segment: str) -> List[str]:
    """
    Tokenize one shell segment, keeping the redirections.

    The routing lexer drops redirections as noise — where the output of a
    command went says nothing about which command ran.  Reading an edit is the
    opposite question, so the segment is lexed on its own here, with quotes
    resolved and an unclosed quote dropping the segment rather than guessing.
    """
    lexer = shlex.shlex(segment, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return []


def _exec_edit_phase(text: str, rules: CodexPhaseConfig):
    """
    Return the edit a shell command makes to a file, when it names one.

    A build of the CLI that has no patch tool edits through the shell:
    ``cat > src/a.py <<'EOF'``, ``… | tee src/a.py`` and ``sed -i`` all change
    a file without a patch envelope, and without a signal for them a turn that
    only did that carries no evidence of the change at all.  The signal is
    deliberately weak — it says *a file was written*, not what was written —
    and carries a reason of its own, so a reader can count how often it fired
    and how often it was right.

    A write the command does not name (a redirect into a substitution, a
    ``/dev/null``, a ``2>`` stream descriptor) is not an edit of the
    repository, and a command that only reads or prints is not an edit either.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call, or the command itself.
    rules : CodexPhaseConfig
        Validated phase rules; the interpreter's write patterns.

    Returns
    -------
    Optional[Tuple[str, str, str, Tuple[str, ...]]]
        ``(implement_mode, REASON_EXEC_EDIT, STRENGTH_WEAK, paths)`` when a
        named file is written, otherwise ``None``.
    """
    command = _command_text(text)
    if not command:
        return None
    shell, _body = _without_here_document(command)
    for segment in _SEQUENCE_OPERATORS_RE.split(shell):
        if not segment.strip():
            continue
        paths = tuple(_written_paths(_shell_words(segment)))
        if paths:
            return rules.implement_mode, REASON_EXEC_EDIT, STRENGTH_WEAK, paths
    return None


def _written_paths(words: List[str]) -> List[str]:
    """
    The files one shell segment writes, in the order it names them.

    A segment writes a file when it redirects output to a path, feeds
    ``tee`` one, or edits one in place with ``sed -i``.  A redirection whose
    target is a substitution, a stream descriptor (``2>``) or ``/dev/null``
    writes nothing the repository holds, and a command that only reads is not
    an edit at all.
    """
    paths: List[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word in (">", ">>"):
            # A bare digit before the operator is a file descriptor (2>), not
            # a path; the descriptor of a stream is never an edit.
            previous = words[index - 1] if index else ""
            target = words[index + 1] if index + 1 < len(words) else ""
            if not previous.isdigit() and _is_write_target(target):
                paths.append(target)
            index += 2
            continue
        if PurePosixPath(word).name == "tee":
            index += 1
            while index < len(words):
                candidate = words[index]
                if candidate.startswith("-"):
                    index += 1
                    continue
                if _is_write_target(candidate):
                    paths.append(candidate)
                index += 1
            continue
        if PurePosixPath(word).name == "sed" and any(
            argument == "-i"
            or (
                argument.startswith("-")
                and not argument.startswith("--")
                and "i" in argument[1:]
            )
            for argument in words[index + 1 :]
        ):
            for argument in words[index + 1 :]:
                if _is_write_target(argument):
                    paths.append(argument)
            break
        index += 1
    return paths


def _is_write_target(word: str) -> bool:
    """Whether one shell word names a file a redirect or ``tee`` writes."""
    if not word or word.startswith(("-", "$", "`", "(", "<", "1", "2")):
        return False
    if word in ("/dev/null", "/dev/stdout", "/dev/stderr"):
        return False
    if word == _SUBSTITUTION_PLACEHOLDER:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._/-]+", word))


def _test_path(path: str, rules: CodexPhaseConfig) -> bool:
    """Whether *path* names a test file under the configured test layout."""
    parts = PurePosixPath(path).parts
    name = parts[-1].lower()
    return (
        any(part.lower() in rules.test_directories for part in parts[:-1])
        or name.startswith(rules.test_filename_prefixes)
        or bool(rules.test_filename_pattern.search(name))
    )


def _patch_phase(text: str, rules: CodexPhaseConfig) -> Tuple[Optional[str], str]:
    """
    Return the phase of a Codex patch envelope, by the paths it touches.

    The envelope is looked for rather than assumed: a current Codex CLI applies
    a patch by piping an envelope to ``apply_patch`` inside ``exec_command``,
    where the name of the tool says nothing and only the decoded arguments do.

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
    if _ambiguous_patch_arguments(text):
        return None, ""
    for envelope in patch_envelopes(text):
        mode, reason = _envelope_phase(envelope, rules)
        if mode is not None:
            return mode, reason
    return None, ""


def _ambiguous_patch_arguments(text: str) -> bool:
    """Whether one call declares two different patches for the same work."""
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(arguments, dict):
        return False
    declared = [
        arguments[key]
        for key in ("patch", "input")
        if key in arguments and isinstance(arguments[key], str)
    ]
    return len(set(declared)) > 1


def patch_envelopes(text: str) -> List[str]:
    """
    Return every patch envelope a tool call carries, decoded.

    An envelope rides in three shapes: the ``patch``/``input`` argument of an
    ``apply_patch`` call, the body of a here-document a shell command pipes to
    ``apply_patch`` — what a current Codex CLI does — and, in older captures,
    the arguments already unencoded.  Arguments arrive JSON-encoded, so the line
    breaks of the envelope are ``\\n`` escapes and a line-anchored pattern over
    the raw call reads nothing.

    A here-document counts only when the command applies a patch: a file the
    agent writes can quote an envelope, and quoting one patched nothing.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call, or the envelope itself.

    Returns
    -------
    List[str]
        The envelope texts, in the order the call carries them.
    """
    envelopes: List[str] = []
    for candidate in _patch_argument_texts(text):
        for body in _envelope_bodies(candidate):
            for envelope in _envelope_spans(body):
                if envelope not in envelopes:
                    envelopes.append(envelope)
    return envelopes


def _patch_argument_texts(text: str) -> List[str]:
    """Return the argument values of one call that can hold an envelope."""
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        return [text]
    if isinstance(arguments, str):
        return [arguments]
    if not isinstance(arguments, dict):
        return [text]
    texts: List[str] = []
    for key in _PATCH_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip() and value not in texts:
            texts.append(value)
    return texts or [text]


def _envelope_bodies(candidate: str) -> List[str]:
    """Return the texts of one argument in which an envelope can sit."""
    if "<<" not in candidate or candidate.lstrip().startswith(_PATCH_BEGIN):
        return [candidate]
    shell, body = _without_here_document(candidate)
    if not body:
        return [candidate]
    if "apply_patch" not in shell and not body.lstrip().startswith(_PATCH_BEGIN):
        return []
    return [body]


def _envelope_spans(text: str) -> List[str]:
    """Cut every ``Begin Patch`` … ``End Patch`` block out of *text*."""
    if _PATCH_BEGIN not in text:
        return []
    lines = text.splitlines()
    spans: List[str] = []
    index = 0
    while index < len(lines):
        if lines[index] != _PATCH_BEGIN:
            index += 1
            continue
        closing = next(
            (
                stop
                for stop in range(index + 1, len(lines))
                if lines[stop] == _PATCH_END
            ),
            None,
        )
        if closing is None:
            break
        spans.append("\n".join(lines[index : closing + 1]))
        index = closing + 1
    return spans


def _envelope_phase(text: str, rules: CodexPhaseConfig) -> Tuple[Optional[str], str]:
    """
    Return the phase of one complete envelope, validated line by line.

    Parameters
    ----------
    text : str
        The envelope, from its first to its last marker line.
    rules : CodexPhaseConfig
        Validated patch and test-path rules.

    Returns
    -------
    Tuple[Optional[str], str]
        ``(mode, reason)``, or ``(None, "")`` for anything that is not a
        well-formed envelope over safe relative paths.
    """
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
            paths.append(line[len("*** Move to: ") :])
        elif operation == "Add" and line.startswith("+"):
            continue
        elif operation == "Update" and (
            line.startswith((" ", "+", "-", "@@")) or line == "*** End of File"
        ):
            continue
        else:
            return None, ""
    if not paths:
        return None, ""
    for path in paths:
        if (
            not path.strip()
            or path != path.strip()
            or "\\" in path
            or not PurePosixPath(path).parts
            or ":" in path
            or any(ord(char) < 32 for char in path)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or path.endswith("/")
        ):
            return None, ""
    mode = (
        rules.test_mode
        if all(_test_path(path, rules) for path in paths)
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
    envelope = text[: match.start()] if match else text
    codes = [int(found[1]) for found in _EXIT_CODE.finditer(envelope)]
    if codes:
        return not all(code != 0 for code in codes)
    if _TRACEBACK.search(envelope):
        return False
    return None


def collect_phase_evidence(
    activity: Tuple[CodexActivity, ...],
    rules: CodexPhaseConfig,
) -> Tuple[PhaseEvidence, ...]:
    """
    Return every phase evidence of the active turn, oldest first.

    Only explicit current announcements, recognized executed commands and Codex
    patch envelopes count; an interpreter a here-document fed with writes counts
    as an edit, and a test failure additionally requires a call linked in this
    turn.  Neutral or unrecognized activity simply produces no evidence, leaving
    the earlier signal in place.

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
                evidence.append(
                    PhaseEvidence(
                        mode=mode,
                        kind=EVIDENCE_ANNOUNCEMENT,
                        reason="announcement",
                        event_id=item.event_id,
                        completed=True,
                    )
                )
        elif item.kind == "function_call":
            edited_paths: Tuple[str, ...] = ()
            if item.name in rules.command_tools:
                # A command that pipes an envelope to ``apply_patch`` patches,
                # whatever the tools of this CLI release are called: the
                # envelope outranks the command line, which cannot see it.
                mode, reason = (
                    _patch_phase(item.text, rules)
                    if _PATCH_BEGIN in item.text
                    else (None, "")
                )
                if mode is not None:
                    strength = STRENGTH_STRONG
                    kind = EVIDENCE_PATCH
                else:
                    signal = _command_signal(item.text, rules)
                    if signal is None:
                        scripted = _here_document_phase(item.text, rules)
                        if scripted[0] is not None:
                            mode, reason = scripted[0], scripted[1]
                            strength = STRENGTH_STRONG
                            kind = EVIDENCE_PATCH
                        else:
                            # Nothing named the command and no script wrote a
                            # file; the last shape left is an edit the shell
                            # itself made — the one sign of changed code a CLI
                            # without a patch tool leaves behind, and only as a
                            # weak signal with a reason of its own.
                            exec_edit = _exec_edit_phase(item.text, rules)
                            if exec_edit is None:
                                continue
                            mode, reason = exec_edit[0], exec_edit[1]
                            strength = STRENGTH_WEAK
                            kind = EVIDENCE_PATCH
                            edited_paths = exec_edit[3]
                    else:
                        mode, reason, strength = signal
                        kind = EVIDENCE_COMMAND
            elif item.name in rules.patch_tools:
                mode, reason = _patch_phase(item.text, rules)
                strength = STRENGTH_STRONG
                kind = EVIDENCE_PATCH
            else:
                continue
            edited_paths = edited_paths if kind == EVIDENCE_PATCH else ()
            if mode is None:
                continue
            if item.call_id:
                calls.setdefault(item.call_id, (item.name, len(evidence)))
            evidence.append(
                PhaseEvidence(
                    mode=mode,
                    kind=kind,
                    reason=reason,
                    event_id=item.event_id,
                    call_id=item.call_id,
                    strength=strength,
                    paths=edited_paths,
                )
            )
        elif item.kind == "function_call_output" and item.call_id and item.name:
            linked = calls.get(item.call_id)
            if linked is None or linked[0] != item.name:
                continue
            settled = evidence[linked[1]]
            status = _execution_status(item.text)
            if status is None:
                continue
            calls.pop(item.call_id)
            evidence[linked[1]] = settled.settle(status)
            if (
                settled.mode == rules.test_mode
                and status is False
                and settled.kind == EVIDENCE_COMMAND
                and linked[1] == len(evidence) - 1
            ):
                evidence.append(
                    PhaseEvidence(
                        mode=rules.failure_mode,
                        kind=EVIDENCE_TEST_FAILURE,
                        reason="test command failed",
                        event_id=item.event_id,
                        call_id=item.call_id,
                        completed=True,
                        succeeded=False,
                    )
                )
    return tuple(evidence)


def _action_ages(
    activity: Tuple[CodexActivity, ...], rules: CodexPhaseConfig
) -> Tuple[Dict[str, int], int]:
    """
    Map each event to how many executed actions the turn had reached.

    Only a command or a patch is an executed action; an assistant utterance
    does not advance the clock, and a result is dated by the call it answers.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The assistant and tool events of the active turn.
    rules : CodexPhaseConfig
        Validated phase rules naming the tools that execute.

    Returns
    -------
    (Dict[str, int], int)
        Event identifier to action index, and the total actions of the turn.
    """
    ages: Dict[str, int] = {}
    step = 0
    for item in activity:
        if item.kind == "function_call":
            executes = (
                item.name in rules.command_tools or item.name in rules.patch_tools
            )
            if not executes:
                continue
            step += 1
            if item.event_id:
                ages[item.event_id] = step
        elif item.kind == "function_call_output" and item.event_id:
            ages[item.event_id] = step
    return ages, step


def detect_phase_evidence(
    activity: Tuple[CodexActivity, ...],
    rules: CodexPhaseConfig,
) -> Optional[PhaseEvidence]:
    """
    Return the latest phase evidence, or ``None`` when there is none.

    A strong signal outranks a weak one however recent the weak one is: a
    linter run after ``git log`` does not make the turn a review of code style.
    Among strong signals, the phase is the one dominating the last
    ``evidence_window`` actions of the turn, the most recent action breaking a
    tie.  One command among several of another kind is a step inside that work,
    not a change of it: an agent committing between two edits is still editing,
    and the model behind a long turn should not answer to every interleaved
    command.  With a window of one the latest action decides alone.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The assistant and tool events after the newest genuine user command.
    rules : CodexPhaseConfig
        Validated phase rules.

    Returns
    -------
    Optional[PhaseEvidence]
        The evidence naming the phase that dominates the recent strong signals
        of the turn, the recent weak ones when there are no strong ones.
    """
    evidence = collect_phase_evidence(activity, rules)
    if not evidence:
        return None
    if rules.signal_ttl is not None:
        # A command describes the moment it ran.  Once the turn has moved on by
        # more than the TTL without another executed signal, it is no longer
        # doing that work, so the aged signal is dropped and the cascade answers
        # from the instruction instead of from a step that has become history.
        ages, step = _action_ages(activity, rules)
        evidence = tuple(
            held
            for held in evidence
            if held.kind == EVIDENCE_ANNOUNCEMENT
            or step - ages.get(held.event_id, step) <= rules.signal_ttl
        )
        if not evidence:
            return None
    strong = [item for item in evidence if item.strength == STRENGTH_STRONG]
    pool = strong or evidence
    return _dominant_recent(pool, max(1, rules.evidence_window))


def _dominant_recent(evidence: List[PhaseEvidence], window: int) -> PhaseEvidence:
    """
    Return the evidence of the phase dominating the recent end of a turn.

    Parameters
    ----------
    evidence : List[PhaseEvidence]
        Equally strong evidence of one turn, oldest first.
    window : int
        How many of the most recent signals are weighed.

    Returns
    -------
    PhaseEvidence
        The latest evidence of the winning phase, so the reported kind and
        reason name a real action the agent took.
    """
    recent = evidence[-window:]
    totals: Dict[str, int] = {}
    for item in recent:
        totals[item.mode] = totals.get(item.mode, 0) + 1
    best = max(totals.values())
    leaders = {mode for mode, total in totals.items() if total == best}
    for item in reversed(recent):
        if item.mode in leaders:
            return item
    return recent[-1]


def detect_phase(
    activity: Tuple[CodexActivity, ...],
    rules: CodexPhaseConfig,
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


#: Kinds of action a command of an unnamed executable still describes.
_ACTION_NAMES: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"^(?:rg|ag|ack|grep|egrep|fgrep)$"), "searched code"),
    (re.compile(r"^(?:find|fd|ls|tree|du)$"), "listed files"),
    (
        re.compile(r"^(?:cat|head|tail|less|more|sed|awk|jq|column|wc)$"),
        "read files",
    ),
    (
        re.compile(r"^(?:ruff|flake8|mypy|pyright|pylint|black|isort|bandit)$"),
        "linted",
    ),
    (re.compile(r"^(?:curl|wget|httpie)$"), "called a service"),
    (re.compile(r"^(?:pip|pip3|uv|poetry|npm|yarn|pnpm)$"), "installed packages"),
)

#: Longest argument a described action may name.
_ACTION_SUBJECT_CHARS = 40


def _action_subject(words: List[str]) -> str:
    """Return one short, single-line argument that says what the action hit."""
    for word in words[1:]:
        if word.startswith("-") or word == _SUBSTITUTION_PLACEHOLDER:
            continue
        subject = word.strip("\"'")
        if subject and len(subject) <= _ACTION_SUBJECT_CHARS and "\n" not in subject:
            return subject
    return ""


def _describe_command(text: str, rules: Optional[CodexPhaseConfig]) -> str:
    """
    Describe an executed command the phase rules did not name.

    A phase rule is the precise description, but without one the command is
    still the agent's action and should not be reported as an anonymous tool
    call: ``ran git mv``, ``searched code`` and ``read files`` say what kind of
    work is happening, which is exactly what a mode is chosen from.  Only the
    shape of the command is read, never the output it produced.

    Parameters
    ----------
    text : str
        The raw JSON arguments of the call.
    rules : CodexPhaseConfig or None
        Phase rules, used to split the command the same way the phase reader does.

    Returns
    -------
    str
        A short clause naming the action, empty when nothing can be said.
    """
    if rules is None:
        return ""
    segments = _split_command(text, rules)
    if not segments:
        return ""
    words = segments[-1]
    executable = PurePosixPath(words[0]).name if words else ""
    if not executable or executable == _SUBSTITUTION_PLACEHOLDER:
        return ""
    for pattern, verb in _ACTION_NAMES:
        if pattern.fullmatch(executable):
            subject = _action_subject(words) if verb == "linted" else ""
            return f"{verb} {subject}".strip()
    return f"ran {executable}".strip()


def describe_activity(
    activity: Tuple[CodexActivity, ...],
    rules: Optional[CodexPhaseConfig] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Describe what the agent is *doing*, without repeating what it read.

    A raw tool output is the content of some file, the body of some page, the
    log of some build — topically about whatever the agent happened to look at,
    and about the work only through the action that produced it.  Embedding it
    lets one large read dominate the description of the current activity.  This
    renders the structure instead: which action ran, which recognized phase it
    is when a rule names it, whether it succeeded, how many files a patch
    touched.  An action the rules do not name is still described by its shape,
    so a long turn of reads, searches and lints says so instead of repeating one
    anonymous tool call.

    Repeated actions collapse into one clause carrying how often they ran, and
    the newest distinct actions are kept: the description stays bounded, stays
    identical while the work stays the same even when tool output changes, and
    spends its budget on different actions rather than on copies of one.

    Parameters
    ----------
    activity : Tuple[CodexActivity, ...]
        The events of the active user turn.
    rules : CodexPhaseConfig, optional
        Phase rules used to name recognized actions.  Without them the
        description still reports tools and statuses.
    limit : int, optional
        How many of the most recent distinct actions to keep.  Defaults to
        ``rules.activity_description_limit``, or three without rules.

    Returns
    -------
    str
        One clause per distinct action, oldest first; empty when nothing is
        describable.
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
        command_tool = bool(rules is not None and item.name in rules.command_tools)
        patch_tool = bool(rules is not None and item.name in rules.patch_tools)
        if item.name and patch_tool:
            clause = (
                f"editing files ({evidence.reason})"
                if evidence is not None
                else "editing files"
            )
        elif evidence is not None and evidence.kind == EVIDENCE_COMMAND:
            clause = f"ran {evidence.reason}"
        elif command_tool:
            clause = _describe_command(item.text, rules) or f"called {item.name}"
        else:
            clause = f"called {item.name or 'a tool'}"
        if (
            evidence is not None
            and evidence.completed
            and evidence.succeeded is not None
        ):
            clause += ": succeeded" if evidence.succeeded else ": failed"
        if clause:
            clauses.append(clause)
    if limit is None:
        limit = rules.activity_description_limit if rules is not None else 3
    counts: Dict[str, int] = {}
    ordered: List[str] = []
    for clause in clauses:
        counts[clause] = counts.get(clause, 0) + 1
        if clause in ordered:
            ordered.remove(clause)
        ordered.append(clause)
    if limit > 0:
        ordered = ordered[-limit:]
    rendered = [
        clause if counts[clause] < 2 else f"{clause} x{counts[clause]}"
        for clause in ordered
    ]
    return "\n".join(rendered)
