"""Conservative, turn-local phase signals from observed Codex activity."""

import json
import re
import shlex
from pathlib import PurePosixPath
from typing import Optional, Tuple

from .payload import CodexActivity
from .phase_config import CodexPhaseConfig


_EXIT_CODE = re.compile(
    r"^\s*(?:(?:process\s+)?exited\s+with\s+code\s+|"
    r"exit[ _-]code\s*[:=]?\s*)(-?\d+)\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _announcement(text: str, rules: CodexPhaseConfig) -> Optional[str]:
    text = text.strip()
    if "\n" in text or "```" in text or rules.uncertain.search(text):
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


def _command_phase(text: str, rules: CodexPhaseConfig) -> Optional[str]:
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
        tokens = list(lexer)
    except (ValueError, TypeError):
        return None
    segments = [[]]
    for token in tokens:
        if token in (";", "&&"):
            if not segments[-1]:
                return None
            segments.append([])
        elif token and all(char in ";&|()" for char in token):
            return None
        else:
            segments[-1].append(token)
    phases = set()
    for words in segments:
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
        matches = {
            rule.mode for rule in rules.commands
            if rule.executable.fullmatch(executable)
            and tuple(args[:len(rule.args_prefix)]) == rule.args_prefix
        }
        if len(matches) != 1:
            return None
        signal = next(iter(matches))
        if signal is not None:
            phases.add(signal)
    return next(iter(phases)) if len(phases) == 1 else None


def _test_path(path: str, rules: CodexPhaseConfig) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1].lower()
    return (
        any(part.lower() in rules.test_directories for part in parts[:-1])
        or name.startswith(rules.test_filename_prefixes)
        or bool(rules.test_filename_pattern.search(name))
    )


def _patch_phase(text: str, rules: CodexPhaseConfig) -> Optional[str]:
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        patches = [arguments[key] for key in ("patch", "input") if key in arguments]
        if not patches or any(not isinstance(patch, str) for patch in patches):
            return None
        if len(set(patches)) != 1:
            return None
        text = patches[0]
    lines = text.strip().splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        return None
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
            return None
    if not paths:
        return None
    for path in paths:
        if (
            not path.strip() or path != path.strip() or "\\" in path
            or not PurePosixPath(path).parts or ":" in path
            or any(ord(char) < 32 for char in path)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or path.endswith("/")
        ):
            return None
    return (
        rules.test_mode if all(_test_path(path, rules) for path in paths)
        else rules.implement_mode
    )


def _failed_test(text: str) -> bool:
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        result = None
    if isinstance(result, dict) and "exit_code" in result:
        code = result["exit_code"]
        return type(code) is int and code != 0
    codes = [int(match[1]) for match in _EXIT_CODE.finditer(text)]
    if codes:
        return all(code != 0 for code in codes)
    return bool(re.search(r"^Traceback \(most recent call last\):", text, re.MULTILINE))


def detect_phase(
    activity: Tuple[CodexActivity, ...], rules: CodexPhaseConfig,
) -> Optional[str]:
    """Return the latest unambiguous phase, without retaining cross-turn state.

    Only explicit current announcements, recognized executed commands and Codex
    patch envelopes count. Test failures require a matching call in this turn.
    Neutral or ambiguous activity leaves the previous clear signal intact.
    """
    if not rules.enabled:
        return None
    phase = None
    calls = {}
    for item in activity:
        signal = None
        if item.kind == "assistant":
            signal = _announcement(item.text, rules)
        elif item.kind == "function_call":
            if item.name in rules.command_tools:
                signal = _command_phase(item.text, rules)
            elif item.name in rules.patch_tools:
                signal = _patch_phase(item.text, rules)
            if item.call_id:
                calls[item.call_id] = (item.name, signal)
        elif item.kind == "function_call_output" and item.name and item.call_id:
            if calls.get(item.call_id) == (item.name, rules.test_mode):
                if item.name in rules.command_tools:
                    if _failed_test(item.text):
                        signal = rules.failure_mode
        if signal is not None:
            phase = signal
    return phase