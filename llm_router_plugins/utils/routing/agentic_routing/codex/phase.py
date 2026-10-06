"""Conservative, turn-local phase signals from observed Codex activity."""

import json
import re
import shlex
from pathlib import PurePosixPath
from typing import Optional, Tuple

from .payload import CodexActivity


_ANNOUNCEMENT_PREFIX = re.compile(
    r"^(?:teraz\s+(?:ja\s+)?|"
    r"now\s+(?:i\s+(?:will\s+|am\s+)|i['’](?:ll|m)\s+)?|"
    r"i\s+(?:will|am)\s+now\s+|i['’](?:ll|m)\s+now\s+)",
    re.IGNORECASE,
)
_ANNOUNCEMENTS = {
    "implement": re.compile(
        r"(?:edytuję|dodam|zaktualizuję|zmienię|"
        r"edit(?:ing)?|add(?:ing)?|updat(?:e|ing)|modify(?:ing)?)\s+"
        r"(?:the\s+)?(?:wpis\s+do\s+)?"
        r"(?:changelog\b|readme\b|docs?\b|documentation\b|dokumentację\b|"
        r"kod\b|code\b|plik\b|pliki\b|files?\b|"
        r"[\w./-]+\.(?:py|js|ts|tsx|jsx|md|rst|txt)\b)"
        r"(?:\.(?:md|rst|txt))?",
        re.IGNORECASE,
    ),
    "test": re.compile(
        r"(?:uruchomię|uruchamiam|run(?:ning)?)\s+"
        r"(?:the\s+)?(?:pytest|python(?:3)?\s+-m\s+(?:pytest|unittest)|"
        r"unittest|npm\s+(?:run\s+)?test|testy|tests|test suite)"
        r"(?:\s+--?[\w./=-]+)*",
        re.IGNORECASE,
    ),
    "debug": re.compile(
        r"(?:diagnozuję|zdiagnozuję|naprawię|naprawiam|"
        r"debug(?:ging)?|diagnos(?:e|ing)|fix(?:ing)?|investigat(?:e|ing))\s+"
        r"(?:the\s+)?(?:błąd|błędy|awarię|bug|error|failure|"
        r"failing test|test failure)",
        re.IGNORECASE,
    ),
    "git_review": re.compile(
        r"(?:przejrzę|przeglądam|sprawdzę|review(?:ing)?|inspect(?:ing)?)\s+"
        r"(?:the\s+)?(?:diff|git\s+(?:diff|log|show|blame)|"
        r"changes|zmiany)",
        re.IGNORECASE,
    ),
}
_UNCERTAIN = re.compile(
    r"\b(?:not|never|don't|won't|cannot|can't|if|maybe|might|would|"
    r"but|then|later|after|before|instead|nie|jeśli|gdy|może|"
    r"potem|później|zamiast|ale|oraz|and|lub|or)\b",
    re.IGNORECASE,
)
_EXIT_CODE = re.compile(
    r"^\s*(?:(?:process\s+)?exited\s+with\s+code\s+|"
    r"exit[ _-]code\s*[:=]?\s*)(-?\d+)\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _announcement(text: str) -> Optional[str]:
    text = text.strip()
    if "\n" in text or "```" in text or _UNCERTAIN.search(text):
        return None
    match = _ANNOUNCEMENT_PREFIX.match(text)
    if not match:
        return None
    action = text[match.end():].rstrip(".! ").replace("`", "")
    phases = [
        phase for phase, pattern in _ANNOUNCEMENTS.items()
        if pattern.fullmatch(action)
    ]
    return phases[0] if len(phases) == 1 else None


def _command_phase(text: str) -> Optional[str]:
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
        if executable in ("pytest", "unittest"):
            phases.add("test")
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
            if len(args) >= 2 and args[:2] in (
                ["-m", "pytest"], ["-m", "unittest"]
            ):
                phases.add("test")
            else:
                return None
        elif executable == "npm" and (
            args[:1] == ["test"] or args[:2] == ["run", "test"]
        ):
            phases.add("test")
        elif executable == "git":
            while args:
                if args[0] == "--no-pager":
                    args = args[1:]
                elif args[0] == "-C" and len(args) >= 2:
                    args = args[2:]
                else:
                    break
            if args and args[0] in ("log", "diff", "show", "blame"):
                phases.add("git_review")
            elif args[:1] != ["status"]:
                return None
        elif executable != "cd":
            return None
    return next(iter(phases)) if len(phases) == 1 else None


def _test_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1].lower()
    return (
        any(part.lower() in ("test", "tests", "__tests__") for part in parts[:-1])
        or name.startswith("test_")
        or bool(re.search(r"(?:_test\.py|\.(?:test|spec)\.[\w]+)$", name))
    )


def _patch_phase(text: str) -> Optional[str]:
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
    return "test" if all(_test_path(path) for path in paths) else "implement"


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


def detect_phase(activity: Tuple[CodexActivity, ...]) -> Optional[str]:
    """Return the latest unambiguous phase, without retaining cross-turn state.

    Only explicit current announcements, recognized executed commands and Codex
    patch envelopes count. Test failures require a matching call in this turn.
    Neutral or ambiguous activity leaves the previous clear signal intact.
    """
    phase = None
    calls = {}
    for item in activity:
        signal = None
        if item.kind == "assistant":
            signal = _announcement(item.text)
        elif item.kind == "function_call":
            if item.name in ("exec_command", "shell_command"):
                signal = _command_phase(item.text)
            elif item.name == "apply_patch":
                signal = _patch_phase(item.text)
            if item.call_id:
                calls[item.call_id] = (item.name, signal)
        elif item.kind == "function_call_output" and item.name and item.call_id:
            if calls.get(item.call_id) == (item.name, "test"):
                if item.name in ("exec_command", "shell_command"):
                    if _failed_test(item.text):
                        signal = "debug"
        if signal is not None:
            phase = signal
    return phase