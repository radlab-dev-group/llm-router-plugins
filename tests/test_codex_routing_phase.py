import json

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import CodexActivity
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import detect_phase


def assistant(text):
    return CodexActivity(kind="assistant", text=text)


def command(text, name="exec_command", call_id="call-1", key="cmd"):
    return CodexActivity(
        kind="function_call", name=name, call_id=call_id,
        text=json.dumps({key: text}),
    )


def output(text, name="exec_command", call_id="call-1"):
    return CodexActivity(
        kind="function_call_output", text=text, name=name, call_id=call_id,
    )


def patch(*paths):
    text = "*** Begin Patch\n"
    for path in paths:
        text += "*** Add File: " + path + "\n+content\n"
    text += "*** End Patch"
    return CodexActivity(kind="function_call", name="apply_patch", text=text)


@pytest.mark.parametrize("text, expected", [
    ("Teraz edytuję CHANGELOG.", "implement"),
    ("Teraz dodam wpis do CHANGELOG.", "implement"),
    ("Now I'm editing `CHANGELOG.md`.", "implement"),
    ("I will now update the documentation.", "implement"),
    ("Teraz uruchomię pytest.", "test"),
    ("Now I'll run python -m pytest.", "test"),
    ("I'll now run npm test.", "test"),
    ("Teraz diagnozuję błąd.", "debug"),
    ("Teraz naprawię błąd.", "debug"),
    ("Now I will fix the failing test.", "debug"),
    ("Teraz przejrzę diff.", "git_review"),
    ("Now I'm reviewing the diff.", "git_review"),
])
def test_explicit_current_announcements(text, expected):
    assert detect_phase((assistant(text),)) == expected


@pytest.mark.parametrize("text", [
    "pytest CHANGELOG debug git diff",
    "Updated CHANGELOG and ran pytest. Everything passed.",
    "I reviewed the diff.",
    "Teraz nie uruchomię pytest.",
    "Now I will not run pytest.",
    "Now I will run pytest later.",
    "Now I will run pytest if needed.",
    "Now I will run pytest and review the diff.",
    "Now I will run pytest, but not yet.",
    "Now I would run pytest.",
    '"Now I will run pytest."',
    "> Now I will run pytest.",
    "```\nNow I will run pytest.\n```",
    "The assistant said: Now I will run pytest.",
    "Summary: Now I will run pytest.",
    "Now I will run pytest.\nNow I will review the diff.",
    "Now I will edit the tests after reviewing the diff.",
    "Now I will explain pytest.",
    "Now I will run pytest succeeded.",
    "Now I will edit CHANGELOG completed.",
    "Now I will review the diff finished.",
    "Now I will run ```pytest```.",
])
def test_noncurrent_uncertain_or_quoted_announcements(text):
    assert detect_phase((assistant(text),)) is None


@pytest.mark.parametrize("text, expected", [
    ("pytest -q tests", "test"),
    ("python -m pytest", "test"),
    ("python3 -m unittest discover", "test"),
    ("/venv/bin/python3.12 -m pytest", "test"),
    ("unittest", "test"),
    ("npm test -- --runInBand", "test"),
    ("npm run test", "test"),
    ("MODE=test pytest -q", "test"),
    ("cd project && pytest -q", "test"),
    ("git log -5", "git_review"),
    ("git diff --stat", "git_review"),
    ("git show HEAD", "git_review"),
    ("git blame src/main.py", "git_review"),
    ("git -C project --no-pager diff", "git_review"),
    ("git status", None),
    ("echo 'pytest -q'", None),
    ("printf 'git diff'", None),
    ("python script.py pytest", None),
    ("npm install", None),
    ("git diff && pytest", None),
    ("pytest || true", None),
    ("echo $(pytest)", None),
    ("pytest | tee output", None),
    ("pytest > output", None),
    ("pytest &", None),
    ("'pytest", None),
    ("echo ';' pytest", None),
    ("echo '&&' pytest", None),
    ("false && pytest", None),
    ("pytest --help", None),
    ("git diff --help", None),
])
def test_executed_command_signals(text, expected):
    assert detect_phase((command(text),)) == expected
    activity = (command(text, name="shell_command", key="command"),)
    assert detect_phase(activity) == expected


@pytest.mark.parametrize("text", [
    "pytest", "[]", "null", '{"cmd": 123}',
    '{"cmd": "pytest", "command": "git diff"}',
])
def test_invalid_command_arguments(text):
    item = CodexActivity(kind="function_call", name="exec_command", text=text)
    assert detect_phase((item,)) is None


def test_advertised_and_unrecognized_tools_are_not_execution():
    activity = (assistant("Available tools: exec_command pytest apply_patch"),)
    assert detect_phase(activity) is None
    assert detect_phase((command("pytest", name="tool_description"),)) is None


@pytest.mark.parametrize("paths, expected", [
    (("src/main.py",), "implement"),
    (("docs/testing.md",), "implement"),
    (("CHANGELOG.md",), "implement"),
    (("tests/test_main.py",), "test"),
    (("src/tests/main.py", "test_helper.py"), "test"),
    (("src/main.test.ts", "src/main.spec.js"), "test"),
    (("tests/test_main.py", "src/main.py"), "implement"),
    (("tests/test_main.py", "README.md"), "implement"),
    (("contest/main.py",), "implement"),
    (("../tests/test_main.py",), None),
    (("/tests/test_main.py",), None),
    (("tests/",), None),
    ((".",), None),
    (("C:/tests/test_main.py",), None),
])
def test_patch_paths(paths, expected):
    assert detect_phase((patch(*paths),)) == expected


@pytest.mark.parametrize("text, expected", [
    ("*** Begin Patch\n*** Delete File: tests/test_old.py\n*** End Patch", "test"),
    ("*** Begin Patch\n*** Update File: tests/test_old.py\n"
     "*** Move to: src/main.py\n@@\n-old\n+new\n*** End Patch", "implement"),
    ("*** Begin Patch\n*** Update File: src/main.py\n@@\n"
     " context\n-old\n+new\n*** End Patch", "implement"),
    ("*** Begin Patch\n*** End Patch", None),
    ("*** Add File: tests/test_main.py\n+content", None),
    ("*** Begin Patch\n*** Add File: tests/test_main.py\n"
     "unmarked content\n*** End Patch", None),
    ("*** Begin Patch\n*** Unknown File: src/main.py\n*** End Patch", None),
    ("I might apply *** Add File: src/main.py", None),
])
def test_patch_envelope_and_operations(text, expected):
    for arguments in (text, json.dumps({"patch": text}), json.dumps({"input": text})):
        item = CodexActivity(kind="function_call", name="apply_patch", text=arguments)
        assert detect_phase((item,)) == expected


@pytest.mark.parametrize("text", [
    "Process exited with code 1",
    "Exit code: 2",
    '{"exit_code": -1, "output": "failed"}',
    "Traceback (most recent call last):\n  ...\nAssertionError",
])
def test_linked_test_failure_is_debug(text):
    assert detect_phase((command("pytest"), output(text))) == "debug"


@pytest.mark.parametrize("text", [
    "debug implement pytest git diff failed error",
    "FAILED tests/test_main.py",
    "Process exited with code 0",
    "Exit code: 1\nExit code: 0",
    "Traceback (most recent call last):\nExit code: 0",
    "Documentation mentions exit code 1 as an example.",
    'Example: {"exit_code": 1}',
    '{"exit_code": false}',
    '{"exit_code": "1"}',
])
def test_output_words_and_uncertain_failure_do_not_change_phase(text):
    assert detect_phase((command("pytest"), output(text))) == "test"


def test_output_requires_matching_executed_test_call():
    failure = "Process exited with code 1"
    assert detect_phase((output(failure),)) is None
    assert detect_phase((command("pytest"), output(failure, name=""))) == "test"
    assert detect_phase((command("pytest"), output(failure, call_id="other"))) == "test"
    activity = (command("pytest"), output(failure, name="shell_command"))
    assert detect_phase(activity) == "test"
    assert detect_phase((command("git diff"), output(failure))) == "git_review"
    assert detect_phase((output(failure), command("pytest"))) == "test"
    activity = (command("pytest", call_id=""), output(failure, call_id=""))
    assert detect_phase(activity) == "test"


def test_latest_clear_signal_wins_and_neutral_activity_preserves_it():
    activity = (
        assistant("Teraz dodam CHANGELOG."),
        command("git status"),
        assistant("Done."),
        command("pytest"),
        output("Process exited with code 1"),
        assistant("Now I will review the diff."),
        command("git diff && pytest"),
        assistant("Summary: pytest failed."),
    )
    assert detect_phase(activity) == "git_review"
    assert detect_phase(activity[:3]) == "implement"
    assert detect_phase(activity[:5]) == "debug"
    assert detect_phase(()) is None


def test_no_cross_turn_memory():
    assert detect_phase((command("pytest"),)) == "test"
    assert detect_phase((output("Exit code: 1"),)) is None