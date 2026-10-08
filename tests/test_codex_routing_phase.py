import json
from pathlib import Path

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    collect_phase_evidence,
    detect_phase,
    detect_phase_evidence,
    describe_activity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase_config import (
    CodexPhaseConfig,
)


def phase_raw(overrides=None):
    path = (
        Path(__file__).resolve().parents[1]
        / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))["settings"]["phase"]
    return {**raw, **(overrides or {})}


@pytest.fixture
def rules():
    return CodexPhaseConfig.from_raw(phase_raw())


def assistant(text):
    return CodexActivity(kind="assistant", text=text)


def command(text, name="exec_command", call_id="call-1", key="cmd"):
    return CodexActivity(
        kind="function_call",
        name=name,
        call_id=call_id,
        text=json.dumps({key: text}),
    )


def output(text, name="exec_command", call_id="call-1"):
    return CodexActivity(
        kind="function_call_output",
        text=text,
        name=name,
        call_id=call_id,
    )


def patch(*paths):
    text = "*** Begin Patch\n"
    for path in paths:
        text += "*** Add File: " + path + "\n+content\n"
    text += "*** End Patch"
    return CodexActivity(kind="function_call", name="apply_patch", text=text)


def test_late_test_failure_does_not_replace_newer_action(rules):
    activity = (
        command("pytest -q"),
        patch("src/main.py"),
        output("Process exited with code 1\nOutput:\nfailure"),
    )
    assert detect_phase(activity, rules) == "implement"
    assert collect_phase_evidence(activity, rules)[0].succeeded is False


def test_duplicate_test_output_does_not_replace_newer_action(rules):
    failed = output("Process exited with code 1\nOutput:\nfailure")
    activity = (command("pytest -q"), failed, patch("src/main.py"), failed)
    assert detect_phase(activity, rules) == "implement"


def test_unknown_execution_status_is_not_described_as_failure(rules):
    activity = (command("pytest"), output("Waiting for execution to finish"))
    assert describe_activity(activity, rules) == "ran pytest"


def test_intermediate_output_does_not_hide_final_failure(rules):
    activity = (
        command("pytest"),
        output("Waiting for execution to finish"),
        output("Process exited with code 1\nOutput:\nfailure"),
    )
    assert detect_phase(activity, rules) == "debug"


@pytest.mark.parametrize(
    "text, expected",
    [
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
    ],
)
def test_explicit_current_announcements(text, expected, rules):
    assert detect_phase((assistant(text),), rules) == expected


@pytest.mark.parametrize(
    "text",
    [
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
    ],
)
def test_noncurrent_uncertain_or_quoted_announcements(text, rules):
    assert detect_phase((assistant(text),), rules) is None


@pytest.mark.parametrize(
    "text, expected",
    [
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
        ("pytest | tee output", "test"),
        ("pytest > output", None),
        ("pytest &", None),
        ("'pytest", None),
        ("echo ';' pytest", None),
        ("echo '&&' pytest", None),
        ("false && pytest", None),
        ("pytest --help", None),
        ("git diff --help", None),
    ],
)
def test_executed_command_signals(text, expected, rules):
    assert detect_phase((command(text),), rules) == expected
    activity = (command(text, name="shell_command", key="command"),)
    assert detect_phase(activity, rules) == expected


@pytest.mark.parametrize(
    "text",
    [
        "pytest",
        "[]",
        "null",
        '{"cmd": 123}',
        '{"cmd": "pytest", "command": "git diff"}',
    ],
)
def test_invalid_command_arguments(text, rules):
    item = CodexActivity(kind="function_call", name="exec_command", text=text)
    assert detect_phase((item,), rules) is None


def test_advertised_and_unrecognized_tools_are_not_execution(rules):
    activity = (assistant("Available tools: exec_command pytest apply_patch"),)
    assert detect_phase(activity, rules) is None
    assert detect_phase((command("pytest", name="tool_description"),), rules) is None


@pytest.mark.parametrize(
    "paths, expected",
    [
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
    ],
)
def test_patch_paths(paths, expected, rules):
    assert detect_phase((patch(*paths),), rules) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "*** Begin Patch\n*** Delete File: tests/test_old.py\n*** End Patch",
            "test",
        ),
        (
            "*** Begin Patch\n*** Update File: tests/test_old.py\n"
            "*** Move to: src/main.py\n@@\n-old\n+new\n*** End Patch",
            "implement",
        ),
        (
            "*** Begin Patch\n*** Update File: src/main.py\n@@\n"
            " context\n-old\n+new\n*** End Patch",
            "implement",
        ),
        ("*** Begin Patch\n*** End Patch", None),
        ("*** Add File: tests/test_main.py\n+content", None),
        (
            "*** Begin Patch\n*** Add File: tests/test_main.py\n"
            "unmarked content\n*** End Patch",
            None,
        ),
        ("*** Begin Patch\n*** Unknown File: src/main.py\n*** End Patch", None),
        ("I might apply *** Add File: src/main.py", None),
    ],
)
def test_patch_envelope_and_operations(text, expected, rules):
    for arguments in (
        text,
        json.dumps({"patch": text}),
        json.dumps({"input": text}),
    ):
        item = CodexActivity(
            kind="function_call", name="apply_patch", text=arguments
        )
        assert detect_phase((item,), rules) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Process exited with code 1",
        "Exit code: 2",
        '{"exit_code": -1, "output": "failed"}',
        "Traceback (most recent call last):\n  ...\nAssertionError",
    ],
)
def test_linked_test_failure_is_debug(text, rules):
    assert detect_phase((command("pytest"), output(text)), rules) == "debug"


@pytest.mark.parametrize(
    "text",
    [
        "debug implement pytest git diff failed error",
        "FAILED tests/test_main.py",
        "Process exited with code 0",
        "Exit code: 1\nExit code: 0",
        "Traceback (most recent call last):\nExit code: 0",
        "Documentation mentions exit code 1 as an example.",
        'Example: {"exit_code": 1}',
        '{"exit_code": false}',
        '{"exit_code": "1"}',
    ],
)
def test_output_words_and_uncertain_failure_do_not_change_phase(text, rules):
    assert detect_phase((command("pytest"), output(text)), rules) == "test"


def test_output_requires_matching_executed_test_call(rules):
    failure = "Process exited with code 1"
    assert detect_phase((output(failure),), rules) is None
    assert (
        detect_phase((command("pytest"), output(failure, name="")), rules) == "test"
    )
    assert (
        detect_phase((command("pytest"), output(failure, call_id="other")), rules)
        == "test"
    )
    activity = (command("pytest"), output(failure, name="shell_command"))
    assert detect_phase(activity, rules) == "test"
    assert (
        detect_phase((command("git diff"), output(failure)), rules) == "git_review"
    )
    assert detect_phase((output(failure), command("pytest")), rules) == "test"
    activity = (command("pytest", call_id=""), output(failure, call_id=""))
    assert detect_phase(activity, rules) == "test"


def test_latest_clear_signal_wins_and_neutral_activity_preserves_it(rules):
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
    assert detect_phase(activity, rules) == "git_review"
    assert detect_phase(activity[:3], rules) == "implement"
    assert detect_phase(activity[:5], rules) == "debug"
    assert detect_phase((), rules) is None


def test_no_cross_turn_memory(rules):
    assert detect_phase((command("pytest"),), rules) == "test"
    assert detect_phase((output("Exit code: 1"),), rules) is None


def test_phase_can_be_disabled_without_disabling_keywords():
    rules = CodexPhaseConfig.from_raw(phase_raw({"enabled": False}))
    assert detect_phase((command("pytest"),), rules) is None


def test_custom_announcements_replace_defaults():
    rules = CodexPhaseConfig.from_raw(
        phase_raw(
            {
                "announcement_prefix": r"^Next:\s+",
                "announcements": {"review": "inspect configuration"},
            }
        )
    )
    assert (
        detect_phase((assistant("Next: inspect configuration"),), rules) == "review"
    )
    assert detect_phase((assistant("Teraz uruchomię pytest."),), rules) is None


def test_custom_command_and_tool_rules_replace_defaults():
    rules = CodexPhaseConfig.from_raw(
        phase_raw(
            {
                "command_tools": ["run_shell"],
                "commands": [
                    {"executable": "cargo", "args_prefix": ["test"], "mode": "test"}
                ],
            }
        )
    )
    assert detect_phase((command("cargo test", name="run_shell"),), rules) == "test"
    assert detect_phase((command("cargo build", name="run_shell"),), rules) is None
    assert detect_phase((command("pytest", name="run_shell"),), rules) is None
    assert detect_phase((command("cargo test"),), rules) is None
    assert (
        detect_phase((command("cargo test | tee log", name="run_shell"),), rules)
        == "test"
    )


def test_conflicting_command_rules_do_not_pick_by_order():
    rules = CodexPhaseConfig.from_raw(
        phase_raw(
            {
                "commands": [
                    {"executable": "cargo", "args_prefix": [], "mode": "implement"},
                    {"executable": "cargo", "args_prefix": ["test"], "mode": "test"},
                ]
            }
        )
    )
    assert detect_phase((command("cargo test"),), rules) is None


def test_custom_test_paths_and_failure_mode():
    rules = CodexPhaseConfig.from_raw(
        phase_raw(
            {
                "test_directories": ["checks"],
                "test_filename_prefixes": ["check_"],
                "test_filename_pattern": r"\.check\.rs$",
                "failure_mode": "review",
            }
        )
    )
    assert detect_phase((patch("checks/main.rs"),), rules) == "test"
    assert detect_phase((patch("src/check_main.rs"),), rules) == "test"
    assert detect_phase((patch("src/main.check.rs"),), rules) == "test"
    assert detect_phase((patch("tests/main.rs"),), rules) == "implement"
    assert (
        detect_phase((command("pytest"), output("Exit code: 1")), rules) == "review"
    )


def test_empty_rule_collections_disable_only_their_signals():
    rules = CodexPhaseConfig.from_raw(
        phase_raw({"announcements": {}, "commands": []})
    )
    assert (
        detect_phase(
            (assistant("Teraz uruchomię pytest."), command("pytest")), rules
        )
        is None
    )
    assert detect_phase((patch("src/main.py"),), rules) == "implement"


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"enabled": "false"},
        {"enabled": 0},
        {"unknown": True},
        {"announcement_prefix": "["},
        {"uncertain": 123},
        {"announcements": []},
        {"announcements": {"test": "["}},
        {"command_tools": "exec_command"},
        {"patch_tools": [1]},
        {"test_directories": [""]},
        {"test_filename_pattern": "["},
        {"commands": {}},
        {"commands": [{"executable": "pytest"}]},
        {"commands": [{"executable": "[", "args_prefix": [], "mode": "test"}]},
        {
            "commands": [
                {"executable": "pytest", "args_prefix": "test", "mode": "test"}
            ]
        },
        {"commands": [{"executable": "pytest", "args_prefix": [], "mode": False}]},
        {"test_mode": ""},
    ],
)
def test_invalid_phase_configuration_is_rejected(raw):
    with pytest.raises(ValueError, match=r"settings\.phase"):
        CodexPhaseConfig.from_raw(phase_raw(raw) if isinstance(raw, dict) else raw)


def test_explicit_rules_require_configured_modes():
    with pytest.raises(ValueError, match="Unknown settings.phase modes"):
        CodexPhaseConfig.from_raw(
            phase_raw({"failure_mode": "missing"}), mode_names=["implement"]
        )


def test_routing_config_uses_only_supplied_phase_rules():
    from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
        CodexRoutingConfig,
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["settings"]["phase"]["enabled"] = False
    config = CodexRoutingConfig._from_raw(raw)
    assert config.phase.enabled is False
    assert config.heuristic_enabled is True
    raw["settings"]["phase"] = {"enabled": False}
    with pytest.raises(ValueError, match="Missing settings.phase fields"):
        CodexRoutingConfig._from_raw(raw)
    raw["settings"]["phase"] = None
    with pytest.raises(ValueError, match=r"settings\.phase"):
        CodexRoutingConfig._from_raw(raw)


# --- compound commands, filters and evidence --------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "cd /srv/work/llm-router && git show e65a66b --stat && "
            "git show e65a66b | head -80",
            "git_review",
        ),
        ("git log --oneline main..HEAD | tee /tmp/log", "git_review"),
        ("MODE=ci python -m pytest -q | tee ci.log", "test"),
        ("cd repo && cd tests && pytest", "test"),
        ("git status && git diff", "git_review"),
    ],
)
def test_safe_compound_commands_keep_the_producer_phase(text, expected, rules):
    assert detect_phase((command(text),), rules) == expected


@pytest.mark.parametrize(
    "text",
    [
        "pytest | sh",
        "pytest | python -c 'print(1)'",
        "git diff | tee log && pytest",
        "pytest && git diff",
        "pytest || true",
        "head -1 tests/test_main.py",
        "git show HEAD | cat | sh",
    ],
)
def test_unsupported_or_conflicting_composites_abstain(text, rules):
    assert detect_phase((command(text),), rules) is None


def test_filters_are_configurable_and_replaceable():
    strict = CodexPhaseConfig.from_raw(phase_raw({"neutral_filters": ["head"]}))
    assert detect_phase((command("pytest | head -5"),), strict) == "test"
    assert detect_phase((command("pytest | tee log"),), strict) is None
    none = CodexPhaseConfig.from_raw(phase_raw({"neutral_filters": []}))
    assert detect_phase((command("pytest | head -5"),), none) is None


def test_invalid_neutral_filter_configuration_is_rejected():
    for raw in (
        {"neutral_filters": "head"},
        {"neutral_filters": [""]},
        {"neutral_filters": ["["]},
        {"announcement_followup_max_chars": -1},
        {"announcement_followup_max_chars": "10"},
    ):
        with pytest.raises(ValueError, match=r"settings\.phase"):
            CodexPhaseConfig.from_raw(phase_raw(raw))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Teraz uruchomię testy.\n", "test"),
        ("Teraz uruchomię testy.\n\nPotem podsumuję wynik.", "test"),
        ("Now I will update the CHANGELOG.\nThe entry covers the fix.", "implement"),
    ],
)
def test_trailing_newlines_and_short_explanations_are_tolerated(
    text, expected, rules
):
    assert detect_phase((assistant(text),), rules) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Teraz uruchomię testy.\n" + "X" * 400,
        "Teraz uruchomię testy.\nTeraz przejrzę diff.",
    ],
)
def test_long_bodies_and_second_announcements_do_not_announce(text, rules):
    assert detect_phase((assistant(text),), rules) is None


def test_followup_budget_is_configurable():
    strict = CodexPhaseConfig.from_raw(
        phase_raw({"announcement_followup_max_chars": 0})
    )
    assert (
        detect_phase(
            (assistant("Teraz uruchomię testy.\nPotem podsumuję."),), strict
        )
        is None
    )
    assert detect_phase((assistant("Teraz uruchomię testy."),), strict) == "test"


def test_failure_is_read_from_the_result_envelope_not_the_captured_body(rules):
    body = "Process exited with code 1\nTraceback (most recent call last):\nAssertionError"
    output_of_a_green_run = (
        "Chunk ID: 4899d0\nWall time: 0.1000 seconds\n"
        "Process exited with code 0\nOriginal token count: 40\nOutput:\n" + body
    )
    assert (
        detect_phase((command("pytest"), output(output_of_a_green_run)), rules)
        == "test"
    )


def test_infrastructure_error_is_not_a_test_failure(rules):
    gateway = (
        "Process exited with code 0\nOutput:\n"
        "error: authentication request failed: 401 Unauthorized\n"
        "connection refused, retrying\n"
    )
    assert detect_phase((command("pytest"), output(gateway)), rules) == "test"
    assert detect_phase((command("git log"), output(gateway)), rules) == "git_review"


def test_evidence_carries_kind_reason_and_settlement(rules):
    activity = (
        command("python -m pytest tests/test_a.py", call_id="c1"),
        output("Process exited with code 1", call_id="c1"),
    )
    evidence = detect_phase_evidence(activity, rules)
    assert evidence.mode == "debug"
    assert evidence.kind == "test_failure"
    assert evidence.call_id == "c1"
    assert evidence.completed is True
    assert evidence.succeeded is False


def test_evidence_of_a_running_command_is_unsettled(rules):
    evidence = detect_phase_evidence((command("python -m pytest"),), rules)
    assert (
        evidence.mode,
        evidence.kind,
        evidence.completed,
        evidence.succeeded,
    ) == (
        "test",
        "command",
        False,
        None,
    )
    assert evidence.reason == "python -m pytest"


def test_evidence_keeps_the_event_identifier_of_its_payload_item():
    with_ids = CodexPhaseConfig.from_raw(phase_raw())
    activity = (
        CodexActivity(
            kind="function_call",
            name="exec_command",
            call_id="c1",
            event_id="fc_01",
            text='{"cmd": "pytest"}',
        ),
        CodexActivity(
            kind="function_call_output",
            name="exec_command",
            call_id="c1",
            event_id="fco_01",
            text="Process exited with code 1",
        ),
    )
    collected = collect_phase_evidence(activity, with_ids)
    assert [item.event_id for item in collected] == ["fc_01", "fco_01"]
    assert collected[0].completed is True


def test_announcement_evidence_explains_itself(rules):
    evidence = detect_phase_evidence((assistant("Teraz uruchomię pytest."),), rules)
    assert (evidence.kind, evidence.reason, evidence.mode) == (
        "announcement",
        "announcement",
        "test",
    )


def test_patch_evidence_names_the_envelope(rules):
    evidence = detect_phase_evidence((patch("src/main.py"),), rules)
    assert evidence.kind == "patch"
    assert evidence.reason.startswith("patch ")
