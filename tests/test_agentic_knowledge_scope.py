"""
Repository scope: one collection per repository, paths relativized.

The scope is the only repository identifier the CLIs expose — the declared
working directory.  It is turned into a path-free key, and paths are made
relative to it so no absolute layout leaks into a knowledge document.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    RepositoryScope,
    ScopeResolver,
    STRATEGY_PARENT_DIRECTORY,
    STRATEGY_REPO_BASENAME,
    declared_working_directory,
)


def test_declares_the_cwd_tag():
    payload = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "<environment_context>\n"
                            "<cwd>/home/u/dev/webapp</cwd>\n"
                            "</environment_context>"
                        ),
                    }
                ],
            }
        ]
    }
    assert declared_working_directory(payload) == "/home/u/dev/webapp"


def test_declares_the_working_directory_line_for_claude():
    payload = {"system": "You are a coder.\nWorking directory: /home/u/dev/webapp"}
    assert declared_working_directory(payload) == "/home/u/dev/webapp"


def test_no_declaration_yields_no_scope():
    assert declared_working_directory({"input": []}) == ""


def test_repo_basename_strategy_uses_the_directory_name():
    resolver = ScopeResolver(strategy=STRATEGY_REPO_BASENAME)
    scope = resolver.resolve({"system": "Working directory: /home/u/dev/webapp"})
    assert scope.key == "webapp"
    assert scope.repo_root == "/home/u/dev/webapp"
    assert scope.known


def test_parent_directory_strategy_qualifies_the_name():
    resolver = ScopeResolver(strategy=STRATEGY_PARENT_DIRECTORY)
    scope = resolver.resolve({"system": "Working directory: /home/u/org/webapp"})
    assert scope.key == "org-webapp"


def test_overrides_win_over_the_strategy():
    resolver = ScopeResolver(overrides={"/home/u/dev/webapp": "the-repo"})
    scope = resolver.resolve({"system": "Working directory: /home/u/dev/webapp"})
    assert scope.key == "the-repo"


def test_paths_are_made_relative_to_the_repo_root():
    scope = RepositoryScope(key="webapp", repo_root="/home/u/dev/webapp")
    assert scope.relative("/home/u/dev/webapp/src/a.py") == "src/a.py"


def test_paths_outside_the_repo_fall_back_to_their_basename():
    scope = RepositoryScope(key="webapp", repo_root="/home/u/dev/webapp")
    assert scope.relative("/etc/passwd") == "passwd"


def test_unknown_strategy_is_rejected_by_config():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )

    try:
        AgenticKnowledgeConfig.from_json('{"scope": {"strategy": "bogus"}}')
    except ValueError:
        return
    raise AssertionError("expected an unknown strategy to be rejected")


def _message(role, text):
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _context(cwd="/home/u/dev/webapp"):
    return f"<environment_context>\n<cwd>{cwd}</cwd>\n</environment_context>"


def test_declaration_behind_the_leading_messages_is_still_read():
    """
    A compaction request re-sends the window, so the context block is no longer
    among the first items — and the scope of the whole conversation depends on
    finding it there.
    """
    developer = _message("developer", "You are a coding agent. " * 900)
    payload = {
        "input": [
            developer,
            _message("developer", "instructions"),
            _message("assistant", "ok"),
            _message("user", _context() + "\ncontinue"),
        ]
    }
    assert declared_working_directory(payload) == "/home/u/dev/webapp"


def test_declaration_deep_in_one_message_is_read():
    payload = {"input": [_message("user", "x" * 30_000 + "\n" + _context())]}
    assert declared_working_directory(payload) == "/home/u/dev/webapp"


def test_a_directory_quoted_in_a_tool_output_is_not_a_declaration():
    """Only what the agent sends as a message declares a repository."""
    payload = {
        "input": [
            _message("user", "what is in there?"),
            {"type": "function_call_output", "output": _context("/other/repo")},
        ]
    }
    assert declared_working_directory(payload) == ""


def test_the_first_declaration_wins_over_a_later_quoted_one():
    payload = {
        "input": [
            _message("user", _context("/home/u/dev/webapp")),
            _message(
                "user", "and in another checkout: " + _context("/home/u/other")
            ),
        ]
    }
    assert declared_working_directory(payload) == "/home/u/dev/webapp"


def test_the_scan_is_bounded_per_message():
    """
    A request path may not pay for a history it does not need: past the ceiling
    of one message the payload is simply not searched.
    """
    payload = {"input": [_message("user", "x" * 300_000 + "\n" + _context())]}
    assert declared_working_directory(payload) == ""


def test_the_scan_is_bounded_per_request():
    lead = [_message("user", "x" * 200_000) for _ in range(12)]
    payload = {"input": lead + [_message("user", _context())]}
    assert declared_working_directory(payload) == ""
