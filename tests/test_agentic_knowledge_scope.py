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
