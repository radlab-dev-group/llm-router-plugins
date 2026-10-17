"""
Error signatures: a stable, normalized identity for a failure.

A signature reduces a stack trace or test report to what matters — the error
class and message, not the line numbers or the machine — so the same failure
recognized two weeks apart still matches a recipe.  A green run yields no
signature at all.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.signature import (
    signature_of,
    looks_like_failure,
)


def test_a_pytest_failure_yields_a_signature():
    text = "FAILED tests/test_a.py::test_x - AssertionError: expected 1 got 2"
    signature = signature_of(text)
    assert signature is not None
    assert signature.signature_hash
    assert "AssertionError" in signature.signature


def test_the_same_failure_has_a_stable_hash_across_runs():
    text = "FAILED tests/test_a.py::test_x - AssertionError: expected 1 got 2"
    assert signature_of(text).signature_hash == signature_of(text).signature_hash


def test_traceback_lines_do_not_change_the_signature():
    # The same exception raised from two different files and line numbers is
    # one signature: the chain and the message survive, the location does not.
    first = (
        "Traceback (most recent call last):\n"
        '  File "src/a.py", line 10, in run\n'
        "AssertionError: expected 1 got 2"
    )
    second = (
        "Traceback (most recent call last):\n"
        '  File "other/a.py", line 999, in run\n'
        "AssertionError: expected 1 got 2"
    )
    assert signature_of(first).signature_hash == signature_of(second).signature_hash


def test_different_failures_have_different_signatures():
    a = signature_of("AssertionError: expected 1 got 2")
    b = signature_of("KeyError: 'session'")
    assert a.signature_hash != b.signature_hash


def test_a_green_run_has_no_signature():
    assert signature_of("1 passed in 0.02s") is None
    assert signature_of("Process exited with code 0") is None


def test_looks_like_failure_matches_failures_only():
    assert looks_like_failure("Process exited with code 1")
    assert looks_like_failure("FAILED tests/test_a.py::test_x")
    assert not looks_like_failure("1 passed")
