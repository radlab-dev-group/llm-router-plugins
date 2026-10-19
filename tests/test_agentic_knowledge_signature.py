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


# --- run statistics must not reach the signature ---------------------------
# A re-run with ``--lf`` or ``-x``, or a sibling that flaked, changes the
# summary counts.  The failure is the same, so the signature must be too, or
# the loop detector never sees a repeat.
_HEAD = "FAILED tests/test_a.py::test_x - AssertionError: boom"


def test_sibling_counts_in_the_summary_do_not_change_the_signature():
    one = f"{_HEAD}\n=== 1 failed, 2 passed in 0.14s ==="
    rerun = f"{_HEAD}\n=== 1 failed in 0.09s ==="
    crowd = f"{_HEAD}\n=== 1 failed, 5 passed, 1 error in 0.30s ==="
    assert signature_of(one).signature_hash == signature_of(rerun).signature_hash
    assert signature_of(one).signature_hash == signature_of(crowd).signature_hash


def test_the_summary_line_is_not_mistaken_for_a_distinct_failure():
    # Only a summary line and a real failure line: the summary collapses, so
    # the signature carries the failure and nothing run-specific.
    signature = signature_of(f"{_HEAD}\n=== 2 failed, 8 passed in 0.20s ===")
    assert "failed" not in signature.signature
    assert "passed" not in signature.signature


# --- an assertion value is the failure, not run noise ----------------------
def test_the_same_assertion_at_a_different_path_or_line_matches():
    first = (
        "Traceback (most recent call last):\n"
        '  File "/a/1.py", line 12, in run\n'
        "AssertionError: assert total == 100"
    )
    moved = (
        "Traceback (most recent call last):\n"
        '  File "/moved/deep/1.py", line 340, in run\n'
        "AssertionError: assert total == 100"
    )
    assert signature_of(first).signature_hash == signature_of(moved).signature_hash


def test_different_asserted_values_have_different_signatures():
    low = signature_of("AssertionError: assert total == 100")
    high = signature_of("AssertionError: assert total == 900")
    assert low.signature_hash != high.signature_hash
    assert "100" in low.signature
    assert "900" in high.signature


def test_reversed_expected_and_got_have_different_signatures():
    a = signature_of("AssertionError: expected 3 got 4")
    b = signature_of("AssertionError: expected 4 got 3")
    assert a.signature_hash != b.signature_hash


def test_path_line_and_address_noise_is_still_stripped():
    # The assertion value survives, but a line number and an object address
    # around it do not, so two runs of one failure still match.
    first = "AssertionError: at foo.py:12 0x7fa3, expected 42"
    moved = "AssertionError: at foo.py:340 0x19bc, expected 42"
    assert signature_of(first).signature_hash == signature_of(moved).signature_hash
    assert "42" in signature_of(first).signature
    assert "0x7fa3" not in signature_of(first).signature


# --- go test and ESLint shapes are recognized ------------------------------
_GO_TEST = (
    "--- FAIL: TestAdd (0.00s)\n"
    "    add_test.go:15: got 3, want 4\n"
    "FAIL\tpkg/math\t0.005s"
)


def test_a_go_test_failure_yields_a_stable_signature():
    assert looks_like_failure(_GO_TEST)
    signature = signature_of(_GO_TEST)
    assert signature is not None
    assert "TestAdd" in signature.signature
    # A re-run only changes durations; the signature is unchanged.
    rerun = (
        "--- FAIL: TestAdd (0.01s)\n"
        "    add_test.go:15: got 3, want 4\n"
        "FAIL\tpkg/math\t0.008s"
    )
    assert signature_of(rerun).signature_hash == signature.signature_hash


def test_an_eslint_failure_yields_a_stable_signature():
    report = (
        "/repo/src/app.js\n"
        "  12:5  error  Unexpected console statement  no-console\n"
        "\n1 problem (1 error, 0 warnings)"
    )
    assert looks_like_failure(report)
    signature = signature_of(report)
    assert signature is not None
    assert "no-console" in signature.signature
    rerun = (
        "/repo/src/app.js\n"
        "  12:5  error  Unexpected console statement  no-console\n"
        "\n1 problem (1 error, 0 warnings)"
    )
    assert signature_of(rerun).signature_hash == signature.signature_hash


# --- exit-code reports agree with looks_like_failure -----------------------
def test_an_exit_code_report_yields_a_signature():
    text = "Process exited with code 1"
    assert looks_like_failure(text)
    signature = signature_of(text)
    assert signature is not None
    assert signature.signature_hash
