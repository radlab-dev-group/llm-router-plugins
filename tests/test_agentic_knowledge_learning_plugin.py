"""
The learned path driven through the real plugin.

The policy tests cover the learner alone; these cover the seams the request
path crosses: the exposure a read writes to the training stream, the graded
label the settle path hands the learner instead of the coarse bit, the
annotation that names the model which served, and the fail-open rule that a
learner that cannot be reached is a request that goes through.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FAILING,
    GREEN,
    FakePublisher,
    FakeStore,
    _command,
    _command_output,
    base_items,
    codex_payload,
    knowledge_config,
    loop_items,
    main_payload,
    recipe_payload,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    LearningConfig,
)


class FakeRedis:
    """The handful of commands the learner uses, in memory."""

    def __init__(self) -> None:
        self.hashes = {}
        self.values = {}
        self.streams = {}
        self.ttls = {}
        self.calls = []

    def hset(self, key, field, value):
        self.calls.append(("hset", key, field))
        self.hashes.setdefault(key, {})[field] = str(value)
        return 1

    def hgetall(self, key):
        self.calls.append(("hgetall", key))
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount=1):
        self.calls.append(("hincrby", key, field, amount))
        current = int(self.hashes.setdefault(key, {}).get(field, "0"))
        self.hashes[key][field] = str(current + amount)
        return current + amount

    def expire(self, key, seconds):
        self.ttls[key] = int(seconds)
        return True

    def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    def set(self, key, value):
        self.calls.append(("set", key))
        self.values[key] = value
        return True

    def xadd(self, key, fields, maxlen=0):
        self.calls.append(("xadd", key))
        stream = self.streams.setdefault(key, [])
        stream.append((f"{len(stream) + 1}-0", dict(fields)))
        if maxlen:
            del stream[: max(0, len(stream) - maxlen)]
        return stream[-1][0]

    def xrange(self, key, min="-", max="+", count=0):
        rows = list(self.streams.get(key, []))
        return rows[:count] if count else rows


class BrokenRedis(FakeRedis):
    """Every command fails; the layer must not notice."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise ConnectionError(f"{name} unavailable")

        return _raise


def make_learning_plugin(client, enabled, hits=()):
    """A real plugin over a fake engine, with the learner on or off."""
    config = knowledge_config()
    config.learning = LearningConfig(
        enabled=enabled,
        min_samples=1000,
        min_observations=1000,
        explore_rate=0.0,
        half_life_seconds=0.0,
    )
    return AgenticKnowledgePlugin(
        config=config,
        store=FakeStore(hits=tuple(hits)),
        cache=InMemoryContextCache(),
        publisher=FakePublisher(),
        redis_client=client,
    )


def test_an_injection_writes_its_features_to_the_training_stream():
    """The exposure is logged before the verdict, and with learning off."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=False, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    rows = client.streams.get("agentic_kb:training") or []
    assert rows, "an injection wrote no training record"
    record = json.loads(rows[0][1]["d"])
    assert record["kind"] == "injection"
    assert record["label"] == 0.0
    assert record["scope"] == "webapp"
    assert record["doc"]
    # The record carries features, never content.
    for banned in ("def ", "src/", "/home/", "traceback", "assert"):
        assert banned not in json.dumps(record)


def test_a_repeated_failure_blames_the_recipe_the_learner_saw():
    """The graded label of the settle path reaches the learner, not only the counter."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    # The same failure comes back: the injected recipe did not resolve it.
    # Turn two re-injects the block, so the pending entry is the new one —
    # the verdict it settled is already folded into the reputation.
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
    stats = plugin._learning.reputation.stats(doc)
    # The posterior moved against the document: a full judgement of failure.
    assert stats["beta"] > 2.0
    assert stats["exposure"] >= 1
    # …and the same verdict is in the stream, graded per document.
    rows = [
        json.loads(payload["d"])
        for _, payload in client.streams.get("agentic_kb:training", [])
    ]
    graded = [row for row in rows if row["label"] < 0]
    assert graded, "no graded label reached the stream"
    assert graded[0]["doc"] == doc
    # The fake hit carries no signature and no paths, so the verdict could
    # only be the block's, marked coarse: the credit is collective, and the
    # model is told so.
    assert graded[0]["kind"] == "coarse"


def test_a_verified_fix_credits_the_recipe_it_names():
    """A turn that mines a recipe credits the document it was shown."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    # The next turn of this session fixes the failure and mines a recipe.
    plugin.apply(recipe_payload(turn_id="turn-2", root_turn_id="gen-2"))
    stats = plugin._learning.reputation.stats(doc)
    assert stats["alpha"] > 2.0
    rows = [
        json.loads(payload["d"])
        for _, payload in client.streams.get("agentic_kb:training", [])
    ]
    credited = [row for row in rows if row["label"] > 0]
    assert credited, "the fix mined this turn credited nothing"
    assert credited[0]["doc"] == doc
    assert credited[0]["kind"] in ("confirmed", "plausible", "coarse")


def test_the_annotation_names_the_model_that_served_the_request():
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["enabled"] is True
    assert learning["range"] == [0.3, 1.5]
    # Nothing was fitted yet, so the null model serves and says so.
    assert learning["model_id"] == ""
    assert learning["labelled"] == 0
    feedback = payload["routing"]["knowledge"]["metrics"]["feedback"]
    assert feedback["source"] == "redis"


def test_the_annotation_names_the_multiplier_of_each_candidate():
    """Why one recipe ranked above a stronger one is answerable per candidate.

    The request annotation carries the learner's account of each hit it
    weighed: the multiplier it applied, the model it read, and whether the
    read was an exploration.  A candidate the learner did not touch — no
    learner, no scope, or a policy off — carries no entry, and a disabled
    policy carries none for every candidate, because it has no vote to
    explain.
    """
    import random

    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        publish_snapshot,
        train_ranker,
    )
    from test_agentic_knowledge_learning import corpus

    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    # Fit and publish a model with real signal, so the annotation has a
    # model to name.
    snapshot, _ = train_ranker(
        corpus(300, signal=True, seed=11),
        "webapp",
        config=plugin._config.learning,
        rng=random.Random(11),
    )
    publish_snapshot(client, snapshot, key_prefix=plugin._config.learning.key_prefix)
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["model_id"] == snapshot.model_id
    assert learning["labelled"] == 300
    # The hits this block injected are weighed by the learner, and the
    # annotation says so per candidate: the multiplier the vote carried, the
    # model that produced it, and whether the read was an exploration.
    hits = {
        name: detail for name, detail in (learning.get("candidates") or {}).items()
    }
    assert hits, "the candidate the learner weighed is not named in the annotation"
    for name, detail in hits.items():
        assert name.startswith("recipe:")
        # The multiplier the candidate's vote actually carries, within the
        # policy's range, with the reasons for it.
        assert 0.3 <= float(detail["mult"]) <= 1.5
        assert detail["reason"] in ("learned", "reputation")
        assert detail["model_id"] == snapshot.model_id
        assert "explore" in detail


def test_a_disabled_learning_section_is_the_layer_it_always_was():
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=False, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["enabled"] is False
    # Nothing was fitted and nothing moves: the annotation names the null
    # model, and the request path did not read a snapshot to rank the block.
    assert learning["model_id"] == ""
    assert not any(
        call[0] == "get" and call[1].startswith("agentic-kb:model:")
        for call in client.calls
    )


def test_a_broken_redis_does_not_cost_a_request():
    """Every learned read is fail-open: the request is answered, state falls back."""
    plugin = make_learning_plugin(
        BrokenRedis(), enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    assert payload["routing"]["knowledge"]["learning"]["enabled"] is True
    # …and the next turn settles against the same in-process state.
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))


# ------------------------------------------------- the attribution window ---
#: A failure the injected block was not shown how to fix, so a turn that
#: repeats it judges nothing the previous request was shown.
OTHER_FAILURE = "FAILED tests/test_z.py::test_other - ValueError: unrelated"


def other_failure(turn_id, root_turn_id):
    """A request stuck on a failure the injected block never named."""
    return codex_payload(
        loop_items(failures=1, failure=OTHER_FAILURE),
        turn_id=turn_id,
        root_turn_id=root_turn_id,
    )


def _graded_labels(client, doc):
    """The non-zero labels one document earned, in stream order."""
    rows = [
        json.loads(payload["d"])
        for _, payload in client.streams.get("agentic_kb:training", [])
    ]
    return [
        (row["label"], row["kind"])
        for row in rows
        if row["doc"] == doc and row["label"] != 0.0
    ]


def test_a_verdict_two_turns_late_is_credited_with_the_square_of_the_discount():
    """The window the plan closed: shown, edited, proved — on three turns.

    The injection of turn one is not judged by turn two, which repeats a
    different failure and injects nothing, and it is not forgotten for it:
    the green test of turn three proves it two turns late, and the credit
    arrives discounted by the delay, ``gamma`` squared, because a fix
    applied three turns after it was shown was less caused by the showing
    than one applied the next turn.
    """
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    # Turn two repeats a failure the block was not shown how to fix, and the
    # engine answers nothing: no verdict, no injection, and the window keeps
    # what turn one showed.
    plugin._store = FakeStore(hits=())
    plugin.apply(other_failure(turn_id="turn-2", root_turn_id="gen-2"))
    window = plugin._pending_feedback["sess-1"]
    assert [int(entry[0]) for entry in window] == [1]
    # Turn three proves turn one's failure fixed.
    plugin._store = FakeStore(hits=(store_hit("recipe", score=0.9),))
    plugin.apply(recipe_payload(turn_id="turn-3", root_turn_id="gen-3"))
    labels = _graded_labels(client, doc)
    assert labels == [(pytest.approx(0.85 ** 2), "coarse")]


def test_a_verdict_arrives_once_even_when_the_window_holds_two_injections():
    """Three showings, one green test, one piece of evidence.

    The same document injected on turn one and turn two is credited once by
    the verdict of turn three, at the nearest injection that showed it —
    counting it twice is the self-reinforcement the monitors watch for,
    manufactured out of the credit rule itself.
    """
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    # Turn two is stuck on a failure the block never named, so it judges
    # nothing that turn one showed — and re-injects the same document, so the
    # window now holds two showings of it.
    plugin.apply(other_failure(turn_id="turn-2", root_turn_id="gen-2"))
    window = plugin._pending_feedback["sess-1"]
    assert [int(entry[0]) for entry in window] == [2, 1]
    plugin.apply(recipe_payload(turn_id="turn-3", root_turn_id="gen-3"))
    labels = _graded_labels(client, doc)
    assert len(labels) == 1
    # The credit is at the delay of the nearest injection, not the oldest.
    assert labels[0][0] == pytest.approx(0.85)
    # …and every entry the verdict read is gone: what remains is the showing
    # of the verdict's own turn, waiting for its own verdict.
    assert [int(entry[0]) for entry in plugin._pending_feedback["sess-1"]] == [3]


def test_a_turn_that_proves_nothing_keeps_the_window_it_was_given():
    """No verdict is not a verdict of zero; the entries wait for one."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    # A different failure, nothing injected: turn two judges nothing that
    # turn one showed, and the layer does not get to say otherwise.
    plugin._store = FakeStore(hits=())
    plugin.apply(other_failure(turn_id="turn-2", root_turn_id="gen-2"))
    window = plugin._pending_feedback["sess-1"]
    assert [int(entry[0]) for entry in window] == [1]
    assert _graded_labels(client, doc) == []


# --------------------------------------------------- adoption and revert ---
def _recipe_hit_with_change():
    """A recipe that names a file and quotes the change that fixed it.

    The header carries the paths the verdict matches against and the fenced
    diff whose ``+`` lines are the side of the adoption comparison the turn
    is checked on.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
        build_document_text,
    )

    return store_hit(
        "recipe",
        score=0.9,
        text=build_document_text(
            "recipe",
            "Failure: KeyError: 'alg'\n"
            "Change that fixed it:\n"
            "```diff\n"
            "- x = 1\n"
            "+ x = 2\n"
            "+ y = 3\n"
            "```",
            signature="KeyError: 'alg'",
            signature_hash="deadbeef",
            paths=("src/export.py",),
            mode="debug",
            test_command="pytest tests/test_a.py",
        ),
    )


def _patch_call(call_id, event_id, envelope):
    return {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": call_id,
        "id": event_id,
        "arguments": envelope,
    }


def _envelope(path, added, removed=""):
    body = "*** Begin Patch\n"
    body += f"*** Update File: {path}\n@@\n"
    for line in str(removed).splitlines():
        if line:
            body += f"-{line}\n"
    for line in str(added).splitlines():
        if line:
            body += f"+{line}\n"
    body += "*** End Patch"
    return body


def test_a_turn_that_applies_the_change_is_credited_as_adopted():
    """The only credit route the layer can observe: shown, adopted, green.

    The turn repeated the lines the document's diff adds, and the same
    command went green, so the credit is a whole one, marked ``adopted`` —
    not the half credit of a document that merely shared a file with the
    turn, and not the coarse one of a block whose documents the verdict
    could not read individually.  The label still carries the window's
    discount, because it arrived one turn after the showing.
    """
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(_recipe_hit_with_change(),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    items = base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", FAILING, code=1),
        _patch_call(
            "p1",
            "pe",
            _envelope("src/export.py", added="x = 2\ny = 3", removed="x = 1"),
        ),
        _command("c2", "e2"),
        _command_output("c2", "o2", GREEN, code=0),
    ]
    plugin.apply(codex_payload(items, turn_id="turn-2", root_turn_id="gen-2"))
    assert _graded_labels(client, doc) == [(pytest.approx(0.85), "adopted")]


def test_a_turn_that_touched_the_file_for_other_reasons_is_unattributed():
    """Sharing a file with a turn is not the turn having applied the change.

    The document named ``src/export.py`` and the turn edited the same file,
    but repeated none of the lines the document's diff adds: the turn had
    its own reason to be there.  That is not evidence either way, so the
    document is logged as an exposure with no label rather than paid a
    credit it did not cause — the filter that stops the layer teaching
    itself that being read is the same as being right.
    """
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(_recipe_hit_with_change(),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    items = base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", FAILING, code=1),
        _patch_call(
            "p1",
            "pe",
            _envelope("src/export.py", added="unrelated = 100", removed="old = 9"),
        ),
        _command("c2", "e2"),
        _command_output("c2", "o2", GREEN, code=0),
    ]
    plugin.apply(codex_payload(items, turn_id="turn-2", root_turn_id="gen-2"))
    assert _graded_labels(client, doc) == []


def test_a_change_the_agent_took_back_debunks_the_document_outright():
    """Applied and then undone is the strongest negative without a repeat.

    A ``git checkout --`` of the very file the document describes leaves the
    repository without the fix the document shows, and nothing about the
    turn's other evidence — a green test elsewhere, a moved-on agent —
    outranks that: the label is a full ``-1.0``, marked ``reverted``, and it
    still carries the window's discount, because the undo arrived one turn
    after the showing.
    """
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(_recipe_hit_with_change(),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0][1][0].document_name
    items = base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", FAILING, code=1),
        _patch_call(
            "p1",
            "pe",
            _envelope("src/export.py", added="x = 2\ny = 3", removed="x = 1"),
        ),
        _command("c2", "e2", "git checkout -- src/export.py"),
        _command_output("c2", "o2", "", code=0),
    ]
    plugin.apply(codex_payload(items, turn_id="turn-2", root_turn_id="gen-2"))
    assert _graded_labels(client, doc) == [(pytest.approx(-0.85), "reverted")]
