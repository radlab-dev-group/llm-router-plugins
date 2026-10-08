"""Regressions for balanced Codex semantic ranking without loading ML models."""

import json
import math
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexMode,
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
    CodexRequest,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import (
    CodexSemanticLayer,
)
from llm_router_plugins.utils.routing.embedder import (
    EmbeddingRouter,
    EmbeddingRouterConfig,
)

_MODE_NAMES = ("plan", "implement", "test", "review", "git_review", "debug")
_SPECIAL_NAMES = ("aux_title", "compaction")
_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "agentic_routing_codex.json"
)
_INVALID_COSINES = (
    None,
    True,
    False,
    "0.8",
    [],
    {},
    float("nan"),
    float("inf"),
    float("-inf"),
    1.01,
    -1.01,
)
_INVALID_SETTINGS = (
    ("aggregation", "unknown"),
    ("aggregation", None),
    ("min_margin", -0.01),
    ("min_margin", 2.01),
    ("min_margin", float("nan")),
    ("min_margin", float("inf")),
    ("min_margin", float("-inf")),
    ("min_margin", True),
    ("min_margin", "0.05"),
    ("min_margin", None),
) + tuple(
    (field, value)
    for field in ("intent_max_chars", "phase_max_chars")
    for value in (0, -1, True, False, 1.5, "2000", None)
)


@pytest.fixture
def raw_config():
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def mode_by_name():
    return {
        name: CodexMode(
            name=name, model_name=f"model-{name}", description=name, examples=()
        )
        for name in _MODE_NAMES + _SPECIAL_NAMES
    }


def _make_router(documents, names=_MODE_NAMES, aggregation="per_target_top_k"):
    kwargs = {}
    if aggregation is not None:
        kwargs["aggregation"] = aggregation
    config = EmbeddingRouterConfig(
        embedding_model="unused-test-model",
        chunk_size=256,
        chunk_overlap=64,
        top_k=3,
        routing_targets=tuple(
            SimpleNamespace(
                name=name, model_name=f"model-{name}", description=name, examples=()
            )
            for name in names
        ),
        **kwargs,
    )
    router = EmbeddingRouter(config)
    scores = np.array([[score for _, score in documents]], dtype=np.float32)
    ids = np.arange(len(documents), dtype=np.int64).reshape(1, -1)
    router._faiss_index = SimpleNamespace(
        ntotal=len(documents),
        search=Mock(side_effect=lambda vector, k: (scores[:, :k], ids[:, :k])),
    )
    router._doc_store = {index: name for index, (name, _) in enumerate(documents)}
    router._model = SimpleNamespace(
        tokenizer=None,
        max_seq_length=None,
        encode=Mock(return_value=np.array([[1.0, 0.0]], dtype=np.float32)),
    )
    router._initialized = True
    return router


def _make_layer(
    mode_by_name,
    router=None,
    threshold=0.51,
    min_margin=0.05,
    intent_max_chars=2000,
    phase_max_chars=2000,
):
    return CodexSemanticLayer(
        router,
        threshold,
        mode_by_name,
        min_margin=min_margin,
        intent_max_chars=intent_max_chars,
        phase_max_chars=phase_max_chars,
    )


def _ranking(winner="plan", similarity=0.8, runner_up=0.6):
    scores = [
        {"target": name, "similarity": similarity if name == winner else runner_up}
        for name in _MODE_NAMES
    ]
    return {
        "target_name": winner,
        "model_name": f"model-{winner}",
        "similarity": similarity,
        "all_scores": scores,
    }


def _assistant(text):
    return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}


@pytest.mark.parametrize("scarce_count", [1, 2, 3])
def test_per_target_top_k_balances_all_six_modes(scarce_count):
    fragments = {
        name: [0.1 - index * 0.1, 0.9 - index * 0.1, -0.9, 0.7 - index * 0.1, -0.95]
        for index, name in enumerate(_MODE_NAMES)
    }
    fragments["debug"] = [-0.8, -0.2, -0.5][:scarce_count]
    documents = [
        (name, score) for name, scores in fragments.items() for score in scores
    ]
    documents += [("aux_title", 1.0), ("compaction", 1.0), ("unconfigured", 1.0)]
    router = _make_router(documents)

    result = router.route("ambiguous request")

    router._faiss_index.search.assert_called_once()
    vector, k = router._faiss_index.search.call_args.args
    assert k == len(documents)
    np.testing.assert_allclose(vector, [[1.0, 0.0]])
    expected = {
        name: sum(sorted(scores, reverse=True)[:scarce_count]) / scarce_count
        for name, scores in fragments.items()
    }
    actual = {entry["target"]: entry["similarity"] for entry in result["all_scores"]}
    assert len(result["all_scores"]) == 6
    assert actual == pytest.approx(expected)
    assert [entry["similarity"] for entry in result["all_scores"]] == sorted(
        actual.values(), reverse=True
    )
    assert result["target_name"] == "plan"
    assert result["model_name"] == "model-plan"
    assert result["similarity"] == pytest.approx(expected["plan"])


def test_shared_k_changes_winner_instead_of_using_each_targets_own_k():
    fragments = {
        "plan": [0.99, 0.9, -0.9],
        "implement": [0.8, 0.79],
        **{name: [0.4, 0.3, 0.2] for name in _MODE_NAMES[2:]},
    }
    router = _make_router(
        [(name, score) for name, scores in fragments.items() for score in scores]
    )

    result = router.route("request")

    scores = {entry["target"]: entry["similarity"] for entry in result["all_scores"]}
    assert scores["plan"] == pytest.approx(0.945)
    assert scores["implement"] == pytest.approx(0.795)
    for name in _MODE_NAMES[2:]:
        assert scores[name] == pytest.approx(0.35)
    assert result["target_name"] == "plan"


def test_per_target_ranking_preserves_negative_cosines():
    router = _make_router(
        [
            (name, score - index * 0.1)
            for index, name in enumerate(_MODE_NAMES)
            for score in (-0.3, -0.1, -0.2)
        ]
    )

    result = router.route("request")

    assert result["target_name"] == "plan"
    assert result["similarity"] == pytest.approx(-0.2)
    assert {
        entry["target"]: entry["similarity"] for entry in result["all_scores"]
    } == pytest.approx(
        {name: -0.2 - index * 0.1 for index, name in enumerate(_MODE_NAMES)}
    )


@pytest.mark.parametrize("aggregation", [None, "global_top_k"])
def test_global_top_k_keeps_legacy_partial_ranking(aggregation):
    router = _make_router(
        [("plan", 0.9), ("implement", 0.8), ("plan", 0.7)]
        + [(name, 0.2) for name in _MODE_NAMES[2:]],
        aggregation=aggregation,
    )
    assert router._config.aggregation == "global_top_k"

    result = router.route("request")

    assert router._faiss_index.search.call_args.args[1] == 3
    assert len(result["all_scores"]) == 2
    assert {
        entry["target"]: entry["similarity"] for entry in result["all_scores"]
    } == pytest.approx({"plan": 0.8, "implement": 0.8})


def test_global_top_k_clamps_search_to_index_size():
    router = _make_router(
        [("plan", 0.8)], names=("plan",), aggregation="global_top_k"
    )
    result = router.route("request")
    assert router._faiss_index.search.call_args.args[1] == 1
    assert result["target_name"] == "plan"
    assert result["similarity"] == pytest.approx(0.8)


@pytest.mark.parametrize("missing_name", _MODE_NAMES)
def test_per_target_top_k_rejects_missing_configured_vectors(missing_name):
    router = _make_router(
        [(name, 0.8) for name in _MODE_NAMES if name != missing_name]
    )
    with pytest.raises(ValueError):
        router.route("request")
    assert router._faiss_index.search.call_args.args[1] == router._faiss_index.ntotal


def test_route_context_encodes_sections_separately_and_searches_once():
    router = _make_router(
        [(name, 0.8 - index * 0.1) for index, name in enumerate(_MODE_NAMES)]
    )
    intent = "intent " * 2000
    phase = "pytest failed"
    router._model.encode.side_effect = [
        np.array([[3.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 4.0]], dtype=np.float32),
    ]
    router._encode_query = Mock(wraps=router._encode_query)

    result = router.route_context((intent, "", " \n ", phase))

    assert router._encode_query.call_args_list == [call(intent), call(phase)]
    assert router._model.encode.call_args_list == [
        call([intent], show_progress_bar=False, convert_to_numpy=True),
        call([phase], show_progress_bar=False, convert_to_numpy=True),
    ]
    router._faiss_index.search.assert_called_once()
    vector, k = router._faiss_index.search.call_args.args
    assert vector.shape == (1, 2)
    np.testing.assert_allclose(
        vector, [[1 / math.sqrt(2), 1 / math.sqrt(2)]], atol=1e-6
    )
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert k == router._faiss_index.ntotal
    assert len(result["all_scores"]) == 6


def test_route_context_single_section_matches_route():
    router = _make_router([("plan", 0.8)], names=("plan",))
    direct = router.route("intent")
    direct_vector = router._faiss_index.search.call_args.args[0].copy()
    router._faiss_index.search.reset_mock()

    assert router.route_context(("", "intent", " ")) == direct
    router._faiss_index.search.assert_called_once()
    np.testing.assert_allclose(
        router._faiss_index.search.call_args.args[0], direct_vector
    )


@pytest.mark.parametrize("parts", [(), ("",), (" ", "\n")])
def test_route_context_rejects_empty_sections_without_search(parts):
    router = _make_router([("plan", 0.8)], names=("plan",))
    with pytest.raises(ValueError):
        router.route_context(parts)
    router._model.encode.assert_not_called()
    router._faiss_index.search.assert_not_called()


@pytest.mark.parametrize("winner", _MODE_NAMES)
def test_accept_complete_ranking_for_each_work_mode(mode_by_name, winner):
    layer = _make_layer(mode_by_name)
    result = _ranking(winner=winner)
    result["all_scores"].reverse()
    mode, similarity = layer.accept(result)
    assert mode is mode_by_name[winner]
    assert similarity == pytest.approx(0.8)


@pytest.mark.parametrize(
    "similarity,runner_up,threshold,min_margin,accepted",
    [
        (0.51, 0.46, 0.51, 0.05, True),
        (0.509, 0.3, 0.51, 0.05, False),
        (0.8, 0.751, 0.51, 0.05, False),
        (0.8, 0.749, 0.51, 0.05, True),
        (0.8, 0.8, 0.51, 0.0, False),
        (0.8, 0.801, 0.51, 0.0, False),
        (0.8, 0.799, 0.51, 0.0, True),
        (-0.2, -0.3, -0.2, 0.05, True),
        (1.0, -1.0, 1.0, 2.0, True),
    ],
)
def test_accept_threshold_and_margin_boundaries(
    mode_by_name, similarity, runner_up, threshold, min_margin, accepted
):
    layer = _make_layer(mode_by_name, threshold=threshold, min_margin=min_margin)
    mode, actual = layer.accept(_ranking(similarity=similarity, runner_up=runner_up))
    assert mode is (mode_by_name["plan"] if accepted else None)
    assert actual == pytest.approx(similarity)


@pytest.mark.parametrize("similarity", [-1.0, 0.51, 1.0, np.float64(0.8)])
def test_accept_single_semantic_mode_needs_no_runner_up(mode_by_name, similarity):
    modes = {name: mode_by_name[name] for name in ("plan",) + _SPECIAL_NAMES}
    layer = _make_layer(modes, threshold=-1.0, min_margin=2.0)
    result = {
        "target_name": "plan",
        "similarity": similarity,
        "all_scores": [{"target": "plan", "similarity": similarity}],
    }
    mode, actual = layer.accept(result)
    assert mode is mode_by_name["plan"]
    assert actual == pytest.approx(similarity)


@pytest.mark.parametrize("missing_name", _MODE_NAMES)
def test_accept_requires_every_configured_semantic_mode(mode_by_name, missing_name):
    result = _ranking()
    result["all_scores"] = [
        entry for entry in result["all_scores"] if entry["target"] != missing_name
    ]
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize("value", _INVALID_COSINES)
def test_accept_rejects_malformed_top_similarity_without_raising(
    mode_by_name, value
):
    result = _ranking()
    result["similarity"] = value
    mode, similarity = _make_layer(mode_by_name).accept(result)
    assert mode is None
    assert similarity == 0.0
    assert math.isfinite(similarity)


@pytest.mark.parametrize("value", _INVALID_COSINES)
def test_accept_rejects_malformed_ranked_similarity_preserving_top(
    mode_by_name, value
):
    result = _ranking()
    result["all_scores"][-1]["similarity"] = value
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize(
    "entries", [None, [], "ranking", {}, [None], ["plan"], [{}]]
)
def test_accept_rejects_malformed_ranking_containers(mode_by_name, entries):
    result = _ranking()
    result["all_scores"] = entries
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize(
    "name", ["plan", "aux_title", "compaction", "unknown", None, 3]
)
def test_accept_rejects_duplicate_or_unconfigured_ranked_targets(mode_by_name, name):
    result = _ranking()
    result["all_scores"].append({"target": name, "similarity": 0.1})
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


def test_accept_rejects_duplicate_target_even_with_six_entries(mode_by_name):
    result = _ranking()
    result["all_scores"][-1] = dict(result["all_scores"][0])
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize("key", ["target", "similarity"])
def test_accept_rejects_missing_ranked_fields(mode_by_name, key):
    result = _ranking()
    del result["all_scores"][-1][key]
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize(
    "target", ["implement", "aux_title", "compaction", "unknown", None]
)
def test_accept_rejects_inconsistent_top_target(mode_by_name, target):
    result = _ranking()
    result["target_name"] = target
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


def test_accept_rejects_inconsistent_top_score(mode_by_name):
    result = _ranking()
    result["all_scores"][0]["similarity"] = 0.9
    assert _make_layer(mode_by_name).accept(result) == (None, 0.8)


@pytest.mark.parametrize("key", ["all_scores", "target_name", "similarity"])
def test_accept_rejects_missing_top_level_fields(mode_by_name, key):
    result = _ranking()
    del result[key]
    assert _make_layer(mode_by_name).accept(result) == (
        None,
        0.0 if key == "similarity" else 0.8,
    )


@pytest.mark.parametrize("result", [None, {}, [], "ranking", 1])
def test_accept_rejects_non_results_without_raising(mode_by_name, result):
    assert _make_layer(mode_by_name).accept(result) == (None, 0.0)


@pytest.mark.parametrize(
    "missing", ["min_margin", "intent_max_chars", "phase_max_chars"]
)
def test_semantic_constructor_requires_new_keyword_arguments(mode_by_name, missing):
    kwargs = {"min_margin": 0.05, "intent_max_chars": 2000, "phase_max_chars": 2000}
    del kwargs[missing]
    with pytest.raises(TypeError):
        CodexSemanticLayer(None, 0.51, mode_by_name, **kwargs)


def test_semantic_parts_keep_intent_and_phase_budgets_independent():
    request = CodexRequest(
        latest_user_text="I" * 10000,
        assistant_messages=[_assistant("old announcement"), _assistant("P" * 10000)],
    )
    parts = CodexSemanticLayer._build_semantic_parts(request, 17, 31)
    assert parts == ("I" * 17, "P" * 31)


def test_semantic_phase_contains_latest_assistant_and_linked_call_output():
    request = CodexRequest(
        latest_user_text="continue this task",
        assistant_messages=[
            _assistant("old announcement"),
            _assistant("running checks"),
        ],
        activity=(
            CodexActivity("function_call", "old command", "shell", "old"),
            CodexActivity("function_call_output", "old output", "shell", "old"),
            CodexActivity("function_call", "pytest", "exec_command", "latest"),
            CodexActivity(
                "function_call_output", "FAILED check", "exec_command", "latest"
            ),
        ),
    )
    assert CodexSemanticLayer._build_semantic_parts(request, 2000, 2000) == (
        "continue this task",
        "running checks\ncalled shell\ncalled exec_command",
    )


def test_semantic_phase_includes_latest_call_before_output_arrives():
    request = CodexRequest(
        latest_user_text="task",
        activity=(
            CodexActivity("function_call", "old command", "shell", "old"),
            CodexActivity("function_call_output", "old output", "shell", "old"),
            CodexActivity("function_call", "pytest", "exec_command", "latest"),
        ),
    )
    assert CodexSemanticLayer._build_semantic_parts(request, 2000, 2000) == (
        "task",
        "called shell\ncalled exec_command",
    )


def test_semantic_phase_reports_a_call_once_and_an_orphan_output_never():
    request = CodexRequest(
        latest_user_text="task",
        activity=(
            CodexActivity("function_call", "old command", "shell", "old"),
            CodexActivity(
                "function_call_output", "unlinked output", "shell", "missing"
            ),
        ),
    )
    assert CodexSemanticLayer._build_semantic_parts(request, 2000, 2000) == (
        "task",
        "called shell",
    )


def test_semantic_phase_describes_the_action_not_the_file_it_read():
    """The topic of a read file must not stand in for the work."""

    def request_with(content):
        return CodexRequest(
            latest_user_text="task",
            activity=(
                CodexActivity(
                    "function_call", '{"cmd":"cat recipe.md"}', "exec_command", "c1"
                ),
                CodexActivity("function_call_output", content, "exec_command", "c1"),
            ),
        )

    lasagne = CodexSemanticLayer._build_semantic_parts(
        request_with("lasagne"), 2000, 2000
    )
    engine = CodexSemanticLayer._build_semantic_parts(
        request_with("turbocharger"), 2000, 2000
    )
    assert lasagne == engine
    assert "lasagne" not in lasagne[1]
    assert "turbocharger" not in engine[1]


def test_semantic_phase_names_recognized_actions_and_their_status(mode_by_name):
    from llm_router_plugins.utils.routing.agentic_routing.codex.phase_config import (
        CodexPhaseConfig,
    )
    import json
    import pathlib as _pathlib

    raw = json.loads(
        (
            _pathlib.Path(__file__).resolve().parents[1]
            / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
        ).read_text(encoding="utf-8")
    )["settings"]["phase"]
    rules = CodexPhaseConfig.from_raw(raw)
    request = CodexRequest(
        latest_user_text="task",
        activity=(
            CodexActivity(
                "function_call",
                json.dumps({"cmd": "python -m pytest"}),
                "exec_command",
                "c1",
            ),
            CodexActivity(
                "function_call_output",
                "Process exited with code 1",
                "exec_command",
                "c1",
            ),
        ),
    )
    _, phase = CodexSemanticLayer._build_semantic_parts(
        request, 2000, 2000, phase_rules=rules
    )
    assert phase == "ran python -m pytest: failed"


def test_semantic_phase_preserves_each_source_under_a_small_budget():
    request = CodexRequest(
        latest_user_text="I" * 10000,
        assistant_messages=[_assistant("A" * 10000)],
        activity=(
            CodexActivity("function_call", "C" * 10000, "exec_command", "latest"),
            CodexActivity(
                "function_call_output", "O" * 10000, "exec_command", "latest"
            ),
        ),
    )
    intent, phase = CodexSemanticLayer._build_semantic_parts(request, 11, 101)
    assert intent == "I" * 11
    assert len(phase) <= 101
    assert "A" in phase and "exec_command" in phase
    assert "C" * 20 not in phase and "O" * 20 not in phase


@pytest.mark.parametrize(
    "intent,phase,expected",
    [
        ("", "", ()),
        (" \n ", " \n ", ()),
        ("intent", "", ("intent",)),
        ("", "phase", ("phase",)),
    ],
)
def test_semantic_parts_return_only_nonempty_sections(intent, phase, expected):
    request = CodexRequest(
        latest_user_text=intent, assistant_messages=[_assistant(phase)]
    )
    assert CodexSemanticLayer._build_semantic_parts(request, 20, 20) == expected


@pytest.mark.parametrize("supports_context", [True, False])
def test_semantic_route_uses_parts_or_legacy_join(
    mode_by_name, monkeypatch, supports_context
):
    result = _ranking()
    router = SimpleNamespace(route=Mock(return_value=result))
    if supports_context:
        router.route_context = Mock(return_value=result)
    layer = _make_layer(
        mode_by_name, router, intent_max_chars=17, phase_max_chars=31
    )
    request = CodexRequest(latest_user_text="task")
    parts = ("intent", "phase")
    builder = Mock(return_value=parts)
    monkeypatch.setattr(layer, "_build_semantic_parts", builder)

    assert layer.route(request) is result

    builder.assert_called_once_with(request, 17, 31, phase_rules=None)
    if supports_context:
        router.route_context.assert_called_once_with(parts)
        router.route.assert_not_called()
    else:
        router.route.assert_called_once_with("intent\nphase")


def test_semantic_route_sends_real_independently_limited_parts(mode_by_name):
    router = SimpleNamespace(
        route_context=Mock(return_value=_ranking()), route=Mock()
    )
    layer = _make_layer(
        mode_by_name, router, intent_max_chars=17, phase_max_chars=31
    )
    request = CodexRequest(
        latest_user_text="I" * 10000, assistant_messages=[_assistant("P" * 10000)]
    )
    assert layer.route(request) is router.route_context.return_value
    router.route_context.assert_called_once_with(("I" * 17, "P" * 31))
    router.route.assert_not_called()


def test_semantic_route_does_not_query_empty_context(mode_by_name):
    router = SimpleNamespace(route_context=Mock(), route=Mock())
    assert _make_layer(mode_by_name, router).route(CodexRequest()) is None
    router.route_context.assert_not_called()
    router.route.assert_not_called()


def test_shipped_config_has_balanced_semantic_defaults(raw_config):
    config = CodexRoutingConfig._from_raw(raw_config)
    config.validate_args()
    assert config.semantic_aggregation == "per_target_top_k"
    assert config.semantic_min_margin == pytest.approx(0.005)
    assert config.semantic_intent_max_chars == 2000
    assert config.semantic_phase_max_chars == 2000
    assert config.top_k == 4
    assert set(config.mode_names) - set(_SPECIAL_NAMES) == set(_MODE_NAMES)


@pytest.mark.parametrize(
    "key", ["aggregation", "min_margin", "intent_max_chars", "phase_max_chars"]
)
def test_config_requires_new_semantic_settings(raw_config, key):
    del raw_config["settings"]["semantic"][key]
    with pytest.raises(KeyError, match=key):
        CodexRoutingConfig._from_raw(raw_config)


@pytest.mark.parametrize("semantic", [None, [], "enabled", True])
def test_config_requires_semantic_object(raw_config, semantic):
    raw_config["settings"]["semantic"] = semantic
    with pytest.raises(ValueError, match="semantic"):
        CodexRoutingConfig._from_raw(raw_config)


@pytest.mark.parametrize("key,value", _INVALID_SETTINGS)
def test_config_parser_rejects_invalid_semantic_settings(raw_config, key, value):
    raw_config["settings"]["semantic"][key] = value
    with pytest.raises(ValueError, match=key):
        CodexRoutingConfig._from_raw(raw_config)


@pytest.mark.parametrize("key,value", _INVALID_SETTINGS)
def test_validate_args_rejects_invalid_semantic_settings(raw_config, key, value):
    config = CodexRoutingConfig._from_raw(raw_config)
    setattr(config, f"semantic_{key}", value)
    with pytest.raises(ValueError, match=key):
        config.validate_args()


@pytest.mark.parametrize("aggregation", ["global_top_k", "per_target_top_k"])
@pytest.mark.parametrize("margin", [0.0, 2.0])
def test_config_accepts_both_strategies_and_margin_endpoints(
    raw_config, aggregation, margin
):
    raw_config["settings"]["semantic"].update(
        aggregation=aggregation,
        min_margin=margin,
        intent_max_chars=1,
        phase_max_chars=1,
    )
    config = CodexRoutingConfig._from_raw(raw_config)
    config.validate_args()
    assert config.semantic_aggregation == aggregation
    assert config.semantic_min_margin == margin
    assert config.semantic_intent_max_chars == config.semantic_phase_max_chars == 1
