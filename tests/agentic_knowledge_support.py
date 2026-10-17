"""
Shared builders and fakes for the agentic knowledge test suite.

Codex replays a whole turn in every request, so the tests need payloads that
carry tool calls and their outputs in the exact wire shape the Codex parser
reads.  The builders below produce those payloads; the fakes stand in for the
search engine, the Redis stream and the publisher, so no test opens a socket or
imports an optional dependency.
"""

import datetime
import json
from collections import namedtuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_OK,
    SearchResult,
    StoreHit,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

# --------------------------------------------------------------------- wire -
FAILING = "FAILED tests/test_a.py::test_x - AssertionError: expected 1 got 2"
GREEN = ""


def turn_metadata(**overrides):
    """Serialize an ``x-codex-turn-metadata`` value."""
    base = {
        "session_id": "sess-1",
        "thread_id": "thread-1",
        "root_turn_id": "turn-1",
        "turn_id": "turn-7",
        "window_id": "thread-1:0",
        "context_window_id": "cw-1",
        "agent_name": "/root",
        "request_kind": "turn",
        "thread_source": "user",
        "sandbox_mode": "read-only",
    }
    base.update(overrides)
    return json.dumps(base)


def _user(text):
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _assistant(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _command(call_id, event_id, command="pytest tests/test_a.py"):
    return {
        "type": "function_call",
        "name": "exec_command",
        "call_id": call_id,
        "id": event_id,
        "arguments": json.dumps({"cmd": command}),
    }


def _command_output(call_id, event_id, body, code=1):
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "id": event_id,
        "output": f"Process exited with code {code}\nOutput:\n{body}",
    }


def _patch(call_id, event_id, paths=("src/a.py",)):
    body = "*** Begin Patch\n"
    for path in paths:
        body += f"*** Update File: {path}\n@@\n-x = 1\n+x = 2\n"
    body += "*** End Patch"
    return {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": call_id,
        "id": event_id,
        "arguments": body,
    }


def codex_payload(items, mode="debug", source="phase", turn_id="turn-7", **metadata):
    """Wrap already-built *items* in a Codex request that routing has annotated."""
    return {
        "model": "auto_codex",
        "instructions": "You are a coding agent.",
        "input": items,
        "client_metadata": {
            "session_id": "sess-1",
            "thread_id": "thread-1",
            "turn_id": turn_id,
            "x-codex-turn-metadata": turn_metadata(**metadata),
        },
        "routing": {
            "plugin": "agentic_routing_codex",
            "source": source,
            "model": "auto_codex",
            "similarity": 0.9,
        },
        "agent_mode": mode,
    }


def base_items(cwd="/repo/webapp"):
    """The environment-context and command items that open a Codex turn."""
    return [
        _user(f"<environment_context>\n  <cwd>{cwd}</cwd>\n</environment_context>"),
        _user("fix the failing export test"),
    ]


def loop_items(failures=3, failure=FAILING):
    """Items with *failures* repetitions of the same failing test command."""
    items = base_items()
    for index in range(failures):
        items.append(_command(f"c{index}", f"e{index}"))
        items.append(_command_output(f"c{index}", f"o{index}", failure, code=1))
    return items


def recipe_items(failure=FAILING, paths=("src/export.py",)):
    """Items describing a failure, a patch, and the same command passing."""
    return base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", failure, code=1),
        _patch("p1", "pe", paths),
        _command("c2", "e2"),
        _command_output("c2", "o2", GREEN, code=0),
    ]


def deadend_items(failure=FAILING):
    """Items describing a failure, a patch, and the same failure again."""
    return base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", failure, code=1),
        _patch("p1", "pe"),
        _command("c2", "e2"),
        _command_output("c2", "o2", failure, code=1),
    ]


def main_payload(mode="debug", failures=1, **kwargs):
    """A main request whose turn failed its test *failures* times."""
    return codex_payload(loop_items(failures), mode=mode, **kwargs)


def recipe_payload(**kwargs):
    """A main request whose turn fixed a failure."""
    return codex_payload(recipe_items(), **kwargs)


def compaction_payload(mode="debug", turn_id="turn-9"):
    """A compaction request carrying a conversation window."""
    items = base_items() + [_assistant("I refactored export.py and ran the tests.")]
    return codex_payload(
        items, mode=mode, turn_id=turn_id, request_kind="compaction"
    )


# ------------------------------------------------------------------ config - -
def knowledge_config(**overrides):
    """A fully wired, enabled config for tests; connection settings fake."""
    config = AgenticKnowledgeConfig.from_file()
    config.enabled = True
    config.read_enabled = True
    config.write_enabled = True
    config.sse_host = "http://engine.test"
    config.redis = RedisConnectionSettings(host="cache.test")
    config.escalation_enabled = False
    for key, value in overrides.items():
        setattr(config, key, value)
    config.validate()
    return config


# ------------------------------------------------------------------ fakes - -
class FakeStore:
    """A search store that returns scripted hits and counts requests."""

    def __init__(self, hits=(), status=STATUS_OK):
        self.hits = tuple(hits)
        self.status = status
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return SearchResult(self.status, hits=self.hits)


def store_hit(category="recipe", name=None, text="a verified fix", score=0.9):
    """One fake search hit with a plugin-shaped document name."""
    day = datetime.date.today().strftime("%Y%m%d")
    return StoreHit(
        collection="agentic-kb-webapp",
        document_name=name or f"{category}:{day}:deadbeef",
        text=text,
        score=score,
        score_normalized=score,
    )


class FakePublisher:
    """A publisher that records events and reports success."""

    def __init__(self):
        self.published = []

    def publish(self, events):
        events = list(events)
        self.published.extend(events)
        from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
            STATUS_OK as OK,
            PublishStatus,
        )

        return PublishStatus(OK, len(events), 0)


class FakeRedis:
    """A minimal Redis double: string keys and the stream commands."""

    def __init__(self, messages=None, fail_on=None):
        self.kv = {}
        self.stream = []
        self.messages = list(messages or [])
        self.acked = []
        self.groups = []
        self.fail_on = set(fail_on or ())

    def _maybe_fail(self, name):
        if name in self.fail_on:
            raise RuntimeError(f"boom:{name}")

    def set(self, key, value, nx=False, ex=None):
        self._maybe_fail("set")
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        self._maybe_fail("get")
        return self.kv.get(key)

    def delete(self, key):
        self._maybe_fail("delete")
        self.kv.pop(key, None)

    def xadd(self, key, fields, maxlen=None, approximate=False):
        self._maybe_fail("xadd")
        self.stream.append((key, fields))
        return f"{len(self.stream)}-0"

    def xgroup_create(self, stream, group, id="$", mkstream=False):
        self._maybe_fail("xgroup_create")
        self.groups.append((stream, group, id, mkstream))

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        self._maybe_fail("xreadgroup")
        if not self.messages:
            return []
        take = int(count or len(self.messages))
        batch, self.messages = self.messages[:take], self.messages[take:]
        return [("stream", batch)]

    def xack(self, stream, group, entry_id):
        self._maybe_fail("xack")
        self.acked.append(entry_id)


DocumentRef = namedtuple("DocumentRef", ["name"])


class FakeSSE:
    """A minimal Semantic Search Engine client double."""

    def __init__(self, names=None, exists=(), fail_methods=()):
        self.names = names or {}
        self.collections = set(exists)
        self.indexed = []
        self.deleted = []
        self.deactivated = []
        self.created = []
        self.fail_methods = set(fail_methods)

    def _maybe_fail(self, name):
        if name in self.fail_methods:
            raise RuntimeError(f"boom:{name}")

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, name, description="", **kwargs):
        assert description and kwargs.get("embedder") and kwargs.get("reranker")
        self.created.append((name, kwargs.get("embedder"), kwargs.get("reranker")))
        self.collections.add(name)

    def index_documents(self, collection, documents):
        self._maybe_fail("index_documents")
        self.indexed.append((collection, documents))
        return {}

    def list_document_details(self, collection):
        return [DocumentRef(name) for name in self.names.get(collection, [])]

    def delete_documents(self, collection, documents=None):
        self.deleted.append((collection, list(documents or [])))

    def deactivate_documents(self, collection, documents=None):
        self.deactivated.append((collection, list(documents or [])))


def knowledge_event(
    kind="recipe", fingerprint="abc123", scope="webapp", text="body"
):
    """One event for the worker and publisher tests."""
    return KnowledgeEvent(
        version=1,
        kind=kind,
        fingerprint=fingerprint,
        scope=scope,
        agent="codex",
        mode="debug",
        thread_id="thread-1",
        generation="turn-7",
        created_at=datetime.datetime(
            2026, 10, 17, tzinfo=datetime.timezone.utc
        ).timestamp(),
        signature="AssertionError",
        signature_hash="h1",
        text=text,
        metadata={"paths": ["a.py"], "thread_id": "thread-1"},
    )


def stream_message(event):
    """Encode an event the way the publisher writes it to the stream."""
    return {"event": json.dumps(event.to_dict())}
