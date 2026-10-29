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


def patch_envelope(paths=("src/a.py",)):
    """A Codex patch envelope touching *paths*."""
    body = "*** Begin Patch\n"
    for path in paths:
        body += f"*** Update File: {path}\n@@\n-x = 1\n+x = 2\n"
    body += "*** End Patch"
    return body


def _patch(call_id, event_id, paths=("src/a.py",)):
    return {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": call_id,
        "id": event_id,
        "arguments": patch_envelope(paths),
    }


def _heredoc_patch(call_id, event_id, paths=("src/a.py",), cwd="/repo"):
    """
    A patch the way a current Codex CLI applies one.

    The envelope rides as a here-document piped to ``apply_patch`` inside an
    ``exec_command``, so no call is named after the patch at all.
    """
    command = f"cd {cwd} && apply_patch <<'PATCH'\n{patch_envelope(paths)}\nPATCH"
    return _command(call_id, event_id, command)


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


def recipe_items(
    failure=FAILING,
    paths=("src/export.py",),
    command="pytest tests/test_a.py",
    green_command=None,
):
    """Items describing a failure, a patch, and the same command passing."""
    return base_items() + [
        _command("c1", "e1", command),
        _command_output("c1", "o1", failure, code=1),
        _patch("p1", "pe", paths),
        _command("c2", "e2", green_command or command),
        _command_output("c2", "o2", GREEN, code=0),
    ]


def heredoc_recipe_items(paths=("src/export.py",), command="pytest tests/test_a.py"):
    """A recipe whose patch arrives as a here-document, not as a patch tool."""
    return base_items() + [
        _command("c1", "e1", command),
        _command_output("c1", "o1", FAILING, code=1),
        _heredoc_patch("p1", "pe", paths),
        _command("c2", "e2", command),
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


def procedure_items(
    paths=("src/export.py",),
    command="pytest tests/test_a.py",
    rounds=1,
):
    """
    Items describing changes their test runs accepted, with no failure at all.

    This is the shape of most correct work: the agent implements, runs the
    tests, and nothing ever broke.  Before this builder such a turn wrote no
    knowledge; each *round* is one patch followed by the green run that
    accepted it, so ``rounds=2`` is two procedures, not one.
    """
    items = base_items()
    for index in range(1, rounds + 1):
        items.append(_patch(f"p{index}", f"pe{index}", paths))
        items.append(_command(f"c{index}", f"e{index}", command))
        items.append(_command_output(f"c{index}", f"oe{index}", GREEN, code=0))
    return items


def fix_then_procedure_items(
    failure=FAILING,
    fixed_paths=("src/export.py",),
    later_paths=("src/other.py",),
    command="pytest tests/test_a.py",
    later_command="pytest tests/test_b.py",
):
    """
    Items describing a failure fixed and, afterwards, new work its tests took.

    Two episodes of one turn, with identifiers that do not collide: the tool
    output maps of a turn are keyed by ``call_id``, so a fixture that reuses
    one lets a later output overwrite an earlier failure and the recipe
    silently loses its signature.
    """
    return base_items() + [
        _command("c1", "e1", command),
        _command_output("c1", "o1", failure, code=1),
        _patch("p1", "pe1", fixed_paths),
        _command("c2", "e2", command),
        _command_output("c2", "o2", GREEN, code=0),
        _patch("p3", "pe3", later_paths),
        _command("c3", "e3", later_command),
        _command_output("c3", "o3", GREEN, code=0),
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
        self.deadlines = []

    def search(self, request, *, deadline=None):
        self.requests.append(request)
        self.deadlines.append(deadline)
        return SearchResult(self.status, hits=self.hits)


def store_hit(category="recipe", name=None, text="a verified fix", score=0.9):
    """One fake search hit with a plugin-shaped document name."""
    day = datetime.date.today().strftime("%Y%m%d")
    return StoreHit(
        collection="agentic_kb_webapp",
        document_name=name or f"{category}:{day}:deadbeef",
        text=text,
        score=score,
        score_normalized=score,
    )


class FakePublisher:
    """A publisher that records events and reports success."""

    def __init__(self):
        self.published = []

    def publish(self, events, *, refresh_claims=False):
        events = list(events)
        self.published.extend(events)
        from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
            STATUS_OK as OK,
            PublishStatus,
        )

        return PublishStatus(OK, len(events), 0)

    def replace_docs(self, scope, events):
        """A forced republish replaces the scope's queued documentation."""
        events = list(events)
        self.published.extend(events)
        from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
            STATUS_OK as OK,
            PublishStatus,
        )

        return PublishStatus(OK, len(events), 0)


class FakeRedis:
    """A minimal Redis double: string keys and the stream commands."""

    def __init__(self, messages=None, fail_on=None, pending=None):
        self.kv = {}
        self.stream = []
        self.messages = list(messages or [])
        self.pending = list(pending or [])
        self.claimed = []
        self.expired = []
        self.reads = []
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

    def delete(self, *keys):
        self._maybe_fail("delete")
        removed = 0
        for key in keys:
            if self.kv.pop(key, None) is not None:
                removed += 1
        return removed

    def scan_iter(self, match=None, count=None):
        self._maybe_fail("scan_iter")
        import fnmatch

        for key in sorted(self.kv):
            if match is None or fnmatch.fnmatchcase(key, match):
                yield key

    def xadd(self, key, fields, maxlen=None, approximate=False):
        self._maybe_fail("xadd")
        self.stream.append((key, fields))
        return f"{len(self.stream)}-0"

    def xgroup_create(self, stream, group, id="$", mkstream=False):
        self._maybe_fail("xgroup_create")
        self.groups.append((stream, group, id, mkstream))

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        self._maybe_fail("xreadgroup")
        self.reads.append(block)
        if not self.messages:
            return []
        take = int(count or len(self.messages))
        batch, self.messages = self.messages[:take], self.messages[take:]
        return [("stream", batch)]

    def incr(self, key):
        self._maybe_fail("incr")
        value = int(self.kv.get(key, 0) or 0) + 1
        self.kv[key] = value
        return value

    def expire(self, key, seconds):
        self._maybe_fail("expire")
        self.expired.append((key, seconds))
        return True

    def xautoclaim(
        self,
        stream,
        group,
        consumer,
        min_idle,
        start_id="0-0",
        count=None,
        just_id=False,
    ):
        self._maybe_fail("xautoclaim")
        take = int(count or len(self.pending))
        batch, self.pending = self.pending[:take], self.pending[take:]
        reply = ["0-0", batch, []]
        self.claimed.append(batch)
        return reply

    def xpending(self, stream, group):
        self._maybe_fail("xpending")
        return {
            "pending": len(self.pending),
            "min": None,
            "max": None,
            "consumers": [],
        }

    def xack(self, stream, group, entry_id):
        self._maybe_fail("xack")
        self.acked.append(entry_id)


DocumentRef = namedtuple("DocumentRef", ["name"])
Collection = namedtuple("Collection", ["name"])


class FakeSSE:
    """A minimal Semantic Search Engine client double."""

    def __init__(
        self,
        names=None,
        exists=(),
        fail_methods=(),
        embedders=("bge-m3",),
        rerankers=("bge-reranker-v2-m3",),
        catalogue=(),
    ):
        self.names = names or {}
        self.collections = set(exists)
        #: Collections the engine reports in a listing (retention discovery);
        #: empty by default so a double that never lists keeps old behaviour.
        self.catalogue = tuple(catalogue or ())
        self.indexed = []
        self.uploaded = []
        #: The ``indexing_options`` of each indexed call, in order: a test can
        #: see what chunking the worker sent with the documents.
        self.indexing_options = []
        self.deleted = []
        self.deactivated = []
        self.created = []
        self.fail_methods = set(fail_methods)
        #: What the fake engine says it serves, for the startup check.  A
        #: ``None`` entry stands for an engine that does not answer the call.
        self.embedders = embedders
        self.rerankers = rerankers

    def _maybe_fail(self, name):
        if name in self.fail_methods:
            raise RuntimeError(f"boom:{name}")

    def collection_exists(self, name):
        return name in self.collections

    def list_collections(self):
        return [Collection(name) for name in self.catalogue]

    def create_collection(self, name, description="", **kwargs):
        assert description and kwargs.get("embedder") and kwargs.get("reranker")
        self.created.append((name, kwargs.get("embedder"), kwargs.get("reranker")))
        self.collections.add(name)

    def list_embedders(self):
        self._maybe_fail("list_embedders")
        if self.embedders is None:
            raise RuntimeError("no such endpoint")
        return list(self.embedders)

    def list_rerankers(self):
        self._maybe_fail("list_rerankers")
        if self.rerankers is None:
            raise RuntimeError("no such endpoint")
        return list(self.rerankers)

    def index_documents(self, collection, documents, indexing_options=None):
        self._maybe_fail("index_documents")
        self.indexed.append((collection, documents))
        self.indexing_options.append(indexing_options)
        return {}

    def upload_files(
        self, collection, files, *, indexing_options=None, file_metadata=None
    ):
        self._maybe_fail("upload_files")
        self.uploaded.append((collection, files, file_metadata))
        self.indexing_options.append(indexing_options)
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


def _shell_write(call_id, event_id, path="src/export.py", body="x = 2"):
    """
    A file the agent writes through the shell, with no patch tool involved.

    This is how a build of the CLI without ``apply_patch`` edits code:
    ``cat > path <<'EOF'`` carries the whole new file in the command, and
    nothing about the call names it as an edit.
    """
    return _command(
        call_id,
        event_id,
        f"cat > {path} <<'EOF'\n{body}\nEOF",
    )


def exec_edit_playbook_items(
    paths=("src/export.py",),
    command="pytest tests/test_a.py",
):
    """A shell-written file whose tests then pass, with no failure at all."""
    items = base_items()
    for index, path in enumerate(paths, start=1):
        items.append(_shell_write(f"w{index}", f"we{index}", path))
    items.append(_command("c1", "e1", command))
    items.append(_command_output("c1", "o1", GREEN, code=0))
    return items


def exec_edit_after_failure_items(
    failure=FAILING,
    path="src/export.py",
    command="pytest tests/test_a.py",
):
    """A failure, a shell-written change, and the same command passing."""
    return base_items() + [
        _command("c1", "e1", command),
        _command_output("c1", "o1", failure, code=1),
        _shell_write("w1", "we1", path),
        _command("c2", "e2", command),
        _command_output("c2", "o2", GREEN, code=0),
    ]
