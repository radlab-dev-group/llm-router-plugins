"""
A Redis client that is built on first use and rebuilt after a fork.

The knowledge layer touches Redis from two places: the request path (the
generation cache and the event publisher).  Both live inside a proxy process
that is often preloaded and then forked — gunicorn's ``preload_app``, uWSGI's
``lazy-apps`` off, a supervising parent warming the plugin before its workers
exist.  A :class:`redis.Redis` object owns a connection pool, and a pool
inherited through ``fork()`` hands the parent's sockets to the child: two
processes reading and writing one TCP connection, which shows up as crossed
replies and ``ConnectionError`` and looks nothing like the cause.

So the client is built lazily and its owning process is remembered.  When a
call arrives from a different pid than the one that built the client, a client
this module built is thrown away and rebuilt here, and a client that was
injected by an embedding application is kept but has its pool reset.  This is
the same discipline ``RedisRoutingStateStore`` applies to the Codex session
memory, shared rather than restated.
"""

import logging
import os
import threading

from typing import Any, Optional

from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = ["RedisClientHandle"]


class RedisClientHandle:
    """
    A lazy, fork-aware holder of one Redis client.

    Parameters
    ----------
    connection : RedisConnectionSettings, optional
        Where to connect; an unconfigured setting means ``get()`` returns
        ``None`` and the feature that needed Redis reports itself unavailable.
    client : Any, optional
        A client to use as-is (tests, or an application that owns its own
        connections).  It is never rebuilt, only reset after a fork.
    logger : logging.Logger, optional
        Where a failed connection is reported, without credentials.
    """

    def __init__(
        self,
        connection: Optional[RedisConnectionSettings] = None,
        client: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Record what to connect to; open nothing yet."""
        self._connection = connection or RedisConnectionSettings()
        self._client = client
        self._injected = client is not None
        self._logger = logger
        self._pid = os.getpid()
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        """Whether a connection could be attempted at all."""
        return self._client is not None or self._connection.configured

    def get(self) -> Any:
        """
        Return the client for this process, building it when needed.

        Returns
        -------
        Any
            A Redis client, or ``None`` when Redis is not configured or the
            client could not be built.
        """
        self._adopt_if_forked()
        if self._client is not None:
            return self._client
        if not self._connection.configured:
            return None
        with self._lock:
            if self._client is None:
                try:
                    import redis  # noqa: PLC0415 - optional dependency

                    self._client = redis.Redis(**self._connection.client_kwargs())
                except Exception as exc:
                    self._warn(
                        "redis client could not be built: %s", type(exc).__name__
                    )
                    return None
            return self._client

    def close(self) -> None:
        """Release the client when this holder built it; never raise."""
        client = self._client
        if client is None or self._injected:
            return
        for name in ("close", "shutdown", "quit"):
            hook = getattr(client, name, None)
            if callable(hook):
                try:
                    hook()
                except Exception:  # closing must not raise
                    return
                return

    # -------------------------------------------------------------- internals -
    def _adopt_if_forked(self) -> None:
        """Take ownership of the connection when the process changed."""
        pid = os.getpid()
        if self._pid == pid:
            return
        self._pid = pid
        self._lock = threading.Lock()
        client = self._client
        if client is None:
            return
        if not self._injected:
            # Built here: the parent's copy belongs to the parent's pool.
            self._client = None
            return
        # Given to us: the application owns the object, so reuse it but drop
        # whatever socket it inherited before this process reads from it.
        pool = getattr(client, "connection_pool", None)
        reset = getattr(pool, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception as exc:
                self._warn("redis pool reset failed: %s", type(exc).__name__)

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)
