"""
Shared helper for extracting user text from routing payloads.

Both ``SemanticBiEncoderRoutingPlugin`` and ``SimpleSemanticRoutingPlugin``
need to locate the user's message inside an incoming payload dict.  They use
the same priority list, so this module centralises that logic in one place.

Typical usage::

    from llm_router_plugins.utils.routing.text_extractor import extract_user_text

    user_text = extract_user_text(payload)
"""

from typing import Any, Dict


def extract_user_text(payload: Dict[str, Any]) -> str:
    """
    Extract the user message text from *payload*.

    The text is extracted using the following priority:

    1. ``payload["messages"][-1]["content"]`` — last message in a chat history
    2. ``payload["user_last_statement"]``
    3. ``payload["query"]``
    4. ``payload["prompt"]``
    5. ``payload["input"]``

    Parameters
    ----------
    payload : dict
        The message payload to extract text from.

    Returns
    -------
    str
        The extracted text, or an empty string if no text is found.

    Raises
    ------
    None
    """
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        if content:
            return str(content)
    for key in ("user_last_statement", "query", "prompt", "input"):
        val = payload.get(key)
        if val:
            return str(val)
    return ""
