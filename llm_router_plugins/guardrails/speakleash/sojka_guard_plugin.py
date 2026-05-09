"""
Sójka Guardrail Plugin

This plugin sends the incoming ``payload`` to the Sojka guardrail service and
parses the JSON response.  The service URL can be configured through the
environment variable ``LLM_ROUTER_GUARDRAIL_Sojka_GUARD_HOST``;

The expected response format is:

{
    "results": {
        "detailed": [
            {
                "chunk_index": 0,
                "chunk_text": "...",
                "label": "safe",
                "safe": true,
                "score": 0.9834
            }
        ],
        "safe": true
    }
}

If the request succeeds, ``apply`` returns a dictionary containing the
extracted fields.  If any error occurs (network error, unexpected payload,
missing keys, etc.) the method returns ``{'success': False}``.
"""

import os
import logging
from typing import Optional

from llm_router_plugins.constants import _DontChangeMe
from llm_router_plugins.guardrails.guardrails_base import GuardrailsBase

GUARDRAIL_SOJKA_GUARD_HOST = str(
    os.environ.get(f"{_DontChangeMe.MAIN_ENV_PREFIX}GUARDRAIL_SOJKA_GUARD_HOST", "")
)


class SojkaGuardPlugin(GuardrailsBase):
    """
    Concrete implementation of :class:`GuardrailsBase` that
    talks to the Sojka guardrail HTTP endpoint.
    """

    name = "sojka_guard"
    host_url = GUARDRAIL_SOJKA_GUARD_HOST
    endpoint_path = "api/guardrails/sojka_guard"

    def __init__(self, logger: Optional[logging.Logger] = None):
        if not len(GUARDRAIL_SOJKA_GUARD_HOST):
            raise RuntimeError(
                f"When you are using `sojka_guard` plugin, you must provide a "
                f"host with model, "
                f"{_DontChangeMe.MAIN_ENV_PREFIX}GUARDRAIL_SOJKA_GUARD_HOST "
                f"must be set to valid host."
            )

        super().__init__(logger=logger)
