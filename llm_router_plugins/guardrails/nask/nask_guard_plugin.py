"""
NASK Guardrail Plugin

This plugin sends the incoming ``payload`` to the NASK guardrail service and
parses the JSON response.  The service URL can be configured through the
environment variable ``LLM_ROUTER_GUARDRAIL_NASK_GUARD_HOST``;

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

---

**Model License:** The model used by this plugin is licensed under **CC BY‑NC‑SA 4.0**.
**Router License:** The LLM router component is licensed under **Apache 2.0**.
Before using the plugin, ensure that your intended use complies with these licenses.

**Authors:** Aleksandra Krasnodębska, Karolina Seweryn, Szymon Łukasik, Wojciech Kusa
(see *PL‑Guard: Benchmarking Language Model Safety for Polish*, 2025).

"""

import os
import logging
from typing import Optional

from llm_router_plugins.constants import _DontChangeMe
from llm_router_plugins.guardrails.guardrails_base import GuardrailsBase

# =============================================================================
# Host with router service where NASK-PIB/HerBERT-PL-Guard model is served
# Read model License before using this model **MODEL LICENSE** CC BY-NC-SA 4.0
GUARDRAIL_NASK_GUARD_HOST = str(
    os.environ.get(f"{_DontChangeMe.MAIN_ENV_PREFIX}GUARDRAIL_NASK_GUARD_HOST", "")
)


class NASKGuardPlugin(GuardrailsBase):
    """
    Concrete implementation of :class:`HttpPluginInterface` that
    talks to the NASK guardrail HTTP endpoint.
    """

    name = "nask_guard"
    host_url = GUARDRAIL_NASK_GUARD_HOST
    endpoint_path = "api/guardrails/nask_guard"

    def __init__(self, logger: Optional[logging.Logger] = None):
        if not len(GUARDRAIL_NASK_GUARD_HOST):
            raise RuntimeError(
                "When you are using `nask_guard` plugin, you must provide a "
                "host with model. GUARDRAIL_NASK_GUARD_HOST must be set to "
                "a valid host."
            )

        super().__init__(logger=logger)
