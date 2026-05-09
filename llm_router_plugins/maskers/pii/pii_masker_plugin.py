""" """

import os
import logging
from typing import Dict, Optional, Tuple

from llm_router_plugins.constants import _DontChangeMe
from llm_router_plugins.maskers.maskers_http_base import HttpMaskersBase

PII_MASKER_HOST = str(
    os.environ.get(f"{_DontChangeMe.MAIN_ENV_PREFIX}MASKER_PII_HOST", "")
)


class PiiMaskerPlugin(HttpMaskersBase):

    name = "pii_masker"
    host_url = PII_MASKER_HOST
    endpoint_path = "api/maskers/pii"

    def __init__(self, logger: Optional[logging.Logger] = None):
        if not len(PII_MASKER_HOST):
            raise RuntimeError(
                f"When you are using `{self.name}` plugin, you must provide a "
                f"host with model, {_DontChangeMe.MAIN_ENV_PREFIX}MASKER_PII_HOST "
                f"must be set to valid host."
            )

        super().__init__(logger=logger)
