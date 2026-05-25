from abc import ABC
from typing import Dict, Tuple, Any

from llm_router_plugins.plugin_interface import HttpPluginInterface


class HttpMaskersBase(HttpPluginInterface, ABC):

    def apply(self, payload: Any) -> Tuple[Any, Dict]:
        """
        Send ``payload`` to the guardrail service, parse the JSON response,
        and expose the most relevant fields.

        Parameters
        ----------
        payload: Dict
            The data that should be evaluated by the guardrail.

        Returns
        -------
        Dict
            ``{'success': True, 'safe': <bool>, 'chunk_index': <int>,
            'chunk_text': <str>, 'label': <str>, 'score': <float>}``
            on success, or ``{'success': False}`` on any error.
        """
        try:
            response = self._request(payload)
            ann_payload = response.get("anonymized", payload)
            mappings = response.get("mappings", {})
            return ann_payload, mappings
        except Exception as exc:
            if self._logger:
                self._logger.error(
                    "%s failed to process payload: %s", self.name, exc
                )
            return "", {}
