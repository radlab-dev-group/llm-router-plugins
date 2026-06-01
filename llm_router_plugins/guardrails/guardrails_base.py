from abc import ABC
from typing import Dict, Tuple, Any

from llm_router_plugins.plugin_interface import HttpPluginInterface


class GuardrailsBase(HttpPluginInterface, ABC):

    def apply(self, payload: Any) -> Tuple[bool, Dict]:
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
            results = response.get("results", {})
            safe_overall: bool = bool(results.get("safe", False))
            return safe_overall, response
        except Exception as exc:
            if self._logger:
                self._logger.error(
                    "%s failed to process payload: %s", self.name, exc
                )
            return False, {}
