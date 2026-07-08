"""
Top‑level definitions for the plugin architecture.

This module defines the abstract :class:`PluginInterface` that all concrete
plugins must inherit from.  The interface standardises how plugins receive a
logger (optional) and how they process incoming payloads via the
:py:meth:`apply` method.
"""

import abc
import logging
import requests

from typing import Dict, Optional, Tuple, Any
from requests.exceptions import RequestException


class PluginInterface(abc.ABC):
    """
    Abstract base class for all plugins.

    Sub‑classes must provide a concrete implementation of the
    :py:meth:`apply` method.  The ``name`` attribute can be overridden by a
    subclass to give the plugin a human‑readable identifier; it defaults to
    ``None`` when not set.
    """

    name = None

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the plugin base class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for plugin use.  If ``None``, the plugin
            will have ``self._logger`` set to ``None`` and can choose
            to skip logging.

        Returns
        -------
        None
        """
        self._logger = logger

    @abc.abstractmethod
    def apply(self, payload: Any) -> Any:
        """
        Process an input payload and return a transformed payload.

        Concrete plugins must implement this method.  The method receives a
        dictionary representing the input data and must return a dictionary
        with the processed result.  Each plugin defines the exact semantics
        of the transformation.

        Parameters
        ----------
        payload : Any
            The incoming payload to process.  The exact shape depends on the
            concrete plugin implementation.

        Returns
        -------
        Any
            The transformed payload.  The type and structure depend on the
            concrete plugin implementation.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by a subclass.
        """
        pass


class HttpPluginInterface(PluginInterface, abc.ABC):
    """
    Abstract base class for plugins that need to communicate with a remote HTTP
    endpoint.

    Sub‑classes must define the ``host_url`` and ``endpoint_path`` properties
    that point to the host and API path they will query.  The ``_request``
    helper performs a POST request with the supplied ``payload`` and returns
    the decoded JSON response.  Any HTTP‑related errors are logged (if a
    logger is available) and re‑raised.

    The concrete ``apply`` method remains abstract – implementations are free
    to post‑process the response as required.

    Attributes
    ----------
    host_url : str
        Base URL of the remote service (e.g. ``"https://api.example.com"``).
    endpoint_path : str
        API path appended to *host_url* (e.g. ``"api/guardrails/nask"``).
    """

    host_url = None
    endpoint_path = None

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the HTTP plugin base class.

        Validates that ``host_url`` and ``endpoint_path`` are set (non‑empty)
        on the subclass before delegating to ``super().__init__``.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for plugin use.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If ``host_url`` or ``endpoint_path`` are not set (``None`` or
            empty string).
        """
        if not self.host_url or not self.endpoint_path:
            raise Exception(
                "host_url and endpoint_path must be set before initialization!"
            )

        super().__init__(logger=logger)

    @property
    def endpoint_url(self) -> str:
        """
        URL of the remote endpoint to which the payload will be sent.

        Constructs the full URL by appending *endpoint_path* to *host_url*,
        trimming any trailing slash from the host.

        Returns
        -------
        str
            The full endpoint URL (e.g. ``"https://api.example.com/api/guardrails/nask"``).
        """
        return self.host_url.rstrip("/") + "/" + self.endpoint_path

    @abc.abstractmethod
    def apply(self, payload: Any) -> Tuple[bool | str, Dict]:
        """
        Process *payload* using the common HTTP request mechanism.

        Concrete plugins should call ``self._request(payload)`` and then
        transform the returned data as needed.

        Parameters
        ----------
        payload : Any
            The incoming payload to send to the remote service.

        Returns
        -------
        Tuple[bool | str, Dict]
            A tuple of (success indicator, response data).  The exact types
            of the success indicator depend on the concrete plugin.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by a subclass.
        """
        pass

    def _request(self, payload: Dict) -> Dict:
        """
        Send *payload* to ``self.host_url`` via an HTTP POST request and return
        the JSON response.

        If the request fails (network error, non‑2xx status), the error is
        logged (if a logger is available) and re‑raised.

        Parameters
        ----------
        payload : dict
            The data to send as the JSON body of the POST request.

        Returns
        -------
        Dict
            The JSON response body decoded as a dictionary.

        Raises
        ------
        requests.exceptions.RequestException
            If the HTTP request fails (e.g. connection error, timeout,
            non‑2xx status code).
        """

        try:
            response = requests.post(self.endpoint_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            if self._logger:
                self._logger.error(
                    "HTTP request to %s failed: %s", self.endpoint_url, exc
                )
            raise
