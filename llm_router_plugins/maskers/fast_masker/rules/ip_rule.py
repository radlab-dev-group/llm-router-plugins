"""
Rule that mask IPv4 and IPv6 addresses.
"""

import re
from typing import Optional, Callable, Tuple, List

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


def _is_valid_ipv4_octet(octet: str) -> bool:
    """Return ``True`` if *octet* is a valid IPv4 octet (0-255)."""
    return 0 <= int(octet) <= 255


def _is_valid_port(port: str) -> bool:
    """Return ``True`` if *port* is a valid TCP/UDP port (1-65535)."""
    try:
        p = int(port)
        return 0 < p <= 65535
    except ValueError:
        return False


class IpRule(BaseRule):
    """
    Detects IPv4, IPv6 addresses **and** the special hostname ``localhost``.
    If a port is present after a colon (e.g. ``localhost:1211`` or
    ``0.0.0.0:2131``) the address and the port are masked **separately**,
    yielding ``{{IP}}:{{PORT}}``.
    """

    # Placeholder for the IP/hostname part
    _IP_PLACEHOLDER = "{{IP}}"
    # Placeholder for the port part (if any)
    _PORT_PLACEHOLDER = "{{PORT}}"

    # IPv4: four octets, each 0‑255 (light validation, not strict)
    _IPv4_REGEX = r"""
        (?:\d{1,3}\.){3}\d{1,3}
    """

    # IPv6: eight groups of 1‑4 hex digits separated by ':'
    _IPv6_REGEX = r"""
        (?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}
    """

    # Hostname we want to treat as an IP address
    _LOCALHOST_REGEX = r"localhost"

    # Combined address regex (IPv4 | IPv6 | localhost)
    _ADDRESS_REGEX = rf"""
        (?P<addr>
            {_LOCALHOST_REGEX}
            |
            {_IPv4_REGEX}
            |
            {_IPv6_REGEX}
        )
    """

    # Optional port (digits after a colon)
    _PORT_REGEX = r"""
        (?:\:(?P<port>\d{1,5}))?
    """

    # Full pattern: address optionally followed by a port
    _IP_REGEX = rf"""
        \b
        {_ADDRESS_REGEX}
        {_PORT_REGEX}
        \b
    """

    def __init__(self):
        super().__init__(
            regex=self._IP_REGEX,
            placeholder=self._IP_PLACEHOLDER,
            flags=re.VERBOSE,
        )

    def apply(
        self, text: str, anonymizer_fn: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, List]:
        """
        Replace each address with ``{{IP}}`` and, if a port is present,
        replace it with ``{{PORT}}`` while preserving the separating colon.
        """
        mappings = []

        def replacer(match: re.Match) -> str:
            # Always replace the address part
            addr = match.group("addr")

            # Validate IPv4 octets (0-255)
            if "." in addr:  # IPv4
                octets = addr.split(".")
                if not all(_is_valid_ipv4_octet(o) for o in octets):
                    return addr  # reject invalid octet range

            # Validate port if present
            port = match.group("port")
            if port and not _is_valid_port(port):
                return addr + f":{port}"  # keep original text

            if anonymizer_fn:
                pseudo_addr = anonymizer_fn(addr, "IP")
                mappings.append({"original": addr, "replacement": pseudo_addr})
                result = "{" + pseudo_addr + "}"
            else:
                mappings.append(
                    {"original": addr, "replacement": self._IP_PLACEHOLDER}
                )
                result = self._IP_PLACEHOLDER

            # If a port was captured, append ``:{{PORT}}``
            if port and _is_valid_port(port):
                if anonymizer_fn:
                    pseudo_port = anonymizer_fn(port, "PORT")
                    mappings.append({"original": port, "replacement": pseudo_port})
                    port_replacement = "{" + pseudo_port + "}"
                else:
                    mappings.append(
                        {"original": port, "replacement": self._PORT_PLACEHOLDER}
                    )
                    port_replacement = self._PORT_PLACEHOLDER
                result = f"{result}:{port_replacement}"
            return result

        return self.pattern.sub(replacer, text), mappings
