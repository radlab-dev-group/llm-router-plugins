"""URL masking rule for the fast masker."""

import re

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


class UrlRule(BaseRule):
    """
    Detects HTTP/HTTPS URLs and www-prefixed domain names.

    Examples of matched patterns:
        https://example.com, http://foo.bar.pl/page, www.wp.pl
        http://192.168.1.1:8080/path (IP-based), http://localhost:3000

    Full URLs (with http/https scheme) accept any TLD.  Standalone domains
    must be prefixed with ``www.`` to reduce false positives in regular text.
    """

    # URL pattern using clean non-VERBOSE regex for reliable matching.
    # Three alternatives:
    #   A: https?:// + domain.tld (any TLD, optional subdomains)
    #   B: https?:// + IP_address_or_localhost + optional_port_path
    #   C: www.domain.tld (only www-prefixed domains)
    _URL_REGEX = (
        r"(?:https?://"  # A: scheme
        r"(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,})"  # domain
        r"(?:[/:][^\s\)]*)?"  # optional path
        "|"
        r"https?://"  # B: scheme
        r"(?:"
        r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}"  # IPv4 address
        r"|localhost"  # OR localhost
        r")"
        r"(?::\d+)?(?:/[^\s\)]*)?"  # optional port + path
        "|"
        r"(?<![A-Za-z0-9_])www\."  # C: www prefix
        r"[A-Za-z0-9-]+\.[A-Za-z]{2,}"  # domain.TLD
        r"(?:[/:][^\s\)]*)?"  # optional path
        r"(?![A-Za-z0-9_\(\)])"  # not followed by word char or (
        r")"
    )

    def __init__(self):
        super().__init__(
            regex=self._URL_REGEX,
            placeholder="{{URL}}",
            flags=re.IGNORECASE,
        )
