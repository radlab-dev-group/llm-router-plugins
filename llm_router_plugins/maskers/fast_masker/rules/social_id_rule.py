"""
Rule that masks generic social‑media identifiers.
"""

import re

from llm_router_plugins.maskers.fast_masker.rules.base_rule import BaseRule


class SocialIdRule(BaseRule):
    """
    Detects Facebook numeric IDs (e.g. ``fbid1234567890``) and masks them with
    ``{{SOCIAL_ID}}``.

    .. note::
       ``@username`` handles are not matched because the ``@`` symbol would
       collide with :class:`EmailRule`.  If @username support is needed in the
       future, introduce it in a separate rule placed *before* EmailRule in the
       pipeline so emails do not intercept the pattern first.
    """

    # Match Facebook numeric IDs only – avoids collision with email @ symbol.
    _SOCIAL_REGEX = r"""
        \b
        fbid\d{8,}\b
    """

    def __init__(self):
        super().__init__(
            regex=self._SOCIAL_REGEX,
            placeholder="{{SOCIAL_ID}}",
            flags=re.IGNORECASE | re.VERBOSE,
        )
