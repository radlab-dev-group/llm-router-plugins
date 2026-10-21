"""
Knowledge plugins for coding agents.

The package holds plugins that enrich an agent request with knowledge the agent
did not ask for and would not have found: verified fixes for the failure it is
currently facing, changes already known not to work in this repository, and
the exact text its conversation held before the context was summarized.

:mod:`agentic_knowledge` is the plugin; the modules shared by its adapters live
inside it.
"""
