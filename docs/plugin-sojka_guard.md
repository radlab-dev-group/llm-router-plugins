# Sojka guardrail guide

`sojka_guard` is the other outbound guardrail. It posts the model's answer to
a Sojka guard service and reads back a per-chunk safety decision.

## What it does

`SojkaGuardPlugin` (`name = "sojka_guard"`) inherits `GuardrailsBase`. It
calls `api/guardrails/sojka_guard`, splits the answer into chunks, and collects
a verdict for each. The response carries, per chunk, a `label`, a `safe`
boolean and a `score`, plus an overall `safe` flag.

On success `apply` returns the extracted fields; on any error (network,
unexpected payload, missing keys) it returns `{'success': False}` rather than
raising, so a guard that is down does not take the router down with it.

## Configuration

- `LLM_ROUTER_GUARDRAIL_SOJKA_GUARD_HOST` — the host of the Sojka guard service.

## When to use it

Use `sojka_guard` when you want a chunked, scored safety review of the answer
from a Sojka service. Choose between it and `nask_guard` on the policy each one
hosts.
