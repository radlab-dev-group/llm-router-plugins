# NASK guardrail guide

Guardrails run on the way **out**: they decide whether the model's answer is
acceptable. `nask_guard` delegates that decision to a remote NASK service.

## What it does

`NASKGuardPlugin` (`name = "nask_guard"`) inherits `GuardrailsBase`. It posts
the generated text to a configured host and reads back a safety verdict, which
it annotates onto the response.

## Configuration

The plugin needs a host and fails fast at construction if it is missing:

- `LLM_ROUTER_GUARDRAIL_NASK_GUARD_HOST` — the host of the NASK guard service.

## When to use it

Use `nask_guard` when outbound content must be screened by a specific external
policy (a NASK-hosted classifier) before it reaches the user. It is one of the
two guardrail plugins in the package; the other is `sojka_guard`.
