# PII masker plugin guide

`pii_masker` is the HTTP masker: it sends the text to a remote PII-masking
service and applies the answer to the payload.

## What it does

`PiiMaskerPlugin` (`name = "pii_masker"`) inherits the shared
`HttpMaskersBase`. It calls `api/maskers/pii` on a configured host and writes
the masked text back into the payload.

## Configuration

The plugin needs a host and fails fast at construction if it is missing:

- `LLM_ROUTER_MASKER_PII_HOST` — the host of the PII masking service.

If the host is not set, building the plugin raises a `RuntimeError` naming the
variable, so a misconfigured deployment is visible at start-up rather than as
an empty masking step.

## When to use it

Use `pii_masker` when the masking logic already lives behind an internal
service (a model or a policy engine) and the router should just delegate. It is
the counterpart of `fast_masker`: the same contract, the remote service doing
the work instead of the local rule set.
