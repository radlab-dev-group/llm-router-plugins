# FastMasker plugin guide

Maskers run on the way **in**: they remove personal data from the text the
model will see. `fast_masker` is the local, rule-based one — no external
service, everything happens in the router process.

## What it does

`FastMaskerPlugin` (`name = "fast_masker"`) walks the payload recursively and
masks every string it finds. It returns two things: the redacted payload and
the mappings, so the masked values can be restored later if a step needs them.

`mask()` therefore returns `(masked_text, mappings)`, and the plugin's
`apply` returns `(masked_payload, mappings)`.

## The rules

Rules are applied in certainty order, highest first, so a checksum-validated
identifier is never half-masked by a weaker pattern. The full table — rule,
placeholder, what it detects and its notes — is in
`llm_router_plugins/maskers/fast_masker/README.md`. The groups, strongest to
weakest:

- checksum-validated identifiers (KRS, NIP, SSN, IBAN),
- strict formats (JWT, invoice, order, transaction reference),
- international phone numbers,
- well-structured values (email, URL, IP and port, bank account),
- business identifiers (invoice, order, transaction reference),
- pattern-based with context (dates, money, time),
- format-based (postal code, health ID, car plate, SIM card, SSL serial),
- generic patterns (Polish phone, social ID).

Each rule replaces its match with a placeholder (`{{EMAIL}}`, `{{IBAN}`,
`{{PHONE_INTERNATIONAL}}`, ...) so the structure survives and a de-anonymizer
can put it back.

## Deanonymization

`FastDeanonymizer` reverses a masking pass using the mappings a mask produced.
It is the inverse of the masker and is what the write path of other layers
relies on when it must store the real value.

## When to use it

Use `fast_masker` when the payload may carry Polish or generic personal data
and you want it gone **before** any model call, with no dependency on an
external masking service. It is deterministic and offline, so it is the default
choice for the anonymization step of the router.
