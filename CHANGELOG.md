# Changelog

## Unreleased

- Chat replies keep their role and `tool_call_id`; tool replies are matched to calls by id, user turns stay
  user turns, and the older `function_call` field is read.
- Redaction: home directories are found after `=`, `:`, `file://` and flags such as `-I`, with non-ASCII and
  spaced names; key blocks without an END line stop at a blank line or diff header; key patterns are
  narrower; passes repeat until nothing changes. The test-file flag is taken from the patch before
  redaction.
- Cleaning counts: `redactions` no longer counts the derived action again, and `chars_elided` is the number
  the elision marker states.
- Split: each instance gets one repository before a `--by repo` split, a group joins validation only when it
  brings the share closer to the target, and empty or uneven sides are warned about.
- Malformed records fail with the file and line; records that UTF-8 cannot encode are skipped and counted; a
  directory without `.traj` files is an error.
- The fetch script creates missing folders; the tiktoken extra is given as a git-URL install.

## 0.1.0

First release.

- Readers for SWE-agent `.traj` files, nebius/SWE-agent-trajectories rows and OpenAI-style message lists with
  tool calls, plain or gzipped; ids from a content hash when the source has none.
- Ten selection filters in a fixed order, each switchable and counted: duplicates (content or action
  sequence), outcome, exit status, empty patch, test-file patches, loops (drop or cut), maximum steps,
  maximum tokens, runs per instance (shortest, best or seeded random, keeping k) and a cap per repository
  that keeps distinct instances first.
- Cleaning transforms, idempotent and counted: ANSI stripping, redaction of key-like strings and home
  directories, head-and-tail shortening of observations under a character budget, whitespace normalisation,
  dropping failed steps, collapsing repeated steps.
- Samples as OpenAI messages (with tool calls) or ShareGPT conversations, one per trajectory or one per step
  under a token budget.
- Funnel report with composition before and after: repositories, Herfindahl-Hirschman concentration,
  outcomes, steps, token totals, duplicates.
- Train and validation split grouped by instance or repository.
- Commands `select`, `funnel`, `convert` and `split`; a fetch script for a seeded sample of the public
  dataset.
