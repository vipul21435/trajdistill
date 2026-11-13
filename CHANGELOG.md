# Changelog

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
