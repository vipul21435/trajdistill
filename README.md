# trajdistill

Select, clean and convert coding-agent trajectories into chat-format training samples, and report what each
filter removed and how the composition of the data changed.

## Why this is hard

Raw agent trajectories are long, repetitive and mostly failures. Used as they are, they teach a model to
loop, fill its context with huge command outputs and over-represent the few repositories that produced the
most runs. Every choice made to avoid that (outcome, length, loops, how many runs per issue, a cap per
repository) changes the data, and those choices are rarely reported. trajdistill applies them in a fixed
order, counts what each one removed and shows the data before and after.

## Install

```
pip install git+https://github.com/vipul21435/trajdistill
```

Python 3.10 or newer, no runtime dependencies. For exact token counts add the tiktoken extra:

```
pip install "trajdistill[tiktoken] @ git+https://github.com/vipul21435/trajdistill"
```

## Quickstart

`examples/sample.jsonl` holds 21 made-up trajectories (`python examples/make_sample.py` rebuilds it).

```
$ trajdistill funnel examples/sample.jsonl --test-patches drop --loops drop --max-steps 20 \
    --per-instance best --max-per-repo 2
rows in: 21   rows out: 4
filter        setting                              before  removed  changed  after
duplicates    exact content                            21        1        0     20
outcome       resolved                                 20       10        0     10
exit_status   off                                      10        0        0     10
empty_patch   required                                 10        1        0      9
test_patches  drop                                      9        1        0      8
loops         drop (action+observation, window 1)       8        1        0      7
max_steps     20                                        7        1        0      6
max_tokens    off                                       6        0        0      6
per_instance  best k=1                                  6        1        0      5
per_repo      2                                         5        1        0      4
patches touching test files (flagged, before filtering): 1

composition                                   before                after
rows                                              21                    4
instances                                         14                    4
instances with several runs                        4                    0
repositories                                       3                    3
largest repository               acme/widgets 57.1 %  acme/widgets 50.0 %
concentration (HHI)                           0.4195               0.3750
effective repositories                           2.4                  2.7
resolved / unresolved / unknown          11 / 10 / 0            4 / 0 / 0
steps median / p90 / max                 6 / 12 / 30        8.5 / 12 / 12
tokens median / p90 / max            198 / 347 / 843    255.5 / 346 / 346
tokens total                                    5106                 1014
exact duplicates                                   1                    0
```

The four commands:

```
trajdistill funnel  INPUT [cleaning] [filters]            # print the report, write nothing
trajdistill select  INPUT -o kept.jsonl --report r.json   # same options, write survivors + report
trajdistill convert kept.jsonl -o samples.jsonl --to openai|sharegpt \
                    [--granularity step --context-budget 8192] [cleaning]
trajdistill split   samples.jsonl --train train.jsonl --val val.jsonl --by instance|repo
```

```
$ trajdistill convert examples/sample.jsonl -o steps.jsonl --to sharegpt --granularity step \
    --context-budget 300
{"trajectories": 21, "samples": 167, "skipped_empty": 0, "skipped_over_budget": 0, "steps_omitted": 249,
 "tool_call_text_dropped": 0, "tokens": 28026}
```

`python examples/pipeline.py` runs the same steps through the Python API.

## How it works

**Reading.** Four formats, guessed from the path and the first record (`--format` overrides): `.traj` files
written by SWE-agent, one file or a directory; rows shaped like the `nebius/SWE-agent-trajectories` dataset;
JSONL rows with an OpenAI-style `messages` list, tool calls included; and the JSONL `select` writes. Any of
them may be gzipped. Each assistant turn becomes a step with its text, its tool calls, the environment's
replies and a canonical action (the tool calls, else the last fenced block). A trajectory without an id gets
`instance#<12 hex digits of a content hash>`, so ids do not depend on where a row sits in the file.

**Cleaning** (all off by default, run before the filters, counted). In this order: strip ANSI escape codes;
redact key-like strings and private-key blocks to `[REDACTED]` and home directories followed by a path
(`/home/<name>/...` and the macOS and Windows user folders, also after `=`, `:`, `file://` or a flag such
as `-I`, but not inside a URL with a host) to `~`; shorten each observation longer than
`--max-observation-chars N` to its head and tail around a marker `[... k characters elided ...]`, so the
result never exceeds N characters (redaction runs again after a cut); normalise whitespace (line endings,
trailing spaces, runs of blank lines); `--drop-failed` removes steps whose reply matches a failure pattern
(by default SWE-agent's edit-linter message and `command not found`; the step after, usually the retry, is
kept); `--collapse-repeats` merges consecutive steps with the same action and replies. `--clean` turns on
ANSI, redaction, whitespace and collapsing at once. Running the pass twice gives the same result as running
it once (a property test checks this).

**Filters**, in this order; each can be switched off and reports rows before, removed and after:

1. duplicates: identical content, or with `--dedupe-key actions` the same action sequence on the same
   instance;
2. outcome: resolved only (the default, so `.traj` input without resolution data needs `--outcome any`);
3. exit status, glob patterns such as `submitted*`;
4. empty final patch;
5. patches that touch test files (by directory or file name): always flagged, dropped with
   `--test-patches drop`;
6. loops: a step is a repeat when it is identical to the step before it, meaning the same action and the
   same observation (or the same action only with `--loop-key action`; `--loop-window w` compares with the
   `w` steps before). `--loops drop` removes the trajectory and `--loops cut` truncates it just before the
   first repeat. With `--collapse-repeats` the default key finds nothing, because collapsing merges those
   steps first;
7. maximum steps;
8. maximum tokens: characters divided by 4 and rounded up, or exact counts with `--tokenizer tiktoken`;
9. runs per instance: all, the shortest (fewest steps, then fewest tokens), the best (resolved first, then
   shortest) or a seeded random pick, keeping `k`;
10. cap per repository. The cap keeps distinct instances first: every instance's first pick (in a seeded
    order) comes before any instance's second.

The input is sorted by instance, id and content hash before filtering and every random choice is a seeded
hash of the id, so the output does not depend on input order (a property test shuffles the input).

**Composition.** Before (as read) and after (as kept): rows, instances, instances with several runs,
repositories, the five largest repositories and their shares, the Herfindahl-Hirschman index
`HHI = sum_i s_i^2` over repository shares `s_i` (1 when one repository holds everything, `1/R` for `R`
equal repositories) and `1/HHI` as an effective number of repositories, outcomes, steps and token summaries
(median, nearest-rank 90th percentile, maximum), token totals and exact duplicates.

**Samples.** `openai` writes `{"messages": [...]}` with `assistant.tool_calls` and `tool` replies carrying
`tool_call_id` when the source has tool calls (ids are made up as `call_<step>_<k>` when missing); replies of
text-command agents become `user` turns. `sharegpt` writes `{"conversations": [...]}` with the roles listed
in the LLaMA-Factory data guide (`system`, `human`, `gpt`, `function_call`, `observation`), which places
model and environment turns in alternating positions; a step with tool calls becomes one `function_call`
turn. Every sample ends on a model turn. With `--granularity step` there is one sample per step t: the
context up to t and step t's model turn as the target. Under `--context-budget` the oldest steps are dropped
until the sample fits, and a short note (`[n earlier steps omitted]`) says so; a step that does not fit even
alone is skipped and counted. A test checks on random trajectories that no sample exceeds its budget.

**Split.** Whole groups (instances, or repositories with `--by repo`) go to one side. Groups are taken in a
seeded order into validation until it holds `--val-fraction` of the rows, so it can overshoot by one group.
No instance is ever on both sides (a property test checks this).

**Background.** SWE-agent (Yang et al. 2024) introduced an agent-computer interface through which language
model agents edit code, navigate repositories and run tests. SWE-Gym (Pan et al. 2024) is an environment of
2,438 Python task instances used to train agents and verifiers, and releases agent trajectories. SWE-smith
(Yang et al. 2025) synthesises task instances at scale (50k from 128 repositories) and releases trajectories
with the model it trains. R2E-Gym (Jain et al. 2025) curates executable environments for training
open-weight agents. The write-up that accompanies the nebius dataset describes its own filtering of
successful trajectories: drop those with clearly incorrect actions (malformed responses, tool misuse,
linter-triggering errors), remove duplicate trajectories per problem keeping unique action sequences, and
keep only the 30 shortest per problem. `--drop-failed`, `--dedupe-key actions` and
`--per-instance shortest -k 30` correspond to those steps; trajdistill adds the counts and the composition
report.

## Results on real public data

Data: 3,000 rows of the `nebius/SWE-agent-trajectories` train split (80,036 rows; licence CC BY 4.0 on its
card), fetched through the Hugging Face dataset viewer API as 120 runs of 25 consecutive rows at seeded
random offsets, without the `eval_logs` column. The sample holds 245 instances from 179 repositories,
81,802 steps, and runs of three models (`swe-agent-llama-70b` 2,819, `-8b` 141, `-405b` 40). No trajectory
is committed here.

```
python scripts/fetch_nebius_sample.py --rows 3000 --seed 0 --out data/nebius-sample.jsonl.gz
trajdistill funnel data/nebius-sample.jsonl.gz
```

Resolved only (the default) keeps 569 of 3,000 rows (19.0 %), 68 instances from 61 repositories. That alone
roughly doubles concentration: the largest repository, pydantic/pydantic, goes from 8.4 % to 11.1 % of rows
and HHI from 0.0157 to 0.0337 (effective repositories 63.5 to 29.7). Tokens fall from 45,058,876 to
5,492,708; the median trajectory goes from 17 to 12 steps. 293 of the 3,000 patches touch test files.

A fuller recipe:

```
trajdistill funnel data/nebius-sample.jsonl.gz --strip-ansi --redact --normalise-whitespace \
  --max-observation-chars 4000 --drop-failed --dedupe-key actions --exit-status 'submitted*' \
  --test-patches drop --loops drop --max-tokens 32000 --per-instance shortest -k 3 --max-per-repo 10
```

Cleaning changes every row: 8,917 observations shortened (7,874,161 characters elided), 11,700 failed steps
dropped (before cleaning the edit-linter message appears in 8,715 replies and `command not found` in 2,985),
452 texts with ANSI codes stripped, 2,218 redactions. Tokens go from 45,058,876 to 36,010,118. Then:

| filter | setting | removed | left |
|---|---|---:|---:|
| duplicates | same actions per instance | 4 | 2,996 |
| outcome | resolved | 2,430 | 566 |
| exit status | `submitted*` | 0 | 566 |
| empty patch | required | 0 | 566 |
| test-file patches | drop (292 flagged after cleaning) | 39 | 527 |
| loops | drop | 15 | 512 |
| max tokens | 32,000 | 0 | 512 |
| per instance | shortest, k = 3 | 351 | 161 |
| per repository | 10 | 8 | 153 |

Share of the largest repository (pydantic/pydantic): 8.4 % of the input, 11.1 % of the 161 rows that reach
the cap, 6.5 % after it. HHI: 0.0157 in, 0.0277 before the cap (the same command without `--max-per-repo`),
0.0211 after; effective repositories 63.5, 36.2 and 47.5. The 153 kept trajectories hold 984,489 tokens
(median 6,155, maximum 15,712) and 8 steps at the median.

Converting and splitting them:

```
trajdistill select data/nebius-sample.jsonl.gz <the options above> -o data/kept.jsonl
trajdistill convert data/kept.jsonl -o data/steps.jsonl --granularity step --context-budget 8192
trajdistill split data/steps.jsonl --train data/train.jsonl --val data/val.jsonl --by repo --val-fraction 0.1
```

gives 1,452 step samples (6,369,268 tokens, 514 earlier steps left out to fit the budget, none skipped),
split into 1,287 train and 165 validation samples from 52 and 7 repositories, with no instance on both sides.
The default funnel over the 3,000 rows took 23 seconds here.

## Limitations

- Readers cover SWE-agent `.traj`, nebius-style rows and OpenAI-style message lists. OpenHands event logs
  (actions and observations as separate events) are not read; they need converting to messages first.
- Failure detection is a list of regular expressions over replies plus a `failed` flag; the defaults are
  SWE-agent strings. When a failed step is dropped, the retry's text may still refer to the failure.
- ShareGPT output does not keep the text that accompanies a tool call (counted as `tool_call_text_dropped`),
  and step samples do not mark which turns to train on: the target is the last turn, and masking earlier
  turns is left to the trainer.
- Redaction is pattern based. It misses secrets without a known prefix, a home directory written without a
  following path (`cd /home/<name>`), macOS user names with spaces, and homes elsewhere (`/root`,
  `/var/home`, a Windows folder seen through `/mnt/c/Users`). It over-matches too: a web path under `/home/` (`GET /home/<page>/...`)
  becomes `~/...`, and a long identifier that starts with one of the key prefixes and mixes digits and capitals
  is replaced. An unterminated private-key block is redacted up to the next blank line or diff header. The
  test-file flag is taken from the patch before redaction.
- Tokens are a character estimate unless tiktoken is installed; the estimate is not any model's tokenizer.
- The sample above is 3.7 % of the dataset in runs of 25 neighbouring rows, which group runs of the same
  instance; its numbers describe that sample, not the whole dataset.
- Tests (89, 97 % line coverage with `uv run --extra dev pytest --cov=trajdistill`) use synthetic fixtures;
  the run above is the only check against real rows.

## Related tools

- [trajlens](https://github.com/vipul21435/trajlens) reads the same trajectory formats for analysis;
  trajdistill keeps its own small readers and is about building training data.
- [sftlint](https://github.com/vipul21435/sftlint) lints chat and tool-calling datasets, such as the samples
  written here.
- [datasieve](https://github.com/vipul21435/datasieve) deduplicates training data; trajdistill only drops
  exact and same-action duplicates.
- [contamscan](https://github.com/vipul21435/contamscan) checks benchmark contamination by n-gram overlap.

References: John Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering",
2024, https://arxiv.org/abs/2405.15793. Jiayi Pan et al., "Training Software Engineering Agents and Verifiers
with SWE-Gym", 2024, https://arxiv.org/abs/2412.21139. John Yang et al., "SWE-smith: Scaling Data for Software
Engineering Agents", 2025, https://arxiv.org/abs/2504.21798. Naman Jain et al., "R2E-Gym: Procedural
Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents", 2025, https://arxiv.org/abs/2504.07164.
Dataset card: https://huggingface.co/datasets/nebius/SWE-agent-trajectories. Write-up: "Leveraging training
and search for better software engineering agents", 2024,
https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents. ShareGPT roles:
https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
