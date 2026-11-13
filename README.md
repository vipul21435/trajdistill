# trajdistill

Select, clean and convert coding-agent trajectories into chat-format training samples, and report what each
filter removed.

Status: early. Reading trajectories and the selection funnel work and are tested; cleaning, sample
conversion, the composition report and splitting are not written yet (see "Not implemented yet").

## Why this is fiddly

Raw agent trajectories are long, repetitive and mostly failures. Used as they are, they teach a model to
loop, fill its context with huge command outputs and over-represent the few repositories that produced the
most runs. Every choice made to avoid that (outcome, length, loops, how many runs per issue, a cap per
repository) changes the data, and those choices are rarely reported. trajdistill applies them in a fixed
order and counts what each one removed.

## Install

```
pip install git+https://github.com/vipul21435/trajdistill
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
```

`trajdistill select INPUT -o kept.jsonl --report funnel.json` takes the same options, writes the surviving
trajectories as JSONL and the funnel as JSON.

## What works so far

Input formats (`--format auto` guesses from the path and the first record):

- SWE-agent `.traj` files, one file or a directory of them;
- rows shaped like the `nebius/SWE-agent-trajectories` dataset (`instance_id`, `target`, `trajectory` of
  `{role, text}` turns, `exit_status`, `generated_patch`), as JSONL or a JSON list;
- JSONL rows with an OpenAI-style `messages` list, tool calls included;
- the JSONL that `select` writes.

Filters, in the order they run; each can be switched off and each reports rows before, removed and after:

1. exact duplicates (same content);
2. outcome: resolved only (default) or any;
3. exit status, glob patterns such as `submitted*`;
4. empty final patch;
5. patches that touch test files: always flagged, dropped with `--test-patches drop`;
6. loops: a step identical to the one before it (action and observation by default, or action only, over a
   window of earlier steps): keep, drop the trajectory, or cut it before the repeat;
7. maximum steps;
8. maximum tokens (characters / 4 by default; `pip install 'trajdistill[tiktoken]'` and `--tokenizer tiktoken`
   for exact counts);
9. runs per instance: all, the shortest, the best (resolved first, then fewest steps) or a seeded random
   pick, keeping `k`;
10. cap per repository, spread across instances so the cap removes repeat runs of one issue first.

Input is sorted by instance, id and content hash before filtering, ids are content hashes when the source
has none, and random choices use a seeded hash of the id, so the output does not depend on input order.
A property test checks this on shuffled inputs.

## Not implemented yet

Cleaning transforms, conversion to OpenAI messages and ShareGPT samples, step-level samples, the
composition report (by repository, outcome, steps and size, with a concentration index), train and
validation splits, an OpenHands event reader, and a run on real public data.

## Related tools

- [trajlens](https://github.com/vipul21435/trajlens) reads the same trajectory formats for analysis;
  trajdistill keeps its own small readers and is about building training data.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
