# Data

The fleet simulation reads five inputs from `data/` under the repo root (`--root`). Four are small and in git. The fifth,
the trace bins (23.8 MB), is not: rebuild it from the public trace (below). The code checks the sha256 of every input
except `chutes_w1_analysis.json`, and refuses a file that does not match.

## The fleet simulation (`cloud_sim.py`, `idle_backfill.py`)

We checked this list by logging every file a headline run opened.

| Path under `data/` | In git | Bytes | sha256 | What it is |
|---|---|---|---|---|
| `traces/derived/bins_w1.csv.gz` | no | 23,799,883 | `0d60071280ac4afe180bf6cd76183786c9d58d68704818625928233b09518f7e` | Requests per (minute, model) over 14 days of the Chutes trace |
| `traces/derived/chutes_w1_analysis.json` | yes | 495,311 | `6929e00a341190ce4327c838109a40851b651fbf76f5ce91fd54ea202067f8f4` | Per-model demand summary of the same 14 days |
| `size_census/census.json` | yes | 190,817 | `348e449344fd5b02b0b37604224a2db1a052c0aa15f9ed7a34be35dc9c1d34cb` | Weights and KV-cache bytes per token for each model, from its public Hugging Face metadata |
| `size_census/demand_R2.json` | yes | 419,812 | `342929b9d4ebe393746d67f0d11fc36763d9727a635bd5ab33eb9defa3c2cb2c` | Per-model memory demand in its busiest held-out minute, built from the files above |
| `lineage_census/census.json` | yes | 52,418 | `8778a4ce6e34c2e8d18739e5bcf10d54808e353c03346db1a6b3e3d04adde6af` | Which models are copies or fine-tunes of the same base model |

## Where they come from

- **The trace.** The [Chutes trace](https://github.com/HarvardMadSys/chutes_workload), from W. Nixon, J. Durbin,
  F. Standhartinger, H. S. Gunawi and J. Yang, "A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing",
  arXiv 2608.13573 (2026). CC BY 4.0. We read 225 of its 6,114 row groups: 14 days of trace time (1970-10-28 to
  1970-11-11 in the trace's own clock), 224,708,741 requests. Days 300–306 train the placement; days 307–313 are held out.
- **The model map.** `chute_models.csv` from the same repository (sha256
  `26a464b848c72a10925a71a3704be7bfca709161185000ea4cfc013902a079cd`). The scripts expect it at
  `data/traces/raw/chute_models.csv`.

**The bins.** Rebuild them with `pyarrow` installed (it reads about 1.4 GB over HTTP):

```bash
python3 scripts/real_trace_analysis.py --fetch --start 1970-10-28T00:00:00 --days 14 --tag w1   # writes data/traces/derived/
zcat data/traces/derived/bins_w1.csv.gz | sha256sum   # should print 3caadfe1ac8b62dd9645d2b827244c9e41594dd02a98621d984bda6f7d622390
```

A rebuilt file carries a new gzip timestamp, so its file sha256 differs from the pin even when the rows match. Once the
content check above passes, set `BINS_SHA` in `scripts/tailpool_sim.py` to the new file's sha256.

**The demand summary.** Built from the bins and the model map:

```bash
python3 scripts/real_trace_analysis.py --analyze --bins data/traces/derived/bins_w1.csv.gz \
  --out data/traces/derived/chutes_w1_analysis.json \
  --window "trace 1970-10-28..1970-11-11 (= 2026-02-05..2026-02-19 if day 0 = 2025-04-11)"
```

Re-run on these bins, this gives the published file, or one that differs in the last digit of a few floats. Which
floats depends on the numpy build: we saw 0, 18 and 237 on three setups, all in `cv_1m` or `summary`. The simulation
reads neither. A rebuilt bins file also changes `summary.bins_sha256`.

**The size census.** `scripts/size_census.py --fetch` reads each model's public Hugging Face metadata and `--derive`
writes `census.json`. The script's docstring gives the flags. A re-fetch can drift as repositories change or disappear.
`demand_R2.json` needs no network. Built from the published census and the bins, it matches the published file byte for
byte:

```bash
python3 scripts/size_census.py --demand --params R2 --root . --cache ~/.cache/bins_w1.npz \
  --census data/size_census/census.json --out data/size_census/demand_R2.json
```

The `--cache` file is the one the simulation builds from the bins on its first run.

**The lineage census.** Built by a probe script that is not in this repo. For each model repository named in the model
map, it read the public Hugging Face file index and config, then took small HTTP range reads of a few weight tensors
(three layers, at most 131,072 values each, never a whole file). Repositories are compared only within one architecture
shape. Two count as **copies** when their weight files have the same hashes in the index, or when at least three shared
slices agree value by value to within one unit in the last place (a constant scale is allowed only on tensors whose
normalisation cancels it). `dup_sets_strict` keeps the copies that a second round of reads confirmed, plus the
byte-identical ones. Two repositories are in the same **lineage** when the median distance between their scale-normalised
slices is below 0.05. The census lists the connected groups. The simulation reads only `dup_sets_strict`,
`lineage_sets` and `repo_cids`.

## Changes from the files behind our runs

The published copies differ from the files our runs used in fields the simulation does not read:

- `chutes_w1_analysis.json`: `summary.bins_file` held a local absolute path. It now holds the path relative to the repo,
  which is what `real_trace_analysis.py` now writes.
- `size_census/census.json` and `demand_R2.json`: an internal reference label was removed from 130 `why` strings, and
  one key was renamed (to `fail31`). `size_census.py` now writes the same forms.

The sha256 pins in `scripts/fabric_sim_v2.py` match the published files. The lineage census is unchanged.

## License

The files under `data/` are derived from the Chutes trace and its model map, and are shared under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the trace. We aggregated requests per minute and per model
and computed the summaries above. Please cite the trace's paper (above) if you use them.

## Other runs

The GPU benchmarks need no data: they download pinned models from Hugging Face. The Aegaeon comparison reads outputs of
earlier GPU runs that are not published yet (see [REPRODUCE.md](REPRODUCE.md) § 7).
