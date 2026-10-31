# Reproduce

Each section names the hardware, the commands, and where the number lands. GPU runs used rented machines (Vast.ai and
Modal). Every harness also has a CPU rehearsal, so you can check the flow before paying for a GPU.

Common setup:

```bash
python3 -m venv .venv && . .venv/bin/activate   # Debian/Ubuntu system pip refuses installs (PEP 668)
python3 -m pip install numpy        # the simulator and scorers need only numpy
python3 -m pip install modal        # only for the Modal runs (then: modal setup)
```

All GPU runs used the stock `vllm/vllm-openai:v0.30.0` image. Unset `ENABLE_KVCACHED` and `KVCACHED_AUTOPATCH` if your
image sets them: they silently patch vLLM.

## 1. Wake from host RAM (0.13 / 0.43 / 0.79 s)

Hardware: 1 GPU with 32 GB (we used an RTX 5090 on PCIe 4 x16, with 80 GB of host RAM).

```bash
# on the GPU box, inside the vLLM 0.30.0 image
mkdir -p /root/c13/harness && cp bench/host_ram_wake/*.py /root/c13/harness/
cd /root/c13 && env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python3 harness/drive.py > run.out 2>&1
# anywhere, on the copied run directory
python3 bench/host_ram_wake/score.py /root/c13
```

`drive.py` downloads three Qwen3 models at pinned revisions, then for each one runs cold starts, warm starts, and 5
sleep → wake → first-token cycles. Set `BENCH_DIR` to run somewhere other than `/root/c13`. The page-in times are in
`rows/m_<model>.json` (`l1[].pagein_to_first_token_s`). `followup.py` is the follow-on we ran after an infrastructure stop
part-way through the MoE model (see its docstring).

CPU check: `python3 bench/host_ram_wake/score.py --self-test`.

## 2. Wake from the next GPU over NVLink (0.060 s vs 0.63 s)

Hardware: 2× A100-SXM4-80GB joined by NVLink (`nvidia-smi topo -m` should show `NV12`), PyTorch with CUDA.

```bash
python3 bench/multigpu/beta_bench.py --probe --out probe.json
```

`probe.json` holds `wake_bytes` (16.4 GB, about the size of Qwen3-8B in bf16), `wake_nvlink_s` (5 timed copies from the
other GPU) and `wake_host_s` (5 timed copies from pinned host RAM).

The full multi-GPU harness (`bench/multigpu/gpu_driver.py`, plan `inputs/plan.json`) also runs the decode-step cells
and the in-engine KV split (vLLM DCP 1 vs 2) behind "What's unresolved". CPU rehearsal:
`bash bench/multigpu/stub/run_stub.sh <empty dir>`.

## 3. Reads from a GPU that is going to sleep (no wait; +0.4 ms per GB)

Hardware: 2× A100-SXM4-80GB with NVLink, vLLM 0.30.0 image.

```bash
python3 scripts/peer_read_during_sleep.py --stub-smoke /tmp/prs_smoke     # CPU rehearsal first
python3 scripts/peer_read_during_sleep.py --box --out run                  # on the GPU box
python3 scripts/peer_read_during_sleep.py --score run                      # 100 reads during the shutdown
python3 scripts/peer_read_during_sleep.py --box --dose --out run_dose      # one read per shutdown, the cost per GB
python3 scripts/peer_read_during_sleep.py --score-dose run_dose
```

## 4. Wake through the controller (0.24–0.33 s; outputs unchanged; 2.25 s through the router)

Hardware: 2× A100-SXM4-80GB with NVLink. We ran it on Modal, twice.

```bash
modal run scripts/modal_run.py --spec bench/s0_modal/spec_s0.example.json   # the full checklist, incl. the router wake
modal run scripts/modal_run.py --spec bench/s0_modal/spec_r3.example.json   # the restore cycles again, 3 per model
```

Results land in `runs/<tag>/`. Each checklist item writes `rows/<item>.json`. The restore times and the output
comparison are in `rows/S0-6.json` (`served PEER`, `total_s`, outputs before vs after sleep). The router wake is
`rows/S0-13.json`: `parts.D` is the whole wake, `parts.yield_s` the wait for the batch job to stop. Both run scripts first
check the sha256 of every file they run against `bench/s0_modal/sha_expected.txt`.

CPU rehearsal (stub engines, no GPU): copy `bench/s0_modal/*`, `scripts/warm_wake_ext.py`, `scripts/warm_wake_stub.py`
and `scripts/fabric_ctl.py` into one empty directory, then run `S0_BACKEND=stub python3 s0_driver.py` and
`S0_BACKEND=stub python3 s0_ctl.py` there. On stub engines S0-14 reads FAIL: no stub batch request is still running when
the reclaim comes, so none is aborted. It passed on the GPU run.

The controller alone: `python3 scripts/fabric_ctl.py --self-test` (CPU, stub engines). It checks the controller's
decisions against the reference rule in `scripts/cowake_handoff.py`.

## 5. H100 decode step (6.9 / 11.5 / 12.2 ms)

Hardware: 1× H100 SXM 80 GB. We ran it on Modal.

```bash
python3 scripts/h100_step.py --stub-smoke /tmp/h100_smoke                 # CPU rehearsal, empty dir
modal run scripts/modal_run.py --spec bench/h100_step/spec_H.example.json
python3 scripts/h100_step.py --score --run-dir runs/h100_step_H/h100_step_H --out score_H.json
```

The cells are batch 1 × 4,096 tokens, batch 8 × 4,096, and batch 2 × 16,384, three passes each. The scorer refuses to run
if `bench/multigpu/gpu_driver.py` or `gpu_score.py` differ from the pinned sha256, or if its frozen predictions no longer
match `scripts/steptime_crossover.py` (which imports `aegaeon_repro.py`).

## 6. Fleet simulation (97 vs 137 models; idle 60–63 % vs 58–60 %)

CPU only. Four of its five inputs are in git under `data/` ([DATA.md](DATA.md)). The fifth, the trace bins, is
rebuilt from the public trace:

```bash
python3 -m pip install pyarrow      # only for this step
python3 scripts/real_trace_analysis.py --fetch --start 1970-10-28T00:00:00 --days 14 --tag w1
zcat data/traces/derived/bins_w1.csv.gz | sha256sum     # should print 3caadfe1ac8b62dd9645d2b827244c9e41594dd02a98621d984bda6f7d622390
sha256sum data/traces/derived/bins_w1.csv.gz            # put this value in BINS_SHA in scripts/tailpool_sim.py
```

The rebuilt file has a new gzip timestamp, so its file sha256 differs from the pinned one even when its rows match. That
is why `BINS_SHA` needs the new value. We re-ran this on 2026-10-23 on a fresh x86 machine (pyarrow 26.0.0): the
fetch took 2 minutes and downloaded about 1.5 GB, the content sha matched, and both jobs below reproduced our archived rows. Then:

```bash
mkdir -p out/fleet out/backfill     # the simulators do not create their output folders
python3 scripts/cloud_sim.py --run --arm P  --fmt int4 --N 128 --tag main --root . --cache ~/.cache/bins_w1.npz --out out/fleet/main_int4_128_P.jsonl
python3 scripts/cloud_sim.py --run --arm BS --fmt int4 --N 128 --tag main --root . --cache ~/.cache/bins_w1.npz --out out/fleet/main_int4_128_BS.jsonl
python3 scripts/cloud_sim.py --summarize out/fleet
```

`P` is the provider default (dedicated GPUs). `BS` is gpu-bunk on stock vLLM. Each row carries all 21 replays (3 seeds ×
7 held-out days). The first run builds the `--cache` file from the trace bins.

Backfill (96–97 % of idle recovered) uses the same inputs:

```bash
python3 scripts/idle_backfill.py --run --arm P --fmt int4 --N 128 --tag main --root . --cache ~/.cache/bins_w1.npz --out out/backfill/main_int4_128_P.jsonl
python3 scripts/idle_backfill.py --summarize out/backfill
```

## 7. Lossless weight compression (exact; 32.70 % smaller; 2.70 GB/s per core)

CPU only. It needs numpy and a C compiler (`cc`); the codec builds its C core on first use, into `~/.cache/lossless_bf16/`.

```bash
python3 scripts/lossless_bf16.py --self-test        # 31 checks on synthetic tensors, no download
```

The eval set is 36 safetensors files from 13 public Hugging Face model repos. `data/lossless/evalset.json` lists each
file with its sha256, its size and our container sizes. Put each repo's files in a folder named `<org>__<name>` (for
example `models/Qwen__Qwen3-4B/`), then:

```bash
python3 scripts/lossless_bf16.py --roundtrip-v2 models --evalset data/lossless/evalset.json --out rt.json
python3 scripts/lossless_bf16.py --summarize-v2 rt.json
```

The round trip checks every input file against the eval set before it encodes anything. It then encodes and decodes each
file under four settings and checks the sha256 of every decoded file. It also checks that the first setting's container
sizes equal ours to the byte; the encoder is deterministic. `--only <org>__<name>` runs one repo. A fine-tune's delta rows
need its base repo (`Qwen__Qwen3-4B` or `Qwen__Qwen3-4B-Base`) in the same folder. In the summary, `models.<name>.INC`
is the bits per weight of the first setting (16 = no saving, so the saving is 1 − bits / 16) and `E_bar.median_dE` the
9-bit symbol's extra saving. The README's 32.70 % is the median saving over the 12 bf16 repos; the 13th
(`mohantesting__Affine-come-…`) is fp16.

Our rows are in `data/lossless/roundtrip_v2.json`, so the summary also runs without a download:

```bash
python3 scripts/lossless_bf16.py --summarize-v2 data/lossless/roundtrip_v2.json
```

A small check with one public file (99.6 MB):

```bash
mkdir -p models/Qwen__Qwen3-4B
curl -L -o models/Qwen__Qwen3-4B/model-00003-of-00003.safetensors \
  https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model-00003-of-00003.safetensors
python3 scripts/lossless_bf16.py --roundtrip-v2 models --evalset data/lossless/evalset.json --only Qwen__Qwen3-4B --out rt_one.json
```

We ran this from this copy of the code on 2026-10-31. The download matched the eval set's sha256, all four settings
decoded exactly, and the container sizes matched our rows to the byte (66,922,470 B; 66,054,697 B with the 9-bit symbol).

Decode speed needs whole checkpoints and a many-core host:

```bash
python3 scripts/lossless_bf16.py --roundtrip models --out rt_v1.json --keep-dir keep --keep-std Qwen__Qwen3-4B
python3 scripts/lossless_bf16.py --bench keep --out bench.json --base-dir models --ref rt_v1.json
```

`--bench` times the decode of each kept container at 1–32 threads (5 timed runs each) and checks the output's sha256
after every run. Our run is in `data/lossless/bench_v1.json`: `bench.cases["Qwen__Qwen3-4B.std"].rows`, then the thread
count, then `gbps_median`.

## 8. What is not reproducible from this repo

- **The re-packing checks.** That covers the 2-second check, the online memory sizing and the shared pool's simulation
  results, plus the 10-second layout caveat. Their code, frozen layouts and job lists are not here. One of these checks
  alone ran 1,187 simulation jobs.
- **The Aegaeon comparison.** `aegaeon_repro.py` and `steptime_crossover.py` are here because the H100 scorer uses them.
  The comparison itself reads step-time rows from about a dozen earlier GPU runs that are not published, so 2 checks in
  `steptime_crossover.py --self-test` fail without them.
- **The in-engine KV split on a second machine** (+30–40 % per step): its harness is not here.

## What is here, and why

This is the slim cut: only the code behind the numbers in the README.

| Files | Why |
|---|---|
| `bench/host_ram_wake/` | § 1 |
| `bench/multigpu/beta_bench.py` | § 2 |
| `bench/multigpu/gpu_driver.py`, `gpu_score.py`, `stub/` | § 5 runs on this driver; the in-engine KV split (§ 2) |
| `bench/multigpu/host_gate.py`, `inputs/` | the in-engine KV split (§ 2) |
| `scripts/peer_read_during_sleep.py` | § 3 |
| `scripts/fabric_ctl.py`, `warm_wake_ext.py`, `warm_wake_stub.py`, `modal_run.py`, `bench/s0_modal/` | § 4 |
| `scripts/cowake_handoff.py`, `rehoming.py`, `neighbour_wake.py` | the rule `fabric_ctl.py --self-test` checks against |
| `scripts/measured_p.py`, `cloud_sim_v2.py` … `cloud_sim_v6.py`, `quiet_giants.py` | needed by the self-tests of the three above; no README number |
| `scripts/h100_step.py`, `steptime_crossover.py`, `aegaeon_repro.py`, `bench/h100_step/` | § 5 |
| `scripts/cloud_sim.py`, `idle_backfill.py`, and the 8 modules they import | § 6 |
| `data/` (four small files) | § 6: the demand summary, size census and lineage census the simulation reads |
| `scripts/lossless_bf16.py`, `data/lossless/` | § 7: the codec, its eval set, our round-trip rows and our decode bench |
| `patches/vllm/` | the shared KV pool; the controller does not need them |

Every module has a CPU self-test: `python3 scripts/<module>.py --self-test`. On a fresh clone they all pass except
`steptime_crossover.py`, which fails the two checks named in § 8. The self-tests do not need the trace bins.

## About this code

The code is what produced the numbers, with two kinds of change. First, comments and docstrings were edited to remove
references to a private lab notebook (task numbers, dates of internal reviews, private file paths). Module names lost
their task-number suffixes, data paths moved under `data/`, and sha256 pins were updated to the published files. Second,
three files are one revision newer than a run they reproduce:

- `warm_wake_ext.py` and `fabric_ctl.py` gained clean-up on shutdown and in their tests, plus three self-test checks,
  after the Modal runs. The changes sit in the self-test, the CPU test rig and the CPU smoke run, not in the wake path.
- `peer_read_during_sleep.py` gained the dose mode between its two runs. The other modes are unchanged, apart from one
  extra field (`dose_f`, null) in each row.

Publishing `data/` changed three small things. None of them changes what the simulation computes. `real_trace_analysis.py`
records the bins path relative to the repo instead of as an absolute path. `size_census.py` no longer writes an internal
reference label, and one output key is renamed (to `fail31`). The two size-census pins in `fabric_sim_v2.py` changed to
match the published files.

Before publishing we also removed internal labels from the code: references to a private ledger in
messages and three names (`challenger_readings`, `I_ECOLD_TAIL`, `SMI_GAP_MIB`), our internal project name (the worker
extension class is now `WarmWakeExt`, the Modal app and volume `gpu-bunk-rows` and `gpu-bunk-results`, patch 0004's
switch `VLLM_GPU_BUNK_TLS_FORWARD_CONTEXT`), and a task number in two input names (`inputs/plan.json`,
`inputs/prompts.jsonl`; the seed inside them is unchanged). The GPU runs used the old names. The sha256 pins in
`scripts/h100_step.py` and `bench/s0_modal/sha_expected.txt` were updated. No number changed: the self-tests, the CPU
rehearsals and both headline jobs give the same results as before.

From a fresh clone plus the trace bins, the two headline jobs (`P` and `BS`, int4, 128 GPUs) reproduce our archived
rows exactly, apart from the timing fields.

`lossless_bf16.py` got the same clean-up: a new file name, comments and help text without internal references, and new
names for two flags (`--roundtrip-v2`, `--summarize-v2`) and one summary key (`FAM_le8`). Its round-trip output no longer
records a private file path. The runs behind § 7 used the old names. Two things still carry an old internal number: the
container's 8-byte magic (`BBL3283\x01`) and the seeds of two self-test generators. Renaming the magic would change every
container's bytes, and new seeds would change the self-test's inputs, so both stay. `script_sha256` in
`data/lossless/roundtrip_v2.json` is the hash of the file before this clean-up. On 2026-10-31 this copy and the file the
runs used gave the same 31 self-test checks and the same container sizes on the public file in § 7.
