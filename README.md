# gpu-bunk

Many quiet AI models, fewer idle GPUs.

## What it is

Most models in an inference provider's catalogue are quiet. They get few requests, yet each one usually holds GPUs of
its own. Those GPUs sit idle most of the time.

gpu-bunk tests one way to fix that. Quiet models share GPUs. A model with no traffic goes to sleep: its weights move to
host RAM, or to spare memory on the GPU next door. When a request arrives, it wakes up. Idle GPU time can then be sold
as batch work that steps aside when live traffic needs the GPU.

This repo holds:

- `patches/vllm/`: patches to vLLM 0.30.0 that let several models share one KV-cache pool in one process.
- `scripts/`: a node controller that parks and wakes models across GPUs on stock vLLM, its engine extension, and a
  fleet simulator that replays a public production trace.
- `bench/`: the GPU benchmarks behind the numbers below.

It is research code, as run. It is not production-ready.

## What we measured

On rented GPUs. Each number comes from one machine unless noted.

| What | Result | Hardware |
|---|---|---|
| Wake a sleeping model from host RAM to its first token (vLLM sleep mode) | 0.13 s (1.7B), 0.43 s (8B), 0.79 s (30B MoE, int4); 5 runs each. Restarting the 8B instead took 55 s, even with warm caches. | 1× RTX 5090, PCIe 4 |
| Copy a 16.4 GB model from the next GPU over NVLink | 0.060 s, against 0.63 s from pinned host RAM (about 10× faster); 5 runs each | 2× A100-SXM4-80GB, NVLink |
| A similar copy (15.8 GiB) while the source GPU puts its own model to sleep | None of 100 reads waited; the slowest took 0.065 s. Serving reads slows that GPU's own shutdown by about 0.4 ms per GB. | 2× A100-SXM4-80GB, NVLink |
| Wake through our controller from the next GPU (8B and 14B models) | 0.24–0.33 s to restore; 3 runs per model. Outputs matched the model before sleep in 8 of 8 cycles. One full wake through the router took 2.25 s, most of it waiting 1.79 s for the GPU's batch job to stop. | 2× A100-SXM4-80GB (Modal) |
| One decode step on an H100, Qwen3-8B | 6.9 ms (batch 1 × 4k tokens), 11.5 ms (batch 8 × 4k), 12.2 ms (batch 2 × 16k); 3 passes each | 1× H100 SXM (Modal) |

In simulation, on one week of a public production trace (the
[Chutes trace](https://github.com/HarvardMadSys/chutes_workload), CC BY 4.0): 3.69 million requests to 375 models.
The fleet is fixed at 128 H100s. It must carry one 1-trillion-parameter MoE plus as many of the other 374 models as fit.
The target is a 10-second p95 time to first token. Ranges cover 21 replays of the held-out week.

| | Provider default (dedicated GPUs) | gpu-bunk (stock vLLM) |
|---|---|---|
| Models hosted (the MoE included) | 97 of 375 | 137 of 375 |
| Requests served within the target | 75.0–77.0 % | 81.6–83.6 % |
| GPU-minutes idle | 60.0–62.6 % | 57.8–59.9 % |

- Sharing fits 40 more models on the same GPUs. It barely cuts idle time, because idle follows demand: 235 of the 351
  small models got no requests in the held-out week, and about 70 % of the sellable idle time sits on GPUs whose models
  got none.
- About 4.5 of the 6–7 extra points of requests come from a single busy model.
- Idle is counted in one-minute buckets, so the real idle share is higher than shown.
- Selling idle GPU-minutes as interruptible batch work recovers 96–97 % of the default's idle time (12,500–13,000 GPU-hours
  a week). No hosted model misses the 10-second target. This assumes someone buys all of that batch time.

## Rough endgame

Open tooling a provider can run on its own fleet:

1. **Wake tiers.** Hot models keep dedicated GPUs. Warm models sleep in a neighbour GPU's spare memory (about 0.3 s to
   restore). Cool models sleep in host RAM (0.1–0.8 s to first token). Cold models stay on disk.
2. **A node controller** that parks idle models, wakes them on demand, and fills parked GPUs with batch work.
3. **A placement simulator** that reads a provider's own traffic. It says which models to share, where to put them, and
   what that saves.

The next milestone is one end-to-end run: a full 8-GPU NVLink node replaying a real day of traffic, with each model's
latency reported by tier.

## What's unresolved

- **A 2-second target does not hold yet.** It passed on our original card layout. When the same models were re-packed
  onto cards differently, it held on only 16–19 of 41 layouts (two sets of layouts). Load-aware placement held 38–39 of
  41. The main cause is one model whose traffic bursts only in the test week. A memory floor set with hindsight fixes all
  41 layouts, so the open problem is sizing memory online, as traffic arrives. The 10-second results have not been
  re-checked this way yet.
- **H100 against token-level switching: it misses, on the strong side.** The comparison turned on one input, a 14B
  model's batch-1 decode step. We re-measured it on a rented H100 SXM. Twelve passes read 11.61–11.63 ms (median
  11.62 ms, 0.2 % spread), and 20 more batch-1 requests agreed. The earlier 11.6–16.0 ms swing did not come back; we do
  not know what caused it. At 11.62 ms both of our step models agree: our re-implementation keeps beating the static
  baseline down to a 0.15× token deadline, two steps tighter than the 0.2× the paper reports. So the published curve
  matches A100-class decode speeds, not H100 ones. Limits: the paper's 13B model is mapped from 8B and 14B measurements
  rather than run, this is one H100 host, and the answer flips if the true step is above about 12.5–13 ms.
- **Pooling KV memory across GPUs costs a lot.** Inside one vLLM engine, splitting a model's KV cache across two NVLink
  GPUs doubles its KV capacity but makes each decode step 24–40 % slower (two machines). Reading KV from a separate
  engine on the next GPU did not pay in our bench.
- **Isolation is untested.** We have not measured noisy neighbours or security between tenants on a shared GPU.
- **Small scale.** Every GPU number comes from one or two GPUs on one or two machines. Everything ran on one engine
  version (vLLM 0.30.0). Some timing runs used dummy weights.

How to reproduce each number: [REPRODUCE.md](REPRODUCE.md). Data sources: [DATA.md](DATA.md).

---

*AI-assisted.*
