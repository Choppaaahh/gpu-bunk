# vLLM patches

Five patches against vLLM v0.30.0 (tag `v0.30.0`, commit `ced6857`). They let several independent models share one
KV-cache block pool inside one process.

| Patch | What it does |
|---|---|
| 0001 | `SharedKVPool`: one block pool, one free queue, one prefix-cache map; each engine gets a view with block ownership and its own prefix-cache namespace. |
| 0002 | Three small seams so an engine built inside the pool's scope draws from the shared pool. |
| 0003 | A meta-scheduler over the per-model schedulers: lend free blocks, reclaim them with per-model floors. |
| 0004 | Per-thread forward context (opt-in) plus a lock and in-flight rule for the meta-scheduler. |
| 0005 | Hold reclaimed blocks for the woken model until its prefill completes. Without it, models trade the same blocks back and forth. |

Apply:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout v0.30.0
git am /path/to/gpu-bunk/patches/vllm/000*.patch
```

These patches are research code. They were tested with sync scheduling only. The node controller in `scripts/` does not
need them: it runs on stock vLLM 0.30.0.
