#!/usr/bin/env python3
"""beta_bench.py: item 1 (beta), the decode-step cost of a tail model whose KV sits partly on the NEIGHBOUR GPU.

WHY: node-scope KV pool (C) saves -144 / -104 / -8 GPUs at beta 0 / 1 / 3 (int4 STATIC, S10), and beta
was never measured. Stock vLLM has no cross-model KV borrowing (DCP only shards ONE model's KV inside its own TP group:
vllm/config/model.py:1445-1474 @ v0.30.0), so beta is measured here on a synthetic decode step at Qwen3-8B's real shapes
(config.json @ b968826d: 36 layers, hidden 4096, inter 12288, 32 q heads, 8 kv heads, head_dim 128, vocab 151936).

Mechanisms (prereg (private report) section 2):
  LOCAL  every request's KV on the model's GPU (dev0). f = 0, the unchanged arm, replayed in every pass.
  PUSH   the query goes to the neighbour (dev1), attention runs there over its KV chunk, (out, lse) come back and are
         merged by log-sum-exp (DistAttention, arXiv 2401.02669 section 4; vLLM DCP's combine is the same merge).
  PULL   the neighbour's chunk of layer i+1 is copied to dev0 while layer i computes (double buffer), then attended locally.
  PULLS  PULL serialized (copy, wait, attend): the conservative reading of PULL.
A fraction f of every request's tokens is remote: L_rem = round(f * L), L_loc = L - L_rem.

Subcommands:
  --self-test                     hand goldens on CPU (fp32): exact merge, byte arithmetic, Qwen3-8B param count, splits
  --dry-run --out F               the whole --run path on CPU with the 'tiny' geometry (dev0 = dev1 = cpu); writes F
  --probe --out F [--cpu]         NVLink / host->GPU / HBM bandwidth probes + the row's (a) weight-copy seconds; writes F
  --run --plan P --lender TAG --out F [--passes N]   the beta grid on cuda:0 + cuda:1; APPENDS one row per (pass, cell)

Output contract: success = exit 0 and the last stdout line starts 'BENCH-DONE' (or 'SELF-TEST PASS' / 'PROBE-DONE');
an instrument check that fails (merge/race error over tolerance, an OOM, a missing GPU) = exit 1 and a 'BENCH-FAIL <why>'
line, rows already written stay in --out (nothing is deleted); bad flags or an unreadable --plan = exit 2. --run and
--dry-run APPEND rows to --out (never truncate); --probe OVERWRITES its one JSON object at --out (QA r1).

Top pitfalls (observed while building it, 2026-10-14):
1. **A strided KV view can make torch.matmul copy the whole cache every step** (a permuted token-major layout is not a
   valid cuBLAS batch). Every KV tensor here is contiguous [B, nkv, Lc, hd]; the remote slab is its own tensor.
2. **A cross-device .to() runs on the SOURCE device's current stream and syncs both current streams** (ATen Copy.cu). The
   copy back from dev1 is issued while dev1's current stream is s1 (after the remote attention), never outside it.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

GEOM = {
    # Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 config.json (read 2026-10-14, HF)
    "qwen3_8b": dict(layers=36, hidden=4096, inter=12288, nq=32, nkv=8, hd=128, vocab=151936, dtype_bytes=2),
    # CPU dry run / self-test only
    "tiny": dict(layers=3, hidden=64, inter=96, nq=4, nkv=2, hd=16, vocab=97, dtype_bytes=4),
}
QWEN3_8B_PARAMS = 8190735360          # HF API safetensors.parameters BF16 for Qwen/Qwen3-8B (read 2026-10-14)
MODES = ("LOCAL", "PUSH", "PULL", "PULLS")
TOL_BF16, TOL_FP32 = 5e-2, 1e-4        # relative L2 on the final hidden state vs LOCAL (prereg I-MERGE / I-RACE)


# ----------------------------------------------------------------------------------------------- pure arithmetic
def kv_bytes_per_token(g):
    """K and V, every layer: 2 * nkv * hd * dtype_bytes * layers (Qwen3-8B bf16: 147,456 B)."""
    return 2 * g["nkv"] * g["hd"] * g["dtype_bytes"] * g["layers"]


def weight_params(g):
    """Linear weights + embed + lm_head + norms (2 x hidden + q_norm/k_norm hd each per layer, final norm)."""
    lin = g["hidden"] * (g["nq"] + 2 * g["nkv"]) * g["hd"] + g["nq"] * g["hd"] * g["hidden"] + 3 * g["hidden"] * g["inter"]
    norms = 2 * g["hidden"] + 2 * g["hd"]
    return g["layers"] * (lin + norms) + 2 * g["vocab"] * g["hidden"] + g["hidden"]


def split_len(L, f):
    """(L_loc, L_rem); f in [0, 1). L_rem = round(f * L)."""
    if not (0.0 <= f < 1.0) or L <= 0:
        raise ValueError(f"need 0 <= f < 1 and L > 0, got f={f} L={L}")
    rem = int(round(f * L))
    return L - rem, rem


def rel_l2(a, b):
    """||a - b|| / ||b|| on python floats or tensors (tensors via .float())."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            a, b = a.float().cpu(), b.float().cpu()
            return float((a - b).norm() / max(float(b.norm()), 1e-30))
    except ImportError:
        pass
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return num / max(math.sqrt(sum(y * y for y in b)), 1e-30)


def tol_failures(row, tol):
    """Instrument checks I-MERGE / I-RACE: the keys whose relative error is over tol. NaN is a failure (not <= tol)."""
    return [k for k in ("merge_relerr", "race_relerr") if row.get(k) is not None and not row[k] <= tol]


def pct(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ----------------------------------------------------------------------------------------------- torch pieces
def attn_partial(q, K, V, scale):
    """q [B, nkv, g, hd]; K, V [B, nkv, Lc, hd] contiguous. Returns (o fp32 [B, nkv, g, hd], lse fp32 [B, nkv, g])."""
    import torch
    s = torch.matmul(q, K.transpose(-1, -2)).float() * scale
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    den = p.sum(-1, keepdim=True)
    o = torch.matmul(p.to(V.dtype), V).float() / den
    return o, (m + torch.log(den)).squeeze(-1)


def lse_merge(o1, l1, o2, l2):
    """Exact softmax merge of two partial attentions over disjoint token sets."""
    import torch
    m = torch.maximum(l1, l2)
    w1, w2 = torch.exp(l1 - m), torch.exp(l2 - m)
    tot = w1 + w2
    return (o1 * w1.unsqueeze(-1) + o2 * w2.unsqueeze(-1)) / tot.unsqueeze(-1), m + torch.log(tot)


class Dev:
    """dev0 (the model's GPU) and dev1 (the neighbour). On CPU both are 'cpu' and streams are no-ops."""

    def __init__(self, cpu):
        import torch
        self.torch = torch
        self.cpu = cpu
        if cpu:
            self.d0 = self.d1 = torch.device("cpu")
            self.dtype = torch.float32
        else:
            if torch.cuda.device_count() < 2:
                raise RuntimeError(f"need 2 GPUs, see {torch.cuda.device_count()}")
            self.d0, self.d1 = torch.device("cuda:0"), torch.device("cuda:1")
            self.dtype = torch.bfloat16
            self.main0 = torch.cuda.current_stream(self.d0)
            self.s1 = torch.cuda.Stream(device=self.d1)
            self.sc = torch.cuda.Stream(device=self.d0)

    def stream(self, which):
        if self.cpu:
            return contextlib.nullcontext()
        return self.torch.cuda.stream({"s1": self.s1, "sc": self.sc}[which])

    def event(self, which=None):
        if self.cpu:
            return None
        e = self.torch.cuda.Event()
        e.record({"s1": self.s1, "sc": self.sc, "main0": self.main0}[which])
        return e

    def wait(self, ev):
        if ev is not None:
            self.main0.wait_event(ev)

    def sync(self):
        if not self.cpu:
            self.torch.cuda.synchronize(self.d0)
            self.torch.cuda.synchronize(self.d1)


class Model:
    """Random weights at the geometry's shapes, seeded; residual + RMS norm keeps the numerics finite."""

    def __init__(self, g, dev, seed=2746):
        import torch
        self.g, self.dev, t = g, dev, torch
        gen = t.Generator(device=dev.d0).manual_seed(seed)
        mk = lambda *s: t.empty(*s, device=dev.d0, dtype=dev.dtype).normal_(0.0, 0.02, generator=gen)
        H, nq, nkv, hd = g["hidden"], g["nq"], g["nkv"], g["hd"]
        self.layers = [dict(qkv=mk(H, (nq + 2 * nkv) * hd), o=mk(nq * hd, H), gu=mk(H, 2 * g["inter"]), d=mk(g["inter"], H))
                       for _ in range(g["layers"])]
        self.lm = mk(H, g["vocab"])
        self.scale = 1.0 / math.sqrt(hd)

    def _norm(self, x):
        xf = x.float()
        return (xf * self.dev.torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def step(self, x, mode, kv):
        """One decode step for B requests. kv: per-layer dict with loc K/V (dev0), rem K/V (dev1), stage buffers (dev0)."""
        t, g, dv = self.dev.torch, self.g, self.dev
        B = x.shape[0]
        nq, nkv, hd = g["nq"], g["nkv"], g["hd"]
        grp = nq // nkv
        nl = len(self.layers)
        evs = [None, None]

        def issue_copy(i):
            if dv.cpu:
                kv[i]["sK"], kv[i]["sV"] = kv[i]["rK"], kv[i]["rV"]
                return
            dv.sc.wait_stream(dv.main0)              # the buffer's previous reader (layer i-2) has finished
            b = kv["bufs"][i % 2]
            with dv.stream("sc"):
                b[0].copy_(kv[i]["rK"], non_blocking=True)
                b[1].copy_(kv[i]["rV"], non_blocking=True)
            evs[i % 2] = dv.event("sc")
            kv[i]["sK"], kv[i]["sV"] = b[0], b[1]

        if mode == "PULL":
            issue_copy(0)
        for i, ly in enumerate(self.layers):
            h = x
            q = (h @ ly["qkv"])[:, : nq * hd].reshape(B, nkv, grp, hd)
            if mode == "PULL" and i + 1 < nl:
                issue_copy(i + 1)
            if mode == "LOCAL":
                o, _ = attn_partial(q, kv[i]["lK"], kv[i]["lV"], self.scale)
            elif mode == "PUSH":
                with dv.stream("s1"):
                    q1 = q.to(dv.d1, non_blocking=True)
                    o1, l1 = attn_partial(q1, kv[i]["rK"], kv[i]["rV"], self.scale)
                o0, l0 = attn_partial(q, kv[i]["lK"], kv[i]["lV"], self.scale)
                with dv.stream("s1"):                   # dev1's current stream = s1: the copy follows the remote attention
                    o1b = o1.to(dv.d0, non_blocking=True)
                    l1b = l1.to(dv.d0, non_blocking=True)
                dv.wait(dv.event("s1"))                 # QA r1: explicit, not only ATen's implicit cross-device sync
                o, _ = lse_merge(o0, l0, o1b, l1b)
            elif mode in ("PULL", "PULLS"):
                if mode == "PULLS":                      # serialized: the copy lands before ANY attention of layer i
                    issue_copy(i)
                    dv.wait(evs[i % 2])
                o0, l0 = attn_partial(q, kv[i]["lK"], kv[i]["lV"], self.scale)
                dv.wait(evs[i % 2])
                o1, l1 = attn_partial(q, kv[i]["sK"], kv[i]["sV"], self.scale)
                o, _ = lse_merge(o0, l0, o1, l1)
            else:
                raise ValueError(f"unknown mode {mode}")
            a = o.to(x.dtype).reshape(B, nq * hd) @ ly["o"]
            gu = h @ ly["gu"]
            m = (t.nn.functional.silu(gu[:, : g["inter"]]) * gu[:, g["inter"]:]) @ ly["d"]
            x = self._norm(h + a + m)
        return x, x @ self.lm


def build_kv(model, B, L, f, seed=7):
    """Reference KV on dev0 (deterministic) split into local (dev0) and remote (dev1) contiguous tensors + PULL buffers.
    LOCAL (f = 0) uses the full reference; every split of the same (B, L) holds the same values, so outputs must match."""
    t, g, dv = model.dev.torch, model.g, model.dev
    L_loc, L_rem = split_len(L, f)
    gen = t.Generator(device=dv.d0).manual_seed(seed)
    kv = {}
    for i in range(g["layers"]):
        K = t.empty(B, g["nkv"], L, g["hd"], device=dv.d0, dtype=dv.dtype).normal_(0.0, 1.0, generator=gen)
        V = t.empty(B, g["nkv"], L, g["hd"], device=dv.d0, dtype=dv.dtype).normal_(0.0, 1.0, generator=gen)
        e = {"lK": K[:, :, :L_loc].contiguous(), "lV": V[:, :, :L_loc].contiguous()}
        if L_rem:
            e["rK"] = K[:, :, L_loc:].contiguous().to(dv.d1)
            e["rV"] = V[:, :, L_loc:].contiguous().to(dv.d1)
        kv[i] = e
        del K, V
    if L_rem and not dv.cpu:
        kv["bufs"] = [(t.empty(B, g["nkv"], L_rem, g["hd"], device=dv.d0, dtype=dv.dtype),
                       t.empty(B, g["nkv"], L_rem, g["hd"], device=dv.d0, dtype=dv.dtype)) for _ in range(2)]
    dv.sync()
    return kv, L_loc, L_rem


def x_input(model, B, salt):
    t, dv = model.dev.torch, model.dev
    gen = t.Generator(device=dv.d0).manual_seed(1000 + salt)
    return t.empty(B, model.g["hidden"], device=dv.d0, dtype=dv.dtype).normal_(0.0, 1.0, generator=gen)


def time_cell(model, kv, mode, B, warm, steps):
    """Returns gpu_ms list (events on dev0's main stream; perf_counter on CPU) and host_ms (one enqueue after a sync)."""
    t, dv = model.dev.torch, model.dev
    x = x_input(model, B, 0)
    for _ in range(warm):
        model.step(x, mode, kv)
    dv.sync()
    h0 = time.perf_counter()
    model.step(x, mode, kv)
    host_ms = (time.perf_counter() - h0) * 1e3
    dv.sync()
    out = []
    if dv.cpu:
        for _ in range(steps):
            a = time.perf_counter()
            model.step(x, mode, kv)
            out.append((time.perf_counter() - a) * 1e3)
        return out, host_ms
    ev = [(t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)) for _ in range(steps)]
    for s in range(steps):
        ev[s][0].record(dv.main0)
        model.step(x, mode, kv)
        ev[s][1].record(dv.main0)
    dv.sync()
    return [a.elapsed_time(b) for a, b in ev], host_ms


def check_vs_local(model, kv, kv_local, mode, B, salt):
    """Relative L2 between this mode's final hidden and LOCAL's on the same input (a fresh salt per call)."""
    x = x_input(model, B, salt)
    h_m, _ = model.step(x, mode, kv)
    h_l, _ = model.step(x, "LOCAL", kv_local)
    model.dev.sync()
    return rel_l2(h_m, h_l)


# ----------------------------------------------------------------------------------------------- the grid
def run_grid(geom, cells, fs, modes, passes, lender, out, cpu, warm, steps, mark=None):
    """Pass-major order: in each pass every (cell) runs LOCAL first, then every (mode, f). Rows appended to `out`."""
    import torch
    dev = Dev(cpu)
    model = Model(geom, dev)
    tol = TOL_FP32 if cpu else TOL_BF16
    kvb = kv_bytes_per_token(geom)
    n = 0
    for p in range(passes):
        for cell in cells:
            B, L = cell["B"], cell["L"]
            kv_local, _, _ = build_kv(model, B, L, 0.0)
            combos = [("LOCAL", 0.0)] + [(m, f) for m in modes if m != "LOCAL" for f in fs if f > 0]
            for mode, f in combos:
                t0 = time.time()
                row = dict(source="beta_bench", trigger="gpu_driver", trust_tier="autonomous-unsupervised",
                           pass_=p, lender=lender, cell=cell["name"], B=B, L=L, mode=mode, f=f,
                           kv_bytes_step=B * L * kvb, geom=geom.get("name", "custom"), ok=False)
                try:
                    kv, L_loc, L_rem = (kv_local, L, 0) if mode == "LOCAL" else build_kv(model, B, L, f)
                    row.update(L_loc=L_loc, L_rem=L_rem, kv_remote_bytes_step=B * L_rem * kvb)
                    if mode != "LOCAL":
                        row["merge_relerr"] = check_vs_local(model, kv, kv_local, mode, B, salt=11 + p)
                    ms, host_ms = time_cell(model, kv, mode, B, warm, steps)
                    if mode != "LOCAL":
                        row["race_relerr"] = check_vs_local(model, kv, kv_local, mode, B, salt=101 + p)
                    row.update(gpu_ms_p50=pct(ms, 50), gpu_ms_p10=pct(ms, 10), gpu_ms_p90=pct(ms, 90), n_steps=len(ms),
                               host_ms=host_ms, t0=t0, t1=time.time())
                    bad = tol_failures(row, tol)
                    row["ok"] = not bad
                    if bad:
                        row["fail"] = f"{','.join(bad)} > {tol}"
                    if kv is not kv_local:
                        del kv
                except (RuntimeError, ValueError) as e:     # torch OOM is a RuntimeError subclass
                    row.update(fail=f"{type(e).__name__}: {str(e)[:200]}", t0=t0, t1=time.time())
                with open(out, "a") as fh:
                    fh.write(json.dumps({("pass" if k == "pass_" else k): v for k, v in row.items()}) + "\n")
                n += 1
                if mark:
                    mark(row)
                if not row["ok"]:
                    print(f"BENCH-FAIL {row['cell']} {mode} f={f}: {row.get('fail')}", flush=True)
                    return False, n
                if not cpu:
                    torch.cuda.empty_cache()
            del kv_local
    return True, n


# ----------------------------------------------------------------------------------------------- probes
def probe(out, cpu, wake_bytes):
    """Bandwidths in GB/s (1e9): p2p each way, H2D pinned per GPU alone and both at once, HBM D2D per GPU; plus the
    row's (a) transfer component: `wake_bytes` copied neighbour->dev0 vs pinned host->dev0, n=5 each (seconds)."""
    import threading
    import torch
    res = {"source": "beta_bench", "trigger": "gpu_driver", "trust_tier": "autonomous-unsupervised", "cpu": cpu}
    if cpu:
        dev = [torch.device("cpu")] * 2
        nb, reps, wake_bytes = 1 << 20, 3, 1 << 20
    else:
        dev = [torch.device("cuda:0"), torch.device("cuda:1")]
        nb, reps = 1 << 30, 10
        res["peer_access_01"] = torch.cuda.can_device_access_peer(0, 1)
        res["peer_access_10"] = torch.cuda.can_device_access_peer(1, 0)

    def sync():
        if not cpu:
            torch.cuda.synchronize(dev[0])
            torch.cuda.synchronize(dev[1])

    def timed(fn, n):
        ts = []
        for _ in range(n):
            sync()
            a = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - a)
        return ts

    a0 = torch.empty(nb, dtype=torch.uint8, device=dev[0])
    a1 = torch.empty(nb, dtype=torch.uint8, device=dev[1])
    for name, (dst, src) in (("p2p_10_gbs", (a0, a1)), ("p2p_01_gbs", (a1, a0))):
        ts = timed(lambda: dst.copy_(src, non_blocking=True), reps)
        res[name] = nb / pct(ts, 50) / 1e9
    for i in (0, 1):
        b = torch.empty(2 * nb, dtype=torch.uint8, device=dev[i])
        ts = timed(lambda: b[nb:].copy_(b[:nb], non_blocking=True), reps)
        res[f"hbm_d2d_gbs_{i}"] = 2 * nb / pct(ts, 50) / 1e9       # read + write
        del b
    pin = torch.empty(nb, dtype=torch.uint8, pin_memory=not cpu)
    for i in (0, 1):
        d = a0 if i == 0 else a1
        ts = timed(lambda: d.copy_(pin, non_blocking=True), reps)
        res[f"h2d_gbs_{i}"] = nb / pct(ts, 50) / 1e9
    pin2 = torch.empty(nb, dtype=torch.uint8, pin_memory=not cpu)

    def both():
        def one(dst, src):
            dst.copy_(src, non_blocking=True)
            if not cpu:
                torch.cuda.synchronize(dst.device)
        th = [threading.Thread(target=one, args=(a0, pin)), threading.Thread(target=one, args=(a1, pin2))]
        [x.start() for x in th]
        [x.join() for x in th]
    ts = timed(both, reps)
    res["h2d_both_gbs_total"] = 2 * nb / pct(ts, 50) / 1e9
    res["h2d_both_ratio"] = res["h2d_both_gbs_total"] / max(1e-9, res["h2d_gbs_0"] + res["h2d_gbs_1"])
    del a0, a1, pin, pin2
    # row (a): a whole model's bytes from the neighbour vs from pinned host RAM (transfer only; stock vLLM has no
    # neighbour-GPU wake path)
    w0 = torch.empty(wake_bytes, dtype=torch.uint8, device=dev[0])
    w1 = torch.empty(wake_bytes, dtype=torch.uint8, device=dev[1])
    wp = torch.empty(wake_bytes, dtype=torch.uint8, pin_memory=not cpu)
    res["wake_bytes"] = wake_bytes
    res["wake_nvlink_s"] = timed(lambda: w0.copy_(w1, non_blocking=True), 5)
    res["wake_host_s"] = timed(lambda: w0.copy_(wp, non_blocking=True), 5)
    del w0, w1, wp
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# ----------------------------------------------------------------------------------------------- self-test
def self_test():
    import torch
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'} {name}{'' if cond else ' :: ' + str(detail)}")
        if not cond:
            fails.append(name)

    # G1 (hand): q=[1,0], K=[[1,0],[0,1]], V=[[1,2],[3,4]], scale 1 -> softmax(1,0) = (e/(e+1), 1/(e+1))
    #   = (0.731059, 0.268941); out = 0.731059*[1,2] + 0.268941*[3,4] = [1.537883, 2.537883]; lse = log(e+1) = 1.313262
    q = torch.tensor([[[[1.0, 0.0]]]])
    K = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    V = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    o, l = attn_partial(q, K, V, 1.0)
    ck("G1 full attention = hand [1.537883, 2.537883], lse 1.313262",
       abs(o[0, 0, 0, 0].item() - 1.537883) < 1e-5 and abs(o[0, 0, 0, 1].item() - 2.537883) < 1e-5
       and abs(l.item() - 1.313262) < 1e-5, (o, l))
    # G2 (hand): split token 0 | token 1: chunk A lse 1, out [1,2]; chunk B lse 0, out [3,4]; merge = G1
    oa, la = attn_partial(q, K[:, :, :1], V[:, :, :1], 1.0)
    ob, lb = attn_partial(q, K[:, :, 1:], V[:, :, 1:], 1.0)
    om, lm = lse_merge(oa, la, ob, lb)
    ck("G2 chunk lse hand (1.0, 0.0) and merge = full",
       abs(la.item() - 1.0) < 1e-6 and abs(lb.item() - 0.0) < 1e-6 and rel_l2(om, o) < 1e-6 and abs(lm.item() - l.item()) < 1e-6,
       (la, lb, om))
    # G3 (hand): Qwen3-8B bf16 KV per token = 2 x 8 x 128 x 2 B x 36 = 147,456 B; S cell 8 x 4096 tokens = 4,831,838,208 B
    g8 = GEOM["qwen3_8b"]
    ck("G3 kv bytes/token 147456 and S cell 4831838208", kv_bytes_per_token(g8) == 147456
       and 8 * 4096 * kv_bytes_per_token(g8) == 4831838208, kv_bytes_per_token(g8))
    # G4 (real data): the geometry reproduces HF's parameter count EXACTLY (8,190,735,360; norms 308,224 included)
    ck("G4 Qwen3-8B params = HF 8190735360", weight_params(g8) == QWEN3_8B_PARAMS, weight_params(g8))
    # G5 (hand): splits L=4096: f 0.125 -> (3584, 512); 0.5 -> (2048, 2048); 0.875 -> (512, 3584)
    ck("G5 splits", [split_len(4096, f) for f in (0.125, 0.5, 0.875)] == [(3584, 512), (2048, 2048), (512, 3584)])
    # G6 ADVERSARIAL: f = 1 (no local tokens) and f < 0 must refuse, never return an empty local side
    bad = []
    for f in (1.0, -0.1):
        try:
            split_len(4096, f)
            bad.append(f)
        except ValueError:
            pass
    ck("G6 adversarial: f=1 and f<0 refused", not bad, bad)
    # G7 ADVERSARIAL: a merge with a wrong-sign lse (a stale or swapped partial) must NOT pass the I-MERGE tolerance
    om_bad, _ = lse_merge(oa, -la, ob, lb)
    ck("G7 adversarial: swapped-lse merge fails the fp32 tolerance", rel_l2(om_bad, o) > TOL_FP32, rel_l2(om_bad, o))
    # G8: the tiny model on CPU, every mode at f 0.5 equals LOCAL within the fp32 tolerance (the I-MERGE path itself)
    dev = Dev(True)
    m = Model(dict(GEOM["tiny"], name="tiny"), dev)
    kvl, _, _ = build_kv(m, 2, 32, 0.0)
    errs = {}
    for mode in ("PUSH", "PULL", "PULLS"):
        kvs, _, _ = build_kv(m, 2, 32, 0.5)
        errs[mode] = check_vs_local(m, kvs, kvl, mode, 2, salt=3)
    ck("G8 tiny CPU: PUSH/PULL/PULLS == LOCAL (fp32)", all(e <= TOL_FP32 for e in errs.values()), errs)
    # G9 pct (hand): [1,2,3,4] p50 = 2.5, p10 = 1.3
    ck("G9 pct hand", pct([4, 1, 3, 2], 50) == 2.5 and abs(pct([1, 2, 3, 4], 10) - 1.3) < 1e-12)
    # G10 ADVERSARIAL: the run's instrument gate fails a large error AND a NaN error (a NaN compares False to anything)
    ck("G10 tol gate: 0.1 and NaN fail, 1e-6 passes, absent keys pass",
       tol_failures({"merge_relerr": 0.1}, TOL_FP32) == ["merge_relerr"]
       and tol_failures({"race_relerr": float("nan")}, TOL_FP32) == ["race_relerr"]
       and tol_failures({"merge_relerr": 1e-6, "race_relerr": 1e-6}, TOL_FP32) == [] and tol_failures({}, TOL_FP32) == [])
    print("SELF-TEST " + ("PASS" if not fails else f"FAIL {fails}"))
    return 0 if not fails else 1


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--plan")
    ap.add_argument("--lender", choices=("IDLE", "BUSY"))
    ap.add_argument("--passes", type=int)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--progress", help="file rewritten after every row (the driver's stall watch)")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.out:
        print("USAGE --out is required", file=sys.stderr)
        return 2
    if a.probe:
        try:
            res = probe(a.out, a.cpu, int(15.26 * 2 ** 30) if not a.cpu else 1 << 20)
        except (RuntimeError, ValueError) as e:
            print(f"BENCH-FAIL probe {type(e).__name__}: {e}")
            return 1
        print("PROBE-DONE " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()
                                          if not isinstance(v, list)}))
        return 0
    if a.dry_run:
        geom = dict(GEOM["tiny"], name="tiny")
        cells = [dict(name="Q", B=1, L=64), dict(name="S", B=4, L=64)]
        ok, n = run_grid(geom, cells, [0.125, 0.5, 0.875], MODES, a.passes or 2, a.lender or "IDLE", a.out, True, 1, 3)
        print(f"BENCH-DONE dry-run rows={n} ok={ok}")
        return 0 if ok else 1
    if not (a.plan and a.lender):
        print("USAGE --run needs --plan and --lender", file=sys.stderr)
        return 2
    try:                                                         # QA r1: an unreadable plan is a usage error, exit 2
        with open(a.plan) as fh:
            plan = json.load(fh)["beta"]
        geom = dict(GEOM[plan["geom"]], name=plan["geom"])
        for k in ("cells", "f", "modes", "passes", "warm_steps", "timed_steps"):
            plan[k]
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"USAGE --plan unreadable: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    def prog(r):
        if a.progress:
            with open(a.progress, "w") as fh:
                fh.write(json.dumps({"n": r.get("cell"), "mode": r["mode"], "f": r["f"], "t": time.time()}))
    try:
        ok, n = run_grid(geom, plan["cells"], plan["f"], plan["modes"], a.passes or plan["passes"], a.lender, a.out,
                         a.cpu, plan["warm_steps"], plan["timed_steps"], mark=prog)
    except RuntimeError as e:
        print(f"BENCH-FAIL setup {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"BENCH-DONE lender={a.lender} rows={n} ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
