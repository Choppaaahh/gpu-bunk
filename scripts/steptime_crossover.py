#!/usr/bin/env python3
"""steptime_crossover.py — a decode step-time map by model size and hardware class, then the Aegaeon-vs-static
crossover re-read per hardware class under the paper's 6-14 B mix, ShareGPT length variance and a KV-swap cost.

Prereg (frozen before this file existed): (private report). On any divergence the
prereg wins. FORK BY IMPORT: scripts/aegaeon_repro.py is imported (`R`), never edited; run_aeg / run_mux_gpu are re-written
here with per-model step / prefill / switch / KV, and at L0 (MIX0, W0, no KV swap) they reproduce exactly (REPLAY gate, G8).

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n" (goldens G1-G10 of prereg §9); exit 1 + FAIL lines.
  --check-lengths [--lengths P]     read-only: verifies the frozen lengths file's sha256 and prints n / means as one JSON object; exit 0.
  --map --out DIR                   WRITES DIR/map.json (BENCH + SERVING cells, every error of prereg §3.3, the §7.2 trust gate, the §3.4
                                    table); prints a short JSON summary; exit 0.
  --b-run --out DIR --phase P [--seeds S] [--workers N]
                                    APPENDS rows to DIR/aeg_rows.jsonl (one per simulated AEG point) and DIR/mux_rows.jsonl (one per MUX
                                    GPU sim); P in replay | primary | ladder | ours | desc | all. Rows already present are skipped (resume).
  --b-one --spec JSON               read-only: one tiny AEG or MUX config (G7's real entry point); one JSON object on stdout; exit 0.
  --summarize DIR                   WRITES DIR/summary.json (verdicts of prereg §7-§8) and prints it; exit 0.
  Bad input (unknown arm / class / model type, lengths sha mismatch, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed while building):
  1. **Two different "8 B byte" constants live in **: the step fit (empirical, Qwen3-8B, 16,381,516,776 B) and the switch object
     ('s 16.39e9 B wake). MIX0 keeps both exactly as had them; mix models use one config-derived number for both.
  2. **MuxServe's 161 / 337 are not ShareGPT_V3 under vLLM's sampler rule** (that reads 250.94 / 231.46, n 82,780). W0 and WS are
     separate frozen levels; never quote one as the other.
  3. **Exact L0 equivalence needs arithmetic order** (trans x tsw, t_pre(L) x n, the step's own sum order): a sum of identical
     floats can differ from a product in the last bit and flip a tie. The fork keeps the product form whenever the terms are identical.
"""
import argparse, collections, gzip, hashlib, heapq, json, math, os, subprocess, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import aegaeon_repro as R   # noqa: E402  (imported unchanged)

PREREG = "data/runs/steptime_crossover_prereg_2026-10-20.md"
PREREG_COMMIT = "56ebf8de5"
RUN_DIR_REL = "data/steptime_crossover"
LENGTHS_REL = RUN_DIR_REL + "/inputs/sharegpt_v3_llama_lengths.json.gz"
LENGTHS_SHA = "e60d1f5a7e95bf17ebd8af77785ac2bd1c17d14daf20520717e3a055a97f5742"
B2957_ROWS_REL = "data/aegaeon_repro/b_rows.jsonl"
B2957_ROWS_SHA = "a108620933c4ca2aa43e08e86f016011b1745a49dd4ea795df6d0f23b77dc3f7"

# ------------------------------------------------------------------------------------------------ frozen inputs (prereg §0, §3)
W_REF = 16381516776.0                     # Qwen3-8B safetensors bytes (bench/multigpu/inputs/plan.json)
V_REF = R.BPT                             # 147,456 B / token
QDL_REF = 32 * 128 * 36                   # Qwen3-8B: query heads x head dim x layers
ANCH_A80 = (R.ANCH_Q, R.ANCH_S, R.ANCH_LC)                                          # run_h2
ANCH_A40 = (0.01356234626164512, 0.02929489196292938, 0.027288601512000674)          # run_h2 (A100-SXM4-40GB)
B_SPEC = {"A100-SXM4-80GB": 2039e9, "A100-SXM4-40GB": 1555e9, "A100-PCIe-80GB": 1935e9, "A40": 696e9, "RTX-A6000": 768e9,
          "RTX-6000-Ada": 960e9, "RTX-4090": 1008e9, "RTX-5090": 1792e9, "RTX-PRO-6000": 1792e9, "H800-SXM": 3350e9}
B_A, B_40, B_H = B_SPEC["A100-SXM4-80GB"], B_SPEC["A100-SXM4-40GB"], B_SPEC["H800-SXM"]
CLASSES = ("A", "HM", "HF")               # crossover classes; "A40" is map-only
W0_MEANS = (161, 337)
WS_MEANS = (250.9416, 231.4567)           # prereg P-7 (4 dp); WV rescales by W0 / WS


def fit3(Q, S, LC):
    """ exact 3-point solve of t = a + kb*b + kl*C, (b, C) = Q (1, 4096), S (8, 32768), LC (2, 32768), where C = B x L is the
    batch's SUMMED context (review: BENCH_CELLS below lists (B, per-sequence L); 8 x 4096 = 2 x 16384 = 32768, same cells)."""
    kb = (S - LC) / 6.0
    kl = (LC - Q - kb) / 28672.0
    a = Q - kb - 4096.0 * kl
    return a, kb, kl


def kappa_solve(a1, B1, a2, B2, W):
    """two measured intercepts a_i = F + kappa*W/B_i -> (kappa, F)."""
    d = W * (1.0 / B2 - 1.0 / B1)
    if not (B1 > 0 and B2 > 0) or d == 0.0:
        raise ValueError("kappa solve needs two distinct positive bandwidths")
    kappa = (a2 - a1) / d
    return kappa, a1 - kappa * W / B1


FIT_A80 = fit3(*ANCH_A80)
FIT_A40 = fit3(*ANCH_A40)
KAPPA, F_FIX = kappa_solve(FIT_A80[0], B_A, FIT_A40[0], B_40, W_REF)
I_HM_REF = R.A_FIT - KAPPA * W_REF * (1.0 / B_A - 1.0 / B_H)       # = F + kappa*W_ref/B_H


def dense_params(h, m, L, nq, nkv, d, V, tied=False):
    attn = h * (nq * d) + 2 * h * (nkv * d) + (nq * d) * h
    per = attn + 3 * h * m + 2 * h
    return L * per + V * h * (1 if tied else 2) + h


# config dims (prereg P-8; Qwen-7B MLP width = intermediate_size / 2 per the QWen MLP)
CONFIGS = {
    "yi6b": dict(h=4096, m=11008, L=32, nq=32, nkv=4, d=128, V=64000, name="Yi-6B"),
    "qwen7b": dict(h=4096, m=11008, L=32, nq=32, nkv=32, d=128, V=151936, name="Qwen-7B"),
    "internlm25_7b": dict(h=4096, m=14336, L=32, nq=32, nkv=8, d=128, V=92544, name="InternLM2.5-7B-chat"),
    "yi9b": dict(h=4096, m=11008, L=48, nq=32, nkv=4, d=128, V=64000, name="Yi-9B"),
    "llama13b": dict(h=5120, m=13824, L=40, nq=40, nkv=40, d=128, V=32000, name="LLaMA-13B"),
    "qwen25_7b": dict(h=3584, m=18944, L=28, nq=28, nkv=4, d=128, V=152064, name="Qwen2.5-7B-Instruct"),
    "llama31_8b": dict(h=4096, m=14336, L=32, nq=32, nkv=8, d=128, V=128256, name="Llama-3.1-8B-Instruct"),
    "llama7b": dict(h=4096, m=11008, L=32, nq=32, nkv=32, d=128, V=32000, name="huggyllama/llama-7b"),
}
MODELS = {"ref8b": dict(W=W_REF, Wsw=R.MODEL_GB * 1e9, v=V_REF, rho=1.0, P=8.0, name="Qwen3-8B (MIX0)")}
for _k, _c in CONFIGS.items():
    _p = dense_params(_c["h"], _c["m"], _c["L"], _c["nq"], _c["nkv"], _c["d"], _c["V"])
    MODELS[_k] = dict(W=2.0 * _p, Wsw=2.0 * _p, v=_c["L"] * 2 * _c["nkv"] * _c["d"] * 2, rho=(_c["nq"] * _c["d"] * _c["L"]) / QDL_REF,
                      P=_p / 1e9, params=_p, name=_c["name"])
MIXES = {"MIX0": ["ref8b"], "MIX3": ["qwen7b", "internlm25_7b", "llama13b"],
         "MIX5": ["yi6b", "qwen7b", "internlm25_7b", "yi9b", "llama13b"]}
LEVELS = {"L0": ("MIX0", "W0", False), "L1": ("MIX3", "W0", False), "L2v": ("MIX3", "WV", False), "L2": ("MIX3", "WS", False),
          "L3": ("MIX3", "WS", True), "L3-5": ("MIX5", "WS", True)}
RATE = {"PAPER": 32e9, "HOST": R.MODEL_GB * 1e9 / R.TSW["HOST"], "NVL": R.MODEL_GB * 1e9 / R.TSW["NVL"]}
AEG_POL = {"PAPER": ("PAPER", R.QMAX), "HOST": ("HOST", R.QMAX), "NVL": ("NVL", R.QMAX), "NVLQ": ("NVL", R.QMAX_NVLQ)}
MUX_BETA = {"MPS": R.BETA["MPS"][0], "STOCK": R.BETA["STOCK"][0]}


def round_overhead(models, ty, tsw_t):
    """Eq. 2-3's c: the summed switch time of every model transition in the round (cyclic), charged at the INCOMING model's T_sw."""
    K = len(models)
    if K <= 1:
        return 0.0
    cnt = collections.Counter(ty(models[i]) for i in range(K) if models[i] != models[i - 1])
    return 0.0 + sum(n * tsw_t[t] for t, n in cnt.items())      # trans * tsw when there is one type


def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------ step / prefill / switch (prereg §3.2, §4)
def model_of(mt):
    if mt not in MODELS:
        raise ValueError(f"unknown model type {mt!r}")
    return MODELS[mt]


def intercept(mt, cls):
    m = model_of(mt)
    if cls == "A":
        return R.A_FIT + KAPPA * (m["W"] - W_REF) / B_A
    if cls == "HM":
        return I_HM_REF + KAPPA * (m["W"] - W_REF) / B_H
    if cls == "A40":
        return FIT_A40[0] + KAPPA * (m["W"] - W_REF) / B_40
    raise ValueError(f"no intercept for class {cls!r}")


def step(mt, cls, b, ctx, allow_map=False):
    """decode step (s) of model type mt on class cls with b sequences and summed context ctx (prereg §3.2)."""
    if cls not in CLASSES and not (allow_map and cls == "A40"):
        raise ValueError(f"unknown hardware class {cls!r}")
    m = model_of(mt)
    if mt == "ref8b" and cls in ("A", "HF"):
        return R.step_time("A" if cls == "A" else "H", b, ctx)          # exactly
    if cls == "HF":
        return step(mt, "A", b, ctx) * R.H_RATIO
    return intercept(mt, cls) + R.KB_FIT * b + R.KL_FIT * ctx * m["rho"]


def tpre(mt, L):
    m = model_of(mt)
    if mt == "ref8b":
        return R.t_pre(L)
    return L * ((m["P"] / R.ACTIVE_REF_B) / R.CH)


def tsw(mt, link):
    m = model_of(mt)
    if link not in RATE:
        raise ValueError(f"unknown link {link!r}")
    if mt == "ref8b":
        return R.TSW[link]
    return m["Wsw"] / RATE[link]


# ------------------------------------------------------------------------------------------------ lengths + requests (prereg §4)
_LEN = {}


def load_lengths(path):
    if path in _LEN:
        return _LEN[path]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != LENGTHS_SHA:
        raise LookupError(f"lengths sha256 mismatch for {path}")
    d = json.loads(gzip.decompress(raw))
    _LEN[path] = (np.asarray(d["prompt"], np.int64), np.asarray(d["output"], np.int64), d)
    return _LEN[path]


def make_requests2(M, seed, workload, lengths, arr_end=R.ARR_END, stream_base=0):
    """ arrivals exactly; lengths from a separate stream [seed, 2959, 7, stream_base + i] in each model's arrival order."""
    reqs = R.make_requests(M, seed, arr_end, stream_base=stream_base)
    if workload == "W0":
        return reqs
    if workload not in ("WS", "WV"):
        raise ValueError(f"unknown workload {workload!r}")
    P, O = lengths[0], lengths[1]
    rngs = {}
    for r in reqs:
        g = rngs.get(r.model)
        if g is None:
            g = rngs[r.model] = np.random.default_rng([seed, 2959, 7, stream_base + r.model])
        j = int(g.integers(0, len(P)))
        p, o = int(P[j]), int(O[j])
        if workload == "WS":
            r.lin, r.lout = p, o
        else:   # WV: Python round (half to even), floor at 1
            r.lin, r.lout = max(1, round(p * W0_MEANS[0] / WS_MEANS[0])), max(1, round(o * W0_MEANS[1] / WS_MEANS[1]))
    return reqs


# ------------------------------------------------------------------------------------------------ AEG (fork of R.run_aeg)
def run_aeg2(M, scale, seed, cls, link, pf, qmax, types, kvswap=False, workload="W0", lengths=None, n_pre=R.N_PRE, n_dec=R.N_DEC,
             reqs=None, arr_end=R.ARR_END, score=(R.SCORE0, R.SCORE1), drain=R.DRAIN):
    """ minimal Aegaeon with per-model step / prefill / switch / KV and the L3 KV-swap cost (prereg §4)."""
    TT, TB = R.TTFT0 * scale, R.TBT0 * scale
    if TB <= 0 or TT <= 0:
        raise ValueError("SLO must be positive")
    if cls not in CLASSES:
        raise ValueError(f"unknown hardware class {cls!r}")
    KT = len(types)
    for t in types:
        model_of(t)
    if reqs is None:
        reqs = make_requests2(M, seed, workload, lengths, arr_end)
    kvrate = R.KVRATE[link]
    TSW_T = {t: tsw(t, link) for t in set(types)}
    kvmul = 2 if kvswap else 1
    t_stop = score[1] + drain
    P = [R.PInst(i) for i in range(n_pre)]
    D = [R.DInst(i) for i in range(n_dec)]
    ev = []
    seq = [0]
    stats = collections.Counter()

    def ty(model):
        return types[model % KT]

    def tsw_m(model):
        return TSW_T[ty(model)]

    def push(t, kind, obj, ver=0):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, obj, ver))

    for r in reqs:
        push(r.a, "ARR", r)

    def pend_pre(g):
        if not g.pending:
            return 0.0
        l0 = g.pending[0].lin
        if all(x.lin == l0 for x in g.pending):
            return tpre(ty(g.model), l0) * len(g.pending)        # t_pre(L_IN) * n form
        return sum(tpre(ty(g.model), x.lin) for x in g.pending)

    def p_load(p, now):
        load = max(0.0, p.busy_until - now) if p.busy else 0.0
        prev = p.loaded
        for g in p.queue:
            if g.model != prev and g is not p.cur:
                load += tsw_m(g.model)
            load += pend_pre(g)
            prev = g.model
        return load

    def p_start(p, now):
        while p.queue and not p.queue[0].pending:
            g0 = p.queue.pop(0)
            if g0.exec_start is not None:
                p.prev_exec_start = g0.exec_start
        if not p.queue:
            p.busy, p.cur = False, None
            p.prev_exec_start = None
            return
        g = p.queue[0]
        r = g.pending.popleft()
        cost = 0.0
        if g.model != p.loaded:
            ts_ = tsw_m(g.model)
            if pf and p.prev_exec_start is not None:
                cost = max(0.0, ts_ - (now - p.prev_exec_start))
            else:
                cost = ts_
            p.loaded = g.model
            stats["p_switch"] += 1
            stats["p_switch_s"] += cost
        start = now + cost
        if g.exec_start is None:
            g.exec_start = start
        end = start + tpre(ty(g.model), r.lin)
        p.busy, p.cur, p.busy_until = True, g, end
        push(end, "PFREE", (p, r))

    def on_arrival(r, now):
        for p in P:
            for g in p.queue:
                if g.model == r.model and g.acc < R.MAX_GPSIZE:
                    g.acc += 1
                    g.pending.append(r)
                    if not p.busy:
                        p_start(p, now)
                    return
        best = min(P, key=lambda p: (p_load(p, now), p.i))
        g = R.Group(r.model)
        g.acc = 1
        g.pending.append(r)
        best.queue.append(g)
        if not best.busy:
            p_start(best, now)

    def on_pfree(p, r, now):
        r.e1 = now
        r.k = 1
        if now <= r.a + TT + 1e-9:
            r.met += 1
            r.met1 = 1
        if r.lout <= 1:
            r.done, r.fin = True, now
        else:
            r.ready = now + r.lin * MODELS[ty(r.model)]["v"] * kvmul / kvrate      # L3: P->C + C->D (prereg §4 (i))
            push(r.ready, "READY", r)
        p.busy = False
        p_start(p, now)

    def d_find_batch(model):
        for d in D:
            for b in d.work:
                if b.model == model and len(b.reqs) + len(b.joiners) < R.BATCH_CAP:
                    return d, b
        return None, None

    def interrupt(d, now):
        if d.seg is None:
            return
        st, t, J = d.seg
        jd = int(math.floor((now - st) / t + 1e-12))
        J2 = min(J, jd + 1)
        if J2 < J:
            d.seg = (st, t, J2)
            d.ver += 1
            push(st + J2 * t, "DWAKE", d, d.ver)

    def on_ready(r, now):
        d, b = d_find_batch(r.model)
        if b is not None:
            if b is d.cur and d.seg is not None:
                b.joiners.append(r)
                interrupt(d, now)
            else:
                b.reqs.append(r)
            if d.idle:
                d.idle = False
                d_wake(d, now)
            return
        d = min(D, key=lambda x: (sum(1 for bb in x.work if bb.reqs or bb.joiners), sum(len(bb.reqs) for bb in x.work), x.i))
        b = R.Batch(r.model, d)
        b.reqs.append(r)
        d.work.append(b)
        if d.idle:
            d.idle = False
            d_wake(d, now)
        elif d.c0 and d.cur is not None and b.model != d.cur.model:
            d.end_after_seg = True
            if d.seg is not None:
                interrupt(d, now)

    def new_round(d):
        d.work = [b for b in d.work if b.reqs]
        if not d.work:
            return False
        order = []
        seen = {}
        for b in d.work:
            seen.setdefault(b.model, []).append(b)
        for b in d.work:
            if b.model in seen:
                order.extend(seen.pop(b.model))
        d.work = order
        d.round, d.ridx = list(order), 0
        ts = [step(ty(b.model), cls, len(b.reqs), sum(x.lin + x.k for x in b.reqs)) for b in order]
        c = round_overhead([b.model for b in order], ty, TSW_T)
        alpha, qs = R.eq23(ts, TB, c, qmax)
        if qs is None:
            d.c0 = True
            for b in order:
                b.q = qmax
        else:
            d.c0 = False
            for b, q in zip(order, qs):
                b.q = q
        stats["rounds"] += 1
        return True

    def end_turn(d, now):
        if d.cur is not None:
            d.prev_turn = now - d.turn_start
        d.cur, d.seg, d.steps, d.end_after_seg = None, None, 0, False
        d.ridx += 1

    def d_wake(d, now):
        while True:
            if now > t_stop:
                return
            if d.cur is None:
                if d.ridx >= len(d.round):
                    if not new_round(d):
                        d.idle, d.prev_turn, d.round, d.ridx = True, None, [], 0
                        return
                b = d.round[d.ridx]
                if not b.reqs:
                    d.ridx += 1
                    continue
                cost = 0.0
                if b.model != d.loaded:
                    ts_ = tsw_m(b.model)
                    if pf and d.prev_turn is not None:
                        cost = max(0.0, ts_ - d.prev_turn)
                    else:
                        cost = ts_
                    d.loaded = b.model
                    stats["d_switch"] += 1
                    stats["d_switch_s"] += cost
                    if kvswap:                                      # prereg §4 (ii): incoming KV on the critical path
                        kv = sum(r.lin + r.k for r in b.reqs) * MODELS[ty(b.model)]["v"] / kvrate
                        cost += kv
                        stats["kv_in_s"] += kv
                        stats["kv_in_n"] += 1
                d.cur, d.turn_start, d.turn_q, d.steps = b, now + cost, b.q, 0
                if cost > 0:
                    d.ver += 1
                    push(now + cost, "DWAKE", d, d.ver)
                    return
                continue
            b = d.cur
            if d.seg is not None:
                st, t, J = d.seg
                d.seg = None
                fin = []
                for r in b.reqs:
                    r.met += R.count_met(st, t, J, r.a, r.k + 1, TT, TB)
                    if r.k == 1 and J > 0:
                        r.fd = st + t
                    r.k += J
                    if r.k >= r.lout:
                        r.done, r.fin = True, st + J * t
                        fin.append(r)
                if fin:
                    b.reqs = [r for r in b.reqs if not r.done]
                d.steps += J
                stats["dec_steps"] += J
                stats["dec_tok"] += J * (len(b.reqs) + len(fin))
                stats["dec_busy_s"] += J * t
                if b.joiners:
                    b.reqs.extend(b.joiners)
                    b.joiners = []
            if not b.reqs:
                end_turn(d, now)
                continue
            tend = d.turn_start + d.turn_q
            if d.steps > 0 and (d.end_after_seg or now >= tend - 1e-12):
                end_turn(d, now)
                continue
            if d.c0 and d.steps > 0 and any(bb.model != b.model and bb.reqs for bb in d.work):
                end_turn(d, now)
                continue
            t = step(ty(b.model), cls, len(b.reqs), sum(r.lin + r.k for r in b.reqs))
            nall = math.ceil((tend - now) / t - 1e-12) if now < tend else 0
            if d.steps == 0:
                nall = max(1, nall)
            J = min(nall, min(r.lout - r.k for r in b.reqs))
            if J <= 0:
                end_turn(d, now)
                continue
            d.seg = (now, t, J)
            d.ver += 1
            push(now + J * t, "DWAKE", d, d.ver)
            return

    while ev:
        t, _, kind, obj, ver = heapq.heappop(ev)
        if t > t_stop:
            break
        if kind == "ARR":
            on_arrival(obj, t)
        elif kind == "PFREE":
            on_pfree(obj[0], obj[1], t)
        elif kind == "READY":
            on_ready(obj, t)
        elif kind == "DWAKE":
            if ver != obj.ver:
                continue
            d_wake(obj, t)
    stats["emitted_dec_tok"] = sum(max(0, r.k - 1) for r in reqs)
    stats["t_last"] = max((r.fin for r in reqs if r.fin is not None), default=0.0)
    return R.score_reqs(reqs, score, stats)


# ------------------------------------------------------------------------------------------------ MUX (fork of R.run_mux_gpu)
def run_mux2(types_gpu, seed, cls, beta, workload="W0", lengths=None, arr_end=None, reqs=None, score=None, drain=R.DRAIN):
    n_models = len(types_gpu)
    if n_models not in (1, 2):
        raise ValueError("a MUX GPU holds 1 or 2 models")
    if cls not in CLASSES:
        raise ValueError(f"unknown hardware class {cls!r}")
    for t in types_gpu:
        model_of(t)
    if arr_end is None:
        arr_end = R.SCORE0 + R.MUX_GPU_EQ * (R.SCORE1 - R.SCORE0)
    if score is None:
        score = (R.SCORE0, arr_end)
    if reqs is None:
        reqs = make_requests2(n_models, seed, workload, lengths, arr_end, stream_base=100000)
    t_stop = score[1] + drain
    for r in reqs:
        r.emit = np.full(r.lout, np.inf)
    eng = [{"wait": collections.deque(), "run": [], "in_step": False, "step": None} for _ in range(n_models)]
    ev = []
    seq = [0]

    def push(t, kind, obj):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, obj))

    for r in reqs:
        push(r.a, "ARR", r)

    def has_work(e):
        return bool(e["wait"]) or bool(e["run"]) or e["in_step"]

    def start(i, now):
        e = eng[i]
        if e["in_step"] or not has_work(e):
            return
        co = any(has_work(eng[j]) for j in range(n_models) if j != i)
        if e["wait"]:
            batch = list(e["wait"]); e["wait"].clear()
            dur = sum(tpre(types_gpu[i], r.lin) for r in batch)
            kind = "PRE"
        else:
            batch = list(e["run"])
            dur = step(types_gpu[i], cls, len(batch), sum(r.lin + r.k for r in batch))
            kind = "DEC"
        if co:
            dur *= (1.0 + beta)
        e["in_step"], e["step"] = True, (kind, batch)
        push(now + dur, "END", i)

    while ev:
        t = ev[0][0]
        if t > t_stop:
            break
        while ev and ev[0][0] <= t + 1e-12:
            _, _, kind, obj = heapq.heappop(ev)
            if kind == "ARR":
                eng[obj.model]["wait"].append(obj)
            else:
                e = eng[obj]
                sk, batch = e["step"]
                e["in_step"], e["step"] = False, None
                for r in batch:
                    r.emit[r.k] = t
                    r.k += 1
                    if r.k >= r.lout:
                        r.done, r.fin = True, t
                    elif sk == "PRE":
                        e["run"].append(r)
                if sk == "DEC":
                    e["run"] = [r for r in e["run"] if not r.done]
        for j in range(n_models):
            start(j, t)
    return reqs, score


def mux_attain_mix(M, K, A1, A2):
    """prereg §6: model i on GPU i mod 16, types (t, t+1 mod K); a GPU's pooled attainment for each of its models; mean over M."""
    if M > R.MUX_CAP:
        return None
    if K == 1:
        return R.mux_attain(M, A1[0], A2[(0, 0)])
    tot = 0.0
    for g in range(min(M, R.N_GPU)):
        t0 = g % K
        if g + R.N_GPU < M:
            tot += 2.0 * A2[(t0, (g + R.N_GPU) % K)]
        else:
            tot += A1[t0]
    return tot / M


def mux_goodput_mix(K, A1, A2):
    g, fails = 0, 0
    for M in R.M_GRID:
        at = mux_attain_mix(M, K, A1, A2)
        if at is not None and at >= R.BAR:
            g, fails = M, 0
        else:
            fails += 1
            if fails >= 2:
                break
    return g


# ------------------------------------------------------------------------------------------------ arms (prereg §5)
def aeg_arms():
    """(arm, level) for every AEG configuration, by phase."""
    ph = collections.defaultdict(list)
    ph["replay"].append(("AEG-HOST-A", "L0"))
    for X in CLASSES:
        ph["primary"].append((f"AEG-PAPER-{X}", "L3"))
        for L in ("L0", "L1", "L2"):
            ph["ladder"].append((f"AEG-PAPER-{X}", L))
    for X in ("A", "HM"):
        for pol in ("HOST", "NVL", "NVLQ"):
            ph["ours"].append((f"AEG-{pol}-{X}", "L3"))
        for L in ("L2v", "L3-5"):
            ph["desc"].append((f"AEG-PAPER-{X}", L))
    return ph


def mux_cfgs():
    """(beta name, class, mix, workload) by phase; KV swap does not touch MUX, so L3 = L2's MUX and L3-5 = (MIX5, WS)."""
    ph = collections.defaultdict(list)
    ph["replay"] += [("MPS", "A", "MIX0", "W0"), ("MPS", "HF", "MIX0", "W0")]
    for X in CLASSES:
        ph["primary"].append(("MPS", X, "MIX3", "WS"))
        ph["ladder"] += [("MPS", X, "MIX0", "W0"), ("MPS", X, "MIX3", "W0")]
    for X in ("A", "HM"):
        ph["ours"].append(("STOCK", X, "MIX3", "WS"))
        ph["desc"] += [("MPS", X, "MIX3", "WV"), ("MPS", X, "MIX5", "WS")]
    return ph


def parse_aeg_arm(arm):
    parts = arm.split("-")
    if len(parts) != 3 or parts[0] != "AEG" or parts[1] not in AEG_POL or parts[2] not in CLASSES:
        raise ValueError(f"unknown AEG arm {arm!r}")
    link, qmax = AEG_POL[parts[1]]
    return parts[1], parts[2], link, qmax


_WORK = {}


def _lengths_for(workload):
    if workload == "W0":
        return None
    return load_lengths(_WORK["lengths_path"])


def aeg_scan(args):
    arm, level, s, seed, lpath = args
    _WORK["lengths_path"] = lpath
    pol, cls, link, qmax = parse_aeg_arm(arm)
    mix, wl, kvs = LEVELS[level]
    lens = _lengths_for(wl)
    rows, fails = [], 0
    for M in R.M_GRID:
        t0 = time.time()
        r = run_aeg2(M, s, seed, cls, link, True, qmax, MIXES[mix], kvswap=kvs, workload=wl, lengths=lens)
        ok = r["attain"] is not None and r["attain"] >= R.BAR
        rows.append(dict(arm=arm, level=level, scale=s, seed=seed, M=M, ok=ok, wall_s=round(time.time() - t0, 2), **r))
        if ok:
            fails = 0
        else:
            fails += 1
            if fails >= 2:
                break
    return rows


def mux_one(args):
    bname, cls, mix, wl, seed, kind, t, lpath = args
    _WORK["lengths_path"] = lpath
    K = len(MIXES[mix])
    types = [MIXES[mix][t]] if kind == "solo" else [MIXES[mix][t], MIXES[mix][(t + 1) % K]]
    t0 = time.time()
    reqs, score = run_mux2(types, seed, cls, MUX_BETA[bname], workload=wl, lengths=_lengths_for(wl))
    sc = R.score_mux(reqs, score)
    return dict(beta=bname, cls=cls, mix=mix, workload=wl, seed=seed, kind=kind, t=t, types=types, beta_v=MUX_BETA[bname],
                wall_s=round(time.time() - t0, 2), n_req=sum(1 for r in reqs if score[0] <= r.a < score[1]),
                unfinished=sum(1 for r in reqs if score[0] <= r.a < score[1] and not r.done),
                attain={str(k): v["attain"] for k, v in sc.items()})


def b_run(out, phases, seeds, workers, lpath):
    import multiprocessing as mp
    os.makedirs(out, exist_ok=True)
    ap, mp_ = os.path.join(out, "aeg_rows.jsonl"), os.path.join(out, "mux_rows.jsonl")
    have, have_m = set(), set()
    if os.path.exists(ap):
        for line in open(ap):
            x = json.loads(line)
            have.add((x["arm"], x["level"], x["scale"], x["seed"]))
    if os.path.exists(mp_):
        for line in open(mp_):
            x = json.loads(line)
            have_m.add((x["beta"], x["cls"], x["mix"], x["workload"], x["seed"], x["kind"], x["t"]))
    A, Mx = aeg_arms(), mux_cfgs()
    arms = [a for p in phases for a in A[p]]
    mcfg = []
    for p in phases:
        for c in Mx[p]:
            if c not in mcfg:
                mcfg.append(c)
    jobs = [(a, L, s, sd, lpath) for a, L in arms for s in R.SCALES for sd in seeds if (a, L, s, sd) not in have]
    mjobs = []
    for bname, cls, mix, wl in mcfg:
        K = len(MIXES[mix])
        for sd in seeds:
            for kind in ("solo", "pair"):
                for t in range(K):
                    key = (bname, cls, mix, wl, sd, kind, t)
                    if key not in have_m:
                        mjobs.append((bname, cls, mix, wl, sd, kind, t, lpath))
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(mux_one, mjobs):
            res.update(prereg=PREREG_COMMIT)
            with open(mp_, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        for rows in pool.imap_unordered(aeg_scan, jobs):
            with open(ap, "a") as fh:
                for r in rows:
                    r["prereg"] = PREREG_COMMIT
                    fh.write(json.dumps(r) + "\n")
            last = rows[-1]
            print(json.dumps({"aeg": last["arm"], "L": last["level"], "s": last["scale"], "seed": last["seed"], "last_M": last["M"],
                              "wall": round(sum(r["wall_s"] for r in rows), 1)}), flush=True)
    return {"aeg_jobs": len(jobs), "mux_jobs": len(mjobs)}


# ------------------------------------------------------------------------------------------------ the map (prereg §3.3, §3.4, §7.1-§7.2)
BENCH_RUNS = [  # label, path (gpu_score.json), card, role
    ("nvlink-pair run_h2", "data/runs/nvlink_beta_2026-10-19/run_h2/gpu_score.json", "A100-SXM4-80GB", "FIT"),
    ("nvlink-pair run_h1", "data/runs/nvlink_beta_2026-10-19/run_h1/gpu_score.json", "A100-SXM4-80GB", "HELD-OUT"),
    ("inengine run_h3", "data/runs/inengine_pool_2026-10-19/run_h3/gpu_score.json", "A100-SXM4-80GB", "HELD-OUT"),
    ("multigpu run_h2", "data/runs/multigpu_probe_2026-10-14/run_h2/gpu_score.json", "A100-SXM4-40GB", "FIT"),
]
BENCH_CELLS = {"Q": (1, 4096), "S": (8, 4096), "LC": (2, 16384)}   # (B, per-sequence prompt L) as bench/multigpu/inputs/plan.json; summed C = B x L
SERVING = [  # label, dir, card, {slot: model key}, precision, engine
    ("phase1 A40", "data/runs/multiplex_phase1_2026-10-11/run2/rows", "A40", {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "bf16", "vllm"),
    ("phase3 A100-SXM4-80GB REF", "data/runs/multiplex_phase3_2026-10-11/run/rows", "A100-SXM4-80GB",
     {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "bf16", "vllm"),
    ("phase3 A100-SXM4-80GB run3b", "data/runs/multiplex_phase3_2026-10-11/run3b/rows", "A100-SXM4-80GB",
     {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "bf16", "vllm"),
    ("ch6 arm4 A100-SXM4-80GB", "data/runs/multiplex_ch6_arm4_2026-10-12/run_sxm4b/rows/orig", "A100-SXM4-80GB",
     {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "bf16", "vllm"),
    ("ch4 A100 80GB PCIe", "data/runs/multiplex_ch4_2026-10-11/run/rows", "A100-PCIe-80GB", {"qwen": "qwen25_7b", "llama": "llama31_8b"},
     "bf16", "vllm"),
    ("ch5 RTX A6000", "data/runs/multiplex_ch5_2026-10-11/run_a6000/rows", "RTX-A6000", {"qwen": "qwen25_7b", "llama": "llama31_8b"},
     "bf16", "vllm"),
    ("ch5 RTX 6000 Ada", "data/runs/multiplex_ch5_2026-10-11/run_ada/rows", "RTX-6000-Ada", {"qwen": "qwen25_7b", "llama": "llama31_8b"},
     "bf16", "vllm"),
    ("ch5 RTX 4090", "data/runs/multiplex_ch5_2026-10-11/run_r4090/rows", "RTX-4090", {"qwen": "qwen25_7b", "llama": "llama31_8b"},
     "bf16", "vllm"),
    ("ch5 RTX 5090", "data/runs/multiplex_ch5_2026-10-11/run_r5090/rows", "RTX-5090", {"qwen": "qwen25_7b", "llama": "llama31_8b"},
     "bf16", "vllm"),
    ("ch6 arm2 RTX PRO 6000 Server", "data/runs/multiplex_ch6_arms236_2026-10-12/run_a2/rows/orig", "RTX-PRO-6000",
     {"llama": "llama31_8b"}, "bf16", "vllm"),
    ("ch5 RTX 6000 Ada FP8", "data/runs/multiplex_ch5_2026-10-11/run_adafp8b/rows", "RTX-6000-Ada",
     {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "fp8", "vllm"),
    ("ch6 arm1 RTX 5090 FP8", "data/runs/multiplex_ch6_arm1_2026-10-12/run_r5090fp8/rows", "RTX-5090",
     {"qwen": "qwen25_7b", "llama": "llama31_8b"}, "fp8", "vllm"),
    ("phase1b A100-SXM4-80GB MuxServe fork", "data/runs/multiplex_phase1b_2026-10-11/run/rows", "A100-SXM4-80GB",
     {"qwen": "llama7b", "llama": "llama7b"}, "fp16", "muxserve"),
]
REF_SERVING = "phase3 A100-SXM4-80GB REF"
TRUST_BAR = 0.20


def serving_cell(root, d, slot):
    p = os.path.join(root, d, f"rows_A-{slot}_s1.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    v = [float(x["tpot"]) for x in map(json.loads, open(p)) if x.get("err") is None and x.get("tpot") is not None]
    if not v:                                       # review: an empty cell is an input error, never a median of nothing
        raise ValueError(f"no valid tpot rows in {p}")
    return float(np.median(v)), len(v), p


def build_map(root):
    out = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT,
           "kappa": KAPPA, "F_s": F_FIX, "I_HM_ref_s": I_HM_REF, "fit_A80": FIT_A80, "fit_A40": FIT_A40, "bench": [], "serving": []}
    for label, path, card, role in BENCH_RUNS:
        d = json.load(open(os.path.join(root, path)))
        pr = d["probe"]
        d2d = (pr["hbm_d2d_gbs_0"] + pr["hbm_d2d_gbs_1"]) / 2.0 * 1e9
        cls = "A" if card == "A100-SXM4-80GB" else "A40"
        for cell, (B, L) in BENCH_CELLS.items():
            tm = d["anchor"][cell]["tpot_s"]
            C = B * (L + 32)
            byt = W_REF + C * V_REF
            tns, tnd = byt / B_SPEC[card], byt / d2d
            tk = step("ref8b", cls, B, C, allow_map=True)
            out["bench"].append({"run": label, "card": card, "role": role, "cell": cell, "B": B, "L": L, "C": C, "t_meas": tm,
                                 "n_passes": len(d["anchor"][cell]["per_pass"]), "spread": d["anchor"][cell]["spread"],
                                 "d2d_gbs": d2d / 1e9, "t_nspec": tns, "t_nd2d": tnd, "t_kappa": tk,
                                 "e_nspec": tns / tm - 1, "e_nd2d": tnd / tm - 1, "e_kappa": tk / tm - 1})
    ref, refW = {}, {}
    cells = []
    for label, d, card, slots, prec, eng in SERVING:
        for slot, mk in slots.items():
            med, n, p = serving_cell(root, d, slot)
            W = MODELS[mk]["W"] / (2.0 if prec == "fp8" else 1.0)
            cells.append(dict(label=label, card=card, slot=slot, model=mk, precision=prec, engine=eng, t_meas=med, n=n,
                              t_nspec_w=W / B_SPEC[card], e_abs=(W / B_SPEC[card]) / med - 1, W=W, path=p))
            if label == REF_SERVING:
                ref[slot], refW[slot] = med, W
    Bref = B_SPEC["A100-SXM4-80GB"]
    gate = []
    for c in cells:
        if c["slot"] in ref and c["label"] != REF_SERVING:
            r_meas = c["t_meas"] / ref[c["slot"]]
            Wr = refW[c["slot"]]           # bytes of the REF row's model in this slot (= c["W"] for a bf16 same-model cell)
            r_n = (c["W"] / B_SPEC[c["card"]]) / (Wr / Bref)                       # = B_REF / B_X for a bf16 same-model cell
            r_k = (F_FIX + KAPPA * c["W"] / B_SPEC[c["card"]]) / (F_FIX + KAPPA * Wr / Bref)
            c.update(r_meas=r_meas, r_n=r_n, r_k=r_k, e_n=r_n / r_meas - 1, e_k=r_k / r_meas - 1)
            if c["precision"] == "bf16" and c["engine"] == "vllm":
                gate.append(c)
    out["serving"] = cells
    med_k, tv = trust_verdict([abs(c["e_k"]) for c in gate])
    en = sorted(abs(c["e_n"]) for c in gate)
    out["trust_gate"] = {"n": len(gate), "median_abs_e_kappa": med_k, "median_abs_e_n": float(np.median(en)) if en else None,
                         "bar": TRUST_BAR, "verdict": tv, "cells": [f"{c['label']}/{c['slot']}" for c in gate]}
    tbl = {}
    for mk in ("ref8b", "qwen25_7b", "llama31_8b", "llama7b", "yi6b", "qwen7b", "internlm25_7b", "yi9b", "llama13b"):
        row = {}
        for cls, lab in (("A", "A100-SXM4-80GB"), ("A40", "A100-SXM4-40GB"), ("HM", "H800-SXM (HM)"), ("HF", "H800-SXM (HF)")):
            row[lab] = {"value_s": step(mk, cls, 1, 4128, allow_map=True), "label": "MODELLED-HF" if cls == "HF" else "MODELLED"}
        tbl[MODELS[mk]["name"]] = row
    for b in out["bench"]:
        if b["cell"] == "Q" and b["role"] == "FIT":
            tbl[MODELS["ref8b"]["name"]][b["card"]] = {"value_s": b["t_meas"], "label": "BENCH-MEASURED (Q cell)", "run": b["run"]}
    for c in cells:
        key = f"{c['card']} [{c['precision']}/{c['engine']}] {c['label']}"
        tbl[MODELS[c["model"]]["name"]].setdefault("SERVING", {})[key] = {"value_s": c["t_meas"], "n": c["n"],
                                                                          "label": "SERVING-MEASURED (TPOT p50, s 1)"}
    out["table"] = tbl
    out["bytes"] = {k: {"params": v.get("params"), "W": v["W"], "v": v["v"], "rho": v["rho"]} for k, v in MODELS.items()}
    return out


# ------------------------------------------------------------------------------------------------ summarize (prereg §7-§8)
def goodputs(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["level"], r["scale"], r["seed"])].append(r)
    g, nonmono = {}, []
    for key, rs in by.items():
        rs.sort(key=lambda x: x["M"])
        best, seen_fail = 0, False
        for r in rs:
            if r["ok"]:
                if seen_fail:
                    nonmono.append(list(key) + [r["M"]])
                best = r["M"]
            else:
                seen_fail = True
        g[key] = best
    return g, nonmono


def mux_goodputs(mrows):
    m = {(r["beta"], r["cls"], r["mix"], r["workload"], r["seed"], r["kind"], r["t"]): r for r in mrows}
    out = {}
    cfgs = {(r["beta"], r["cls"], r["mix"], r["workload"]) for r in mrows}
    for bname, cls, mix, wl in cfgs:
        K = len(MIXES[mix])
        for sd in R.SEEDS:
            need = [(bname, cls, mix, wl, sd, k, t) for k in ("solo", "pair") for t in range(K)]
            if not all(x in m for x in need):
                continue
            for s in R.SCALES:
                A1 = [m[(bname, cls, mix, wl, sd, "solo", t)]["attain"][str(s)] for t in range(K)]
                A2 = {(t, (t + 1) % K): m[(bname, cls, mix, wl, sd, "pair", t)]["attain"][str(s)] for t in range(K)}
                out[(bname, cls, mix, wl, sd, s)] = mux_goodput_mix(K, A1, A2)
    return out


def step_index(c):
    return len(R.SCALES) if c == "<0.1" else R.SCALES.index(c)


def verdict_class(c, gA):
    if c == 0.2:
        shape = gA.get(0.5, 0) >= 48 and gA.get(0.3, 0) >= 36 and 64 <= gA.get(1.0, 0) <= 76
        return "LANDS" if shape else "LANDS-CROSSOVER-ONLY"
    if c in (0.3, 0.15):
        return "NEAR"
    if c in (0.1, "<0.1"):
        return "MISSES-STRONG"
    if c in (1.0, 0.5):
        return "MISSES-WEAK"
    return None


def completeness(rows, mrows):
    """an AEG scan is complete iff it ends on two consecutive fails or reaches the top of the M grid; a MUX config needs every
    (seed, kind, t) sim. Lists what is not (QA r1, glm)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["level"], r["scale"], r["seed"])].append(r)
    bad = []
    for k, rs in by.items():
        rs.sort(key=lambda x: x["M"])
        ms = [x["M"] for x in rs]
        expect = list(R.M_GRID[:len(rs)])
        ends = (len(rs) >= 2 and not rs[-1]["ok"] and not rs[-2]["ok"]) or rs[-1]["M"] == R.M_GRID[-1]
        if ms != expect or not ends:
            bad.append(list(k))
    have = collections.defaultdict(set)
    for r in mrows:
        have[(r["beta"], r["cls"], r["mix"], r["workload"])].add((r["seed"], r["kind"], r["t"]))
    mbad = []
    for cfg, got in have.items():
        K = len(MIXES[cfg[2]])
        need = {(sd, k, t) for sd in R.SEEDS for k in ("solo", "pair") for t in range(K)}
        if need - got:
            mbad.append(list(cfg) + [len(need - got)])
    return {"aeg_scans_incomplete": bad, "mux_cfgs_incomplete": mbad}


def attrib(g1, g2, sp1, sp2):
    """prereg §6 ATTRIBUTABLE: |median delta| > the LARGER of the two arms' seed spreads."""
    d = g2 - g1
    sp = max(sp1, sp2)
    return {"delta": d, "max_spread": sp, "attributable": abs(d) > sp}


def trust_verdict(abs_errs, bar=TRUST_BAR):
    """prereg §7.2: PASS iff the median |e_kappa| over the gate cells <= bar; no cell = FAIL."""
    if not abs_errs:
        return None, "FAIL"
    m = float(np.median(sorted(abs_errs)))
    return m, ("PASS" if m <= bar else "FAIL")


def challenger_readings(a_v, hm_v):
    """prereg §7.3, scored under both readings of 'LANDS' (strict class / LANDS-CROSSOVER-ONLY counted) per the undefined-term rule."""
    out = {}
    for name, incl in (("strict", False), ("inclusive", True)):
        lands_hm = hm_v == "LANDS" or (incl and hm_v == "LANDS-CROSSOVER-ONLY")
        ok_a = a_v in ("LANDS", "NEAR") or (incl and a_v == "LANDS-CROSSOVER-ONLY")
        out[name] = "CONTRADICTED" if lands_hm else ("SUPPORTED" if ok_a and hm_v == "MISSES-STRONG" else "UNRESOLVED")
    out["readings_differ"] = out["strict"] != out["inclusive"]
    return out


def summarize(out, root):
    ap, mp_ = os.path.join(out, "aeg_rows.jsonl"), os.path.join(out, "mux_rows.jsonl")
    res = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT}
    mpath = os.path.join(out, "map.json")
    if os.path.exists(mpath):
        mm = json.load(open(mpath))
        res["trust_gate"] = mm.get("trust_gate")
    rows = [json.loads(x) for x in open(ap)] if os.path.exists(ap) else []
    mrows = [json.loads(x) for x in open(mp_)] if os.path.exists(mp_) else []
    g, nonmono = goodputs(rows)
    mg = mux_goodputs(mrows)
    res["nonmonotone_scans"] = nonmono
    res["incomplete"] = completeness(rows, mrows)     # review: a truncated scan or a missing MUX sim is surfaced, never silent
    med, spr, per = {}, {}, collections.defaultdict(dict)
    for (arm, L) in sorted({(k[0], k[1]) for k in g}):
        key = f"{arm}@{L}"
        med[key], spr[key] = {}, {}
        for s in R.SCALES:
            vals = [g[(arm, L, s, sd)] for sd in R.SEEDS if (arm, L, s, sd) in g]
            if len(vals) == len(R.SEEDS):
                med[key][s] = float(np.median(vals))
                spr[key][s] = max(vals) - min(vals)
            for sd in R.SEEDS:
                if (arm, L, s, sd) in g:
                    per[key].setdefault(sd, {})[s] = g[(arm, L, s, sd)]
    mmed, mper = {}, collections.defaultdict(dict)
    for (bname, cls, mix, wl) in sorted({k[:4] for k in mg}):
        key = f"MUX-{bname}-{cls}@{mix}/{wl}"
        mmed[key] = {}
        for s in R.SCALES:
            vals = [mg[(bname, cls, mix, wl, sd, s)] for sd in R.SEEDS if (bname, cls, mix, wl, sd, s) in mg]
            if len(vals) == len(R.SEEDS):
                mmed[key][s] = float(np.median(vals))
            for sd in R.SEEDS:
                if (bname, cls, mix, wl, sd, s) in mg:
                    mper[key].setdefault(sd, {})[s] = mg[(bname, cls, mix, wl, sd, s)]
    res["aeg_goodput_median"] = {k: {str(s): v for s, v in d.items()} for k, d in med.items()}
    res["aeg_goodput_spread"] = {k: {str(s): v for s, v in d.items()} for k, d in spr.items()}
    res["aeg_goodput_per_seed"] = {k: {str(sd): {str(s): v for s, v in dd.items()} for sd, dd in d.items()} for k, d in per.items()}
    res["mux_goodput_median"] = {k: {str(s): v for s, v in d.items()} for k, d in mmed.items()}
    res["mux_goodput_per_seed"] = {k: {str(sd): {str(s): v for s, v in dd.items()} for sd, dd in d.items()} for k, d in mper.items()}

    def mux_key(beta, cls, L):
        mix, wl, _ = LEVELS[L]
        return f"MUX-{beta}-{cls}@{mix}/{wl}"

    def pair(arm, L, beta):
        cls = parse_aeg_arm(arm)[1]
        ak, mk = f"{arm}@{L}", mux_key(beta, cls, L)
        if ak not in med or mk not in mmed or len(med[ak]) < len(R.SCALES) or len(mmed[mk]) < len(R.SCALES):
            return None
        c, nm = R.crossover(med[ak], mmed[mk])
        seeds_c = []
        for sd in R.SEEDS:
            a_, m_ = per[ak].get(sd), mper[mk].get(sd)
            seeds_c.append(R.crossover(a_, m_)[0] if a_ and m_ and len(a_) == len(R.SCALES) and len(m_) == len(R.SCALES) else None)
        idx = [step_index(x) for x in seeds_c if x is not None]
        return {"aeg": ak, "mux": mk, "crossover": c, "nonmonotone": nm, "per_seed": seeds_c,
                "unstable": (max(idx) - min(idx) > 1) if idx else None}

    def attributable(k1, k2, s):
        if k1 not in med or k2 not in med or s not in med[k1] or s not in med[k2]:
            return None
        return attrib(med[k1][s], med[k2][s], spr[k1][s], spr[k2][s])

    tg = (res.get("trust_gate") or {}).get("verdict")
    cls_v = {}
    for X in CLASSES:
        p = pair(f"AEG-PAPER-{X}", "L3", "MPS")
        if not p:
            cls_v[X] = {"verdict": "UNRESOLVABLE (missing arm)"}
            continue
        v = verdict_class(p["crossover"], med[f"AEG-PAPER-{X}@L3"])
        label = "intercept MEASURED for 8 B, mix MODELLED" if X == "A" else "MODELLED"
        if X in ("HM", "HF") and tg != "PASS":
            reading = "UNRESOLVED-MODELLED"
        else:
            reading = v
        cls_v[X] = dict(p, verdict=v, reading=reading, label=label, g_aeg=med[f"AEG-PAPER-{X}@L3"])
    res["crossover_per_class_L3"] = cls_v

    res["challenger_readings"] = challenger_readings(cls_v.get("A", {}).get("verdict"), cls_v.get("HM", {}).get("verdict"))

    ladder = {}
    for X in CLASSES:
        arm = f"AEG-PAPER-{X}"
        steps = [("L0", "L1"), ("L1", "L2"), ("L2", "L3"), ("L1", "L2v"), ("L2v", "L2")]
        lad = {"medians": {L: med.get(f"{arm}@{L}") for L in ("L0", "L1", "L2v", "L2", "L3")},
               "mux": {L: mmed.get(mux_key("MPS", X, L)) for L in ("L0", "L1", "L2v", "L2", "L3")},
               "crossovers": {L: (pair(arm, L, "MPS") or {}).get("crossover") for L in ("L0", "L1", "L2v", "L2", "L3")}, "steps": {}}
        for a_, b_ in steps:
            lad["steps"][f"{a_}->{b_}"] = {str(s): attributable(f"{arm}@{a_}", f"{arm}@{b_}", s) for s in R.SCALES}
        ladder[X] = lad
    res["ladder"] = ladder

    ours = {}
    for X in ("A", "HM"):
        o = {}
        for pol in ("HOST", "NVL", "NVLQ"):
            for beta in ("MPS", "STOCK"):
                o[f"AEG-{pol}-{X} vs MUX-{beta}"] = pair(f"AEG-{pol}-{X}", "L3", beta)
        for ref_pol, tier_pol, name in (("HOST", "NVL", "NVLink moves it"), ("NVL", "NVLQ", "the quantum pays")):
            for beta in ("MPS", "STOCK"):
                pr, pt = o.get(f"AEG-{ref_pol}-{X} vs MUX-{beta}"), o.get(f"AEG-{tier_pol}-{X} vs MUX-{beta}")
                if not pr or not pt:
                    continue
                cr, ct = pr["crossover"], pt["crossover"]
                moved = step_index(ct) - step_index(cr) >= 1
                at = attributable(f"AEG-{ref_pol}-{X}@L3", f"AEG-{tier_pol}-{X}@L3", cr) if cr != "<0.1" else None
                v = "DOES NOT MOVE IT"
                if moved:
                    v = "MOVES IT" if (at and at["attributable"]) else "UNATTRIBUTABLE"
                o[f"{name} ({beta})"] = {"ref_cross": cr, "tier_cross": ct, "at_ref_cross": at, "verdict": v}
        ours[X] = o
    res["ours_L3"] = ours

    # REPLAY gate (prereg §8)
    rep = {"verdict": None}
    p2957 = os.path.join(root, B2957_ROWS_REL)
    if os.path.exists(p2957):
        old = {}
        for x in map(json.loads, open(p2957)):
            if x["arm"] == "AEG-HOST-A":
                old[(x["scale"], x["seed"], x["M"])] = x["attain"]
        new = {(r["scale"], r["seed"], r["M"]): r["attain"] for r in rows if r["arm"] == "AEG-HOST-A" and r["level"] == "L0"}
        mism = [list(k) for k in old if k not in new or new[k] != old[k]] + [list(k) for k in new if k not in old]
        want = {1.0: 72, 0.5: 48, 0.3: 40, 0.2: 28, 0.15: 24, 0.1: 0}
        gseeds = {sd: per.get("AEG-HOST-A@L0", {}).get(sd) for sd in R.SEEDS}
        g_ok = all(gseeds[sd] and all(gseeds[sd].get(s) == w for s, w in want.items()) for sd in R.SEEDS)
        ma, mh = mmed.get("MUX-MPS-A@MIX0/W0"), mmed.get("MUX-MPS-HF@MIX0/W0")
        mux_ok = (ma == {1.0: 32.0, 0.5: 32.0, 0.3: 32.0, 0.2: 32.0, 0.15: 32.0, 0.1: 20.0}
                  and mh == {s: 32.0 for s in R.SCALES})
        rep = {"rows_compared": len(old), "row_mismatches": mism[:20], "n_mismatch": len(mism), "goodput_per_seed_ok": g_ok,
               "mux_L0_ok": mux_ok, "mux_A": ma, "mux_HF": mh,
               "verdict": "PASS" if (not mism and g_ok and mux_ok) else "FAIL (VOID)"}
    res["replay_gate"] = rep
    return res


# ------------------------------------------------------------------------------------------------ --b-one (G7's real entry point)
def b_one(spec, root):
    sp = json.loads(spec)
    types = sp.get("types", ["ref8b"])
    reqs = [R.Req(i, int(m), float(a), int(sp.get("lin", R.L_IN)), int(sp.get("lout", R.L_OUT))) for i, (a, m) in enumerate(sp["reqs"])]
    if sp["policy"] == "aeg":
        r = run_aeg2(len({x.model for x in reqs}), float(sp["scale"]), 0, sp["cls"], sp["link"], bool(sp.get("pf", True)),
                     float(sp.get("qmax", R.QMAX)), types, kvswap=bool(sp.get("kvswap", False)), n_pre=int(sp.get("n_pre", 1)),
                     n_dec=int(sp.get("n_dec", 1)), reqs=reqs, score=(-1.0, 1e9), drain=float(sp.get("drain", 60.0)))
        r["emit"] = [{"id": x.id, "e1": x.e1, "ready": x.ready, "fd": x.fd, "fin": x.fin, "met": x.met} for x in reqs]
        return r
    if sp["policy"] == "mux":
        reqs2, score = run_mux2(types, 0, sp["cls"], float(sp["beta"]), reqs=reqs, score=(-1.0, 1e9), drain=float(sp.get("drain", 60.0)))
        return {"emit": [[float(v) for v in x.emit] for x in reqs2],
                "attain": {str(k): v["attain"] for k, v in R.score_mux(reqs2, score).items()}}
    raise ValueError("policy must be aeg or mux")


# ------------------------------------------------------------------------------------------------ self-test (prereg §9)
def self_test(root):
    res = []

    def chk(name, ok, got=None):
        res.append((name, bool(ok), got))

    def close(x, y, tol):
        return abs(x - y) <= tol

    # G1 fits, kappa, F
    a80, kb80, kl80 = FIT_A80
    a40, kb40, kl40 = FIT_A40
    chk("G1 A80 fit", close(a80, 0.009107822733941803, 1e-15) and close(kb80, 0.000338947016095358, 1e-15)
        and close(kl80, 4.6267392100077835e-07, 1e-18), FIT_A80)
    chk("G1 A80 fit == 's", a80 == R.A_FIT and kb80 == R.KB_FIT and kl80 == R.KL_FIT)
    chk("G1 A40 fit", close(a40, 0.011314839732890226, 1e-15) and close(kb40, 0.0003343817418214508, 1e-15)
        and close(kl40, 4.6707148118492263e-07, 1e-18), FIT_A40)
    for nm, (a, kb, kl), (Q, S, LC) in (("A80", FIT_A80, ANCH_A80), ("A40", FIT_A40, ANCH_A40)):
        chk(f"G1 {nm} fit returns its anchors", close(a + kb + kl * 4096, Q, 1e-12) and close(a + 8 * kb + kl * 32768, S, 1e-12)
            and close(a + 2 * kb + kl * 32768, LC, 1e-12))
    chk("G1 kappa 0.8825794778140704, F 0.0020170966319484167", close(KAPPA, 0.8825794778140704, 1e-12)
        and close(F_FIX, 0.0020170966319484167, 1e-12), (KAPPA, F_FIX))
    chk("G1 I(ref, HM) 0.006332914698206481", close(I_HM_REF, 0.006332914698206481, 1e-12)
        and close(F_FIX + KAPPA * W_REF / B_H, I_HM_REF, 1e-15), I_HM_REF)
    chk("G1 A40 intercept = F + kappa W/B40", close(intercept("ref8b", "A40"), a40, 1e-15))
    # G2 equivalence to step
    ok2 = all(step("ref8b", "A", b, c) == R.step_time("A", b, c) and step("ref8b", "HF", b, c) == R.step_time("H", b, c)
              for b, c in ((1, 4096), (8, 32768), (3, 1000)))
    chk("G2 ref step == R.step_time (A, H) exactly", ok2)
    chk("G2 ref tpre / tsw == 's", tpre("ref8b", 161) == R.t_pre(161) and all(tsw("ref8b", k) == R.TSW[k] for k in R.TSW))
    # G3 a MODELLED step
    chk("G3 t_HM(llama13b, 2, 730)", close(step("llama13b", "HM", 2, 730), 0.01002232030426489, 1e-12), step("llama13b", "HM", 2, 730))
    chk("G3 t_A(llama13b, 2, 730)", close(step("llama13b", "A", 2, 730), 0.014431903004997908, 1e-12))
    chk("G3 t_HF(llama13b, 2, 730)", close(step("llama13b", "HF", 2, 730), 0.008784074694683801, 1e-12))
    # G4 configs
    want = {"yi6b": (6061035520, 65536, 0.8888888888888888), "qwen7b": (7720931328, 524288, 0.8888888888888888),
            "internlm25_7b": (7737708544, 131072, 0.8888888888888888), "yi9b": (8829407232, 98304, 1.3333333333333333),
            "llama13b": (13015864320, 819200, 1.3888888888888888), "qwen25_7b": (7615487488, None, None),
            "llama31_8b": (8030261248, None, None), "llama7b": (6738415616, None, None)}
    ok4 = all(MODELS[k]["params"] == p and MODELS[k]["W"] == 2.0 * p and (v is None or MODELS[k]["v"] == v)
              and (rho is None or close(MODELS[k]["rho"], rho, 1e-15)) for k, (p, v, rho) in want.items())
    chk("G4 config params / bytes / KV / rho", ok4)
    chk("G4 Table 1 KV 524,288 / 131,072 / 819,200", (MODELS["qwen7b"]["v"], MODELS["internlm25_7b"]["v"], MODELS["llama13b"]["v"])
        == (524288, 131072, 819200))
    # G5 switch
    chk("G5 llama13b tsw PAPER / HOST / NVL", close(tsw("llama13b", "PAPER"), 0.81349152, 1e-12)
        and close(tsw("llama13b", "HOST"), 1.0006094596217205, 1e-12) and close(tsw("llama13b", "NVL"), 0.09529613901159242, 1e-12),
        (tsw("llama13b", "PAPER"), tsw("llama13b", "HOST"), tsw("llama13b", "NVL")))
    # G6 adversarial
    for nm, fn in (("unknown class B200", lambda: step("ref8b", "B200", 1, 1)),
                   ("A40 not a crossover class", lambda: step("ref8b", "A40", 1, 1)),
                   ("unknown model type", lambda: step("llama70b", "A", 1, 1)),
                   ("degenerate kappa solve", lambda: kappa_solve(0.01, 2e12, 0.01, 2e12, W_REF)),
                   ("MUX GPU with 3 models", lambda: run_mux2(["ref8b"] * 3, 0, "A", 0.0)),
                   ("AEG unknown class", lambda: run_aeg2(1, 1.0, 0, "B200", "HOST", True, 4.0, ["ref8b"]))):
        try:
            fn()
            chk(f"G6 {nm} raises", False)
        except ValueError:
            chk(f"G6 {nm} raises", True)
        except Exception as ex:      # noqa: BLE001  a wrong exception type is a recorded FAIL
            chk(f"G6 {nm} raises ValueError (got {type(ex).__name__})", False, repr(ex))
    chk("G6(iv) 'H-MID = H-FAST at the reference model' FAILS", not close(intercept("ref8b", "HM"), R.A_FIT * R.H_RATIO, 1e-6),
        (intercept("ref8b", "HM"), R.A_FIT * R.H_RATIO))
    me = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.json.gz")
        with gzip.open(bad, "wb") as fh:
            fh.write(b'{"prompt":[1],"output":[1]}')
        p = subprocess.run([sys.executable, me, "--check-lengths", "--lengths", bad], capture_output=True, text=True)
        chk("G6(ii) bad lengths sha -> exit 2 + one ERROR line", p.returncode == 2 and p.stderr.count("ERROR:") == 1 and not p.stdout,
            (p.returncode, p.stderr[-200:]))

    def cli(spec):
        p = subprocess.run([sys.executable, me, "--b-one", "--spec", json.dumps(spec)], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-500:])
        return json.loads(p.stdout)
    # G7 real entry point
    try:
        base = dict(policy="aeg", reqs=[[0.0, 0]], types=["llama13b"], cls="HM", link="PAPER", kvswap=True, lin=161, lout=3, pf=True,
                    qmax=4.0, n_pre=1, n_dec=1)
        r = cli(dict(base, scale=0.1))
        e = r["emit"][0]
        chk("G7 e1 / ready / fd / fin", close(e["e1"], 0.8462194067376655, 1e-9) and close(e["ready"], 0.8544626067376655, 1e-9)
            and close(e["fd"], 1.6814197017103787, 1e-9) and close(e["fin"], 1.6907380766830922, 1e-9), e)
        chk("G7 s0.1 met 1/3", e["met"] == 1, e["met"])
        for sc in (1.0, 0.2):
            rr = cli(dict(base, scale=sc))
            chk(f"G7 s{sc} met 3/3", rr["emit"][0]["met"] == 3, rr["emit"][0]["met"])
        r = cli(dict(base, scale=0.1, kvswap=False))
        e = r["emit"][0]
        chk("G7 KV-swap off: ready 0.8503410067376654, fin 1.6824692766830922", close(e["ready"], 0.8503410067376654, 1e-9)
            and close(e["fin"], 1.6824692766830922, 1e-9), e)
    except Exception as ex:      # noqa: BLE001
        chk("G7 CLI", False, repr(ex))
    # G8 equivalence (AEG-HOST-A seed 0, every stored row; and full output vs R.run_aeg)
    try:
        p2957 = os.path.join(root, B2957_ROWS_REL)
        raw = open(p2957, "rb").read()
        chk("G8 b_rows sha256", hashlib.sha256(raw).hexdigest() == B2957_ROWS_SHA)
        old = [json.loads(x) for x in raw.decode().splitlines() if x]
        old = [x for x in old if x["arm"] == "AEG-HOST-A" and x["seed"] == 0]
        bad = []
        gs = {}
        for x in old:
            r = run_aeg2(x["M"], x["scale"], 0, "A", "HOST", True, R.QMAX, ["ref8b"])
            if r["attain"] != x["attain"]:
                bad.append((x["scale"], x["M"], r["attain"], x["attain"]))
            if r["attain"] is not None and r["attain"] >= R.BAR:
                gs[x["scale"]] = max(gs.get(x["scale"], 0), x["M"])
        chk(f"G8 fork == attain on {len(old)} stored AEG-HOST-A seed-0 rows", not bad and len(old) == 60, bad[:3])
        chk("G8 seed 0 reads 72/48/40/28/24/0", [gs.get(s, 0) for s in R.SCALES] == [72, 48, 40, 28, 24, 0], gs)
        a = R.run_aeg(8, 0.2, 1, "A", R.TSW["HOST"], R.KVRATE["HOST"], True, R.QMAX)
        b = run_aeg2(8, 0.2, 1, "A", "HOST", True, R.QMAX, ["ref8b"])
        chk("G8 fork full output == R.run_aeg (M 8, s 0.2, seed 1)", a == b, [k for k in a if a.get(k) != b.get(k)])
        a = R.run_aeg(12, 0.15, 2, "H", R.TSW["PAPER"], R.KVRATE["PAPER"], True, R.QMAX)
        b = run_aeg2(12, 0.15, 2, "HF", "PAPER", True, R.QMAX, ["ref8b"])
        chk("G8 fork (HF, PAPER) == R.run_aeg (H, PAPER)", a == b, [k for k in a if a.get(k) != b.get(k)])
        ra, sa = R.run_mux_gpu(2, 0, "A", MUX_BETA["MPS"], arr_end=900.0)
        rb, sb = run_mux2(["ref8b", "ref8b"], 0, "A", MUX_BETA["MPS"], arr_end=900.0)
        chk("G8 MUX fork == R.run_mux_gpu (2 models, 900 s)", sa == sb and all(np.array_equal(x.emit, y.emit) for x, y in zip(ra, rb))
            and len(ra) == len(rb))
    except Exception as ex:      # noqa: BLE001
        chk("G8 equivalence", False, repr(ex))
    # G9 MUX composition
    A1 = [1.0, 0.9, 0.8]
    A2 = {(0, 1): 0.65, (1, 2): 0.45, (2, 0): 0.25}
    chk("G9 M 20 -> 0.74", close(mux_attain_mix(20, 3, A1, A2), 0.74, 1e-12), mux_attain_mix(20, 3, A1, A2))
    chk("G9 M 5 -> 0.92", close(mux_attain_mix(5, 3, A1, A2), 0.92, 1e-12))
    chk("G9 M 33 -> INFEASIBLE", mux_attain_mix(33, 3, A1, A2) is None)
    chk("G9 MIX0 composition == R.mux_attain", all(mux_attain_mix(M, 1, [0.97], {(0, 0): 0.91}) == R.mux_attain(M, 0.97, 0.91)
                                                  for M in (8, 16, 20, 32, 36)))
    # G10 lengths
    try:
        P, O, _ = load_lengths(os.path.join(root, LENGTHS_REL))
        chk("G10 lengths n / means / min / max", len(P) == 82780 and len(O) == 82780 and round(float(P.mean()), 4) == 250.9416
            and round(float(O.mean()), 4) == 231.4567 and (P.min(), O.min(), P.max(), O.max()) == (4, 4, 1024, 2013),
            (len(P), float(P.mean()), float(O.mean())))
        rq = make_requests2(3, 0, "WS", (P, O), arr_end=200.0)
        rq2 = make_requests2(3, 0, "WS", (P, O), arr_end=200.0)
        r0 = R.make_requests(3, 0, 200.0)
        chk("G10 WS keeps 's arrivals, lengths deterministic", [(x.a, x.model) for x in rq] == [(x.a, x.model) for x in r0]
            and [(x.lin, x.lout) for x in rq] == [(x.lin, x.lout) for x in rq2] and any(x.lout != 337 for x in rq))
    except Exception as ex:      # noqa: BLE001
        chk("G10 lengths", False, repr(ex))
    # G11 (added at build time) round overhead with mixed types (Eq. 2's c), hand: tsw = bytes / 32e9
    ty3 = lambda i: MIXES["MIX3"][i % 3]
    T3 = {t: tsw(t, "PAPER") for t in MIXES["MIX3"]}
    chk("G11 c([q7, l13]) = 0.482558208 + 0.81349152", close(round_overhead([0, 2], ty3, T3), 1.296049728, 1e-12),
        round_overhead([0, 2], ty3, T3))
    chk("G11 c([q7, q7', il7]) = 2 x 0.482558208 + 0.483606784", close(round_overhead([0, 3, 1], ty3, T3), 1.4487232, 1e-12),
        round_overhead([0, 3, 1], ty3, T3))
    chk("G11 c(one model) = 0; c([m, m]) = 0", round_overhead([5], ty3, T3) == 0.0 and round_overhead([4, 4], ty3, T3) == 0.0)
    # G12 (added) the §6 verdict classes and ATTRIBUTABLE
    g_ok = {1.0: 72.0, 0.5: 48.0, 0.3: 36.0}
    chk("G12 verdict classes", (verdict_class(0.2, g_ok), verdict_class(0.2, {**g_ok, 1.0: 96.0}), verdict_class(0.15, g_ok),
                                verdict_class(0.3, g_ok), verdict_class("<0.1", g_ok), verdict_class(0.1, g_ok),
                                verdict_class(0.5, g_ok), verdict_class(1.0, g_ok))
        == ("LANDS", "LANDS-CROSSOVER-ONLY", "NEAR", "NEAR", "MISSES-STRONG", "MISSES-STRONG", "MISSES-WEAK", "MISSES-WEAK"))
    chk("G12 shape edge: g(0.3) 32 fails, g(1.0) 64 / 76 pass, 60 / 80 fail",
        verdict_class(0.2, {**g_ok, 0.3: 32.0}) == "LANDS-CROSSOVER-ONLY"
        and verdict_class(0.2, {**g_ok, 1.0: 64.0}) == "LANDS" and verdict_class(0.2, {**g_ok, 1.0: 76.0}) == "LANDS"
        and verdict_class(0.2, {**g_ok, 1.0: 60.0}) == "LANDS-CROSSOVER-ONLY"
        and verdict_class(0.2, {**g_ok, 1.0: 80.0}) == "LANDS-CROSSOVER-ONLY")
    chk("G12 ATTRIBUTABLE: 8 > max(0, 4) yes; 4 > max(4, 0) no", attrib(28, 36, 0, 4)["attributable"]
        and not attrib(28, 32, 4, 0)["attributable"] and attrib(28, 20, 0, 0)["attributable"])
    # G13 (added) the §7.2 trust gate
    chk("G13 trust gate: median 0.15 PASS, 0.20 PASS, 0.25 FAIL, empty FAIL",
        trust_verdict([0.1, 0.3, 0.15])[1] == "PASS" and trust_verdict([0.2, 0.1, 0.3])[1] == "PASS"
        and trust_verdict([0.25, 0.3, 0.1])[1] == "FAIL" and trust_verdict([])[1] == "FAIL")
    # G14 (added) under both readings
    chk("G14 challenger readings", challenger_readings("NEAR", "MISSES-STRONG") == {"strict": "SUPPORTED", "inclusive": "SUPPORTED",
                                                                       "readings_differ": False}
        and challenger_readings("LANDS-CROSSOVER-ONLY", "MISSES-STRONG") == {"strict": "UNRESOLVED", "inclusive": "SUPPORTED",
                                                                       "readings_differ": True}
        and challenger_readings("MISSES-WEAK", "LANDS")["strict"] == "CONTRADICTED"
        and challenger_readings("LANDS", "LANDS-CROSSOVER-ONLY") == {"strict": "UNRESOLVED", "inclusive": "CONTRADICTED",
                                                              "readings_differ": True}
        and challenger_readings("LANDS", "NEAR")["strict"] == "UNRESOLVED")
    n_pass = sum(1 for _, ok, _ in res if ok)
    for name, ok, got in res:
        if not ok:
            print(f"FAIL {name}: {got}")
    print(f"SELF-TEST {'PASS' if n_pass == len(res) else 'FAIL'} {n_pass}/{len(res)}")
    return 0 if n_pass == len(res) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--check-lengths", action="store_true")
    g.add_argument("--map", action="store_true")
    g.add_argument("--b-run", action="store_true")
    g.add_argument("--b-one", action="store_true")
    g.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--out")
    ap.add_argument("--phase", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--lengths")
    ap.add_argument("--spec")
    a = ap.parse_args()
    lpath = a.lengths or os.path.join(a.root, LENGTHS_REL)
    if a.self_test:
        return self_test(a.root)
    if a.check_lengths:
        try:
            P, O, d = load_lengths(lpath)
        except (LookupError, FileNotFoundError, ValueError, OSError) as ex:
            die(str(ex))
        print(json.dumps({"path": lpath, "sha256": LENGTHS_SHA, "n": int(len(P)), "prompt_mean": float(P.mean()),
                          "output_mean": float(O.mean())}))
        return 0
    if a.b_one:
        if not a.spec:
            die("--b-one needs --spec")
        try:
            print(json.dumps(b_one(a.spec, a.root)))
        except (ValueError, KeyError, TypeError) as ex:       # review: a malformed spec is exit 2, never a traceback
            die(f"--b-one spec: {ex!r}")
        return 0
    if a.summarize:
        if not os.path.isdir(a.summarize):
            die(f"no such dir {a.summarize}")
        res = summarize(a.summarize, a.root)
        with open(os.path.join(a.summarize, "summary.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(json.dumps(res, indent=1, default=str))
        return 0
    if not a.out:
        die("--out is required")
    os.makedirs(a.out, exist_ok=True)
    if a.map:
        try:
            m = build_map(a.root)
        except (OSError, KeyError, ValueError) as ex:        # review: missing / corrupt / empty input = exit 2
            die(f"map input: {ex!r}")
        with open(os.path.join(a.out, "map.json"), "w") as fh:
            json.dump(m, fh, indent=1)
        print(json.dumps({"trust_gate": m["trust_gate"], "n_bench": len(m["bench"]), "n_serving": len(m["serving"])}, indent=1))
        return 0
    if a.b_run:
        phases = ["replay", "primary", "ladder", "ours", "desc"] if a.phase == "all" else [x for x in a.phase.split(",") if x]
        if not phases or any(p not in ("replay", "primary", "ladder", "ours", "desc") for p in phases):
            die(f"--phase must be replay|primary|ladder|ours|desc|all (got {a.phase!r})")
        try:
            load_lengths(lpath)
        except (LookupError, FileNotFoundError, OSError) as ex:
            die(str(ex))
        seeds = tuple(int(x) for x in a.seeds.split(",") if x != "")
        print(json.dumps(b_run(a.out, phases, seeds, a.workers, lpath)))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
