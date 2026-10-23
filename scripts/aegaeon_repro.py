#!/usr/bin/env python3
"""aegaeon_repro.py — reproduce two Aegaeon (SOSP'25) results in our own simulation stack.

(a) The active-model count E[m] = M(1 - e^(-lam T)) (Theorem 3.1) against what our cloud sim's own active notion reads on the same
    inputs (CS.busy_matrix, imported unchanged) and against our Chutes trace (World, imported unchanged).
(b) The strict-SLO crossover of Fig. 13: a minimal token-level Aegaeon scheduler (grouped FCFS prefill, MAX_GPSIZE 8, batch 1;
    weighted round-robin decode with Eqs. 2-3, QMAX 4 s, alpha >= 0.5) vs static placement (<= 2 models per GPU), with our measured
    step times, wake times and co-residency beta.

Prereg (frozen before this file existed): (private report). On any divergence the
prereg wins. FORK BY IMPORT: cloud_sim / node_sim / tailpool_sim are imported, never edited or copied.

Output contract:
  --self-test                        exit 0 + "SELF-TEST PASS n/n" (goldens G1-G9 of prereg §8); exit 1 + FAIL lines.
  --a-paper --out DIR [--seeds S]    WRITES DIR/a_paper.json (A-CF, A-I1, A-SIM, A-PRINT); prints it; exit 0.
  --a-trace --out DIR [--seeds S] [--root R] [--cache C]
                                     WRITES DIR/a_trace.json (A-TRACE + A-RAW); prints it; exit 0. --cache is the bins npz the
                                     trace loader writes on first use (it WRITES that file if absent).
  --b-run --out DIR --arms A,B [--seeds S] [--scales X] [--workers N]
                                     APPENDS one JSON row per simulated point to DIR/b_rows.jsonl (AEG) and DIR/mux_rows.jsonl
                                     (MUX); exit 0. Rows already present (same key) are skipped, so a rerun resumes.
  --b-one --spec JSON                read-only: one tiny AEG or MUX config from a JSON string (G7/G8's real entry point); one JSON
                                     object on stdout; exit 0.
  --summarize DIR                    read-only: prereg §5 / §6.5 verdicts over DIR's files; one JSON object on stdout; exit 0.
  Bad input (unknown arm, missing file, bins sha mismatch) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed while building):
  1. **The paper's printed 46.55 is not its formula at its printed inputs** (46.27): the target is the formula; the print is a
     rounding check (prereg §0.7). The self-test asserts the brief's 46.55 FAILS at +-0.01.
  2. **Our cloud sim's "busy" is minute occupancy** (busy_matrix marks whole minutes): it reads M(1 - e^(-lam(60 + T))) on Poisson
     input, about 2x Theorem 3.1 at Aegaeon's operating point. Never quote it as an active-model count without the Delta.
  3. **Two different betas:** the paper's App. A.2 beta (0.625) is a PCIe factor; ours (beta_co) is a co-residency step inflation.
"""
import argparse, collections, gzip, heapq, json, math, os, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PREREG = "data/runs/aegaeon_repro_prereg_2026-10-20.md"
PREREG_COMMIT = "3edb6c6d9"

# ------------------------------------------------------------------------------------------------ frozen inputs (prereg §2-3)
PAPER_M, PAPER_LAM, PAPER_T = 100, 0.037, 16.79
MAX_GPSIZE = 8
QMAX = 4.0
ALPHA_MIN = 0.5
N_PRE, N_DEC = 6, 10
N_GPU = 16
MUX_CAP = 2 * N_GPU
BATCH_CAP = 256
TTFT0, TBT0 = 10.0, 0.1
SCALES = (1.0, 0.5, 0.3, 0.2, 0.15, 0.1)
PAPER_CROSS = 0.2
M_GRID = tuple(range(8, 97, 4))
RPS = 0.1
L_IN, L_OUT = 161, 337
ARR_END, SCORE0, SCORE1, DRAIN = 1500.0, 300.0, 1500.0, 600.0
MUX_GPU_EQ = 16
BAR = 0.90
SEEDS = (0, 1, 2)
HELD_MIN = 442080                   # node_sim.HELD (trace day 307), checked in a_trace
WEEK_S = 7 * 86400.0

# run_h2 gpu_score.json anchor.*.tpot_s (Qwen/Qwen3-8B, real vLLM engine, A100-SXM4-80GB)
ANCH_Q, ANCH_S, ANCH_LC = 0.01134188213045635, 0.026980297906058173, 0.024946615809486025
KB_FIT = (ANCH_S - ANCH_LC) / 6.0
KL_FIT = (ANCH_LC - ANCH_Q - KB_FIT) / 28672.0
A_FIT = ANCH_Q - KB_FIT - 4096.0 * KL_FIT
H_RATIO = 2039.0 / 3350.0           # ASSUMED: HBM spec ratio A100-80GB SXM : H100/H800 SXM
from real_trace_analysis import C_REF, ACTIVE_REF_B   # noqa: E402  (the cloud sim's own constants, imported unchanged)
CH = C_REF * 1.0 * 1.1              # World.Ch at NPHYS0 (cref_mult 1.0, big_mult = m_h 1.1)
PPT = (8.0 / ACTIVE_REF_B) / CH     # prefill seconds per input token, 8 B active (World.service's tail prefill term)
BPT = 147456                        # KV bytes / token, 8 B GQA bf16 (fabric v1 prereg)
MODEL_GB = 16.39                    # 's measured object
TSW = {"PAPER": MODEL_GB / 32.0, "HOST": 0.63, "NVL": 0.060}
KVRATE = {"PAPER": 32.0e9, "HOST": 26.0e9, "NVL": 273.0e9}
BETA = {"MPS": (0.7040124446344402, 0.9891460421887985), "STOCK": (5.7434701780985735, 5.851676424895361), "IDEAL": (0.0, 0.0)}
QMAX_NVLQ = QMAX * TSW["NVL"] / TSW["PAPER"]

AEG_ARMS = {   # name: (hw, tsw key, prefetch, qmax)
    "AEG-PAPER": ("H", "PAPER", True, QMAX),
    "AEG-HOST-A": ("A", "HOST", True, QMAX),
    "AEG-NVL-A": ("A", "NVL", True, QMAX),
    "AEG-NVLQ-A": ("A", "NVL", True, QMAX_NVLQ),
    "AEG-HOST-H": ("H", "HOST", True, QMAX),
    "AEG-NVL-H": ("H", "NVL", True, QMAX),
    "AEG-HOST-A-NOPF": ("A", "HOST", False, QMAX),
    "AEG-NVL-A-NOPF": ("A", "NVL", False, QMAX),
}
MUX_ARMS = {}   # name: (hw, beta)
for _hw in ("H", "A"):
    for _k, (_lo, _hi) in BETA.items():
        MUX_ARMS[f"MUX-{_k}-{_hw}"] = (_hw, _lo)
        if _hi != _lo:
            MUX_ARMS[f"MUX-{_k}-{_hw}-EDGE"] = (_hw, _hi)


def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------ (a) closed forms + counters
def closed_form(M, lam, T):
    if not (lam > 0) or T < 0 or M < 0:
        raise ValueError(f"closed_form needs lam > 0, T >= 0, M >= 0 (got {lam}, {T}, {M})")
    return M * (1.0 - math.exp(-lam * T))


def minute_form(M, lam, T, delta=60.0):
    if not (lam > 0) or T < 0 or T >= delta:
        raise ValueError("minute_form needs lam > 0 and 0 <= T < delta")
    return M * (1.0 - math.exp(-lam * (delta + T)))


def union_active(a, e, model, w0, w1):
    """E[m]_cont: time-average over [w0, w1) of the number of models with some interval [a, e) covering t (prereg §4)."""
    a = np.clip(np.asarray(a, float), w0, w1)
    e = np.clip(np.asarray(e, float), w0, w1)
    model = np.asarray(model, np.int64)
    if not len(a):
        return 0.0
    if np.any(e < a):
        raise ValueError("an interval ends before it starts")
    big = (w1 - w0) * 4.0 + 1.0
    off = (model.astype(float)) * big
    a2, e2 = a - w0 + off, e - w0 + off
    o = np.lexsort((a2, model))
    a2, e2 = a2[o], e2[o]
    cm = np.maximum.accumulate(e2)
    prev = np.concatenate(([-np.inf], cm[:-1]))
    cov = np.maximum(0.0, e2 - np.maximum(a2, prev))
    return float(cov.sum() / (w1 - w0))


def poisson_arrivals(M, lam, H, seed, stream=0):
    if not (lam > 0) or H <= 0:
        raise ValueError("poisson_arrivals needs lam > 0 and H > 0")
    rng = np.random.default_rng([seed, 2957, stream])
    n = rng.poisson(lam * H, size=M)
    model = np.repeat(np.arange(M), n)
    a = rng.random(int(n.sum())) * H
    return model, a


def a_paper(seeds):
    import cloud_sim as CS     # noqa: imported unchanged
    M, lam, T, H, W0 = PAPER_M, PAPER_LAM, PAPER_T, 100000.0, 100.0
    out = {"closed_form": closed_form(M, lam, T), "minute_form": minute_form(M, lam, T),
           "print": {"printed": 46.55, "lam_lo": closed_form(M, 0.0365, T), "lam_hi": closed_form(M, 0.0375, T),
                     "lam_for_print": -math.log(1 - 0.4655) / T, "T_for_print": -math.log(1 - 0.4655) / lam}}
    out["print"]["verdict"] = ("CONSISTENT-WITH-ROUNDING" if out["print"]["lam_lo"] <= 46.55 <= out["print"]["lam_hi"]
                               else "INCONSISTENT")
    rows = []
    for sd in seeds:
        model, a = poisson_arrivals(M, lam, H, sd, stream=1)
        det = union_active(a, a + T, model, W0, H)
        rng = np.random.default_rng([sd, 2957, 2])
        ex = union_active(a, a + rng.exponential(T, len(a)), model, W0, H)
        n_min = int(H // 60)
        mi = np.floor(a / 60.0).astype(np.int64)
        k = mi < n_min
        mi, mo, off = mi[k], model[k], (a[k] - 60.0 * mi[k])
        key = mo * (n_min + 1) + mi
        o = np.lexsort((off, key))
        key_s, off_s = key[o], off[o]
        last_idx = np.r_[np.nonzero(np.diff(key_s))[0], len(key_s) - 1]
        ck = key_s[last_idx]
        B = CS.busy_matrix(ck % (n_min + 1), off_s[last_idx], np.full(len(ck), T), ck // (n_min + 1), M, n_min)
        m0 = int(math.ceil(W0 / 60.0))
        emin = float(B.sum(axis=0)[m0:].mean())
        rows.append({"seed": sd, "n_arrivals": int(len(a)), "E_cont_det": det, "E_cont_exp": ex, "E_min": emin,
                     "n_cells": int(len(ck))})
    det = [r["E_cont_det"] for r in rows]; ex = [r["E_cont_exp"] for r in rows]; em = [r["E_min"] for r in rows]
    cf, mf = out["closed_form"], out["minute_form"]
    i1 = abs(np.mean(det) - cf) <= 0.5 and abs(np.mean(ex) - cf) <= 0.5
    if abs(np.mean(em) - cf) <= 1.0:
        sim_v = "REPRODUCES"
    elif abs(np.mean(em) - mf) <= 1.0:
        sim_v = "QUANTIZED"
    else:
        sim_v = "UNEXPLAINED"
    head = ("REPRODUCES" if sim_v == "REPRODUCES" else
            "NOT COMPARABLE AS-IS; REPRODUCES AT DELTA -> 0" if (sim_v == "QUANTIZED" and i1) else
            "VOID (A-I1 FAIL)" if not i1 else "UNEXPLAINED")
    out.update(seeds=list(seeds), horizon_s=H, window=[W0, H], rows=rows,
               A_I1={"mean_det": float(np.mean(det)), "mean_exp": float(np.mean(ex)), "spread_det": [min(det), max(det)],
                     "spread_exp": [min(ex), max(ex)], "verdict": "PASS" if i1 else "FAIL"},
               A_SIM={"mean": float(np.mean(em)), "spread": [min(em), max(em)], "ratio_to_theorem": float(np.mean(em)) / cf,
                      "verdict": sim_v},
               headline_a=head)
    return out


def load_world(root, cache):
    import node_sim as NS        # noqa: imported unchanged
    import cloud_sim as CS       # noqa: imported unchanged
    return CS.World(dict(NS.NPHYS0), dict(CS.EXTRA0), root, cache), CS, NS


def trace_arrays(W, seed):
    """held-out cells of World: (k, abs arrival times per request, model per request, service per request, cell index per request)."""
    import tailpool_sim as T     # noqa: imported unchanged
    tr = W.cx["tr"]
    k, last_k = W.arrivals(seed)
    first, last, _ = T.arrivals(tr, seed)
    if not np.allclose(last[k], last_k):
        raise RuntimeError("World.arrivals and tailpool.arrivals disagree")
    mid = tr.mid[k]
    isg = np.zeros(W.cx["M"], bool)
    isg[list(W.mg)] = True
    s_cell = np.where(isg[mid], W.service("giant"), W.service("tail"))
    n = tr.n[k].astype(np.int64)
    tot = int(n.sum())
    a = np.empty(tot); mo = np.empty(tot, np.int64); s = np.empty(tot); ci = np.empty(tot, np.int64)
    p = 0
    base = (tr.minute[k] - HELD_MIN) * 60.0
    for j, i in enumerate(k):
        offs = T.cell_times(tr, int(i), first, last, seed)
        q = len(offs)
        a[p:p + q] = base[j] + offs
        mo[p:p + q] = mid[j]; s[p:p + q] = s_cell[j]; ci[p:p + q] = j
        p += q
    if p != tot:
        raise RuntimeError(f"offsets {p} != requests {tot}")
    return k, a, mo, s, ci, s_cell


def a_trace(root, cache, seeds):
    W, CS, NS = load_world(root, cache)
    if NS.HELD != HELD_MIN or CS.HELD != HELD_MIN:
        raise RuntimeError("HELD minute moved")
    M = W.cx["M"]
    WEEK = WEEK_S
    rows = []
    het = hom = None
    raw = None
    for sd in seeds:
        k, a, mo, s, ci, s_cell = trace_arrays(W, sd)
        econt = union_active(a, a + s, mo, 0.0, WEEK)
        B = W.busy(sd)
        emin = float(B.sum(axis=0).mean())
        if het is None:
            n_i = np.bincount(mo, minlength=M).astype(float)
            sw_i = np.bincount(mo, weights=s, minlength=M)
            lam_i = n_i / WEEK
            T_i = np.divide(sw_i, n_i, out=np.zeros(M), where=n_i > 0)
            het = float(np.sum(1.0 - np.exp(-lam_i * T_i)))
            lam_bar = n_i.sum() / (M * WEEK)
            T_bar = float(sw_i.sum() / n_i.sum())
            hom = closed_form(M, lam_bar, T_bar)
            m_obs = int((n_i > 0).sum())
            desc = {"M": M, "M_obs": m_obs, "n_req": int(n_i.sum()), "lam_bar": lam_bar, "T_bar": T_bar,
                    "T_p50": float(np.median(s)), "T_p90": float(np.quantile(s, 0.9)),
                    "lam_i_max": float(lam_i.max()), "n_models_lamT_gt_1": int((lam_i * T_i > 1).sum())}
            if int(n_i.sum()) != 3687246:
                raise RuntimeError("held-out request count != 3,687,246")
        rows.append({"seed": sd, "E_cont": econt, "E_min": emin})
        if raw is None:
            raw = a_raw(root, W, sd, k, a, mo, s, s_cell)
    ec = [r["E_cont"] for r in rows]; em = [r["E_min"] for r in rows]
    mean_c = float(np.mean(ec))
    def rel(x, y):
        return (x - y) / y
    v_het = ("THEOREM-HOLDS-ON-TRACE" if abs(rel(mean_c, het)) <= 0.10 else
             ("POISSON-BREAKS-LOW" if mean_c < het else "POISSON-BREAKS-HIGH"))
    v_hom = "HOM-HOLDS" if abs(rel(mean_c, hom)) <= 0.10 else ("HOM-BREAKS-LOW" if mean_c < hom else "HOM-BREAKS-HIGH")
    return {"seeds": list(seeds), "window_s": WEEK, "desc": desc, "rows": rows, "HET": het, "HOM": hom,
            "E_cont_mean": mean_c, "E_cont_spread": [min(ec), max(ec)], "E_min_mean": float(np.mean(em)),
            "E_min_spread": [min(em), max(em)], "ratio_cont_to_HET": mean_c / het, "ratio_cont_to_HOM": mean_c / hom,
            "ratio_min_to_cont": float(np.mean(em)) / mean_c, "verdict_HET": v_het, "verdict_HOM": v_hom, "A_RAW": raw}


def a_raw(root, W, seed, k, a, mo, s, s_cell):
    """A-RAW (descriptive): the 9 raw-us chutes, continuous active from raw timestamps vs from the synthetic offsets.
    I-RAW: per model, the raw held-out request count must equal the bins' (I-BINS, re-checked here)."""
    path = os.path.join(root, "data/runs/nowait_peer_read_2026-10-20/p2/req_307_314.csv.gz")
    if not os.path.exists(path):
        return {"error": "raw file missing"}
    tr = W.cx["tr"]
    cid_m = W.cx["cid_m"]
    t0_us, t1_us = 307 * 86400 * 10**6, 314 * 86400 * 10**6
    rc, ru = [], []
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            c, u = line.rstrip("\n").split(",")
            u = int(u)
            if t0_us <= u < t1_us:
                rc.append(c); ru.append(u)
    mids = {c: cid_m[c] for c in sorted(set(rc)) if c in cid_m}
    mset = set(mids.values())
    cell_min = (tr.minute[k] - HELD_MIN)
    cell_key = {}
    for j in range(len(k)):
        m = int(tr.mid[k[j]])
        if m in mset:
            cell_key[(m, int(cell_min[j]))] = j
    ra, rmo, rs = [], [], []
    no_cell = 0
    for c, u in zip(rc, ru):
        if c not in mids:
            continue
        m = mids[c]
        t = u / 1e6 - 307 * 86400.0
        j = cell_key.get((m, int(t // 60)))
        if j is None:
            no_cell += 1
            continue
        ra.append(t); rmo.append(m); rs.append(float(s_cell[j]))
    ra, rmo, rs = np.array(ra), np.array(rmo, np.int64), np.array(rs)
    per = {}
    count_bad = 0
    for c, m in mids.items():
        r_ = rmo == m
        s_ = mo == m
        n_bins = int(s_.sum())
        if int(r_.sum()) != n_bins:
            count_bad += 1
        per[c] = {"model": int(m), "n_raw": int(r_.sum()), "n_bins": n_bins,
                  "raw": union_active(ra[r_], ra[r_] + rs[r_], np.zeros(int(r_.sum()), np.int64), 0.0, WEEK_S),
                  "synthetic": union_active(a[s_], a[s_] + s[s_], np.zeros(n_bins, np.int64), 0.0, WEEK_S)}
    tot_raw = sum(v["raw"] for v in per.values()); tot_syn = sum(v["synthetic"] for v in per.values())
    return {"seed": seed, "n_chutes": len(mids), "n_raw_in_week": int(len(ra)), "raw_rows_without_cell": no_cell,
            "I_RAW_models_count_mismatch": count_bad, "sum_raw": tot_raw, "sum_synthetic": tot_syn,
            "ratio_raw_to_synthetic": (tot_raw / tot_syn) if tot_syn else None, "per_chute": per}


# ------------------------------------------------------------------------------------------------ (b) shared pieces
def step_time(hw, b, ctx):
    t = A_FIT + KB_FIT * b + KL_FIT * ctx
    return t * H_RATIO if hw == "H" else t


def t_pre(L):
    return L * PPT


def count_met(T0, t, J, a, kf, TT, TB):
    """met tokens among J decode steps starting at T0 (step t), first emitted token index kf (prereg §4 deadlines)."""
    if J < 0 or t <= 0 or TB <= 0:
        raise ValueError("count_met needs J >= 0, t > 0, TB > 0")
    if J == 0:
        return 0
    sig = a + TT + (kf - 2) * TB - T0
    g = TB - t
    eps = 1e-9
    if abs(g) <= 1e-15:
        return J if sig >= -eps else 0
    if g > 0:
        jmin = max(1, math.ceil(-sig / g - eps))
        return max(0, J - jmin + 1)
    jmax = min(J, math.floor(sig / (-g) + eps))
    return max(0, jmax)


def eq23(ts, tbt, c, qmax):
    """Eqs. 2-3: per-batch quotas for step times ts, TBT tbt, summed switch overhead c. Returns (alpha, [q])."""
    if tbt <= 0 or any(t <= 0 for t in ts):
        raise ValueError("eq23 needs positive TBT and step times")
    n = [tbt / t for t in ts]
    S = sum(1.0 / x for x in n)
    alpha = max(c / (min(n) * qmax) + S, ALPHA_MIN)
    if c <= 0:
        return alpha, None
    return alpha, [c / (nk * (alpha - S)) for nk in n]


def make_requests(M, seed, arr_end=ARR_END, lin=L_IN, lout=L_OUT, stream_base=0):
    reqs = []
    for i in range(M):
        rng = np.random.default_rng([seed, 2957, stream_base + i])
        t = 0.0
        while True:
            t += rng.exponential(1.0 / RPS)
            if t >= arr_end:
                break
            reqs.append([t, i])
    reqs.sort()
    return [Req(j, m, a, lin, lout) for j, (a, m) in enumerate(reqs)]


class Req:
    __slots__ = ("id", "model", "a", "lin", "lout", "k", "met", "done", "fin", "e1", "emit", "met1", "ready", "fd")

    def __init__(self, i, model, a, lin, lout):
        self.id, self.model, self.a, self.lin, self.lout = i, model, a, lin, lout
        self.k = 0
        self.met = 0
        self.done = False
        self.fin = None
        self.e1 = None
        self.emit = None
        self.met1 = 0           # diagnostics only: first token met (0/1), decode-ready time, first decode token time
        self.ready = None
        self.fd = None


# ------------------------------------------------------------------------------------------------ (b) AEG
class Group:
    __slots__ = ("model", "acc", "pending", "exec_start")

    def __init__(self, model):
        self.model, self.acc, self.pending, self.exec_start = model, 0, collections.deque(), None


class PInst:
    def __init__(self, i):
        self.i, self.loaded, self.queue, self.busy, self.cur, self.prev_exec_start, self.busy_until = i, None, [], False, None, None, 0.0


class Batch:
    __slots__ = ("model", "reqs", "joiners", "q", "dinst")

    def __init__(self, model, d):
        self.model, self.reqs, self.joiners, self.q, self.dinst = model, [], [], None, d


class DInst:
    def __init__(self, i):
        self.i, self.loaded, self.work = i, None, []
        self.round, self.ridx, self.c0 = [], 0, False
        self.cur, self.turn_start, self.turn_q, self.steps = None, 0.0, 0.0, 0
        self.seg = None          # (start, t, J)
        self.ver = 0
        self.idle = True
        self.prev_turn = None
        self.end_after_seg = False


def run_aeg(M, scale, seed, hw, tsw, kvrate, pf, qmax, n_pre=N_PRE, n_dec=N_DEC, reqs=None, arr_end=ARR_END,
            score=(SCORE0, SCORE1), drain=DRAIN):
    TT, TB = TTFT0 * scale, TBT0 * scale
    if TB <= 0 or TT <= 0:
        raise ValueError("SLO must be positive")
    if reqs is None:
        reqs = make_requests(M, seed, arr_end)
    t_stop = score[1] + drain
    P = [PInst(i) for i in range(n_pre)]
    D = [DInst(i) for i in range(n_dec)]
    ev = []
    seq = [0]
    stats = collections.Counter()

    def push(t, kind, obj, ver=0):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, obj, ver))

    for r in reqs:
        push(r.a, "ARR", r)

    # ---------------- prefill
    def p_load(p, now):
        load = max(0.0, p.busy_until - now) if p.busy else 0.0
        prev = p.loaded
        for g in p.queue:
            if g.model != prev and g is not p.cur:
                load += tsw
            load += t_pre(L_IN) * len(g.pending)
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
            if pf and p.prev_exec_start is not None:
                cost = max(0.0, tsw - (now - p.prev_exec_start))
            else:
                cost = tsw
            p.loaded = g.model
            stats["p_switch"] += 1
            stats["p_switch_s"] += cost
        start = now + cost
        if g.exec_start is None:
            g.exec_start = start
        end = start + t_pre(r.lin)
        p.busy, p.cur, p.busy_until = True, g, end
        push(end, "PFREE", (p, r))

    def on_arrival(r, now):
        for p in P:
            for g in p.queue:
                if g.model == r.model and g.acc < MAX_GPSIZE:
                    g.acc += 1
                    g.pending.append(r)
                    if not p.busy:
                        p_start(p, now)
                    return
        best = min(P, key=lambda p: (p_load(p, now), p.i))
        g = Group(r.model)
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
            r.ready = now + r.lin * BPT / kvrate
            push(r.ready, "READY", r)
        p.busy = False
        p_start(p, now)

    # ---------------- decode
    def d_find_batch(model):
        for d in D:
            for b in d.work:
                if b.model == model and len(b.reqs) + len(b.joiners) < BATCH_CAP:
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
        b = Batch(r.model, d)
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
        ts = [step_time(hw, len(b.reqs), sum(x.lin + x.k for x in b.reqs)) for b in order]
        K = len(order)
        trans = sum(1 for i in range(K) if order[i].model != order[i - 1].model) if K > 1 else 0
        c = trans * tsw
        alpha, qs = eq23(ts, TB, c, qmax)
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
                    if pf and d.prev_turn is not None:
                        cost = max(0.0, tsw - d.prev_turn)
                    else:
                        cost = tsw
                    d.loaded = b.model
                    stats["d_switch"] += 1
                    stats["d_switch_s"] += cost
                d.cur, d.turn_start, d.turn_q, d.steps = b, now + cost, b.q, 0
                if cost > 0:
                    d.ver += 1
                    push(now + cost, "DWAKE", d, d.ver)
                    return
                continue
            b = d.cur
            if d.seg is not None:     # a segment just ended at `now`: account it
                st, t, J = d.seg
                d.seg = None
                fin = []
                for r in b.reqs:
                    r.met += count_met(st, t, J, r.a, r.k + 1, TT, TB)
                    if r.k == 1 and J > 0:
                        r.fd = st + t
                    r.k += J
                    if r.k >= r.lout:
                        r.done, r.fin = True, st + J * t
                        fin.append(r)
                if fin:
                    b.reqs = [r for r in b.reqs if not r.done]
                d.steps += J
                stats["dec_steps"] += J                       # I-WORK diagnostics (no effect on the schedule)
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
            t = step_time(hw, len(b.reqs), sum(r.lin + r.k for r in b.reqs))
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
    return score_reqs(reqs, score, stats)


def score_reqs(reqs, score, stats=None):
    met = tok = unfin = n = 0
    for r in reqs:
        if score[0] <= r.a < score[1]:
            n += 1
            met += r.met
            tok += r.lout
            if not r.done:
                unfin += 1
    out = {"n_req": n, "tokens": tok, "met": met, "attain": (met / tok) if tok else None, "unfinished": unfin}
    sc = [r for r in reqs if score[0] <= r.a < score[1]]
    ttft = np.array([r.e1 - r.a for r in sc if r.e1 is not None])
    wait = np.array([r.fd - r.ready for r in sc if r.fd is not None and r.ready is not None])
    if len(ttft):
        out["diag"] = {"ttft_p50": float(np.median(ttft)), "ttft_p90": float(np.quantile(ttft, 0.9)),
                       "first_token_met": sum(r.met1 for r in sc) / len(sc),
                       "decode_met": ((met - sum(r.met1 for r in sc)) / (tok - len(sc))) if tok > len(sc) else None,
                       "wait_first_decode_p50": float(np.median(wait)) if len(wait) else None,
                       "wait_first_decode_p90": float(np.quantile(wait, 0.9)) if len(wait) else None}
    if stats:
        out["stats"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}
    return out


# ------------------------------------------------------------------------------------------------ (b) MUX (static)
def run_mux_gpu(n_models, seed, hw, beta, arr_end=None, reqs=None, score=None, drain=DRAIN):
    """one GPU with n_models (1 or 2) co-resident engines; per-step DES. Returns per-request emission times for scoring."""
    if n_models not in (1, 2):
        raise ValueError("a MUX GPU holds 1 or 2 models")
    if arr_end is None:
        arr_end = SCORE0 + MUX_GPU_EQ * (SCORE1 - SCORE0)
    if score is None:
        score = (SCORE0, arr_end)
    if reqs is None:
        reqs = make_requests(n_models, seed, arr_end, stream_base=100000)
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
        return bool(e["wait"]) or bool(e["run"]) or e["in_step"]   # a request inside a prefill step is running (prereg §4)

    def start(i, now):
        e = eng[i]
        if e["in_step"] or not has_work(e):
            return
        co = any(has_work(eng[j]) for j in range(n_models) if j != i)
        if e["wait"]:
            batch = list(e["wait"]); e["wait"].clear()
            dur = sum(t_pre(r.lin) for r in batch)
            kind = "PRE"
        else:
            batch = list(e["run"])
            dur = step_time(hw, len(batch), sum(r.lin + r.k for r in batch))
            kind = "DEC"
        if co:
            dur *= (1.0 + beta)
        e["in_step"], e["step"] = True, (kind, batch)
        push(now + dur, "END", i)

    while ev:
        t = ev[0][0]
        if t > t_stop:
            break
        # every event at this timestamp is applied before any step starts (order-independent co-activity)
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


def score_mux(reqs, score, scales=SCALES):
    out = {}
    for s in scales:
        TT, TB = TTFT0 * s, TBT0 * s
        met = tok = 0
        for r in reqs:
            if score[0] <= r.a < score[1]:
                dl = r.a + TT + np.arange(r.lout) * TB
                met += int(np.sum(r.emit <= dl + 1e-9))
                tok += r.lout
        out[s] = {"met": met, "tokens": tok, "attain": met / tok if tok else None}
    return out


def mux_attain(M, A1, A2):
    if M > MUX_CAP:
        return None            # INFEASIBLE
    if M <= N_GPU:
        return A1
    return (2 * (M - N_GPU) * A2 + (MUX_CAP - M) * A1) / M


def mux_goodput(A1, A2):
    g, fails = 0, 0
    for M in M_GRID:
        at = mux_attain(M, A1, A2)
        ok = at is not None and at >= BAR
        if ok:
            g, fails = M, 0
        else:
            fails += 1
            if fails >= 2:
                break
    return g


# ------------------------------------------------------------------------------------------------ (b) grid driver
def aeg_scan(args):
    arm, s, seed = args
    hw, tk, pf, qm = AEG_ARMS[arm]
    rows, g, fails = [], 0, 0
    for M in M_GRID:
        t0 = time.time()
        r = run_aeg(M, s, seed, hw, TSW[tk], KVRATE[tk], pf, qm)
        ok = r["attain"] is not None and r["attain"] >= BAR
        rows.append(dict(arm=arm, scale=s, seed=seed, M=M, ok=ok, wall_s=round(time.time() - t0, 2), **r))
        if ok:
            g, fails = M, 0
        else:
            fails += 1
            if fails >= 2:
                break
    return rows


def mux_one(args):
    arm, seed, nm = args
    hw, beta = MUX_ARMS[arm]
    t0 = time.time()
    reqs, score = run_mux_gpu(nm, seed, hw, beta)
    sc = score_mux(reqs, score)
    return dict(arm=arm, seed=seed, n_models=nm, beta=beta, hw=hw, wall_s=round(time.time() - t0, 2),
                n_req=sum(1 for r in reqs if score[0] <= r.a < score[1]),
                unfinished=sum(1 for r in reqs if score[0] <= r.a < score[1] and not r.done),
                attain={str(k): v["attain"] for k, v in sc.items()})


def b_run(out, arms, seeds, scales, workers):
    import multiprocessing as mp
    os.makedirs(out, exist_ok=True)
    bp, mp_ = os.path.join(out, "b_rows.jsonl"), os.path.join(out, "mux_rows.jsonl")
    have = set()
    for p in (bp,):
        if os.path.exists(p):
            for line in open(p):
                x = json.loads(line)
                have.add((x["arm"], x["scale"], x["seed"]))
    have_m = set()
    if os.path.exists(mp_):
        for line in open(mp_):
            x = json.loads(line)
            have_m.add((x["arm"], x["seed"], x["n_models"]))
    jobs = [(a, s, sd) for a in arms if a in AEG_ARMS for s in scales for sd in seeds if (a, s, sd) not in have]
    mjobs = [(a, sd, nm) for a in arms if a in MUX_ARMS for sd in seeds for nm in (1, 2) if (a, sd, nm) not in have_m]
    bad = [a for a in arms if a not in AEG_ARMS and a not in MUX_ARMS]
    if bad:
        die(f"unknown arm(s) {bad}")
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(mux_one, mjobs):
            res.update(prereg=PREREG_COMMIT)
            with open(mp_, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(json.dumps({"mux": res["arm"], "seed": res["seed"], "n": res["n_models"], "wall": res["wall_s"]}), flush=True)
        for rows in pool.imap_unordered(aeg_scan, jobs):
            with open(bp, "a") as fh:
                for r in rows:
                    r["prereg"] = PREREG_COMMIT
                    fh.write(json.dumps(r) + "\n")
            last = rows[-1]
            print(json.dumps({"aeg": last["arm"], "s": last["scale"], "seed": last["seed"], "last_M": last["M"],
                              "wall": round(sum(r["wall_s"] for r in rows), 1)}), flush=True)
    return {"aeg_jobs": len(jobs), "mux_jobs": len(mjobs)}


# ------------------------------------------------------------------------------------------------ summarize (prereg §5, §6.5)
def goodput_from_rows(rows):
    g = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["scale"], r["seed"])].append(r)
    nonmono = []
    for key, rs in by.items():
        rs.sort(key=lambda x: x["M"])
        best, fails, seen_fail = 0, 0, False
        for r in rs:
            if r["ok"]:
                if seen_fail:
                    nonmono.append(list(key) + [r["M"]])
                best, fails = r["M"], 0
            else:
                seen_fail = True
                fails += 1
        g[key] = best
    return g, nonmono


def crossover(gA, gM):
    """gA, gM: {scale: goodput}. first scale (1.0 -> 0.1) where AEG does not outperform."""
    c, nonmono = None, False
    for s in SCALES:
        if s not in gA or s not in gM:
            return None, None
        if gA[s] <= gM[s]:
            if c is None:
                c = s
        elif c is not None:
            nonmono = True
    return (c if c is not None else "<0.1"), nonmono


def summarize(out):
    bp, mp_ = os.path.join(out, "b_rows.jsonl"), os.path.join(out, "mux_rows.jsonl")
    res = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT}
    for f in ("a_paper.json", "a_trace.json"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            res[f.split(".")[0]] = json.load(open(p))
    if not os.path.exists(bp) or not os.path.exists(mp_):
        return res
    rows = [json.loads(x) for x in open(bp)]
    mrows = [json.loads(x) for x in open(mp_)]
    g, nonmono = goodput_from_rows(rows)
    agg = collections.defaultdict(dict)
    spread = collections.defaultdict(dict)
    per_seed = collections.defaultdict(lambda: collections.defaultdict(dict))
    arms = sorted({k[0] for k in g})
    for a in arms:
        for s in SCALES:
            vals = [g[(a, s, sd)] for sd in SEEDS if (a, s, sd) in g]
            if len(vals) == len(SEEDS):
                agg[a][s] = float(np.median(vals))
                spread[a][s] = max(vals) - min(vals)
            for sd in SEEDS:
                if (a, s, sd) in g:
                    per_seed[a][sd][s] = g[(a, s, sd)]
    # MUX goodput per arm / seed / scale (A1 from the arm's 1-model GPU = the IDEAL-equivalent solo run of the same HW)
    m = {(r["arm"], r["seed"], r["n_models"]): r for r in mrows}
    mux_g = collections.defaultdict(dict)
    mux_seed = collections.defaultdict(lambda: collections.defaultdict(dict))
    for a, (hw, beta) in MUX_ARMS.items():
        for s in SCALES:
            vals = []
            for sd in SEEDS:
                r1, r2 = m.get((a, sd, 1)), m.get((a, sd, 2))
                if r1 is None or r2 is None:
                    continue
                gg = mux_goodput(r1["attain"][str(s)], r2["attain"][str(s)])
                vals.append(gg)
                mux_seed[a][sd][s] = gg
            if len(vals) == len(SEEDS):
                mux_g[a][s] = float(np.median(vals))
    res["aeg_goodput_median"] = {a: {str(s): v for s, v in d.items()} for a, d in agg.items()}
    res["aeg_goodput_spread"] = {a: {str(s): v for s, v in d.items()} for a, d in spread.items()}
    res["aeg_goodput_per_seed"] = {a: {str(sd): {str(s): v for s, v in d.items()} for sd, d in dd.items()}
                                   for a, dd in per_seed.items()}
    res["mux_goodput_median"] = {a: {str(s): v for s, v in d.items()} for a, d in mux_g.items()}
    res["nonmonotone_scans"] = nonmono
    # I-MUX1: the 1-model GPU never co-runs, so its attainment must be identical across beta arms of one HW and seed
    i_mux1 = True
    for hw in ("H", "A"):
        for sd in SEEDS:
            vals = {json.dumps(r["attain"], sort_keys=True) for (a_, sd_, nm_), r in m.items()
                    if nm_ == 1 and sd_ == sd and MUX_ARMS.get(a_, ("?",))[0] == hw}
            if len(vals) > 1:
                i_mux1 = False
    res["I_MUX1"] = "PASS" if i_mux1 else "FAIL"
    pairs = {}
    for aa in arms:
        for mm in mux_g:
            if AEG_ARMS[aa][0] != MUX_ARMS[mm][0]:
                continue
            c, nm = crossover(agg[aa], mux_g[mm])
            if c is None:
                continue
            seeds_c = []
            for sd in SEEDS:
                cs, _ = crossover(per_seed[aa][sd], mux_seed[mm][sd]) if sd in per_seed[aa] and sd in mux_seed[mm] else (None, None)
                seeds_c.append(cs)
            pairs[f"{aa} vs {mm}"] = {"crossover": c, "nonmonotone": nm, "per_seed": seeds_c}
    res["pairs"] = pairs

    def step_index(c):
        return len(SCALES) if c == "<0.1" else SCALES.index(c)

    rp = pairs.get("AEG-PAPER vs MUX-MPS-H")
    if rp:
        idx = [step_index(c) for c in rp["per_seed"] if c is not None]
        unstable = (max(idx) - min(idx) > 1) if idx else None
        rep = rp["crossover"] in (0.3, 0.2, 0.15)
        direction = None
        if not rep:
            direction = "LOOSER" if (rp["crossover"] != "<0.1" and rp["crossover"] >= 0.5) else "STRICTER"
        ga = agg.get("AEG-PAPER", {})
        res["b_reproduction"] = {"pair": "AEG-PAPER vs MUX-MPS-H", "crossover": rp["crossover"], "paper": PAPER_CROSS,
                                 "verdict": "REPRODUCES" if rep else f"DOES-NOT-REPRODUCE ({direction})",
                                 "unstable": unstable, "per_seed": rp["per_seed"],
                                 "shape": {"g_1.0_vs_70": ga.get(1.0), "g_0.5_ge_48": (ga.get(0.5, 0) >= 48) if 0.5 in ga else None,
                                           "g_0.3_ge_36": (ga.get(0.3, 0) >= 36) if 0.3 in ga else None}}
    nv = {}
    for base in ("MUX-MPS-A", "MUX-STOCK-A"):
        h = pairs.get(f"AEG-HOST-A vs {base}")
        for tier in ("AEG-NVL-A", "AEG-NVLQ-A"):
            n_ = pairs.get(f"{tier} vs {base}")
            if not h or not n_:
                continue
            ch, cn = h["crossover"], n_["crossover"]
            moved = step_index(cn) - step_index(ch) >= 1
            verdict = "DOES NOT MOVE IT"
            delta = sprd = None
            if ch != "<0.1":
                delta = agg[tier].get(ch, 0) - agg["AEG-HOST-A"].get(ch, 0)
                sprd = spread["AEG-HOST-A"].get(ch)
            if moved:
                verdict = "MOVES IT" if (delta is not None and sprd is not None and delta > sprd) else "UNATTRIBUTABLE"
            nv[f"{tier} vs AEG-HOST-A ({base})"] = {"host_cross": ch, "tier_cross": cn, "goodput_delta_at_host_cross": delta,
                                                    "host_spread_at_host_cross": sprd, "verdict": verdict}
    res["nvlink"] = nv
    return res


# ------------------------------------------------------------------------------------------------ --b-one (G7/G8 real entry point)
def b_one(spec):
    sp = json.loads(spec)
    reqs = [Req(i, int(m), float(a), int(sp.get("lin", L_IN)), int(sp.get("lout", L_OUT))) for i, (a, m) in enumerate(sp["reqs"])]
    if sp["policy"] == "aeg":
        r = run_aeg(len({x.model for x in reqs}), float(sp["scale"]), 0, sp["hw"], float(sp["tsw"]), float(sp["kvrate"]),
                    bool(sp.get("pf", True)), float(sp.get("qmax", QMAX)), n_pre=int(sp.get("n_pre", 1)),
                    n_dec=int(sp.get("n_dec", 1)), reqs=reqs, score=(-1.0, 1e9), drain=float(sp.get("drain", 60.0)))
        r["emit"] = [{"id": x.id, "e1": x.e1, "fin": x.fin, "met": x.met} for x in reqs]
        return r
    if sp["policy"] == "mux":
        reqs, score = run_mux_gpu(int(sp["n_models"]), 0, sp["hw"], float(sp["beta"]), reqs=reqs, score=(-1.0, 1e9),
                                  drain=float(sp.get("drain", 60.0)))
        return {"emit": [[float(v) for v in x.emit] for x in reqs],
                "attain": {str(k): v["attain"] for k, v in score_mux(reqs, score).items()}}
    raise ValueError("policy must be aeg or mux")


# ------------------------------------------------------------------------------------------------ self-test (prereg §8)
def self_test():
    res = []

    def chk(name, ok, got=None):
        res.append((name, bool(ok), got))

    # G1
    chk("G1 closed form 46.271683", abs(closed_form(100, 0.037, 16.79) - 46.271683) <= 1e-5, closed_form(100, 0.037, 16.79))
    chk("G1 lam 0.0365 / 0.0375", abs(closed_form(100, 0.0365, 16.79) - 45.818735) <= 1e-5
        and abs(closed_form(100, 0.0375, 16.79) - 46.720844) <= 1e-5)
    chk("G1 minute form 94.164615", abs(minute_form(100, 0.037, 16.79) - 94.164615) <= 1e-5)
    # G2: the paper's worked example
    alpha, qs = eq23([0.025] * 3, 0.1, 3.0, 3.0)
    chk("G2 alpha 1.0, q 3.0", abs(alpha - 1.0) < 1e-12 and all(abs(q - 3.0) < 1e-12 for q in qs), (alpha, qs))
    toks = 3.0 / 0.025
    chk("G2 120 tokens / turn, round 12 s", abs(toks - 120) < 1e-9 and abs(3 * qs[0] + 3.0 - 12.0) < 1e-12)
    # replay the LAST batch of the round (worst placed: its turns start at 9, 21, 33 s after switch, turn, switch, turn, switch)
    # through the turn accounting in steady state: a = 0, the first token at 0, one round (12 s) of buffered slack (TTFT 12).
    tot = met = 0
    for rnd in range(3):
        met += count_met(9.0 + 12.0 * rnd, 0.025, 120, 0.0, 2 + 120 * rnd, 12.0, 0.1)
        tot += 120
    chk("G2 worst-placed batch: all tokens met over 3 rounds", met == tot, (met, tot))
    # the binding token is the turn's FIRST (9.025 s vs TTFT + 0.1): slack 8.93 meets all, 8.90 misses exactly one (hand: 119)
    chk("G2 worst-placed batch needs TTFT >= 8.925: 8.93 -> 120, 8.90 -> 119",
        count_met(9.0, 0.025, 120, 0.0, 2, 8.93, 0.1) == 120 and count_met(9.0, 0.025, 120, 0.0, 2, 8.90, 0.1) == 119)
    # G3
    chk("G3 step fit Q", abs(step_time("A", 1, 4096) - ANCH_Q) <= 1e-12)
    chk("G3 step fit S", abs(step_time("A", 8, 32768) - ANCH_S) <= 1e-12)
    chk("G3 step fit LC", abs(step_time("A", 2, 32768) - ANCH_LC) <= 1e-12)
    chk("G3 C_REF 18,777.0147 (prereg) and ACTIVE_REF_B 3.1", abs(C_REF - 18777.0147) < 1e-3 and ACTIVE_REF_B == 3.1,
        (C_REF, ACTIVE_REF_B))
    chk("G3 prefill 161 tokens 0.0201157 s", abs(t_pre(161) - 0.020115690) <= 1e-8, t_pre(161))
    # G4
    chk("G4 29 met", count_met(3.0, 0.006, 100, 0.0, 2, 2.0, 0.02) == 29, count_met(3.0, 0.006, 100, 0.0, 2, 2.0, 0.02))
    rng = np.random.default_rng(7)
    okb = True
    for _ in range(3000):
        T0 = rng.random() * 5; t = rng.choice([0.005, 0.01, 0.02, 0.03, 0.0137]); J = int(rng.integers(0, 60))
        a = rng.random() * 2; kf = int(rng.integers(2, 50)); TT = rng.choice([1.0, 2.0, 3.0]); TB = rng.choice([0.01, 0.02, 0.0137])
        brute = sum(1 for j in range(1, J + 1) if T0 + j * t <= a + TT + (kf + j - 2) * TB + 1e-9)
        if brute != count_met(T0, t, J, a, kf, TT, TB):
            okb = False
            break
    chk("G4 brute force 3000 cases (t<d, t>d, t=d)", okb)
    # G5
    chk("G5 union 0.666667", abs(union_active([0, 5, 20], [10, 15, 25], [0, 0, 1], 0.0, 30.0) - 20 / 30) <= 1e-12)
    # G6 adversarial
    chk("G6(i) the brief's 46.55 FAILS at +-0.01", not (abs(closed_form(100, 0.037, 16.79) - 46.55) <= 0.01))
    for nm, fn in (("lam<=0", lambda: closed_form(100, 0.0, 16.79)), ("T<0", lambda: closed_form(100, 0.037, -1)),
                   ("TBT<=0", lambda: eq23([0.01], 0.0, 1.0, 4.0)), ("count_met t<=0", lambda: count_met(0, 0, 1, 0, 2, 1, 1)),
                   ("MUX 3 models/GPU", lambda: run_mux_gpu(3, 0, "A", 0.0))):
        try:
            fn()
            chk(f"G6(ii) {nm} raises", False)
        except ValueError:
            chk(f"G6(ii) {nm} raises", True)
        except Exception as ex:      # noqa: BLE001  review: a wrong exception type is a recorded FAIL
            chk(f"G6(ii) {nm} raises ValueError (got {type(ex).__name__})", False, repr(ex))
    chk("G6(iii) MUX M=36 INFEASIBLE", mux_attain(36, 1.0, 1.0) is None and mux_goodput(1.0, 1.0) == 32)
    # G7 real entry point (CLI)
    me = os.path.abspath(__file__)

    def cli(spec):
        p = subprocess.run([sys.executable, me, "--b-one", "--spec", json.dumps(spec)], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-500:])
        return json.loads(p.stdout)
    base = dict(policy="aeg", reqs=[[0.0, 0]], lin=161, lout=3, hw="A", pf=True, n_pre=1, n_dec=1)
    try:
        r = cli(dict(base, scale=0.1, tsw=0.5, kvrate=26.0e9))
        e = r["emit"][0]
        chk("G7 e1 0.52011569", abs(e["e1"] - 0.52011569) <= 1e-7, e["e1"])
        chk("G7 e3 1.04007223", abs(e["fin"] - 1.04007223) <= 1e-7, e["fin"])
        chk("G7 s0.1 met 1/3", e["met"] == 1, e["met"])
        for sc in (1.0, 0.2):
            rr = cli(dict(base, scale=sc, tsw=0.5, kvrate=26.0e9))
            chk(f"G7 s{sc} met 3/3", rr["emit"][0]["met"] == 3, rr["emit"][0]["met"])
        r = cli(dict(base, scale=0.1, tsw=0.06, kvrate=273.0e9))
        chk("G7 NVL e3 0.15924610, met 3/3", abs(r["emit"][0]["fin"] - 0.15924610) <= 1e-7 and r["emit"][0]["met"] == 3,
            r["emit"][0])
    except Exception as ex:      # noqa: BLE001
        chk("G7 CLI", False, repr(ex))
    # G8 MUX
    try:
        r = cli(dict(policy="mux", reqs=[[0.0, 0]], lin=161, lout=3, hw="A", n_models=1, beta=0.0))
        em = r["emit"][0]
        chk("G8 solo e1 0.02011569, e2 0.02963741", abs(em[0] - 0.02011569) <= 1e-8 and abs(em[1] - 0.02963741) <= 1e-8, em)
        r = cli(dict(policy="mux", reqs=[[0.0, 0], [0.0, 1]], lin=161, lout=2, hw="A", n_models=2, beta=1.0))
        ok8 = all(abs(x[0] - 0.04023138) <= 1e-8 and abs(x[1] - 0.05927483) <= 1e-8 for x in r["emit"])
        chk("G8 co-active beta 1: e1 0.04023138, e2 0.05927483", ok8, r["emit"])
    except Exception as ex:      # noqa: BLE001
        chk("G8 CLI", False, repr(ex))
    # G9 busy_matrix plumbing
    try:
        import cloud_sim as CS
        B = CS.busy_matrix([0, 0], [50.0, 30.0], [16.79, 16.79], [0, 1], 2, 3)
        chk("G9 offset 50 -> minutes 0,1; offset 30 -> minute 0", B[0].tolist() == [True, True, False]
            and B[1].tolist() == [True, False, False], B.tolist())
    except Exception as ex:      # noqa: BLE001
        chk("G9 busy_matrix", False, repr(ex))
    # extra: a two-batch decode instance honours Eq. 2-3 turns (structural: two models alternate, both finish)
    r = run_aeg(2, 1.0, 0, "A", 0.5, 26.0e9, True, QMAX, n_pre=1, n_dec=1,
                reqs=[Req(0, 0, 0.0, 161, 50), Req(1, 1, 0.0, 161, 50)], score=(-1, 1e9), drain=60.0)
    chk("GX two models on one decode instance both finish", r["unfinished"] == 0 and r["n_req"] == 2, r)
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
    g.add_argument("--a-paper", action="store_true")
    g.add_argument("--a-trace", action="store_true")
    g.add_argument("--b-run", action="store_true")
    g.add_argument("--b-one", action="store_true")
    g.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--out")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--cache")
    ap.add_argument("--spec")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(",") if x != "")
    if a.self_test:
        return self_test()
    if a.b_one:
        if not a.spec:
            die("--b-one needs --spec")
        print(json.dumps(b_one(a.spec)))
        return 0
    if a.summarize:
        if not os.path.isdir(a.summarize):
            die(f"no such dir {a.summarize}")
        print(json.dumps(summarize(a.summarize), indent=1, default=str))
        return 0
    if not a.out:
        die("--out is required")
    os.makedirs(a.out, exist_ok=True)
    if a.a_paper:
        res = a_paper(seeds)
        res.update(prereg=PREREG_COMMIT)
        with open(os.path.join(a.out, "a_paper.json"), "w") as fh:     # QA r1 (flash-lite): context manager
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return 0
    if a.a_trace:
        if not a.cache:
            die("--a-trace needs --cache (the bins npz path)")
        bins = os.path.join(a.root, "data/traces/derived/bins_w1.csv.gz")
        if not os.path.exists(bins):
            die(f"missing {bins}")
        res = a_trace(a.root, a.cache, seeds)
        res.update(prereg=PREREG_COMMIT)
        with open(os.path.join(a.out, "a_trace.json"), "w") as fh:     # QA r1 (flash-lite): context manager
            json.dump(res, fh, indent=1, default=float)
        print(json.dumps(res, indent=1, default=float))
        return 0
    if a.b_run:
        arms = [x for x in a.arms.split(",") if x]
        if not arms:
            die("--b-run needs --arms")
        scales = tuple(float(x) for x in a.scales.split(","))
        for s in scales:
            if s not in SCALES:
                die(f"scale {s} not in the frozen grid {SCALES}")
        print(json.dumps(b_run(a.out, arms, seeds, scales, a.workers)))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
