#!/usr/bin/env python3
"""tailpool_v2.py — PRE-STAGED tail pool at honest costs (sleeper GPU charge, per-card host-RAM cap,
compute-aware vs compute-blind homes, size mixes, a burstier arrival arm, TRAIN-only tail selection).

Prereg (frozen before this file existed): (private report). On any divergence
the prereg wins. Reuses scripts/tailpool_sim.py (NOT edited) for the trace, arrivals, queue
constants, dedicated baseline and scoring.

Output contract:
  --self-test              exit 0 + "SELF-TEST PASS n/n" on the hand-computed goldens V1-V8; exit 1 + FAIL lines otherwise.
  --describe --scenario S  read-only: tail-set counts, default-guess count, drift, out-of-scope, K_min_feasible per
                           (H, placement), one JSON object on stdout.
  --arm NAME --seeds 0,1,2 [--ks ...] --out F
                           runs one prereg pool arm over the K grid, appends one JSON line per (K, seed) to --out (WRITES).
  --baselines --scenario S --arrivals POISSON|BURST --seeds ... --out F
                           DEDICATED + S2Z_EQUAL (T_idle 60/300) + S2Z_55 rows (WRITES --out).
  --summarize DIR          read-only: prereg Q1-Q6 + costs + prototype rule over every *.jsonl in DIR, JSON on stdout.
  Bad input (sha mismatch, missing file, an infeasible home assignment requested explicitly) = exit 2 + one "ERROR:" line.

Top pitfalls (observed while building): a sleeper is bound to its home card, so placement is static and the RAM cap is
per card, not pooled; BURST at w=60 must reproduce POISSON exactly (checked in --self-test).
"""
import argparse, collections, csv, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tailpool_sim as T  # noqa: E402  (reused as-is)
from real_trace_analysis import parse_size, C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

SLEEPER_GIB = 1.0          #: 0.9-1.1 GiB held on the card by a sleeping engine
SLEEP_S = 0.40             #: later sleep (demote, mode=wait) of an 8B
PAGEIN_FIXED = 0.02        # page-in = 0.02 s + bytes / 22.6 GB/s
PAGEIN_GBPS = 22.6
GIB = 1.073741824          # GiB -> GB
FIRST_SLEEP_S = {"fp8": 11.3, "int4": 28.4}   # boot cost per model (8B fp8 / MoE int4), reported only
H_ARMS = {"H32": 29.8, "H84": 78.2, "H200": 200.0}
H_PIN_FRAC = {"H32": 0.6, "H84": 0.6, "H200": 0.8}
BURST_W = 6.0
S2Z_RESTART = 55.0
TRAIN_END = T.HELDOUT_MIN
N_TRAIN_BINS5 = (T.HELDOUT_MIN - T.MIN0) // 5     # 2016
K_GRID = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
SEEDS = (0, 1, 2)
OFFERS_CSV = "data/tailpool_v2/vast_5090_offers_2026-10-13T163607Z.csv"

# arm -> (scenario, H key, placement, burst_w or None, sleeper GiB)
ARMS = {
    "PRIMARY": ("P8", "H84", "AWARE", None, 1.0),
    "BLIND": ("P8", "H84", "BLIND", None, 1.0),
    "H32": ("P8", "H32", "AWARE", None, 1.0),
    "H200": ("P8", "H200", "AWARE", None, 1.0),
    "BURST": ("P8", "H84", "AWARE", BURST_W, 1.0),
    "BURST_BLIND": ("P8", "H84", "BLIND", BURST_W, 1.0),
    "RS": ("RS", "H84", "AWARE", None, 1.0),
    "RS_U3": ("RS_U3", "H84", "AWARE", None, 1.0),
    "RS_U32": ("RS_U32", "H84", "AWARE", None, 1.0),
    "RS_BURST": ("RS", "H84", "AWARE", BURST_W, 1.0),
    "U3": ("U3", "H84", "AWARE", None, 1.0),
    "S11": ("P8", "H84", "AWARE", None, 1.1),
}


# ------------------------------------------------------------------------------------------------ sizes / tail
def scen_sizes(name, scen):
    """(total_B, active_B, prec, default_guess) or None (frontier, size not in name)."""
    if scen == "U3":
        return 3.0, 3.0, "fp8", False
    tb, ab, src = parse_size(name)
    if src == "frontier-moe-size-not-in-name":
        return None
    if tb is None:
        if scen in ("P8", "RS"):
            return 8.0, 8.0, "fp8", True
        if scen == "RS_U3":
            return 3.0, 3.0, "fp8", True
        if scen == "RS_U32":
            return 32.0, 32.0, "int4", True
        raise ValueError(f"unknown scenario {scen}")
    if scen == "P8":
        return tb, ab, "fp8", False
    if scen in ("RS", "RS_U3", "RS_U32"):
        return tb, ab, ("fp8" if tb <= 14.0 else "int4"), False
    raise ValueError(f"unknown scenario {scen}")


def train_stats(arrays):
    """Per bins-index: (TRAIN p95 5-min work rate tok/s over 2016 bins incl. zeros, TRAIN requests, HELD-OUT p95 rate)."""
    minute, midx, n, it, ot, ids = arrays
    work = it + ot
    out = {}
    for lo, hi, tag in ((T.MIN0, TRAIN_END, "train"), (TRAIN_END, TRAIN_END + (TRAIN_END - T.MIN0), "held")):
        k = (minute >= lo) & (minute < hi)
        b = (minute[k] - lo) // 5
        nb = (hi - lo) // 5
        mat = collections.defaultdict(lambda: np.zeros(nb))
        req = collections.Counter()
        for j, bb, wv, nn in zip(midx[k], b, work[k], n[k]):
            mat[int(j)][int(bb)] += wv
            req[int(j)] += int(nn)
        for j, arr in mat.items():
            d = out.setdefault(ids[j], {})
            d[tag + "_p95"] = float(np.percentile(arr / 300.0, 95))
            d[tag + "_req"] = req[j]
    return out


def tail_set_v2(names, stats, scen, card_gib=T.CARD_GIB):
    """prereg §1 TRAIN-only tail. Returns ({cid: (tb, ab, prec, w, f)}, {cid: {"u", "guess", "u_held"}}, counts)."""
    tail, meta, cnt = {}, {}, collections.Counter()
    for cid, name in names.items():
        st = stats.get(cid, {})
        if st.get("train_req", 0) < 1:
            cnt["not_in_train"] += 1
            continue
        sz = scen_sizes(name, scen)
        if sz is None:
            cnt["unpoolable_frontier"] += 1
            continue
        tb, ab, prec, guess = sz
        u = st["train_p95"] / (C_REF * ACTIVE_REF_B / ab)
        if u >= 1:
            cnt["head_ge_1_card_train"] += 1
            continue
        w, f = T.footprint(tb, prec)
        if f > card_gib:
            cnt["unpoolable_too_big"] += 1
            continue
        tail[cid] = (tb, ab, prec, w, f)
        meta[cid] = {"u": u, "guess": guess, "u_held": st.get("held_p95", 0.0) / (C_REF * ACTIVE_REF_B / ab)}
        cnt["tail"] += 1
        cnt["default_guess"] += int(guess)
    return tail, meta, dict(cnt)


# ------------------------------------------------------------------------------------------------ homes
def assign_homes(models, units, K, H, s=SLEEPER_GIB, card_gib=T.CARD_GIB, mode="AWARE"):
    """Static home card per model (prereg §1). models[m] = (cid, tb, ab, prec, w, f). Raises ValueError if infeasible."""
    M = len(models)
    ram_free = [H] * K
    n_h = [0] * K
    fmax = [0.0] * K
    load = [0.0] * K
    homes = [-1] * M

    def ok(c, m):
        w, f = models[m][4], models[m][5]
        return ram_free[c] >= w - 1e-9 and n_h[c] * s + max(fmax[c], f) <= card_gib + 1e-9

    if mode == "BLIND":
        order = sorted(range(M), key=lambda m: models[m][0])
    elif mode == "AWARE":
        order = sorted(range(M), key=lambda m: (-units[m], models[m][0]))
    else:
        raise ValueError(f"unknown placement {mode}")
    for m in order:
        w = models[m][4]
        if w > H + 1e-9:
            raise ValueError(f"host RAM cap exceeded: model {models[m][0]} weights {w:.2f} GiB > {H} GiB per card")
        feas = [c for c in range(K) if ok(c, m)]
        if not feas:
            if any(ram_free[c] >= w - 1e-9 for c in range(K)):
                raise ValueError(f"sleeper GPU cap exceeded at K={K}: model {models[m][0]} fits no card's GPU budget")
            raise ValueError(f"host RAM cap exceeded at K={K}: model {models[m][0]} ({w:.2f} GiB) fits no card")
        if mode == "BLIND":
            c = max(feas, key=lambda j: (ram_free[j], -j))
        else:
            c = min(feas, key=lambda j: (load[j], -ram_free[j], j))
        homes[m] = c
        ram_free[c] -= w; n_h[c] += 1; fmax[c] = max(fmax[c], models[m][5]); load[c] += units[m]
    return homes


def k_min_feasible(models, units, H, s=SLEEPER_GIB, mode="AWARE", card_gib=T.CARD_GIB, k_max=4096):
    lb = max(1, math.ceil(sum(m[4] for m in models) / H - 1e-9))
    for K in range(lb, k_max + 1):
        try:
            assign_homes(models, units, K, H, s, card_gib, mode)
            return K
        except ValueError as e:
            if "weights" in str(e):     # a single model larger than H: infeasible at every K
                raise
    raise ValueError(f"no feasible K <= {k_max}")


def binding_constraint(models, homes, K, H, s=SLEEPER_GIB, card_gib=T.CARD_GIB):
    """Per card: which budget is tighter as a fraction used — 'RAM' (a) or 'GPU' (b)."""
    ram = [0.0] * K; n = [0] * K; fm = [0.0] * K
    for m, c in enumerate(homes):
        ram[c] += models[m][4]; n[c] += 1; fm[c] = max(fm[c], models[m][5])
    out = collections.Counter()
    for c in range(K):
        a = ram[c] / H
        b = ((n[c] - 1) * s + fm[c]) / card_gib if n[c] else 0.0
        out["RAM" if a >= b else "GPU"] += 1
    return dict(out)


# ------------------------------------------------------------------------------------------------ arrivals
def arrivals_v2(trace, seed, burst_w=None):
    """POISSON (burst_w None) = arrivals. BURST: n arrivals i.i.d. inside one window [s0, s0+w], s0 ~ U[0, 60-w]
    from a separate RNG stream, so w=60 reproduces POISSON exactly (golden V5)."""
    if trace.explicit is not None or burst_w is None:   # explicit (golden) traces carry their own times; burst_w then
        return T.arrivals(trace, seed)                  # only adds the in-burst queue term (review, by design)
    rng = np.random.default_rng([seed, 2683])
    n = trace.n.astype(float)
    v1 = rng.random(len(n)); v2 = rng.random(len(n))
    s0 = np.random.default_rng([seed, 2690]).random(len(n)) * (60.0 - burst_w)
    first = s0 + burst_w * (1.0 - (1.0 - v1) ** (1.0 / n))
    rest = np.maximum(n - 1.0, 1.0)
    last = np.where(n > 1, first + (s0 + burst_w - first) * v2 ** (1.0 / rest), first)
    order = np.lexsort((first, trace.minute))
    return first, last, order


def _burst_extra(work_model, burst_w):
    if burst_w is None:
        return 0.0
    return 0.5 * max(0.0, work_model - C_REF * burst_w) / C_REF


# ------------------------------------------------------------------------------------------------ pre-staged pool
def simulate_prestaged(trace, homes, K, seed, s=SLEEPER_GIB, card_gib=T.CARD_GIB, burst_w=None,
                       score_from_min=T.HELDOUT_MIN, min0=T.MIN0, sleep_s=SLEEP_S, gbps=PAGEIN_GBPS,
                       fixed=PAGEIN_FIXED):
    """Every model asleep on its home card at boot; wake = page-in (+0.40 s per demoted card-mate). No cold loads."""
    models = trace.models
    M = len(models)
    if len(homes) != M or any(h < 0 or h >= K for h in homes):
        raise ValueError("homes must give every model a card in [0, K)")
    first, last, order = arrivals_v2(trace, seed, burst_w)
    gpu = [0.0] * K
    for m in range(M):
        gpu[homes[m]] += s
    awake = [False] * M
    ready = [0.0] * M
    last_arr = [-1e18] * M
    card_awake = [set() for _ in range(K)]
    backlog = [0.0] * K
    cap_min = C_REF * 60.0
    out = [collections.Counter() for _ in range(M)]
    st = collections.Counter()
    N = len(order)
    i_pos = 0
    prev_mi = None
    while i_pos < N:
        mi = int(trace.minute[order[i_pos]])
        if prev_mi is not None and mi - prev_mi > 1:
            for c in range(K):
                backlog[c] = max(0.0, backlog[c] - cap_min * (mi - prev_mi - 1))
        prev_mi = mi
        j = i_pos
        while j < N and int(trace.minute[order[j]]) == mi:
            j += 1
        W = [0.0] * K
        pend = []
        base = (mi - min0) * 60.0
        for p in range(i_pos, j):
            i = int(order[p])
            m = int(trace.mid[i]); n = int(trace.n[i])
            t0 = base + float(first[i])
            tb, ab, prec, w, f = models[m][1:]
            c = homes[m]
            served = True
            if not awake[m]:
                need = f - s
                vict = []
                if gpu[c] + need > card_gib + 1e-9:
                    cand = sorted((last_arr[r], r) for r in card_awake[c]
                                  if not (ready[r] > t0 or last_arr[r] > t0 - T.PROTECT_S))
                    free = card_gib - gpu[c]
                    for _, r in cand:
                        if free >= need - 1e-9:
                            break
                        vict.append(r); free += models[r][5] - s
                    if free < need - 1e-9:
                        served = False
                if served:
                    for r in vict:
                        awake[r] = False; card_awake[c].discard(r); gpu[c] -= models[r][5] - s
                        st["evictions"] += 1
                    L = len(vict) * sleep_s + fixed + w * GIB / gbps
                    awake[m] = True; card_awake[c].add(m); gpu[c] += need; ready[m] = t0 + L
                    st["wakes"] += 1
            last_arr[m] = base + float(last[i])
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF
            if not served:
                st["unserved_req"] += n
                pend.append((m, -1, n, pre, None, 0.0)); continue
            wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
            W[c] += wm
            extra = _burst_extra(wm, burst_w)
            if ready[m] <= t0:
                pend.append((m, c, n, pre, None, extra))
            else:
                ts = T.cell_times(trace, i, first, last, seed) + base
                pend.append((m, c, n, pre, np.maximum(0.0, ready[m] - ts), extra))
        q = [0.0] * K
        for c in range(K):
            q[c] = (backlog[c] + 0.5 * max(0.0, W[c] - cap_min)) / C_REF
            backlog[c] = max(0.0, backlog[c] + W[c] - cap_min)
        if mi >= score_from_min:
            for m, c, n, pre, waits, extra in pend:
                if c < 0:
                    out[m][T.INF_KEY] += n; continue
                if waits is None:
                    out[m][T._key(q[c] + extra + pre)] += n
                else:
                    for wv in waits:
                        out[m][T._key(float(wv) + q[c] + extra + pre)] += 1
        i_pos = j
    return out, dict(st)


def simulate_dedicated_v2(trace, burst_w=None, score_from_min=T.HELDOUT_MIN):
    """ dedicated; under BURST the model's own in-burst term is added (same arrival arm as the pool)."""
    if burst_w is None:
        return T.simulate_dedicated(trace, score_from_min)
    M = len(trace.models)
    out = [collections.Counter() for _ in range(M)]
    backlog = [0.0] * M
    last_min = [None] * M
    cap_min = C_REF * 60.0
    for i in range(len(trace.n)):
        m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        ab = trace.models[m][2]
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        if last_min[m] is not None:
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        last_min[m] = mi
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / C_REF + _burst_extra(Wm, burst_w)
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        if mi >= score_from_min:
            out[m][T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF)] += n
    return out


def simulate_s2z_v2(trace, t_snap, t_idle, seed, burst_w=None, score_from_min=T.HELDOUT_MIN, min0=T.MIN0):
    """ scale-to-zero with a PER-MODEL start term t_snap[m] (S2Z_EQUAL) and the arrival arm."""
    M = len(trace.models)
    first, last, order = arrivals_v2(trace, seed, burst_w)
    alive_until = [-1e18] * M
    ready = [0.0] * M
    start_t = [0.0] * M
    backlog = [0.0] * M
    last_min = [None] * M
    cap_min = C_REF * 60.0
    out = [collections.Counter() for _ in range(M)]
    intervals = []
    st = collections.Counter()
    for p in range(len(order)):
        i = int(order[p]); m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        base = (mi - min0) * 60.0
        t0 = base + float(first[i])
        ab = trace.models[m][2]
        if last_min[m] is not None:
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        if t0 > alive_until[m] and backlog[m] <= 0.0:
            if alive_until[m] > -1e17:
                intervals.append((start_t[m], alive_until[m]))
            start_t[m] = t0; ready[m] = t0 + t_snap[m]; st["cold_starts"] += 1
        alive_until[m] = base + float(last[i]) + t_idle
        last_min[m] = mi
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / C_REF + _burst_extra(Wm, burst_w)
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF
        if mi >= score_from_min:
            if ready[m] <= t0:
                out[m][T._key(q + pre)] += n
            else:
                ts = T.cell_times(trace, i, first, last, seed) + base
                for wv in np.maximum(0.0, ready[m] - ts):
                    out[m][T._key(float(wv) + q + pre)] += 1
    for m in range(M):
        if alive_until[m] > -1e17:
            intervals.append((start_t[m], alive_until[m]))
    return out, intervals, dict(st)


# ------------------------------------------------------------------------------------------------ cost
def usd_per_mtok(cards, usd_per_gpu_h, tokens, hours=168.0):
    if tokens <= 0:
        raise ValueError("tokens must be > 0")
    return cards * usd_per_gpu_h * hours / (tokens / 1e6)


def price_tier(rows, min_ram_gib_per_gpu):
    sel = [float(r["usd_per_gpu_h"]) for r in rows if float(r["host_ram_gib_per_gpu"]) >= min_ram_gib_per_gpu]
    if not sel:
        return {"n": 0, "median": None, "min": None, "thin": True}
    return {"n": len(sel), "median": statistics.median(sel), "min": min(sel), "thin": len(sel) < 5,
            "min_ram_gib_per_gpu": round(min_ram_gib_per_gpu, 1)}


# ------------------------------------------------------------------------------------------------ self-test
def self_test():
    fails, n = [], 0

    def chk(name, got, want, tol=1e-6):
        nonlocal n
        n += 1
        if isinstance(want, list):
            ok = isinstance(got, list) and len(got) == len(want) and all(
                (a == b) if not isinstance(b, float) else abs(a - b) <= tol for a, b in zip(got, want))
        elif isinstance(want, float):
            ok = got is not None and abs(got - want) <= tol
        else:
            ok = got == want
        if not ok:
            fails.append(f"FAIL {name}: got {got!r} want {want!r}")

    def vals(counter):
        out = []
        for k in sorted(counter):
            out += [k / 100.0] * counter[k]
        return out

    def raises(name, fn, needle):
        nonlocal n
        n += 1
        try:
            fn()
            fails.append(f"FAIL {name}: did not raise")
        except ValueError as e:
            if needle not in str(e):
                fails.append(f"FAIL {name}: raised {e!r}, want '{needle}'")

    mk = lambda cid, b, prec="fp8": (cid, b, b, prec) + T.footprint(b, prec)
    # V1 sleeper charge reduces awake capacity
    models = [mk("A", 8.0), mk("B", 8.0), mk("D", 8.0), mk("E", 8.0)]
    reqs = [(0, 0, 0, 0), (100, 1, 0, 0), (200, 0, 0, 0)]
    for s, wa, wb, ev in ((0.0, [0.0, 0.43], [0.43], 0), (1.0, [0.43, 0.83], [0.83], 2)):
        tr = T.Trace.explicit_cells(models, reqs)
        o, stt = simulate_prestaged(tr, [0, 0, 0, 0], 1, 0, s=s, card_gib=21.0, score_from_min=0, min0=0)
        chk(f"V1 s={s} A", sorted(vals(o[0])), wa); chk(f"V1 s={s} B", vals(o[1]), wb)
        chk(f"V1 s={s} evictions", stt.get("evictions", 0), ev)
        chk(f"V1 s={s} no cold path", stt.get("unserved_req", 0), 0)
    # V2 RAM cap fails loudly
    two = [mk("A", 8.0), mk("B", 8.0)]
    raises("V2 K=1 RAM", lambda: assign_homes(two, [0.1, 0.1], 1, 10.0), "host RAM cap exceeded")
    chk("V2 K=2 ok", assign_homes(two, [0.1, 0.1], 2, 10.0), [0, 1])
    raises("V2 single model > H", lambda: assign_homes([mk("Z", 32.0, "int4")], [0.1], 4, 10.0), "host RAM cap exceeded")
    raises("V2 k_min single > H", lambda: k_min_feasible([mk("Z", 32.0, "int4")], [0.1], 10.0), "host RAM cap exceeded")
    chk("V2 k_min two", k_min_feasible(two, [0.1, 0.1], 10.0), 2)
    # V3 sleeper GPU cap fails loudly
    m20 = [mk(f"m{i:02d}", 3.0) for i in range(20)]
    chk("V3 20x3B K=1", assign_homes(m20, [0.0] * 20, 1, 1000.0), [0] * 20)
    m25 = [mk(f"m{i:02d}", 3.0) for i in range(25)]
    raises("V3 25x3B K=1", lambda: assign_homes(m25, [0.0] * 25, 1, 1000.0), "sleeper GPU cap exceeded")
    chk("V3 25x3B k_min", k_min_feasible(m25, [0.0] * 25, 1000.0), 2)
    # V4 placement
    four = [mk("a", 3.0), mk("b", 3.0), mk("c", 3.0), mk("d", 3.0)]
    u = [0.9, 0.1, 0.8, 0.1]
    chk("V4 BLIND", assign_homes(four, u, 2, 1000.0, mode="BLIND"), [0, 1, 0, 1])
    chk("V4 AWARE", assign_homes(four, u, 2, 1000.0, mode="AWARE"), [0, 1, 1, 0])
    # V5 BURST w=60 == POISSON; w=6 inside its window
    tr = T.Trace([0, 0, 1, 2], [0, 1, 0, 1], [5, 3, 1, 40], [10, 10, 10, 10], [5, 5, 5, 5], [mk("A", 3.0), mk("B", 3.0)])
    a = T.arrivals(tr, 3); b = arrivals_v2(tr, 3, 60.0)
    chk("V5 w=60 first", bool(np.allclose(a[0], b[0], rtol=0, atol=1e-12)), True)
    chk("V5 w=60 last", bool(np.allclose(a[1], b[1], rtol=0, atol=1e-12)), True)
    chk("V5 w=60 order", bool((a[2] == b[2]).all()), True)
    f6, l6, _ = arrivals_v2(tr, 3, 6.0)
    chk("V5 w=6 span", bool(((l6 - f6) <= 6.0 + 1e-9).all() and (f6 >= 0).all() and (l6 < 60.0).all()), True)
    allin = True
    for i in range(len(tr.n)):
        ts = T.cell_times(tr, i, f6, l6, 3)
        allin &= bool(ts.max() - ts.min() <= 6.0 + 1e-9)
    chk("V5 w=6 cell times in window", allin, True)
    # BURST queue term: one 3.1B-active model, resident, minute work 2*C_REF*60 -> fluid 30 s; burst adds
    # 0.5*(2*C_REF*60 - C_REF*6)/C_REF = 57 s -> 87 s
    A31 = ("A", 3.1, 3.1, "fp8") + T.footprint(3.1, "fp8")
    tr = T.Trace.explicit_cells([A31], [(0, 0, 0, 2 * C_REF * 60)])
    o, _ = simulate_prestaged(tr, [0], 1, 0, s=1.0, burst_w=6.0, score_from_min=0, min0=0)
    chk("V5 burst queue (+page-in 0.02+3.317*GiB/22.6)", vals(o[0]), [87.18], 0.011)
    # V6 S2Z per-model start reproduces G7 / G7b
    A3 = mk("A", 3.0)
    tr = T.Trace.explicit_cells([A3], [(0, 0, 0, 0), (100, 0, 0, 0), (500, 0, 0, 0)])
    o, iv, stt = simulate_s2z_v2(tr, [5.0], 300.0, 0, score_from_min=0, min0=0)
    chk("V6 G7", sorted(vals(o[0])), [0.0, 5.0, 5.0]); chk("V6 G7 starts", stt.get("cold_starts", 0), 2)
    tr = T.Trace.explicit_cells([A31], [(0, 0, 0, 3 * C_REF * 60), (90, 0, 0, 0)])
    o, iv, stt = simulate_s2z_v2(tr, [5.0], 30.0, 0, score_from_min=0, min0=0)
    chk("V6 G7b", sorted(vals(o[0])), [65.0, 120.0], 0.011)
    # dedicated_v2 under POISSON is exactly
    tr = T.Trace.explicit_cells([A31], [(0, 0, 0, 2 * C_REF * 60), (61, 0, 0, 0)])
    chk("dedicated POISSON == ", [dict(x) for x in simulate_dedicated_v2(tr, None, 0)],
        [dict(x) for x in T.simulate_dedicated(tr, 0)])
    # V7 cost
    chk("V7 $/M tok", usd_per_mtok(2, 0.5, 1.68e6), 100.0, 1e-9)
    rows = [{"usd_per_gpu_h": "0.4", "host_ram_gib_per_gpu": "30"}, {"usd_per_gpu_h": "0.6", "host_ram_gib_per_gpu": "130"},
            {"usd_per_gpu_h": "0.8", "host_ram_gib_per_gpu": "260"}]
    chk("V7 price tier >=130", price_tier(rows, 130.0)["median"], 0.7, 1e-9)
    # V8 TRAIN tail: 'H' heavy only in HELD-OUT -> tail; 'T' heavy in TRAIN -> head
    big = 2 * C_REF * 300      # 5-min work = 2 cards at 3.1B active, 3.1B-sized name
    mins = [T.MIN0 + 5 * k for k in range(0, 2016, 2)]          # every other 5-min TRAIN bin (>5 % of bins)
    held = [TRAIN_END + 5 * k for k in range(0, 2016, 2)]
    minute = np.array([T.MIN0] + held + mins, np.int64)
    midx = np.array([0] + [0] * len(held) + [1] * len(mins), np.int32)
    nn = np.ones(len(minute), np.int64)
    it = np.array([1.0] + [big] * len(held) + [big] * len(mins)); ot = np.zeros(len(minute))
    stx = train_stats((minute, midx, nn, it, ot, ["H", "T"]))
    tl, meta, cnt = tail_set_v2({"H": "x/m-3.1B", "T": "y/m-3.1B"}, stx, "P8")
    chk("V8 held-out-heavy stays tail", "H" in tl, True)
    chk("V8 train-heavy is head", "T" in tl, False)
    chk("V8 drift visible", meta.get("H", {}).get("u_held", 0.0) >= 1.0, True)
    # scenario sizes
    chk("RS 24B -> int4", scen_sizes("org/Foo-24B", "RS")[2], "int4")
    chk("RS unparsed -> 8B guess", scen_sizes("org/foo", "RS")[3], True)
    chk("RS_U32 unparsed -> 32B int4", scen_sizes("org/foo", "RS_U32")[:3], (32.0, 32.0, "int4"))
    # determinism
    tr = T.Trace([0, 0, 1], [0, 1, 0], [5, 3, 1], [10, 10, 10], [5, 5, 5], [mk("A", 3.0), mk("B", 3.0)])
    r1 = simulate_prestaged(tr, [0, 0], 1, 7, card_gib=8.0, score_from_min=0, min0=0)
    r2 = simulate_prestaged(tr, [0, 0], 1, 7, card_gib=8.0, score_from_min=0, min0=0)
    chk("determinism", json.dumps([dict(x) for x in r1[0]]) == json.dumps([dict(x) for x in r2[0]]), True)
    chk("request conservation", sum(sum(x.values()) for x in r1[0]), 9)
    for f in fails:
        print(f)
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n - len(fails)}/{n}")
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ CLI
_CACHE = {}


def load(root, cache):
    if "arrays" not in _CACHE:
        pm = json.load(open(os.path.join(root, T.TRACE_DIR, "derived/chutes_w1_analysis.json")))["per_model"]
        arrays = T.load_bins_np(os.path.join(root, T.TRACE_DIR, "derived/bins_w1.csv.gz"), cache)
        _CACHE["names"] = {c: r["name"] for c, r in pm.items()}
        _CACHE["arrays"] = arrays
        _CACHE["stats"] = train_stats(arrays)
    return _CACHE["names"], _CACHE["arrays"], _CACHE["stats"]


def build(scen, root, cache):
    names, arrays, stats = load(root, cache)
    tail, meta, cnt = tail_set_v2(names, stats, scen)
    tr = T.Trace.from_bins(arrays, tail)
    units = [meta[m[0]]["u"] for m in tr.models]
    return tr, units, meta, cnt


def heldout_tokens(tr):
    k = tr.minute >= T.HELDOUT_MIN
    return float((tr.it[k] + tr.ot[k]).sum()), int(tr.n[k].sum())


def describe(scen, root, cache):
    names, arrays, stats = load(root, cache)
    tr, units, meta, cnt = build(scen, root, cache)
    tok, req = heldout_tokens(tr)
    minute, midx, n, it, ot, ids = arrays
    k = minute >= T.HELDOUT_MIN
    held_req = collections.Counter()
    for j, nn in zip(midx[k], n[k]):
        held_req[ids[j]] += int(nn)
    tailset = {m[0] for m in tr.models}
    oos = {c: r for c, r in held_req.items() if c not in tailset}
    out = {"scenario": scen, "counts": cnt, "heldout_tail_req": req, "heldout_tail_tokens": tok,
           "sum_weights_gib": round(sum(m[4] for m in tr.models), 2),
           "size_mix": dict(collections.Counter(f"{m[1]:g}B-{m[3]}" for m in tr.models).most_common(12)),
           "drift_u_held_ge_1": sum(1 for c in tailset if meta[c]["u_held"] >= 1.0),
           "out_of_scope_heldout_models": len(oos), "out_of_scope_heldout_req": sum(oos.values()),
           "k_min_feasible": {}}
    for hk, H in H_ARMS.items():
        for mode in ("AWARE", "BLIND"):
            kmin = k_min_feasible(tr.models, units, H, SLEEPER_GIB, mode)
            homes = assign_homes(tr.models, units, kmin, H, SLEEPER_GIB, mode=mode)
            out["k_min_feasible"][f"{hk}|{mode}"] = {"K": kmin, "binding": binding_constraint(tr.models, homes, kmin, H)}
    return out


def k_grid_for(kmin, ks=None):
    base = ks if ks else K_GRID
    return sorted({kmin} | {K for K in base if K >= kmin})


def run_arm(arm, seeds, root, cache, sink, ks=None):
    scen, hk, mode, bw, s = ARMS[arm]
    H = H_ARMS[hk]
    tr, units, meta, cnt = build(scen, root, cache)
    ded = simulate_dedicated_v2(tr, bw)
    kmin = k_min_feasible(tr.models, units, H, s, mode)
    tok, req = heldout_tokens(tr)
    for K in k_grid_for(kmin, ks):
        homes = assign_homes(tr.models, units, K, H, s, mode=mode)
        per_card = collections.Counter(homes)
        ram = collections.defaultdict(float)
        for m, c in enumerate(homes):
            ram[c] += tr.models[m][4]
        for seed in seeds:
            t = time.time()
            out, st = simulate_prestaged(tr, homes, K, seed, s=s, burst_w=bw)
            r = {"arm": arm, "scenario": scen, "H": hk, "H_gib": H, "placement": mode, "burst_w": bw, "sleeper_gib": s,
                 "K": K, "K_min_feasible": kmin, "seed": seed, "stats": st, "tail_counts": cnt,
                 "slo": {str(x): T.score(out, ded, x) for x in T.SLOS},
                 "heldout_req": int(sum(sum(c.values()) for c in out)), "heldout_req_bins": req, "heldout_tokens": tok,
                 "models_per_card_median": statistics.median([per_card.get(c, 0) for c in range(K)]),
                 "ram_used_per_card_median_gib": round(statistics.median([ram.get(c, 0.0) for c in range(K)]), 2),
                 "binding": binding_constraint(tr.models, homes, K, H, s),
                 "n_distinct_p95": len({T._pct(c, 0.95) for c in out if sum(c.values())}),
                 "elapsed_s": round(time.time() - t, 1)}
            line = json.dumps(r, sort_keys=True)
            print(line, flush=True)
            sink.write(line + "\n"); sink.flush()


def run_baselines(scen, arrivals_arm, seeds, root, cache, sink):
    bw = BURST_W if arrivals_arm == "BURST" else None
    tr, units, meta, cnt = build(scen, root, cache)
    ded = simulate_dedicated_v2(tr, bw)
    tok, req = heldout_tokens(tr)
    sumw = sum(m[4] for m in tr.models)
    common = {"scenario": scen, "arrivals": arrivals_arm, "heldout_tokens": tok, "heldout_req_bins": req,
              "tail_counts": cnt, "sum_weights_gib": sumw}
    r = dict(common, arm="DEDICATED", cards=len(tr.models), slo={str(x): T.score(ded, ded, x) for x in T.SLOS},
             heldout_req=int(sum(sum(c.values()) for c in ded)))
    sink.write(json.dumps(r, sort_keys=True) + "\n"); print(json.dumps(r, sort_keys=True), flush=True)
    span = (T.HELDOUT_MIN - T.MIN0) * 60.0
    eq = [PAGEIN_FIXED + m[4] * GIB / PAGEIN_GBPS for m in tr.models]
    for seed in seeds:
        for tag, snaps, idles in (("S2Z_EQUAL", eq, (60.0, 300.0)), ("S2Z_55", [S2Z_RESTART] * len(tr.models), (300.0,))):
            for t_idle in idles:
                out, iv, st = simulate_s2z_v2(tr, snaps, t_idle, seed, bw)
                p99, mx, mean = T.concurrency(iv, span, 2 * span)
                r = dict(common, arm=tag, t_idle=t_idle, seed=seed, cards_p99=p99, cards_max=mx, cards_mean=mean,
                         honest_cards={hk: max(p99, math.ceil(sumw / H)) for hk, H in H_ARMS.items()},
                         stats=st, slo={str(x): T.score(out, ded, x) for x in T.SLOS},
                         heldout_req=int(sum(sum(c.values()) for c in out)))
                line = json.dumps(r, sort_keys=True)
                print(line, flush=True); sink.write(line + "\n"); sink.flush()


# ------------------------------------------------------------------------------------------------ summarize
def _min_k(rows, slo, thr_m):
    by = collections.defaultdict(dict)
    for d in rows:
        v = d["slo"][str(slo)]
        by[d["K"]][d["seed"]] = (v["M_frac"] or 0.0, v["R_ratio"] or 0.0)
    for K in sorted(by):
        if all(sd in by[K] for sd in SEEDS) and all(by[K][sd][0] >= thr_m and by[K][sd][1] >= 0.99 for sd in SEEDS):
            return K
    return None


def summarize(d, root):
    rows = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            rows += T._rows(os.path.join(d, f))
    offers = list(csv.DictReader(open(os.path.join(root, OFFERS_CSV))))
    prices = {hk: price_tier(offers, H / H_PIN_FRAC[hk]) for hk, H in H_ARMS.items()}
    prices["ANY"] = price_tier(offers, 0.0)
    out = {"n_rows": len(rows), "prices_usd_per_gpu_h": prices, "arms": {}, "baselines": {}}
    pool = collections.defaultdict(list)
    for r in rows:
        if r["arm"] in ARMS:
            pool[r["arm"]].append(r)
    for arm, rs in sorted(pool.items()):
        a = {"K_min_feasible": rs[0]["K_min_feasible"], "tail_n": rs[0]["tail_counts"].get("tail"),
             "default_guess": rs[0]["tail_counts"].get("default_guess"), "heldout_tokens": rs[0]["heldout_tokens"],
             "seeds_run": sorted({r["seed"] for r in rs})}
        for slo in T.SLOS:
            nom = _min_k(rs, slo, 0.95); res = _min_k(rs, slo, 0.96)
            curve = {}
            for K in sorted({r["K"] for r in rs}):
                ms = [r["slo"][str(slo)]["M_frac"] for r in rs if r["K"] == K]
                rr = [r["slo"][str(slo)]["R_ratio"] for r in rs if r["K"] == K]
                curve[K] = {"M_min": round(min(ms), 4), "M_max": round(max(ms), 4), "R_min": round(min(rr), 4),
                            "n_seeds": len(ms)}
            hk = rs[0]["H"]
            cost = None
            if nom is not None and prices[hk]["median"]:
                cost = round(usd_per_mtok(nom, prices[hk]["median"], rs[0]["heldout_tokens"]), 4)
            mid = next((r for r in rs if r["K"] == (res or nom) and r["seed"] == 0), None)
            a[str(slo)] = {"min_k_nominal": nom, "min_k_resolved": res,
                           "verdict": ("NOT SERVED <= 256" if nom is None else "RESOLVED" if nom == res
                                       else "UNRESOLVED (on-bar)"),
                           "usd_per_mtok_at_nominal": cost, "curve": curve,
                           "at_min_k": None if mid is None else {
                               "models_per_card_median": mid["models_per_card_median"],
                               "ram_used_per_card_median_gib": mid["ram_used_per_card_median_gib"],
                               "binding": mid["binding"], "stats": mid["stats"],
                               "host_ram_provisioned_gib": round(mid["K"] * mid["H_gib"], 1)}}
        out["arms"][arm] = a
    base = collections.defaultdict(list)
    for r in rows:
        if r["arm"] in ("DEDICATED", "S2Z_EQUAL", "S2Z_55"):
            base[(r["scenario"], r["arrivals"], r["arm"], r.get("t_idle"))].append(r)
    for (sc, arr, arm, ti), rs in sorted(base.items(), key=lambda x: str(x[0])):
        tok = rs[0]["heldout_tokens"]
        key = f"{sc}|{arr}|{arm}|{ti}"
        if arm == "DEDICATED":
            out["baselines"][key] = {"cards": rs[0]["cards"],
                                     "usd_per_mtok": round(usd_per_mtok(rs[0]["cards"], prices["ANY"]["median"], tok), 4),
                                     "infeasible_even_dedicated": {s: rs[0]["slo"][s]["infeasible_even_dedicated"]
                                                                   for s in rs[0]["slo"]}}
            continue
        pr = prices["H84"]["median"] if arm == "S2Z_EQUAL" else prices["ANY"]["median"]
        mean = statistics.mean(r["cards_mean"] for r in rs)
        out["baselines"][key] = {
            "seeds": sorted(r["seed"] for r in rs), "cards_mean": round(mean, 2),
            "cards_p99_max": max(r["cards_p99"] for r in rs), "cards_max": max(r["cards_max"] for r in rs),
            "honest_cards": rs[0].get("honest_cards"),
            "M": {s: [round(r["slo"][s]["M_frac"], 4) for r in rs] for s in rs[0]["slo"]},
            "R": {s: [round(r["slo"][s]["R_ratio"], 4) for r in rs] for s in rs[0]["slo"]},
            "serves_all_seeds": {s: all(r["slo"][s]["serves"] for r in rs) for s in rs[0]["slo"]},
            "usd_per_mtok_mean_cards": round(usd_per_mtok(mean, pr, tok), 4),
            "usd_per_mtok_honest_H84": (round(usd_per_mtok(rs[0]["honest_cards"]["H84"], pr, tok), 4)
                                        if rs[0].get("honest_cards") else None)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--scenario", default="P8", choices=["P8", "RS", "RS_U3", "RS_U32", "U3"])
    ap.add_argument("--arrivals", default="POISSON", choices=["POISSON", "BURST"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ks", default=None, help="comma K list (K_min_feasible is always added)")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/tailpool/bins_w1.npz"))
    ap.add_argument("--out", help="append JSON lines here (WRITES)")
    ap.add_argument("--summarize", metavar="DIR")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    try:
        if a.summarize:
            print(json.dumps(summarize(a.summarize, a.root), indent=1, default=str))
            return 0
        if a.describe:
            print(json.dumps(describe(a.scenario, a.root, a.cache), indent=1))
            return 0
        if not a.out or not (a.arm or a.baselines):
            ap.print_help()
            return 2
        seeds = [int(x) for x in a.seeds.split(",")]
        ks = [int(x) for x in a.ks.split(",")] if a.ks else None
        with open(a.out, "a") as sink:
            if a.arm:
                run_arm(a.arm, seeds, a.root, a.cache, sink, ks)
            else:
                run_baselines(a.scenario, a.arrivals, seeds, a.root, a.cache, sink)
    except (ValueError, OSError, KeyError) as e:   # review: a malformed row exits 2, not a traceback
        print(f"ERROR: {e!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
