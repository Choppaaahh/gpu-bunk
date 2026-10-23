#!/usr/bin/env python3
"""cloud_sim_v4.py — make elastic result decision-grade. KA-5 / KA-1 become DECIDING arms; a frozen TAIL gate
(max wake wait + per-model p99 above a floor) stands beside the p95 gate; the biggest engines get two named arms (HOLD: their slots are
never parked; LR53: every tail engine pages in at the 53.8 GB/s line rate, MODELLED); the PEAK GPUs held in any minute (after a warm-up)
is reported as the rented-fleet size; the unmeasured batch-preemption delay p is swept over a frozen 11-point grid.

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. FORK BY IMPORT: evaluate, job path and static_row run UNCHANGED (v3.static_row); `overlay4` is v3's
overlay with additive keys only, so every v3 variant is recomputed with v3's semantics and must equal rows (I-FIX2794).

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n" (v3's GE1-GE7 + v4's GV1-GV5); exit 1 + FAIL lines. No world.
  --run --cell D1|D2|CP [--seeds 0,1,2] [--tag T] --root R --cache C --out F
                                    WRITES (appends) ONE JSON line to F: static scan row at the cell's N + "v4" (every v3
                                    variant + the 88 v4 variants, the in-job I-checks); exit 0. A failed in-job I-check exits 1.
  --ifix NEW_DIR OLD_DIR            read-only: I-FIX2794 (v4 rows vs rows in OLD_DIR), I-REPLAY, I-ALIAS, I-PIDLE, I-PMONO,
                                    I- and the in-job checks; one JSON object on stdout; exit 0 PASS / 1 FAIL.
  --summarize DIR                   read-only: prereg §5 verdicts over DIR/*.jsonl; one JSON object on stdout; exit 0.
  Bad input (unknown cell, missing file) or a held-out manifest mismatch = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **Every keep-alive policy starts the week HELD** (v3 held_keepalive `held |= t <= K - 1`), so a raw max of held GPUs is N for every
     arm (the TRAIN premise read, 2026-10-16) — PEAK_HELD reads minutes t >= WARM (240) only.
  2. **`ssh host '... &'` holds the session open** — launch jobs with `ssh -n` + `setsid -f`.
  3. **A prereg mutant with no site in the code** **a mutant surviving the goldens** — the mutation-test file anchors every
     mutant on a verbatim line of THIS file, run before any data.
"""
import argparse, collections, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import cloud_sim_v2 as V     # noqa: E402  (not edited)
import cloud_sim_v3 as V3    # noqa: E402  (not edited)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
SLOS = CS.SLOS
NREPS = V3.NREPS
BAR = V3.BAR
S2 = 2.0
P_POINT = V3.P_POINT                 # 0.79 s, point (UNMEASURED)
P_GRID = (0.0, 0.25, 0.5, 0.79, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)   # prereg §3, frozen
H2D_TAIL = V3.H2D_TAIL               # 22.6 GB/s MEASURED
H2D_LR = 53.8                        # line rate (NS.H2D_PCIE), MODELLED, not ours
WARM = 240                           # prereg §4: PEAK_HELD excludes the forced-held warm-up (KA-HIST cap)
P99_FLOOR = 100                      # prereg §4: per-model p99 only at n >= 100 requests per replicate
INF = V3.INF
CELLS = V3.CELLS
DEC_POLS = ("KA-5", "KA-1")
DEC_ARMS = ("BASE", "HOLD", "LR53")
ARMS = {"BASE": (H2D_TAIL, None), "HOLD": (H2D_TAIL, "BIG22"), "LR53": (H2D_LR, None), "LR53H": (H2D_LR, "BIG53")}
HOLD_GB = {"BIG22": H2D_TAIL, "BIG53": H2D_LR}
# variant -> (policy, p, tail h2d, cold, hold set)
VARIANTS = {k: v + (None,) for k, v in V3.VARIANTS.items()}
for _pol in DEC_POLS:
    for _arm, (_gb, _hold) in ARMS.items():
        for _p in P_GRID:
            VARIANTS[f"{_pol}|{_arm}|p{_p:g}"] = (_pol, _p, _gb, False, _hold)
V3_NAMES = tuple(V3.VARIANTS)
Q2M = ("E-HOUR", "E-PARK", "HE-R", "KA-1", "KA-5", "KA-60", "KA-HIST", "KA-1|HOLD|p0.79", "KA-5|HOLD|p0.79")
Q2L = Q2M + ("KA-1|LR53|p0.79", "KA-5|LR53|p0.79", "KA-1|LR53H|p0.79", "KA-5|LR53H|p0.79")
TIMING = V3.TIMING
GRANK = {"GATE-HOLDS": 0, "GATE-UNRESOLVED": 1, "GATE-FAILS": 2}


# ------------------------------------------------------------------------------------------------ hold sets + policies (prereg §3)
def big_set(slots, tail_gb):
    """BIG(rate): a slot holds an engine whose own wake at the point p cannot fit S2 (P_POINT + page-in > S2). Structure only."""
    out = []
    for sl in slots:
        gb = tail_gb if sl["kind"] == "tail" else sl["gbps"]
        w = max(V3.page_in(x, gb) for x in sl["units"].values())
        out.append(bool(P_POINT + w > S2))
    return out


def slot_policy(pol, b, k_hist, held_home):
    """a held-home slot is never parked (no wake); every other slot runs v3's policy unchanged."""
    if held_home:
        return V3.held_static(b)
    return V3.policy_arrays(pol, b, k_hist)


# ------------------------------------------------------------------------------------------------ tail metrics (prereg §4)
def pq_bounds(arr, dl, f):
    """(WORST, BEST) nearest-rank p_f keys over every pairing of delays to requests; v3's worst_p95 / best_p95 at f = 0.95."""
    n = len(arr)
    q = math.ceil(f * n)
    a = n - q + 1
    top = arr[q - 1:]
    d = np.sort(np.asarray(dl, np.int64))[::-1][:a]
    dd = np.zeros(a, np.int64)
    dd[:len(d)] = d
    s = np.where(top >= INF, INF, top + dd)
    worst = int(s.min())
    j = len(dl) - (n - q)
    if j <= 0:
        return worst, int(arr[q - 1])
    L = arr[:q]
    e = np.zeros(q, np.int64)
    e[q - j:] = np.sort(np.asarray(dl, np.int64))[:j]
    s2 = np.where(L >= INF, INF, L + e[::-1])
    return worst, int(s2.max())


def static_pq(week, days, f):
    """per LODO replicate: nearest-rank p_f (s | None | inf) of week - day d, read straight from the Counters (cstats's method)."""
    keys = np.array(sorted(week), np.int64)
    Wv = np.array([week[k] for k in sorted(week)], np.int64)
    idx = {int(k): j for j, k in enumerate(keys)}
    out = []
    for d in range(NDAYS):
        R = Wv.copy()
        for k, c in days[d].items():
            if c:
                R[idx[int(k)]] -= c
        tot = int(R.sum())
        if tot == 0:
            out.append(None)
            continue
        j = int(np.searchsorted(np.cumsum(R), math.ceil(f * tot), side="left"))
        out.append(V3._p(int(keys[j])))
    return out


def compose_p99(week, days, dl_by_day):
    """per LODO replicate: (n, WORST p99 s, BEST p99 s) of the static Counter (week - day d) + the delays of days != d."""
    keys = np.array(sorted(week), np.int64)
    Wv = np.array([week[k] for k in keys], np.int64)
    idx = {int(k): j for j, k in enumerate(keys)}
    out = dict(n=[], worst=[], best=[])
    for d in range(NDAYS):
        Dv = np.zeros(len(keys), np.int64)
        for k, c in days[d].items():
            if c:
                Dv[idx[int(k)]] = c
        arr = np.repeat(keys, Wv - Dv)
        dl = np.concatenate([np.asarray(dl_by_day[x], np.int64) for x in range(NDAYS) if x != d] + [np.zeros(0, np.int64)])
        out["n"].append(int(len(arr)))
        if len(arr) == 0:
            out["worst"].append(None)
            out["best"].append(None)
            continue
        if len(dl) > len(arr):
            raise RuntimeError("more delayed requests than requests")
        w, b = pq_bounds(arr, dl, 0.99)
        out["worst"].append(V3._p(w))
        out["best"].append(V3._p(b))
    return out


def p99_rep_pass(models, S):
    """models: [(n, static p99, WORST p99, BEST p99)] for one replicate -> (WORST all pass, BEST all pass, in-scope count, WORST max)."""
    scope = [(w, b) for n, sp, w, b in models if n >= P99_FLOOR and sp is not None and sp <= S]
    wmax = max((w for w, _ in scope), default=None)
    return all(w <= S for w, _ in scope), all(b <= S for _, b in scope), len(scope), wmax


def mww_reps(day_max_cs):
    """per LODO replicate: the max wake wait (s) over the days != d, from per-day maxima in centiseconds."""
    return [max([int(day_max_cs[x]) for x in range(NDAYS) if x != d] + [0]) / 100.0 for d in range(NDAYS)]


# ------------------------------------------------------------------------------------------------ peak held (prereg §4)
def peak_check(held_tot, busy_tot):
    """I-PEAK: held >= busy in every minute (raises loudly otherwise)."""
    h, b = np.asarray(held_tot), np.asarray(busy_tot)
    bad = np.flatnonzero(h < b)
    if len(bad):
        t = int(bad[0])
        raise RuntimeError(f"I-PEAK: minute {t} holds {int(h[t])} GPUs for {int(b[t])} busy")
    return True


def peak_of(series, warm, mask=None):
    s = np.asarray(series)
    keep = np.arange(len(s)) >= warm
    if mask is not None:
        keep &= mask
    return int(s[keep].max()) if keep.any() else None


def peak_reps(series, warm, day_len=DAY_MIN):
    """per LODO replicate d: the max of the series over minutes t >= warm of the days != d."""
    n = len(series)
    day = np.arange(n) // day_len
    return [peak_of(series, warm, day != d) for d in range(n // day_len)]


# ------------------------------------------------------------------------------------------------ verdicts (prereg §5)
def tail_gate(mww, p99w, p99b, S):
    if not (len(mww) == len(p99w) == len(p99b) == NREPS):
        raise ValueError(f"I-REPS: {len(mww)} / {len(p99w)} / {len(p99b)} tail replicates, need {NREPS}")
    if all(m <= S for m in mww) and all(p99w):
        return "TAIL-HOLDS"
    if any(m > S for m in mww) or not all(p99b):
        return "TAIL-FAILS"
    return "TAIL-UNRESOLVED"


def gate4(p95_gate, tg):
    if p95_gate == "SLO-HOLDS" and tg == "TAIL-HOLDS":
        return "GATE-HOLDS"
    if p95_gate == "SLO-FAILS" or tg == "TAIL-FAILS":
        return "GATE-FAILS"
    return "GATE-UNRESOLVED"


def headline4(g, idle):
    b = V3.bar_v(idle)
    if g == "GATE-HOLDS" and b == "UNDER-BAR":
        return "MEETS THE BAR AT SLO"
    if g == "GATE-HOLDS":
        return "HOLDS SLO, ABOVE THE BAR"
    if g == "GATE-FAILS":
        return "BREAKS THE SLO"
    return "UNRESOLVED"


def p_hold(gmap, grid=P_GRID):
    """P_HOLD = the largest grid p with GATE-HOLDS at every grid p' <= p (None if not at grid[0]); P_BREAK = the next point + verdict."""
    seq = [gmap[p] for p in grid]
    k = 0
    while k < len(seq) and seq[k] == "GATE-HOLDS":
        k += 1
    ph = grid[k - 1] if k else None
    pb = [grid[k], seq[k]] if k < len(seq) else None
    nonmono = any(GRANK[seq[i + 1]] < GRANK[seq[i]] for i in range(len(seq) - 1))
    return dict(p_hold=ph, p_break=pb, nonmono=nonmono)


def read_p(ph, pm):
    """prereg §5 Q1: read a future measured p off the grid; no interpolation."""
    if ph["p_hold"] is not None and pm <= ph["p_hold"]:
        return "HOLDS"
    if ph["p_break"] is not None and pm >= ph["p_break"][0]:
        return f"{ph['p_break'][1]} OR WORSE"
    return "re-run at the measured p"


def closest4(cands):
    """Q2: cands [(name, gate4, idle list)] -> (name, max, min) of the GATE-HOLDS policy with the smallest max idle, or None."""
    ok = [(max(i), min(i), n) for n, g, i in cands if g == "GATE-HOLDS"]
    if not ok:
        return None
    mx, mn, nm = min(ok)
    return nm, mx, mn


# ------------------------------------------------------------------------------------------------ the overlay (v3's, additive keys only)
def overlay4(wd, row, rec, snap, r, seeds, variants=None, check_score0=True):
    variants = variants or list(VARIANTS)
    cx = wd.cx
    tr = cx["tr"]
    M = cx["M"]
    adm = [cx["cid_m"][c] for c in snap["admitted"]]
    hosted = [wd.am] + adm
    allm = list(range(M))
    rst = {m: wd.ref_stats(m, r) for m in allm}
    rd = wd.req_day()
    held_mask = tr.minute >= HELD
    kk = np.flatnonzero(held_mask)
    train = ~held_mask
    t_start = time.time()
    res = {v: dict(idle_el=[], held_gpuh=[], parked_gpuh=[], parked_ge15=[], parked_ge60=[], fixed_idle_phi1=[], wakes=[],
                   wake_hit=[], wake_hit_share=[], peak_wakes_min=[], slo={v_: {str(x): collections.defaultdict(list) for x in SLOS}
                                                                          for v_ in ("worst", "best")}, fails={},
                   mww=[], p99={str(x): dict(worst_pass=[], best_pass=[], scope=[], worst_max=[]) for x in SLOS},
                   peak_held=[], peak_busy=[], busy_day=[], held_day=[], held_home_slots=[], held_home_gpus=[])
           for v in variants}
    checks = dict(I_BUSY=True, I_HELD=True, I_STATIC0=True, I_SCORE0=None, I_HOST_parked_blocked=0, I_WAKE=True, I_PEAK=True,
                  I_SCORE0_P99=None)
    req_rep_tot = []
    hold_info = {}
    for si, seed in enumerate(seeds):
        fi = rec["fi"][si]
        slots = V3.build_slots(wd, fi)
        hsets = {h: big_set(slots, gb) for h, gb in HOLD_GB.items()}
        hold_info[str(seed)] = {h: dict(slots=int(sum(x)), gpus=int(sum(sl["g"] for sl, f in zip(slots, x) if f)),
                                        kinds=dict(collections.Counter(sl["kind"] for sl, f in zip(slots, x) if f)))
                                for h, x in hsets.items()}
        bs = [V3.slot_busy(sl, fi) for sl in slots]
        tot = np.zeros(WEEK_MIN, np.int64)
        for sl, b in zip(slots, bs):
            tot += sl["g"] * b.astype(np.int64)
        ref = V.busy_gpus_per_min(fi["B"], fi["slots"], fi["acnt"], fi["r"], fi["g"], WEEK_MIN)
        if not np.array_equal(tot, ref):
            checks["I_BUSY"] = False
            raise RuntimeError("I-BUSY: slot busy series disagree with 's per-minute busy GPUs")
        first, last, _ = T.arrivals(tr, seed)
        cells_of = collections.defaultdict(list)
        for i in kk:
            cells_of[int(tr.mid[i])].append(int(i))
        slot_cells = []
        for sl in slots:
            ids = sorted((c for m in sl["models"] for c in cells_of.get(m, [])), key=lambda c: (int(tr.minute[c]), float(first[c]), c))
            mins = np.array([int(tr.minute[c]) - HELD for c in ids], np.int64)
            slot_cells.append((mins, ids))
        offs_cache = {}

        def offs(c):
            if c not in offs_cache:
                offs_cache[c] = T.cell_times(tr, c, first, last, seed)
            return offs_cache[c]
        khs = []
        for sl in slots:
            k = train & np.isin(tr.mid, sl["models"])
            khs.append(V3.k_hist(tr.minute[k]))
        sc = V3.static_counters(wd, rec, snap, r, seed)
        st_stats = {m: CS.cstats(*sc[m]) for m in hosted}
        st_p99 = {m: static_pq(sc[m][0], sc[m][1], 0.99) for m in hosted}
        if check_score0:
            z = [np.zeros(0, np.int64)] * NDAYS
            ok0 = True
            ok99 = True
            for m in hosted:
                cw = V3.compose(sc[m][0], sc[m][1], z)["worst"]
                s0 = st_stats[m]
                if cw["n"] != s0["n"] or cw["p95"] != s0["p95"] or any(cw["ok"][str(x)] != s0["ok"][str(x)] for x in SLOS):
                    ok0 = False
                    break
                c99 = compose_p99(sc[m][0], sc[m][1], z)
                if c99["worst"] != st_p99[m] or c99["best"] != st_p99[m]:
                    ok99 = False
                    break
            for x in SLOS:
                sr = CS.score_reps(hosted, st_stats, rst, allm, x)
                for k_, v_ in sr.items():
                    if row["slo"][str(x)][k_][si * NDAYS:(si + 1) * NDAYS] != v_:
                        ok0 = False
            checks["I_SCORE0"] = ok0 if checks["I_SCORE0"] in (None, True) else False
            checks["I_SCORE0_P99"] = ok99 if checks["I_SCORE0_P99"] in (None, True) else False
            if not ok0:
                raise RuntimeError("I-SCORE0: zero-delay composition does not reproduce the row's slo block")
            if not ok99:
                raise RuntimeError("I-SCORE0-P99: zero-delay p99 composition != the static p99 from the Counters")
        rep_req = [int(sum(int(rd[:, m].sum() - rd[x, m]) for m in hosted)) for x in range(NDAYS)]
        req_rep_tot.extend(rep_req)
        pol_cache = {}
        for v in variants:
            pol, p, h2d, cold, hold = VARIANTS[v]
            hset = hsets[hold] if hold else [False] * len(slots)
            busy_d = np.zeros(NDAYS, np.int64)
            held_d = np.zeros(NDAYS, np.int64)
            park_d = np.zeros(NDAYS, np.int64)
            p15_d = np.zeros(NDAYS, np.int64)
            p60_d = np.zeros(NDAYS, np.int64)
            wake_d = np.zeros(NDAYS, np.int64)
            wake_min = np.zeros(WEEK_MIN, np.int64)
            held_tot = np.zeros(WEEK_MIN, np.int64)
            cell_delay = {}
            for s_i, (sl, b) in enumerate(zip(slots, bs)):
                key = (pol, s_i, bool(hset[s_i]))
                if key not in pol_cache:
                    if not hset[s_i] and not sl["host_ok"] and pol != "STATIC":
                        checks["I_HOST_parked_blocked"] += 1
                        pol_cache[key] = V3.held_static(b)
                    else:
                        pol_cache[key] = slot_policy(pol, b, khs[s_i], bool(hset[s_i]))
                held, wake = pol_cache[key]
                if (b & ~held).any():
                    checks["I_HELD"] = False
                    raise RuntimeError(f"I-HELD: busy minute not held ({v}, slot {s_i})")
                g = sl["g"]
                held_tot += g * held.astype(np.int64)
                busy_d += g * b.reshape(NDAYS, DAY_MIN).sum(axis=1)
                held_d += g * held.reshape(NDAYS, DAY_MIN).sum(axis=1)
                pk = ~held
                park_d += g * pk.reshape(NDAYS, DAY_MIN).sum(axis=1)
                if pk.any():
                    edges = np.flatnonzero(np.diff(np.concatenate([[0], pk.astype(np.int8), [0]])))
                    runlen = np.zeros(WEEK_MIN, np.int64)
                    for a_, z_ in zip(edges[::2], edges[1::2]):
                        runlen[a_:z_] = z_ - a_
                    p15_d += g * (pk & (runlen >= 15)).reshape(NDAYS, DAY_MIN).sum(axis=1)
                    p60_d += g * (pk & (runlen >= 60)).reshape(NDAYS, DAY_MIN).sum(axis=1)
                wk_t = np.flatnonzero(wake)
                wake_d += np.bincount(wk_t // DAY_MIN, minlength=NDAYS)[:NDAYS]
                wake_min[wk_t] += 1
                if not len(wk_t):
                    continue
                mins, ids = slot_cells[s_i]
                kind = "tail" if sl["kind"] == "tail" else "trigger"
                gb = (h2d if sl["kind"] == "tail" else sl["gbps"]) if h2d is not None else None
                for t in wk_t:
                    te = V3.window_end(pol, held, int(t))
                    a_, z_ = np.searchsorted(mins, t, "left"), np.searchsorted(mins, te, "right")
                    cl = [(int(mins[q]), float(first[ids[q]]), int(tr.mid[ids[q]]), sl["unit_of"][int(tr.mid[ids[q]])], ids[q])
                          for q in range(a_, z_)]
                    if not cl or cl[0][0] != t:
                        checks["I_WAKE"] = False
                    for cid, m, mi, dd in V3.window_delays(kind, cl, int(t), p if p is not None else 0.0,
                                                           (lambda u, _sl=sl, _gb=gb: V3.page_in(_sl["units"][u], _gb)), cold, offs):
                        prev = cell_delay.get(cid)
                        cell_delay[cid] = dd if prev is None else np.maximum(prev, dd)
            o = res[v]
            o["idle_el"].extend(V3.lodo(busy_d, held_d))
            B_, H_, P_ = busy_d.sum(), held_d.sum(), park_d.sum()
            o["held_gpuh"].extend([(H_ - held_d[d]) / 60.0 for d in range(NDAYS)])
            o["parked_gpuh"].extend([(P_ - park_d[d]) / 60.0 for d in range(NDAYS)])
            o["parked_ge15"].extend([None if P_ - park_d[d] == 0 else float(p15_d.sum() - p15_d[d]) / float(P_ - park_d[d])
                                     for d in range(NDAYS)])
            o["parked_ge60"].extend([None if P_ - park_d[d] == 0 else float(p60_d.sum() - p60_d[d]) / float(P_ - park_d[d])
                                     for d in range(NDAYS)])
            o["fixed_idle_phi1"].extend([float((H_ - held_d[d]) - (B_ - busy_d[d])) / float(row["N"] * DAY_MIN * (NDAYS - 1))
                                         for d in range(NDAYS)])
            o["wakes"].extend([int(wake_d.sum() - wake_d[d]) for d in range(NDAYS)])
            o["peak_wakes_min"].append(int(wake_min.max()))
            # v4: peak held (I-PEAK), per-day held / busy for the rented-fleet idle
            peak_check(held_tot, tot)
            o["peak_held"].extend(peak_reps(held_tot, WARM))
            o["peak_busy"].extend(peak_reps(tot, WARM))
            o["busy_day"].append([int(x) for x in busy_d])
            o["held_day"].append([int(x) for x in held_d])
            o["held_home_slots"].append(int(sum(hset)))
            o["held_home_gpus"].append(int(sum(sl["g"] for sl, f in zip(slots, hset) if f)))
            dl = collections.defaultdict(lambda: [[] for _ in range(NDAYS)])
            for cid, dd in cell_delay.items():
                dc = np.round(dd * 100.0).astype(np.int64)
                dc = dc[dc > 0]
                if len(dc):
                    dl[int(tr.mid[cid])][(int(tr.minute[cid]) - HELD) // DAY_MIN].extend(dc.tolist())
            hit_d = np.zeros(NDAYS, np.int64)
            day_max = np.zeros(NDAYS, np.int64)
            for m, per in dl.items():
                for d in range(NDAYS):
                    hit_d[d] += len(per[d])
                    if per[d]:
                        day_max[d] = max(day_max[d], max(per[d]))
            o["wake_hit"].extend([int(hit_d.sum() - hit_d[d]) for d in range(NDAYS)])
            o["wake_hit_share"].extend([(int(hit_d.sum() - hit_d[d]) / rep_req[d]) if rep_req[d] else None for d in range(NDAYS)])
            o["mww"].extend(mww_reps(day_max))
            ost = {"worst": dict(st_stats), "best": dict(st_stats)}
            c99 = {}
            for m, per in dl.items():
                if m not in sc:
                    raise RuntimeError(f"a delay on a model that is not hosted: {m}")
                dlm = [np.asarray(x, np.int64) for x in per]
                cmp_ = V3.compose(sc[m][0], sc[m][1], dlm)
                ost["worst"][m] = cmp_["worst"]
                ost["best"][m] = cmp_["best"]
                c99[m] = compose_p99(sc[m][0], sc[m][1], dlm)
            for x in SLOS:
                q99 = o["p99"][str(x)]
                for d in range(NDAYS):
                    mods = []
                    for m in hosted:
                        n_, sp = st_stats[m]["n"][d], st_p99[m][d]
                        w_, b_ = (c99[m]["worst"][d], c99[m]["best"][d]) if m in c99 else (sp, sp)
                        mods.append((n_, sp, w_, b_))
                    pw, pb, ns, wm = p99_rep_pass(mods, x)
                    q99["worst_pass"].append(pw)
                    q99["best_pass"].append(pb)
                    q99["scope"].append(ns)
                    q99["worst_max"].append(wm if wm is None or wm != math.inf else 1e9)
            for vv in ("worst", "best"):
                for x in SLOS:
                    srr = CS.score_reps(hosted, ost[vv], rst, allm, x)
                    for k_, v_ in srr.items():
                        o["slo"][vv][str(x)][k_].extend(v_)
                    if vv == "worst":
                        f = o["fails"].setdefault(str(x), collections.Counter())
                        mx = o["fails"].setdefault(f"{x}_maxp95", {})
                        for m in hosted:
                            for d in range(NDAYS):
                                rr, oo = rst[m], ost["worst"][m]
                                if rr["n"][d] >= 20 and rr["p95"][d] <= x and oo["p95"][d] > x:
                                    f[tr.models[m][0][:8]] += 1
                                    k8 = tr.models[m][0][:8]
                                    mx[k8] = max(mx.get(k8, 0.0), oo["p95"][d] if oo["p95"][d] != math.inf else 1e9)
    out = {}
    for v in variants:
        o = res[v]
        g = {}
        tg, g4 = {}, {}
        for x in SLOS:
            g[str(x)] = V3.slo_gate(o["slo"]["worst"][str(x)]["pass"], o["slo"]["best"][str(x)]["pass"])
            tg[str(x)] = tail_gate(o["mww"], o["p99"][str(x)]["worst_pass"], o["p99"][str(x)]["best_pass"], x)
            g4[str(x)] = gate4(g[str(x)], tg[str(x)])
        o["gate"] = g
        o["tail_gate"] = tg
        o["gate4"] = g4
        o["slo"] = {vv: {x: dict(d_) for x, d_ in o["slo"][vv].items()} for vv in o["slo"]}
        o["fails"] = {k: (sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:8] if isinstance(c, collections.Counter) else
                          dict(sorted(c.items(), key=lambda kv: -kv[1])[:8])) for k, c in o["fails"].items()}
        o["idle_el"] = [None if x is None else round(x, 6) for x in o["idle_el"]]
        out[v] = o
    st = out.get("STATIC")
    if st is not None:
        ia = row["idle_alloc"]
        checks["I_STATIC0"] = len(ia) == len(st["idle_el"]) and all(abs(a - b) < 5e-7 for a, b in zip(st["idle_el"], ia))
        if not checks["I_STATIC0"]:
            raise RuntimeError("I-STATIC0: STATIC IDLE_EL != the row's idle_alloc")
    return dict(variants=out, checks=checks, req_rep_tot=req_rep_tot, overlay_s=round(time.time() - t_start, 1), hold_info=hold_info,
                consts=dict(P_POINT=P_POINT, H2D_TAIL=H2D_TAIL, H2D_TAIL_SENS=V3.H2D_TAIL_SENS, COLD_S=V3.COLD_S, KA_PARK=V3.KA_PARK,
                            BAR=BAR, tph_h2d=wd.tph["h2d"], P_GRID=list(P_GRID), H2D_LR=H2D_LR, WARM=WARM, P99_FLOOR=P99_FLOOR, S2=S2))


def run_cell(cell, seeds, tag, root, cache, out):
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell}")
    wd, row, rec, snap, r = V3.static_row(cell, seeds, tag, root, cache)
    if abs(wd.tph["h2d"] - H2D_TAIL) > 1e-9:
        raise RuntimeError(f"prereg §2: the cloud sim's tail h2d is {wd.tph['h2d']}, not {H2D_TAIL}")
    row["cell"] = cell
    row["v4"] = overlay4(wd, row, rec, snap, r, seeds)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks over rows
def _strip(x):
    return V3._strip(x)


def fix_diffs(new, old):
    """I-FIX2794 for one (tag, cell): static keys + every v3 variant's keys + req_rep_tot + v3's checks/consts, timing excluded."""
    diffs = []
    for k in sorted(set(old) - {"v3"} - TIMING):
        if _strip(new.get(k)) != _strip(old.get(k)):
            diffs.append(k)
    nv, ov = new["v4"], old["v3"]
    for V_, od in ov["variants"].items():
        nd = nv["variants"].get(V_)
        if nd is None:
            diffs.append(f"variant {V_} missing")
            continue
        for k, x in od.items():
            if _strip(nd.get(k)) != _strip(x):
                diffs.append(f"{V_}.{k}")
    if nv["req_rep_tot"] != ov["req_rep_tot"]:
        diffs.append("req_rep_tot")
    for blk in ("checks", "consts"):
        for k, x in ov[blk].items():
            if nv[blk].get(k) != x:
                diffs.append(f"{blk}.{k}")
    return diffs


def mono_ok(row):
    """I-PMONO + I-PIDLE over one row's v4 variants."""
    V4 = row["v4"]["variants"]
    bad = []
    for pol in DEC_POLS:
        for arm in ARMS:
            names = [f"{pol}|{arm}|p{p:g}" for p in P_GRID]
            base = V4[names[0]]
            for nm in names[1:]:
                for k in ("idle_el", "held_gpuh", "peak_held"):
                    if V4[nm][k] != base[k]:
                        bad.append(f"I-PIDLE {nm}.{k}")
            for a, b in zip(names, names[1:]):
                if any(y < x for x, y in zip(V4[a]["mww"], V4[b]["mww"])):
                    bad.append(f"I-PMONO mww {a}->{b}")
                for s in V4[a]["gate4"]:
                    if GRANK[V4[b]["gate4"][s]] < GRANK[V4[a]["gate4"][s]]:
                        bad.append(f"I-PMONO gate4 {a}->{b} S{s}")
                    if any(y and not x for x, y in zip(V4[a]["p99"][s]["worst_pass"], V4[b]["p99"][s]["worst_pass"])):
                        bad.append(f"I-PMONO p99 {a}->{b} S{s}")
        for p in P_GRID:
            a, b = V4[f"{pol}|BASE|p{p:g}"], V4[f"{pol}|LR53|p{p:g}"]
            for k in ("idle_el", "held_gpuh", "peak_held"):
                if a[k] != b[k]:
                    bad.append(f"I-PIDLE BASE vs LR53 {pol} p{p:g}.{k}")
        al, ka = V4[f"{pol}|BASE|p0.79"], V4[pol]
        for k in ka:
            if _strip(al.get(k)) != _strip(ka[k]):
                bad.append(f"I-ALIAS {pol}.{k}")
    return bad


def ifix(new_dir, old_dir):
    rows = V3._rows(new_dir)
    by = {(r["tag"], r["cell"]): r for r in rows}
    olds = V3._rows(old_dir)
    res, ok = {}, True
    for o in olds:
        key = (o["tag"], o["cell"])
        n = by.get(key)
        if n is None:
            res[f"I_FIX2794_{key[0]}_{key[1]}"] = "MISSING"
            ok = False
            continue
        dfs = fix_diffs(n, o)
        res[f"I_FIX2794_{key[0]}_{key[1]}"] = dict(variants=len(o["v3"]["variants"]), diffs=dfs[:8], n_diffs=len(dfs))
        ok = ok and not dfs
    res["I_FIX2794_rows"] = len(olds)
    ok = ok and len(olds) == 6
    for (tag, cell), r in sorted(by.items()):
        ch = r["v4"]["checks"]
        res[f"checks_{tag}_{cell}"] = ch
        ok = ok and all(ch[k] for k in ("I_BUSY", "I_HELD", "I_STATIC0", "I_SCORE0", "I_WAKE", "I_PEAK", "I_SCORE0_P99"))
        bad = mono_ok(r)
        res[f"mono_{tag}_{cell}"] = bad[:8]
        ok = ok and not bad
    d1 = by.get(("main", "D1"))
    if d1 is None:
        ok = False
        res["I_ECOLD_TAIL"] = "MISSING"
    else:
        ec = d1["v4"]["variants"]["E-COLD"]["tail_gate"]
        res["I_ECOLD_TAIL"] = ec
        ok = ok and all(v == "TAIL-FAILS" for v in ec.values())
    for cell in ("D1", "D2"):
        a = by.get(("main", cell))
        for t in ("rep_b", "rep_c"):
            b = by.get((t, cell))
            if a is None or b is None:
                res[f"I_REPLAY_{cell}_{t}"] = "MISSING"
                ok = False
                continue
            sa, sb = _strip(dict(a, tag=None)), _strip(dict(b, tag=None))
            diff = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
            res[f"I_REPLAY_{cell}_{t}"] = dict(diffs=diff[:5])
            ok = ok and not diff
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summarize (prereg §5)
def _rng(v):
    return V3._rng(v)


def rented(o, N):
    R = max(o["peak_held"])
    phi0, phi1 = [], []
    mins = DAY_MIN * (NDAYS - 1)
    for bd, hd in zip(o["busy_day"], o["held_day"]):
        for d in range(NDAYS):
            b = sum(bd) - bd[d]
            h = sum(hd) - hd[d]
            phi0.append(1.0 - b / float(R * mins))
            phi1.append((h - b) / float(R * mins))
    return dict(rented=R, N=N, saved=N - R, verdict="FEWER-RENTED" if R < N else "SAME-AS-STATIC", peak_held=_rng(o["peak_held"]),
                peak_busy=_rng(o["peak_busy"]), idle_el=_rng(o["idle_el"]), phi0=_rng(phi0), phi0_bar=V3.bar_v(phi0),
                phi1=_rng(phi1), phi1_bar=V3.bar_v(phi1))


def arm_rows(row, pol, arm):
    V4 = row["v4"]["variants"]
    out = {}
    for s in (str(x) for x in SLOS):
        gm = {p: V4[f"{pol}|{arm}|p{p:g}"]["gate4"][s] for p in P_GRID}
        ph = p_hold(gm)
        pt = V4[f"{pol}|{arm}|p{P_POINT:g}"]
        idle = pt["idle_el"]
        sent = (f"{row['cell']} S{s} {pol}|{arm}: {V3.bar_v(idle)} ({_rng(idle)}); GATE-HOLDS for p <= "
                f"{ph['p_hold'] if ph['p_hold'] is not None else 'NONE'} s (grid); at P_BREAK: {ph['p_break']}")
        out[s] = dict(bar=V3.bar_v(idle), idle=_rng(idle), headline_point=headline4(pt["gate4"][s], idle), **ph,
                      gate_by_p={f"{p:g}": gm[p] for p in P_GRID},
                      p95_by_p={f"{p:g}": V4[f"{pol}|{arm}|p{p:g}"]["gate"][s] for p in P_GRID},
                      tail_by_p={f"{p:g}": V4[f"{pol}|{arm}|p{p:g}"]["tail_gate"][s] for p in P_GRID},
                      mww_by_p={f"{p:g}": _rng(V4[f"{pol}|{arm}|p{p:g}"]["mww"]) for p in P_GRID},
                      p99_scope_point=_rng(pt["p99"][s]["scope"]), p99_worst_max_point=_rng(pt["p99"][s]["worst_max"]),
                      sentence=sent)
    return out


def cell_v4(row):
    V4 = row["v4"]["variants"]
    out = dict(q1={}, desc={}, q2={}, q4={}, v3_settings={}, hold_info=row["v4"]["hold_info"])
    for pol in DEC_POLS:
        for arm in ARMS:
            (out["q1"] if arm in DEC_ARMS else out["desc"])[f"{pol}|{arm}"] = arm_rows(row, pol, arm)
            out["q4"][f"{pol}|{arm}"] = rented(V4[f"{pol}|{arm}|p{P_POINT:g}"], row["N"])
    for s in (str(x) for x in SLOS):
        q = {}
        for nm, st in (("M", Q2M), ("L", Q2L)):
            c = closest4([(v, V4[v]["gate4"][s], V4[v]["idle_el"]) for v in st])
            q[nm] = ("NO ELASTIC POLICY HOLDS THE SLO WITH ITS TAIL" if c is None else
                     dict(closest=c[0], idle=[round(c[2], 4), round(c[1], 4)], dist=[round(c[2] - BAR, 4), round(c[1] - BAR, 4)]))
        out["q2"][s] = q
    for v in V3_NAMES:
        o = V4[v]
        out["v3_settings"][v] = dict(idle=_rng(o["idle_el"]), gate=o["gate"], tail_gate=o["tail_gate"], gate4=o["gate4"],
                                     mww=_rng(o["mww"]), peak_held=_rng(o["peak_held"]), rented=max(o["peak_held"]))
    return out


def verdict_key(c):
    """the verdicts the seed-shift compares (prereg §6)."""
    k = {}
    for a, rr in c["q1"].items():
        for s, x in rr.items():
            k[f"q1|{a}|{s}"] = [x["bar"], x["p_hold"], x["headline_point"]]
    for s, q in c["q2"].items():
        for nm, x in q.items():
            k[f"q2|{nm}|{s}"] = x if isinstance(x, str) else x["closest"]
    for a, x in c["q4"].items():
        if a.split("|")[1] in DEC_ARMS:
            k[f"q4|{a}"] = x["verdict"]
    return k


def summarize(d):
    rows = V3._rows(d)
    by = {(r["tag"], r["cell"]): r for r in rows}
    out = dict(cells={}, q3={}, seedshift={})
    for cell in ("D1", "D2"):
        r = by.get(("main", cell))
        if r:
            out["cells"][cell] = cell_v4(r)
            out["cells"][cell]["N"] = r["N"]
            out["cells"][cell]["static_idle_alloc"] = _rng(r["idle_alloc"])
        ss = by.get(("seedshift", cell))
        if r and ss:
            a, b = verdict_key(cell_v4(r)), verdict_key(cell_v4(ss))
            out["seedshift"][cell] = dict(UNATTRIBUTABLE={k: [a[k], b.get(k)] for k in a if a[k] != b.get(k)},
                                          n_compared=len(a), q1_seedshift={k: v for k, v in b.items() if k.startswith("q1")})
    d1 = by.get(("main", "D1"))
    if d1:
        e = d1["v4"]["variants"]["E-COLD"]
        out["q3"] = dict(tail_gate=e["tail_gate"], gate4=e["gate4"], p95_gate=e["gate"], mww=_rng(e["mww"]))
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    rc3 = V3.self_test()
    res = []

    def chk(name, got, exp, tol=None):
        if tol is None:
            ok = got == exp
        else:
            ok = len(got) == len(exp) and all(abs(a - b) <= tol for a, b in zip(got, exp))
        res.append((name, ok, got, exp))
    R = NREPS
    # GV1: passes p95, breaks the tail -> FAILS
    arr20 = np.full(20, 10, np.int64)
    chk("GV1 p95 WORST hides one 55 s wake", V3.worst_p95(arr20, [5500]), 10)
    chk("GV1 n=20 has nothing in p99 scope", p99_rep_pass([(20, 0.10, 55.10, 55.10)], 2.0), (True, True, 0, None))
    p95g = V3.slo_gate([True] * R, [True] * R)
    tg1 = tail_gate([55.0] * R, [True] * R, [True] * R, 2.0)
    chk("GV1 p95 gate", p95g, "SLO-HOLDS")
    chk("GV1 tail gate", tg1, "TAIL-FAILS")
    chk("GV1 GATE", gate4(p95g, tg1), "GATE-FAILS")
    chk("GV1 headline at idle 0.05", headline4(gate4(p95g, tg1), [0.05] * R), "BREAKS THE SLO")
    # GV1b: p99 pairing bounds
    arrb = np.array([10] * 197 + [80, 90, 100], np.int64)
    chk("GV1b WORST / BEST p99 (cs)", pq_bounds(arrb, [150, 120, 90], 0.99), (190, 100))
    chk("GV1b zero delay = static p99", pq_bounds(arrb, [], 0.99), (80, 80))
    arr4 = np.arange(10, 201, 10, dtype=np.int64)
    chk("GV1b q=0.95 equals v3's bounds (GE4)", [pq_bounds(arr4, [50, 30, 20], 0.95), (V3.worst_p95(arr4, [50, 30, 20]),
                                                                                     V3.best_p95(arr4, [50, 30, 20]))],
        [(230, 190), (230, 190)])
    chk("GV1b q=0.95 equals v3 (GE4b)", pq_bounds(np.full(20, 10, np.int64), [100, 5], 0.95), (V3.worst_p95(np.full(20, 10, np.int64),
                                                                                                         [100, 5]), 15))
    # GV1c: tail UNRESOLVED (through compose_p99 with LODO: the 200 requests sit on day 1, the delays on day 0)
    arrc = np.array([10] * 197 + [100] * 3, np.int64)
    chk("GV1c WORST / BEST p99 (cs)", pq_bounds(arrc, [150, 150, 150], 0.99), (250, 160))
    wk = collections.Counter({10: 197 + 7, 100: 3})
    dys = [collections.Counter() for _ in range(NDAYS)]
    dys[0] = collections.Counter({10: 7})
    c99 = compose_p99(wk, dys, [[150, 150, 150] if d == 1 else [] for d in range(NDAYS)])
    chk("GV1c compose_p99 replicate 0 (day 0 dropped, day-1 delays kept)", [c99["n"][0], c99["worst"][0], c99["best"][0]], [200, 2.5, 1.6])
    chk("GV1c compose_p99 replicate 1 (its delays dropped)", [c99["worst"][1], c99["best"][1]], [1.0, 1.0])
    chk("GV1c static p99 from the Counters", static_pq(wk, dys, 0.99)[0], 1.0)
    pw, pb, ns, wm = p99_rep_pass([(200, 1.0, 2.5, 1.6)], 2.0)
    chk("GV1c p99 replicate pass", (pw, pb, ns, wm), (False, True, 1, 2.5))
    tgc = tail_gate([1.5] * R, [False] * R, [True] * R, 2.0)
    chk("GV1c tail gate", tgc, "TAIL-UNRESOLVED")
    chk("GV1c GATE", gate4("SLO-HOLDS", tgc), "GATE-UNRESOLVED")
    chk("GV1c scope: n=99 out", p99_rep_pass([(99, 1.0, 2.5, 1.6)], 2.0)[2], 0)
    chk("GV1c scope: static p99 2.5 out at S2, in at S10", [p99_rep_pass([(200, 2.5, 3.0, 2.6)], 2.0)[2],
                                                            p99_rep_pass([(200, 2.5, 3.0, 2.6)], 10.0)[2]], [0, 1])
    # GV2: a held home never wakes
    bA = np.array([0, 1, 0, 0, 1, 0, 0, 0], bool)
    hA, wA = slot_policy("KA-1", bA, None, True)
    chk("GV2 held home: held all, no wake", [hA.astype(int).tolist(), np.flatnonzero(wA).tolist()], [[1] * 8, []])
    bB = np.array([1, 0, 0, 1], bool)
    hB, wB = slot_policy("KA-1", bB, None, False)
    chk("GV2 not held: KA-1", [hB.astype(int).tolist(), np.flatnonzero(wB).tolist()], [[1, 1, 0, 1], [3]])
    sl = lambda w: dict(kind="tail", units={"u": w}, gbps=None)  # noqa: E731
    chk("GV2b BIG boundary at 22.6 GB/s", big_set([sl(25.2), sl(25.0)], H2D_TAIL), [True, False])
    chk("GV2b not BIG at 53.8 GB/s", big_set([sl(25.2)], H2D_LR), [False])
    chk("GV2b page-in values", [round(V3.page_in(25.2, 22.6), 4), round(V3.page_in(25.0, 22.6), 4)], [1.2173, 1.2078])
    chk("GV2b giant uses its own G x 53.8 (G 4: 0.32 s, not BIG; at the tail rate it would be 2.87 s)",
        big_set([dict(kind="giant", units={"g": 60.0}, gbps=4 * 53.8)], H2D_TAIL), [False])
    # GV3: adversarial peak
    try:
        peak_check([8, 8, 8, 8], [5, 7, 9, 4])
        raised = False
    except RuntimeError:
        raised = True
    chk("GV3 held below the busiest minute raises I-PEAK", raised, True)
    chk("GV3 peak held / busy", [peak_check([8, 8, 9, 8], [5, 7, 9, 4]), peak_of([8, 8, 9, 8], 0), peak_of([5, 7, 9, 4], 0)], [True, 9, 9])
    chk("GV3 warm-up excluded", peak_of([12, 12, 5, 6, 7, 5], 2), 7)
    chk("GV3 LODO peaks", peak_reps([9, 4, 5, 6, 7, 3], 1, day_len=3), [7, 5])
    # GV4: P_HOLD
    H, U, F = "GATE-HOLDS", "GATE-UNRESOLVED", "GATE-FAILS"
    g4a = dict(zip(P_GRID, [H, H, H, H, F, F, F, F, F, F, F]))
    pa = p_hold(g4a)
    chk("GV4 P_HOLD / P_BREAK", [pa["p_hold"], pa["p_break"], pa["nonmono"]], [0.79, [1.0, F], False])
    pb_ = p_hold(dict(zip(P_GRID, [H, U, H] + [F] * 8)))
    chk("GV4 non-monotone prefix", [pb_["p_hold"], pb_["nonmono"]], [0.0, True])
    chk("GV4 NONE", p_hold(dict(zip(P_GRID, [F] * 11)))["p_hold"], None)
    chk("GV4 read p", [read_p(pa, 0.6), read_p(pa, 0.9), read_p(pa, 1.2)], ["HOLDS", "re-run at the measured p", "GATE-FAILS OR WORSE"])
    chk("GV4 closest4 needs GATE-HOLDS", [closest4([("X", F, [0.01] * R), ("Y", H, [0.3] * R)]), closest4([("X", U, [0.01] * R)])],
        [("Y", 0.3, 0.3), None])
    # GV5: MWW per replicate
    chk("GV5 replicate maxima", mww_reps([129, 0, 300, 0, 0, 0, 0]), [3.0, 3.0, 1.29, 3.0, 3.0, 3.0, 3.0])
    of = {0: np.array([5.0, 5.4, 9.0]), 1: np.array([6.0])}
    g6 = V3.window_delays("tail", [(10, 5.0, 1, "U1", 0), (10, 6.0, 2, "U2", 1)], 10, 0.79, {"U1": 0.50, "U2": 0.30}.get, False, of.get)
    dmx = max(int(np.round(x[3] * 100.0).max()) for x in g6)
    chk("GV5 through the delay engine (GE6)", mww_reps([dmx, 0, 0, 0, 0, 0, 0])[1], 1.29)
    # variant table sanity (prereg §3)
    chk("variants: 18 v3 + 88 v4", [len(V3_NAMES), len(VARIANTS)], [18, 106])
    chk("I-ALIAS settings", [VARIANTS["KA-5|BASE|p0.79"][:4], VARIANTS["KA-5"][:4]], [("KA-5", 0.79, 22.6, False)] * 2)
    n_ok = sum(1 for r in res if r[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    allok = n_ok == len(res) and rc3 == 0
    print(f"SELF-TEST {'PASS' if allok else 'FAIL'} {n_ok}/{len(res)} (v4) + v3 rc {rc3}")
    return 0 if allok else 1


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--ifix", nargs=2, metavar=("NEW_DIR", "OLD_DIR"))
    m.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--cell")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.ifix:
            for p in a.ifix:
                if not os.path.isdir(p):
                    raise ValueError(f"no such dir {p}")
            r = ifix(*a.ifix)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"no such dir {a.summarize}")
            print(json.dumps(summarize(a.summarize)))
            return 0
        for k in ("cell", "root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        for p in (a.root, a.cache):
            if not os.path.exists(p):
                raise ValueError(f"missing {p}")
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if len(seeds) * NDAYS != NREPS:
            raise ValueError(f"{len(seeds)} seeds x {NDAYS} days != {NREPS} replicates")
        return run_cell(a.cell, seeds, a.tag, a.root, a.cache, a.out)
    except (ValueError, V.ManifestError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
