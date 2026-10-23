#!/usr/bin/env python3
"""cloud_sim_v3.py — an SLO-tested ELASTIC arm on demand-sized fleets. Every slot (a tail card or dedicated
tail GPU, a giant group, an anchor replica) is HELD (engines awake, GPU serving) or PARKED (engines asleep in pinned host RAM, GPU
released to batch); a request on a parked slot reclaims it and pays a wake (batch preemption p + per-engine page-in, one at a time
per slot). Scored at S2 / S10 on held-out days 307-313 with 21 replicates; idle over HELD GPU-time.

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. FORK BY IMPORT: evaluate and job path run UNCHANGED; v3 RECORDS their per-model static TTFT Counters, busy
matrix and slot list (wrapping CS.run_tail / CS.run_dgroups / CS.fleet_idle for the duration of one evaluate) and overlays the
elastic policies on them. Elastic TTFT = static TTFT + the request's wake residual; per model and replicate the SLO is bounded over
every pairing of delays to requests (WORST / BEST), scored by own score_reps.

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n"; exit 1 + FAIL lines. Pure goldens GE1-GE7 (no world, no trace).
  --run --cell D1|D2|CP [--seeds 0,1,2] [--tag T] --root R --cache C --out F
                                    WRITES (appends) ONE JSON line to F: static scan row at the cell's N + "v3" (every
                                    policy variant, the I-checks run inside the job); exit 0. A failed in-job I-check exits 1.
  --ifix NEW_DIR OLD_DIR            read-only: I-FIX2758 (static rows vs scan rows) + I-REPLAY + the in-job checks; one JSON
                                    object on stdout; exit 0 PASS / 1 FAIL.
  --summarize DIR                   read-only: prereg §5 verdicts over DIR/*.jsonl; one JSON object on stdout; exit 0.
  Bad input (unknown cell, missing file) or a held-out manifest mismatch = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed earlier, inherited here):
  1. **`ssh host '... &'` holds the session open** — launch jobs with `ssh -n` + `setsid -f`.
  2. **The seed is blind for all-awake arms** — the static tail is seed-blind; here the seed moves arrival offsets, so wake
     residuals and the giant sleep path vary with it.
  3. **A prereg mutant with no site in the code** — the mutation-test file anchors every mutant on a verbatim line of THIS file.
"""
import argparse, collections, contextlib, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import cloud_sim_v2 as V     # noqa: E402  (not edited)
import node_sim as NS        # noqa: E402  (not edited)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
SLOS = CS.SLOS
NREPS = 21
HOUR = 60
BAR = 0.10                           # prereg §5 / §10: strict <, on the max over 21 replicates
P_POINT = 0.79                       # prereg §0: preemption point (UNMEASURED, swept)
P_SWEEP = (0.0, 5.0, 30.0)
H2D_TAIL = 22.6                      # prereg §0: measured (the cloud sim's own tph h2d)
H2D_TAIL_SENS = 53.8                 # prereg §0: line rate (NS.H2D_PCIE), not ours
COLD_S = 55.0                        # prereg §0: fastest measured restart; provisioning 0 s (optimistic)
PAGE_FIXED = 0.02                    # tailpool amendment 1 cool_fixed
GIB = 1.073741824
KA_PARK = 15
INF = T.INF_KEY
CELLS = {"D1": ("BT", "CAT128P", 112), "D2": ("CAS", "ALL", 368), "CP": ("P", "CAT128P", 128)}
DECIDING = ("E-HOUR", "E-PARK")
WARM_SET = ("E-HOUR", "E-PARK", "HE-R", "KA-1", "KA-5", "KA-60", "KA-HIST")
# variant -> (policy, p, tail h2d, cold)
VARIANTS = {"STATIC": ("STATIC", None, None, False),
            "E-HOUR": ("HOUR-P", P_POINT, H2D_TAIL, False), "E-PARK": ("KA-15", P_POINT, H2D_TAIL, False),
            "E-COLD": ("HOUR-P", 0.0, None, True),
            "HE-O": ("HOUR-O", None, None, False), "HE-R": ("HOUR-R", P_POINT, H2D_TAIL, False),
            "KA-1": ("KA-1", P_POINT, H2D_TAIL, False), "KA-5": ("KA-5", P_POINT, H2D_TAIL, False),
            "KA-60": ("KA-60", P_POINT, H2D_TAIL, False), "KA-HIST": ("KA-HIST", P_POINT, H2D_TAIL, False)}
for _b, _pol in (("E-HOUR", "HOUR-P"), ("E-PARK", "KA-15")):
    for _p in P_SWEEP:
        VARIANTS[f"{_b}|p{_p:g}"] = (_pol, _p, H2D_TAIL, False)
    VARIANTS[f"{_b}|h{H2D_TAIL_SENS:g}"] = (_pol, P_POINT, H2D_TAIL_SENS, False)
TIMING = {"elapsed_s", "sim_s", "overlay_s"}


# ------------------------------------------------------------------------------------------------ policies (prereg §3)
def held_static(b):
    n = len(b)
    return np.ones(n, bool), np.zeros(n, bool)


def held_keepalive(b, K):
    """KA-K / E-PARK: held iff busy in [t - K, t]; the week starts held (minute -1 counts as busy). wake = busy on a parked slot."""
    b = np.asarray(b, bool)
    n = len(b)
    cs = np.concatenate([[0], np.cumsum(b)])
    t = np.arange(n)
    lo = np.maximum(t - K, 0)
    held = (cs[t + 1] - cs[lo]) > 0
    held |= t <= K - 1
    prev = np.concatenate([[True], held[:-1]])
    return held, b & ~prev


def held_hour(b, mode, hour=HOUR):
    """E-HOUR (mode P: planned from the PREVIOUS hour), HE-R (mode R: carryover only), HE-O (mode O: oracle, no wakes).
    Hour 0 of the week is held; a parked slot is reclaimed at its first busy minute and held to the hour's end."""
    b = np.asarray(b, bool)
    n = len(b)
    held = np.zeros(n, bool)
    wake = np.zeros(n, bool)
    for H in range(n // hour):
        a, z = H * hour, (H + 1) * hour
        if mode == "O":
            held[a:z] = b[a:z].any()
            continue
        planned = H == 0 or bool(b[a] and b[a - 1])
        if mode == "P" and H > 0 and b[a - hour:a].any():
            planned = True
        if planned:
            held[a:z] = True
            continue
        idx = np.flatnonzero(b[a:z])
        if len(idx):
            t = a + int(idx[0])
            held[t:z] = True
            wake[t] = True
    return held, wake


def policy_arrays(pol, b, k_hist=None):
    if pol == "STATIC":
        return held_static(b)
    if pol.startswith("HOUR-"):
        return held_hour(b, pol[-1])
    if pol == "KA-HIST":
        return held_keepalive(b, k_hist)
    return held_keepalive(b, int(pol.split("-")[1]))


def window_end(pol, held, t):
    """the wake window: to the hour's end for hour policies, to the held run's end for keep-alive policies."""
    if pol.startswith("HOUR-"):
        return (t // HOUR + 1) * HOUR - 1
    z = np.flatnonzero(~held[t:])
    return len(held) - 1 if len(z) == 0 else t + int(z[0]) - 1


def k_hist(train_minutes):
    """KA-HIST (TOP-PRO, Shahrad et al. ATC '20): K = clamp(ceil(p99 of TRAIN gaps between request minutes), 1, 240); < 10 gaps -> 60."""
    m = np.unique(np.asarray(train_minutes, np.int64))
    g = np.diff(m)
    if len(g) < 10:
        return 60
    return int(min(240, max(1, math.ceil(float(np.percentile(g, 99))))))


# ------------------------------------------------------------------------------------------------ wakes (prereg §0 / §3)
def page_in(w_gib, gbps):
    return PAGE_FIXED + w_gib * GIB / gbps


def wake_ready(first_by_unit, w_by_unit, gpu_ready):
    """each engine pages in at its own first request, one at a time per slot: -> ready time per unit."""
    ready = {}
    link = -math.inf
    for u in sorted(first_by_unit, key=lambda x: (first_by_unit[x], str(x))):
        start = max(first_by_unit[u], gpu_ready, link)
        ready[u] = start + w_by_unit[u]
        link = ready[u]
    return ready


def residuals(arrivals, ready):
    """delay per request: ready - arrival for a request arriving before its engine is ready, else 0."""
    a = np.asarray(arrivals, float)
    return np.where(a < ready, ready - a, 0.0)


def window_delays(kind, cells, t_star, p, wfun, cold, offs):
    """one wake on one slot. cells: [(minute, first offset s, model, unit, cell id)] of the slot inside the wake window, time order.
    kind 'tail': every engine pages in at its first request; 'trigger' (giant group / anchor replica): only the triggering engine.
    -> [(cell id, model, minute, delays s)]."""
    at = [c for c in cells if c[0] == t_star]
    if not at:
        raise RuntimeError(f"I-WAKE: no cell of the slot starts at wake minute {t_star}")
    T0 = min(60.0 * c[0] + c[1] for c in at)
    gpu = T0 if cold else T0 + p
    out = []
    if kind == "tail":
        A = {}
        for c in cells:
            t = 60.0 * c[0] + c[1]
            if c[3] not in A or t < A[c[3]]:
                A[c[3]] = t
        ready = wake_ready(A, {u: (COLD_S if cold else wfun(u)) for u in A}, gpu)
        for c in cells:
            r = ready[c[3]]
            if 60.0 * c[0] + c[1] < r:
                out.append((c[4], c[2], c[0], residuals(60.0 * c[0] + offs(c[4]), r)))
        return out
    trig = min(at, key=lambda c: (60.0 * c[0] + c[1], c[4]))
    u = trig[3]
    r = gpu + (COLD_S if cold else wfun(u))
    for c in cells:
        if c[3] == u and 60.0 * c[0] + c[1] < r:
            out.append((c[4], c[2], c[0], residuals(60.0 * c[0] + offs(c[4]), r)))
    return out


# ------------------------------------------------------------------------------------------------ composition (prereg §4)
def worst_p95(arr, dl):
    """max over pairings of the nearest-rank p95 of arr + delays: the top n - q + 1 static keys anti-paired with the largest delays."""
    n = len(arr)
    q = math.ceil(0.95 * n)
    a = n - q + 1
    top = arr[q - 1:]
    d = np.sort(np.asarray(dl, np.int64))[::-1][:a]
    dd = np.zeros(a, np.int64)
    dd[:len(d)] = d
    s = np.where(top >= INF, INF, top + dd)
    return int(s.min())


def best_p95(arr, dl):
    """min over pairings: the n - q largest static keys absorb the largest delays; the rest go anti-paired into the q smallest."""
    n = len(arr)
    q = math.ceil(0.95 * n)
    j = len(dl) - (n - q)
    if j <= 0:
        return int(arr[q - 1])
    L = arr[:q]
    e = np.zeros(q, np.int64)
    e[q - j:] = np.sort(np.asarray(dl, np.int64))[:j]
    s = np.where(L >= INF, INF, L + e[::-1])
    return int(s.max())


def max_push(sl, d):
    """the most requests at SLO (slack sl, ascending) a delay (d, ascending) can push over: largest t with d[-t:] > sl[:t]."""
    lo, hi = 0, min(len(sl), len(d))
    while lo < hi:
        t = (lo + hi + 1) // 2
        if np.all(d[len(d) - t:] > sl[:t]):
            lo = t
        else:
            hi = t - 1
    return lo


def worst_ok(arr, dl, sk):
    okv = arr[arr <= sk]
    return int(len(okv) - max_push(np.sort(sk - okv), np.sort(np.asarray(dl, np.int64))))


def best_ok(arr, dl, sk):
    okv = arr[arr <= sk]
    r = len(dl) - (len(arr) - len(okv))
    if r <= 0:
        return int(len(okv))
    sl = np.sort(sk - okv)[::-1][:r]
    d = np.sort(np.asarray(dl, np.int64))[:r]
    return int(len(okv) - int(np.sum(d > sl)))


def _p(k):
    return math.inf if k >= INF else k / 100.0


def compose(week, days, dl_by_day):
    """per LODO replicate: the static Counter (week - day d) + the delays (centiseconds) of days != d -> two cstats-shaped dicts
    (WORST, BEST) with n, p95, ok (exactly CS.cstats's fields score_reps reads)."""
    keys = np.array(sorted(week), np.int64)
    Wv = np.array([week[k] for k in keys], np.int64)
    idx = {int(k): j for j, k in enumerate(keys)}
    out = {v: dict(n=[], p95=[], ok={str(x): [] for x in SLOS}) for v in ("worst", "best")}
    for d in range(NDAYS):
        Dv = np.zeros(len(keys), np.int64)
        for k, c in days[d].items():
            if c:
                Dv[idx[int(k)]] = c
        cnt = Wv - Dv
        if (cnt < 0).any():
            raise RuntimeError("a day Counter exceeds its week Counter")
        arr = np.repeat(keys, cnt)
        dl = np.concatenate([np.asarray(dl_by_day[x], np.int64) for x in range(NDAYS) if x != d] + [np.zeros(0, np.int64)])
        n = len(arr)
        if len(dl) > n:
            raise RuntimeError("more delayed requests than requests")
        for v in ("worst", "best"):
            out[v]["n"].append(int(n))
        if n == 0:
            for v in ("worst", "best"):
                out[v]["p95"].append(None)
                for x in SLOS:
                    out[v]["ok"][str(x)].append(0)
            continue
        out["worst"]["p95"].append(_p(worst_p95(arr, dl)))
        out["best"]["p95"].append(_p(best_p95(arr, dl)))
        for x in SLOS:
            sk = T._key(x)
            out["worst"]["ok"][str(x)].append(worst_ok(arr, dl, sk))
            out["best"]["ok"][str(x)].append(best_ok(arr, dl, sk))
    return out


# ------------------------------------------------------------------------------------------------ verdicts (prereg §5)
def lodo(busy_d, held_d):
    B, H = int(sum(busy_d)), int(sum(held_d))
    return [None if H - int(held_d[d]) <= 0 else 1.0 - (B - int(busy_d[d])) / float(H - int(held_d[d])) for d in range(len(busy_d))]


def slo_gate(worst_pass, best_pass):
    if len(worst_pass) != NREPS or len(best_pass) != NREPS:
        raise ValueError(f"I-REPS: {len(worst_pass)} / {len(best_pass)} replicates, need {NREPS}")
    if all(worst_pass):
        return "SLO-HOLDS"
    if not all(best_pass):
        return "SLO-FAILS"
    return "SLO-UNRESOLVED"


def bar_v(idle):
    if len(idle) != NREPS:
        raise ValueError(f"I-REPS: {len(idle)} idle replicates, need {NREPS}")
    if max(idle) < BAR:
        return "UNDER-BAR"
    if min(idle) >= BAR:
        return "ABOVE-BAR"
    return "STRADDLES"


def headline(gate, idle):
    b = bar_v(idle)
    if gate == "SLO-HOLDS" and b == "UNDER-BAR":
        return "MEETS THE BAR AT SLO"
    if gate == "SLO-HOLDS":
        return "HOLDS SLO, ABOVE THE BAR"
    if gate == "SLO-FAILS":
        return "BREAKS THE SLO"
    return "UNRESOLVED"


def closest(cands):
    """Q2: cands [(name, gate, idle list)] -> (name, max, min) of the SLO-HOLDS policy with the smallest max idle, or None."""
    ok = [(max(i), min(i), n) for n, g, i in cands if g == "SLO-HOLDS"]
    if not ok:
        return None
    mx, mn, nm = min(ok)
    return nm, mx, mn


def range_v(x, p, lo="FEWER", hi="MORE"):
    if max(x) < min(p):
        return lo
    if min(x) > max(p):
        return hi
    return "UNRESOLVED"


# ------------------------------------------------------------------------------------------------ recording the unchanged evaluate
@contextlib.contextmanager
def record(rec):
    o_tail, o_grp, o_fi = CS.run_tail, CS.run_dgroups, CS.fleet_idle

    def tail(*a, **k):
        wk, dy, met = o_tail(*a, **k)
        rec["tail"].append((wk, dy))
        return wk, dy, met

    def grp(cx, np_, groups, seed):
        r = o_grp(cx, np_, groups, seed)
        rec["giant"][seed] = r
        return r

    def fi(B, slots, anc_cnt, anc_r, anc_g, N, n_min, day_len):
        rec["fi"].append(dict(B=B, slots=[(g, list(ms)) for g, ms in slots], acnt=np.asarray(anc_cnt, np.int64), r=anc_r, g=anc_g,
                              N=N))
        return o_fi(B, slots, anc_cnt, anc_r, anc_g, N, n_min, day_len)
    CS.run_tail, CS.run_dgroups, CS.fleet_idle = tail, grp, fi
    try:
        yield
    finally:
        CS.run_tail, CS.run_dgroups, CS.fleet_idle = o_tail, o_grp, o_fi


def static_row(cell, seeds, tag, root, cache):
    """ run_scan's row at the cell's N (same code path), plus the recording. -> (wd, row, rec, snap, r)."""
    arm, cat, N = CELLS[cell]
    fmt = "int4"
    over, extra = {}, dict(CS.EXTRA0)
    wd, base_arm, homing, ca, tau, v2 = V.job_setup(arm, fmt, over, extra, root, cache, None, V.WIN_POINT, 7)
    v2["seeds"] = list(seeds)
    beta = wd.np["beta"]
    ids = V.catalogue(wd, cat)
    cands = [c for c in wd.cands if c["cid"] in ids]
    base = dict(arm=base_arm, fmt=fmt, tag=tag, cat=cat, nphys=wd.np, extra=extra, n_cat=len(ids), v2=v2)
    rec = dict(tail=[], giant={}, fi=[])
    t0 = time.time()
    with V.giant_tau(tau):
        info, snaps = V.plan2(wd, base_arm, homing, N, ca, [len(cands)], cands)
        if info["flag"]:
            raise RuntimeError(f"cell {cell}: anchor flag {info['flag']}")
        sn = snaps[len(cands)]
        complete = set(sn["admitted"]) == ids
        if not complete:
            raise RuntimeError(f"cell {cell}: admission incomplete at N={N}")
        row = dict(base, N=N, r_min=info["r_min"], complete=complete, n_admitted=len(sn["admitted"]))
        with record(rec):
            row.update(V.evaluate2(wd, base_arm, sn, N, info["r"], beta, seeds, ca))
        row.update(base, N=N, complete=True)
        row["pass"] = {s: bool(all(row["slo"][s]["pass"])) for s in row["slo"]}
    row["elapsed_s"] = round(time.time() - t0, 1)
    if len(rec["fi"]) != len(seeds):
        raise RuntimeError(f"recorded {len(rec['fi'])} fleet_idle calls for {len(seeds)} seeds")
    return wd, row, rec, sn, info["r"]


# ------------------------------------------------------------------------------------------------ the overlay
def build_slots(wd, fi):
    """slots: [dict(kind tail|giant|anchor, g, models, units {u: W GiB}, unit_of {m: u}, gbps, host_ok, j)] from the recorded
    fleet_idle call (allocated slots only) + one per anchor replica."""
    cx = wd.cx
    tr, tab, S = cx["tr"], cx["tab"], cx["S"]
    out = []
    for g, ms in fi["slots"]:
        if not ms:
            continue
        if all(m in wd.mg for m in ms):
            units = {m: tab[tr.models[m][0]]["W"] for m in ms}
            host = sum(w * wd.np["r_pin"] + g * CS.O_PROC for w in units.values())
            out.append(dict(kind="giant", g=g, models=list(ms), units=units, unit_of={m: m for m in ms}, gbps=g * NS.H2D_PCIE,
                            host_ok=host <= NS.d_host_cap(g, wd.np) + 1e-9))
        else:
            uo = {m: S["unit_of"][m] for m in ms}
            units = {u: wd.uinfo[u]["w"] for u in set(uo.values())}
            pinned = sum(units.values()) * wd.np["r_pin"]
            out.append(dict(kind="tail", g=g, models=list(ms), units=units, unit_of=uo, gbps=None,
                            host_ok=pinned <= wd.tph["h_pin"] + 1e-9))
    d = tab[NS.ANCHOR]
    host = d["W"] * wd.np["r_pin"] + fi["g"] * CS.O_PROC
    for j in range(1, int(fi["r"]) + 1):
        out.append(dict(kind="anchor", g=fi["g"], models=[wd.am], units={wd.am: d["W"]}, unit_of={wd.am: wd.am},
                        gbps=fi["g"] * NS.H2D_PCIE, host_ok=host <= NS.d_host_cap(fi["g"], wd.np) + 1e-9, j=j))
    return out


def slot_busy(sl, fi):
    if sl["kind"] == "anchor":
        return fi["acnt"][:WEEK_MIN] >= sl["j"]
    return np.any(fi["B"][sl["models"]], axis=0)[:WEEK_MIN]


def static_counters(wd, rec, snap, r, seed):
    """model -> (week Counter, [7 day Counters]) exactly as evaluate scored it for this seed."""
    cx = wd.cx
    out = {}
    tail_w, tail_d = {}, {}
    for wk, dy in rec["tail"]:
        tail_w.update(wk)
        tail_d.update(dy)
    gw, gd, _ = rec["giant"].get(seed, ({}, {}, {}))
    for c in snap["admitted"]:
        m = cx["cid_m"][c]
        if m in tail_w:
            out[m] = (tail_w[m], tail_d[m])
        elif m in gw:
            out[m] = (gw[m], gd[m])
        else:                                   # DED tail or DED giant: the dedicated reference (tail_from_ref / gm DED)
            out[m] = (wd.ref_w[m], [wd.ref_d[x][m] for x in range(NDAYS)])
    w, dy = wd.anchor_out(r)
    out[wd.am] = (w, dy)
    return out


def overlay(wd, row, rec, snap, r, seeds, variants=None, check_score0=True):
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
                                                                          for v_ in ("worst", "best")}, fails={})
           for v in variants}
    checks = dict(I_BUSY=True, I_HELD=True, I_STATIC0=True, I_SCORE0=None, I_HOST_parked_blocked=0, I_WAKE=True)
    req_rep_tot = []
    for si, seed in enumerate(seeds):
        fi = rec["fi"][si]
        slots = build_slots(wd, fi)
        # I-BUSY against per-minute busy GPUs
        bs = [slot_busy(sl, fi) for sl in slots]
        tot = np.zeros(WEEK_MIN, np.int64)
        for sl, b in zip(slots, bs):
            tot += sl["g"] * b.astype(np.int64)
        ref = V.busy_gpus_per_min(fi["B"], fi["slots"], fi["acnt"], fi["r"], fi["g"], WEEK_MIN)
        if not np.array_equal(tot, ref):
            checks["I_BUSY"] = False
            raise RuntimeError("I-BUSY: slot busy series disagree with 's per-minute busy GPUs")
        # cells per slot (held-out), first/last offsets for this seed
        first, last, _ = T.arrivals(tr, seed)
        mod_slot = {}
        for s_i, sl in enumerate(slots):
            if sl["kind"] != "anchor":
                for m in sl["models"]:
                    mod_slot[m] = s_i
        cells_of = collections.defaultdict(list)
        for i in kk:
            m = int(tr.mid[i])
            cells_of[m].append(int(i))
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
        # KA-HIST per slot from TRAIN request minutes
        khs = []
        for sl in slots:
            k = train & np.isin(tr.mid, sl["models"])
            khs.append(k_hist(tr.minute[k]))
        # static counters for this seed
        sc = static_counters(wd, rec, snap, r, seed)
        st_stats = {m: CS.cstats(*sc[m]) for m in hosted}
        # I-SCORE0 (every seed): zero-delay composition reproduces cstats and the row's slo block
        if check_score0:
            z = [np.zeros(0, np.int64)] * NDAYS
            ok0 = True
            for m in hosted:
                cw = compose(sc[m][0], sc[m][1], z)["worst"]
                s0 = st_stats[m]
                if cw["n"] != s0["n"] or cw["p95"] != s0["p95"] or any(cw["ok"][str(x)] != s0["ok"][str(x)] for x in SLOS):
                    ok0 = False
                    break
            for x in SLOS:
                sr = CS.score_reps(hosted, st_stats, rst, allm, x)
                for k_, v_ in sr.items():
                    if row["slo"][str(x)][k_][si * NDAYS:(si + 1) * NDAYS] != v_:
                        ok0 = False
            checks["I_SCORE0"] = ok0 if checks["I_SCORE0"] in (None, True) else False
            if not ok0:
                raise RuntimeError("I-SCORE0: zero-delay composition does not reproduce the row's slo block")
        rep_req = [int(sum(int(rd[:, m].sum() - rd[x, m]) for m in hosted)) for x in range(NDAYS)]
        req_rep_tot.extend(rep_req)
        pol_cache = {}
        for v in variants:
            pol, p, h2d, cold = VARIANTS[v]
            busy_d = np.zeros(NDAYS, np.int64)
            held_d = np.zeros(NDAYS, np.int64)
            park_d = np.zeros(NDAYS, np.int64)
            p15_d = np.zeros(NDAYS, np.int64)
            p60_d = np.zeros(NDAYS, np.int64)
            wake_d = np.zeros(NDAYS, np.int64)
            wake_min = np.zeros(WEEK_MIN, np.int64)
            cell_delay = {}
            for s_i, (sl, b) in enumerate(zip(slots, bs)):
                key = (pol, s_i)
                if key not in pol_cache:
                    if not sl["host_ok"] and pol != "STATIC":
                        checks["I_HOST_parked_blocked"] += 1
                        pol_cache[key] = held_static(b)
                    else:
                        pol_cache[key] = policy_arrays(pol, b, khs[s_i])
                held, wake = pol_cache[key]
                if (b & ~held).any():
                    checks["I_HELD"] = False
                    raise RuntimeError(f"I-HELD: busy minute not held ({v}, slot {s_i})")
                g = sl["g"]
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
                    te = window_end(pol, held, int(t))
                    a_, z_ = np.searchsorted(mins, t, "left"), np.searchsorted(mins, te, "right")
                    cl = [(int(mins[q]), float(first[ids[q]]), int(tr.mid[ids[q]]), sl["unit_of"][int(tr.mid[ids[q]])], ids[q])
                          for q in range(a_, z_)]
                    if not cl or cl[0][0] != t:
                        checks["I_WAKE"] = False
                    for cid, m, mi, dd in window_delays(kind, cl, int(t), p if p is not None else 0.0,
                                                        (lambda u, _sl=sl, _gb=gb: page_in(_sl["units"][u], _gb)), cold, offs):
                        prev = cell_delay.get(cid)
                        cell_delay[cid] = dd if prev is None else np.maximum(prev, dd)
            o = res[v]
            o["idle_el"].extend(lodo(busy_d, held_d))
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
            # delays -> per model per day (centiseconds, > 0)
            dl = collections.defaultdict(lambda: [[] for _ in range(NDAYS)])
            for cid, dd in cell_delay.items():
                dc = np.round(dd * 100.0).astype(np.int64)
                dc = dc[dc > 0]
                if len(dc):
                    dl[int(tr.mid[cid])][(int(tr.minute[cid]) - HELD) // DAY_MIN].extend(dc.tolist())
            hit_d = np.zeros(NDAYS, np.int64)
            for m, per in dl.items():
                for d in range(NDAYS):
                    hit_d[d] += len(per[d])
            o["wake_hit"].extend([int(hit_d.sum() - hit_d[d]) for d in range(NDAYS)])
            o["wake_hit_share"].extend([(int(hit_d.sum() - hit_d[d]) / rep_req[d]) if rep_req[d] else None for d in range(NDAYS)])
            ost = {"worst": dict(st_stats), "best": dict(st_stats)}
            for m, per in dl.items():
                if m not in sc:
                    raise RuntimeError(f"a delay on a model that is not hosted: {m}")
                cmp_ = compose(sc[m][0], sc[m][1], [np.asarray(x, np.int64) for x in per])
                ost["worst"][m] = cmp_["worst"]
                ost["best"][m] = cmp_["best"]
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
        for x in SLOS:
            g[str(x)] = slo_gate(o["slo"]["worst"][str(x)]["pass"], o["slo"]["best"][str(x)]["pass"])
        o["gate"] = g
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
    return dict(variants=out, checks=checks, req_rep_tot=req_rep_tot, overlay_s=round(time.time() - t_start, 1),
                consts=dict(P_POINT=P_POINT, H2D_TAIL=H2D_TAIL, H2D_TAIL_SENS=H2D_TAIL_SENS, COLD_S=COLD_S, KA_PARK=KA_PARK, BAR=BAR,
                            tph_h2d=wd.tph["h2d"]))


def run_cell(cell, seeds, tag, root, cache, out):
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell}")
    wd, row, rec, snap, r = static_row(cell, seeds, tag, root, cache)
    if abs(wd.tph["h2d"] - H2D_TAIL) > 1e-9:
        raise RuntimeError(f"prereg §0: the cloud sim's tail h2d is {wd.tph['h2d']}, not {H2D_TAIL}")
    row["cell"] = cell
    row["v3"] = overlay(wd, row, rec, snap, r, seeds)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------ checks over rows
def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _rows(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            out += _load(os.path.join(d, f))
    return out


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def ifix(new_dir, old_dir):
    """I-FIX2758 (static part of the main rows vs scan rows at the same N), I-REPLAY (main / rep_b / rep_c), in-job checks."""
    rows = _rows(new_dir)
    by = {(r["tag"], r["cell"]): r for r in rows}
    res, ok = {}, True
    for cell, (arm, cat, N) in CELLS.items():
        new = by.get(("main", cell))
        fn = os.path.join(old_dir, f"scan_{cat}_int4_{arm}.jsonl")
        if new is None or not os.path.exists(fn):
            res[f"I_FIX2758_{cell}"] = "MISSING"
            ok = False
            continue
        old = [x for x in _load(fn) if x.get("N") == N]
        if len(old) != 1:
            res[f"I_FIX2758_{cell}"] = f"{len(old)} old rows at N={N}"
            ok = False
            continue
        keys = set(old[0]) - TIMING
        diff = [k for k in sorted(keys) if new.get(k) != old[0].get(k)]
        res[f"I_FIX2758_{cell}"] = dict(keys=len(keys), diffs=diff[:5])
        ok = ok and not diff
        ch = new["v3"]["checks"]
        res[f"checks_{cell}"] = ch
        ok = ok and all(ch[k] for k in ("I_BUSY", "I_HELD", "I_STATIC0", "I_SCORE0", "I_WAKE"))
    a = by.get(("main", "D1"))
    for t in ("rep_b", "rep_c"):
        b = by.get((t, "D1"))
        if a is None or b is None:
            res[f"I_REPLAY_{t}"] = "MISSING"
            ok = False
            continue
        sa, sb = _strip(dict(a, tag=None)), _strip(dict(b, tag=None))
        sa["v2"] = dict(sa["v2"])
        sb["v2"] = dict(sb["v2"])
        diff = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
        res[f"I_REPLAY_{t}"] = dict(diffs=diff[:5])
        ok = ok and not diff
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summarize (prereg §5)
def _rng(v):
    v = [x for x in v if x is not None]
    return [round(min(v), 4), round(max(v), 4)] if v else None


def cell_verdicts(row):
    V3 = row["v3"]["variants"]
    st = V3["STATIC"]
    out = dict(q1={}, q2={}, table={})
    for v, o in V3.items():
        out["table"][v] = dict(idle_el=_rng(o["idle_el"]), held_gpuh=_rng(o["held_gpuh"]), parked_ge60=_rng(o["parked_ge60"]),
                               fixed_idle_phi1=_rng(o["fixed_idle_phi1"]), wakes=_rng(o["wakes"]),
                               wake_hit_share=_rng(o["wake_hit_share"]), peak_wakes_min=o["peak_wakes_min"], gate=o["gate"],
                               at_worst={s: _rng(o["slo"]["worst"][s]["at"]) for s in o["gate"]},
                               RS_worst={s: _rng(o["slo"]["worst"][s]["RS"]) for s in o["gate"]},
                               M_worst_min={s: min(x for x in o["slo"]["worst"][s]["M"] if x is not None)
                                            if any(x is not None for x in o["slo"]["worst"][s]["M"]) else None for s in o["gate"]},
                               fails=o.get("fails"))
    for s in (str(x) for x in SLOS):
        for v in DECIDING:
            o = V3[v]
            out["q1"][f"{v}|{s}"] = dict(headline=headline(o["gate"][s], o["idle_el"]), gate=o["gate"][s], bar=bar_v(o["idle_el"]),
                                         idle=_rng(o["idle_el"]), dist=[round(x - BAR, 4) for x in _rng(o["idle_el"])],
                                         vs_static=CS.verdict_idle(o["idle_el"], st["idle_el"]), static_idle=_rng(st["idle_el"]))
        c = closest([(v, V3[v]["gate"][s], V3[v]["idle_el"]) for v in WARM_SET])
        out["q2"][s] = ("NO ELASTIC POLICY HOLDS THE SLO" if c is None else
                        dict(closest=c[0], idle=[round(c[2], 4), round(c[1], 4)], dist=[round(c[2] - BAR, 4), round(c[1] - BAR, 4)]))
    return out


def summarize(d):
    rows = _rows(d)
    by = {(r["tag"], r["cell"]): r for r in rows}
    out = dict(cells={}, q3={}, q4={}, seedshift={})
    for cell in CELLS:
        r = by.get(("main", cell))
        if r:
            out["cells"][cell] = cell_verdicts(r)
            out["cells"][cell]["count_floor_el_hour"] = _rng(r["v2x"]["el_hour"])
            out["cells"][cell]["static_idle_row"] = _rng(r["idle"])
    d1, cp = by.get(("main", "D1")), by.get(("main", "CP"))
    if d1:
        for s in (str(x) for x in SLOS):
            out["q3"][s] = d1["v3"]["variants"]["E-COLD"]["gate"][s]
    if d1 and cp:
        for v in DECIDING:
            a, b = d1["v3"]["variants"][v], cp["v3"]["variants"][v]
            out["q4"][v] = dict(held=range_v(a["held_gpuh"], b["held_gpuh"], "FEWER-HELD", "MORE-HELD"),
                                held_BT=_rng(a["held_gpuh"]), held_P=_rng(b["held_gpuh"]),
                                idle=CS.verdict_idle(a["idle_el"], b["idle_el"]), gates_BT=a["gate"], gates_P=b["gate"])
    ss = by.get(("seedshift", "D1"))
    if d1 and ss:
        a, b = cell_verdicts(d1), cell_verdicts(ss)
        flips = {}
        for k in a["q1"]:
            flips[f"q1|{k}"] = [a["q1"][k]["headline"], b["q1"][k]["headline"]]
        for k in a["q2"]:
            ca = a["q2"][k] if isinstance(a["q2"][k], str) else a["q2"][k]["closest"]
            cb = b["q2"][k] if isinstance(b["q2"][k], str) else b["q2"][k]["closest"]
            flips[f"q2|{k}"] = [ca, cb]
        out["seedshift"] = dict(pairs=flips, UNATTRIBUTABLE=sorted(k for k, v in flips.items() if v[0] != v[1]),
                                q1_seedshift={k: dict(headline=v["headline"], idle=v["idle"]) for k, v in b["q1"].items()})
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    res = []

    def chk(name, got, exp, tol=None):
        if tol is None:
            ok = got == exp
        else:
            ok = len(got) == len(exp) and all(abs(a - b) <= tol for a, b in zip(got, exp))
        res.append((name, ok, got, exp))
    # GE1: elastic idles less (2 slots, 2 days x 120 min, 60-minute hours)
    nm = 240
    bA = np.zeros(nm, bool)
    bA[0:60] = True
    bB = np.zeros(nm, bool)
    bB[130] = True
    hA, wA = held_hour(bA, "P")
    hB, wB = held_hour(bB, "P")
    busy_d = [int(bA[:120].sum() + bB[:120].sum()), int(bA[120:].sum() + bB[120:].sum())]
    held_d = [int(hA[:120].sum() + hB[:120].sum()), int(hA[120:].sum() + hB[120:].sum())]
    chk("GE1 E-HOUR held per day", held_d, [180, 110])
    chk("GE1 E-HOUR LODO idle", lodo(busy_d, held_d), [0.990909, 0.666667], 1e-6)
    chk("GE1 STATIC LODO idle", lodo(busy_d, [240, 240]), [0.995833, 0.75], 1e-6)
    chk("GE1 wakes", [int(x) for x in np.flatnonzero(wA)] + [int(x) for x in np.flatnonzero(wB)], [130])
    chk("GE1 B held 130-179 then hour 3", [int(hB[129]), int(hB[130]), int(hB[179]), int(hB[180]), int(hB[239])], [0, 1, 1, 1, 1])
    # GE2: a wake larger than the SLO must break it
    arr = np.full(20, 10, np.int64)
    chk("GE2 k=2 worst/best p95", [worst_p95(arr, [300, 300]), best_p95(arr, [300, 300])], [310, 310])
    chk("GE2 k=1 worst/best p95", [worst_p95(arr, [300]), best_p95(arr, [300])], [10, 10])
    g2 = window_delays("tail", [(10, 5.0, 7, "u", 0)], 10, 2.5, lambda u: 0.5, False, lambda c: np.array([5.0]))
    chk("GE2 engine delay", [round(float(x), 6) for x in g2[0][3]], [3.0])
    # GE2 through compose + score_reps (one eligible model, 7 days, 20 requests per replicate)
    wk = collections.Counter({10: 21 + 1})        # 22 requests over the week
    dys = [collections.Counter({10: 1}) for _ in range(NDAYS)]
    dys[0] = collections.Counter({10: 16})
    # replicate 0 (drops day 0): 6 requests; replicates 1..6 drop 1 -> 21 requests
    cm = compose(wk, dys, [[300, 300] if d == 0 else [] for d in range(NDAYS)])
    chk("GE2 compose worst p95 (replicate 1: day-0 delays present)", cm["worst"]["p95"][1], 3.1)
    chk("GE2 compose replicate 0 drops day 0's delays", cm["worst"]["p95"][0], 0.1)
    # GE3: adversarial verdicts
    lo_idle = [0.05] * NREPS
    wf = [True] * NREPS
    wf[3] = False
    chk("GE3 low idle, worst+best fail -> BREAKS", headline(slo_gate(wf, wf), lo_idle), "BREAKS THE SLO")
    chk("GE3 low idle, worst fails, best passes -> UNRESOLVED", headline(slo_gate(wf, [True] * NREPS), lo_idle), "UNRESOLVED")
    chk("GE3 low idle + SLO holds -> MEETS", headline(slo_gate([True] * NREPS, [True] * NREPS), lo_idle), "MEETS THE BAR AT SLO")
    chk("GE3 closest excludes SLO-FAILS", closest([("X", "SLO-FAILS", [0.01] * NREPS), ("Y", "SLO-HOLDS", [0.30] * NREPS)]),
        ("Y", 0.30, 0.30))
    chk("GE3 closest none", closest([("X", "SLO-UNRESOLVED", [0.01] * NREPS)]), None)
    chk("GE3 bar straddles", bar_v([0.05] * 20 + [0.12]), "STRADDLES")
    # GE4: pairing bounds
    arr4 = np.arange(10, 201, 10, dtype=np.int64)
    chk("GE4 worst/best p95", [worst_p95(arr4, [50, 30, 20]), best_p95(arr4, [50, 30, 20])], [230, 190])
    chk("GE4 worst/best ok at 2.0", [worst_ok(arr4, [50, 30, 20], 200), best_ok(arr4, [50, 30, 20], 200)], [17, 20])
    chk("GE4 inf stays inf", worst_p95(np.array([10] * 19 + [INF], np.int64), [5]), 15)
    chk("GE4b BEST puts the SMALLEST delays into the low set (pre-data addition, kills M8)", best_p95(np.full(20, 10, np.int64),
                                                                                                    [100, 5]), 15)
    # GE5: keep-alive
    b5 = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0], bool)
    h5, w5 = held_keepalive(b5, 2)
    chk("GE5 K=2 held", h5.astype(int).tolist(), [1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
    chk("GE5 K=2 wakes", np.flatnonzero(w5).tolist(), [4])
    h0, w0 = held_keepalive(b5, 0)
    chk("GE5 K=0 held = busy, wake at 4", [h0.astype(int).tolist(), np.flatnonzero(w0).tolist()],
        [b5.astype(int).tolist(), [4]])
    chk("GE5 window end (keep-alive run)", window_end("KA-2", h5, 4), 7)
    chk("GE5 window end (hour)", window_end("HOUR-P", h5, 4), 59)
    # GE6: the delay engine, link serialization
    of = {0: np.array([5.0, 5.4, 9.0]), 1: np.array([6.0])}
    g6 = window_delays("tail", [(10, 5.0, 1, "U1", 0), (10, 6.0, 2, "U2", 1)], 10, 0.79, {"U1": 0.50, "U2": 0.30}.get, False, of.get)
    chk("GE6 U1 delays", [round(float(x), 6) for x in g6[0][3]], [1.29, 0.89, 0.0])
    chk("GE6 U2 delay", [round(float(x), 6) for x in g6[1][3]], [0.59])
    # GE7: giant group (trigger member only) and anchor window
    of7 = {0: np.array([5.0]), 1: np.array([30.0])}
    g7 = window_delays("trigger", [(10, 5.0, 1, "A", 0), (10, 30.0, 2, "B", 1)], 10, 0.79, {"A": 0.70, "B": 0.70}.get, False, of7.get)
    chk("GE7 giant: only the trigger member pays", [(x[1], [round(float(y), 6) for y in x[3]]) for x in g7], [(1, [1.49])])
    of7b = {0: np.array([5.0, 6.0, 7.0])}
    g7b = window_delays("trigger", [(10, 5.0, 9, "anc", 0)], 10, 0.79, {"anc": 0.71}.get, False, of7b.get)
    chk("GE7 anchor window", [round(float(y), 6) for y in g7b[0][3]], [1.5, 0.5, 0.0])
    # cold: no p, 55 s per engine, serialized
    g7c = window_delays("tail", [(10, 5.0, 1, "U1", 0), (10, 6.0, 2, "U2", 1)], 10, 9.0, {"U1": 0.5, "U2": 0.3}.get, True, of.get)
    chk("GE7 cold serialized (U1 ready 660, U2 starts 660, ready 715)", [round(float(g7c[1][3][0]), 6)], [109.0])
    # KA-HIST
    chk("KA-HIST p99 gap", [k_hist(list(range(0, 200, 3))), k_hist([0, 5])], [3, 60])
    n_ok = sum(1 for r in res if r[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    print(f"SELF-TEST {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")
    return 0 if n_ok == len(res) else 1


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
