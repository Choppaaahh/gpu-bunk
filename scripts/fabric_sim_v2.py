#!/usr/bin/env python3
"""fabric_sim_v2.py — the gpu-bunk fabric sim with ELASTIC KV and LOAD SHEDDING, on the census-sized tail.

Prereg (frozen before this file existed): (private report). On any divergence
the prereg wins. Imports v1 (`fabric_sim`), and the census script UNCHANGED. `simulate()` here is a FORK of v1's: with every v2 feature off it equals v1 byte for byte (self-test GF,
instrument check I1v2). The three named mechanisms (prereg §1):
  POOL     one process per card, one shared KV block pool, lend-and-reclaim with floors (our shared-KV-pool patches; all-awake only)
  BALLOON  per-process KV pages mapped on demand from the card's free memory (kvcached; IDEALIZED = upper bound)
  RESIZE   stock relaunch with a new --kv-cache-memory-bytes every H min, re-splitting the card by trailing demand
plus SHED (router admission control at tau: shed requests score +inf).

Output contract:
  --self-test            exit 0 + "SELF-TEST PASS n/n" (prereg §9 goldens GA-GQ, GF); exit 1 + FAIL lines otherwise.
  --check-i1 DIR --root R --cache F
                         read-only: I1v2, v1 sizes + every v2 feature off vs v1's committed rows in DIR/*.jsonl; one JSON
                         object; exit 0 if PASS, 1 otherwise.
  --check-i5 --root R --cache F
                         read-only: I5, REF_5090 with census R1 sizes + held-out occupancy vs the census's dedicated counts;
                         one JSON object; exit 0 if PASS, 1 otherwise.
  --run --arm A [--ks 8,16 | --kgrid] [--seeds 0,1,2] [--set k=v ...] [--tag T] --out F
                         simulates and APPENDS one JSON line per (K, seed) to F (WRITES F); exit 0.
  --summarize DIR [--root R]
                         read-only: prereg §7 Q1-Q4 over DIR/*.jsonl; one JSON object on stdout (Q4 reads the frozen
                         offers CSV under the root, default = this repo; absent -> Q4 = {"skipped": reason}).
  Bad input (sha mismatch, unknown arm/param, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed while building v1 / the census, inherited here):
  1. **Float order is load-bearing for byte-equivalence** — the v1 KV block is kept verbatim when no v2 feature is on
     (`if mech is None and shed_tau is None`), and every added term is an exact `+ 0.0` when off (pitfall 1).
  2. **A unit index of -1 wraps numpy** — floors, sizing and train_live are computed on the FULL unit list and only then
     compacted to the units that fit the card (NO-CARD units leave the placement, their requests score +inf).
  3. **`ssh host 'write && … &'` backgrounds the whole chain** — launch runners in separate calls on a CPU server.
"""
import argparse, collections, heapq, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric_sim as F     # noqa: E402  (v1 not edited)
import tailpool_sim as T   # noqa: E402  (not edited)
import tailpool_v2 as V    # noqa: E402  (not edited)
import wakestorm_sim as W  # noqa: E402  (not edited)
import size_census as SC   # noqa: E402  (a/b not edited; bpt_effective only)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

G30 = F.G30
GIB = F.GIB
HELD = F.HELD
OVH = F.OVH
INF_T = F.INF_T
KV_UTIL = F.KV_UTIL
CENSUS_JSON = "data/size_census/census.json"
CENSUS_SHA = "348e449344fd5b02b0b37604224a2db1a052c0aa15f9ed7a34be35dc9c1d34cb"
DEMAND_JSON = "data/size_census/demand_R2.json"
DEMAND_SHA = "342929b9d4ebe393746d67f0d11fc36763d9727a635bd5ab33eb9defa3c2cb2c"
I5_WANT = {"S10": {"eligible": 104, "feasible_honest": 79}, "S2": {"eligible": 104, "feasible_honest": 69}}
CLASSES = ("FITS-ONE-CARD", "NEEDS-BIG-CARD", "NEEDS-REPLICAS", "WEIGHTS-EXCEED-BIG-CARD")
K_GRID_5090 = F.K_GRID_5090
K_GRID_BIG = F.K_GRID_BIG
V1_BAR = 0.833                     # v1's ceiling, the common-bar table (prereg §7)

# prereg §6: v1's physics (tpot_sat = the calibrated 0.06) + v2's new keys; every key overridable with --set k=v
PHYS0 = dict(F.PHYS0, tpot_sat=0.06, sizes="census", reading="R2", bpt_occ="train", restart_s=37.0, resize_h=60.0,
             resize_thr=0.25)
PHYS_STR = F.PHYS_STR | {"sizes", "reading", "bpt_occ"}

# prereg §5 arms. v2 keys: mech None|pool|balloon|resize, shed_tau None|2.0|10.0, placement adds SPREAD
LEV2 = dict(mech=None, shed_tau=None)
SLEEP = dict(F.ALL)
AWAKE = dict(kv="size", int8=True, ram=True, placement="SPREAD", all_awake=True, units="dup")
BASE_ARMS = {
    "ALL": dict(SLEEP),
    "BAL_SZ": dict(SLEEP, mech="balloon"),
    "BAL_LEN": dict(SLEEP, kv="floor", mech="balloon"),
    "ALL_SHED2": dict(SLEEP, shed_tau=2.0),
    "ALL_SHED10": dict(SLEEP, shed_tau=10.0),
    "BAL_SZ_SHED2": dict(SLEEP, mech="balloon", shed_tau=2.0),
    "BAL_SZ_SHED10": dict(SLEEP, mech="balloon", shed_tau=10.0),
    "AWAKE": dict(AWAKE),
    "POOL_SZ": dict(AWAKE, mech="pool"),
    "POOL_LEN": dict(AWAKE, kv="floor", mech="pool"),
    "RESIZE": dict(AWAKE, mech="resize"),
    "AWAKE_SHED2": dict(AWAKE, shed_tau=2.0),
    "AWAKE_SHED10": dict(AWAKE, shed_tau=10.0),
    "POOL_SZ_SHED2": dict(AWAKE, mech="pool", shed_tau=2.0),
    "POOL_SZ_SHED10": dict(AWAKE, mech="pool", shed_tau=10.0),
}
CTRL = {"BAL_SZ": ("ALL",), "BAL_LEN": ("ALL",), "ALL_SHED2": ("ALL",), "ALL_SHED10": ("ALL",),
        "BAL_SZ_SHED2": ("ALL", "BAL_SZ", "ALL_SHED2"), "BAL_SZ_SHED10": ("ALL", "BAL_SZ", "ALL_SHED10"),
        "POOL_SZ": ("AWAKE",), "POOL_LEN": ("AWAKE",), "RESIZE": ("AWAKE",), "AWAKE_SHED2": ("AWAKE",),
        "AWAKE_SHED10": ("AWAKE",), "POOL_SZ_SHED2": ("AWAKE", "POOL_SZ", "AWAKE_SHED2"),
        "POOL_SZ_SHED10": ("AWAKE", "POOL_SZ", "AWAKE_SHED10")}
STACK_CTRL = {"ALL": "ALL", "AWAKE": "AWAKE"}
REPRO = {"V1_ALL": "ALL", "V1_BIG_ALL": "BIG_ALL", "V1_BIG_AWAKE_KV": "BIG_AWAKE_KV"}


def arm_cfg2(arm):
    """-> (cfg, base_name). 'BIG_X' = arm X on the 96 GB card; 'V1_*' = v1's own arm config (reproductions, I1v2)."""
    if arm in REPRO:
        c = F.arm_cfg(REPRO[arm])
        c.update(LEV2)
        return c, arm
    big = arm.startswith("BIG_")
    b = arm[4:] if big else arm
    if b not in BASE_ARMS:
        raise ValueError(f"unknown arm {arm}")
    c = dict(W.BASE)
    c.update(F.LEV)
    c.update(LEV2)
    c.update(BASE_ARMS[b])
    c["card"] = "BIG" if big else "5090"
    return c, b


def phys_with(over=None):
    p = dict(PHYS0)
    for k, v in (over or {}).items():
        if k not in PHYS0:
            raise ValueError(f"unknown param {k}")
        p[k] = v if k in PHYS_STR else float(v)
    for k, ok in (("sizes", ("census", "p8")), ("reading", ("R1", "R2")), ("bpt_occ", ("train", "heldout"))):
        if p[k] not in ok:
            raise ValueError(f"{k} must be one of {ok}, got {p[k]!r} (QA r1)")
    return p


# ------------------------------------------------------------------------------------------------ pure rules (§4)
def elastic_caps(fb, need, spare):
    """prereg §4.2: floor + grant; grants proportional to excess when the total excess exceeds the spare (GiB)."""
    S = max(0.0, spare)
    e = [max(0.0, n - f) for f, n in zip(fb, need)]
    tot = sum(e)
    g = e if tot <= S else [S * x / tot for x in e]
    return [f + x for f, x in zip(fb, g)], g


def shed_fraction(bk, s_in, N, kvtok, tau):
    """prereg §4.6: the admitted fraction a of this minute's new KV demand so the KV wait stays <= tau."""
    cap = 60.0 * kvtok
    if bk > tau * kvtok:
        return 0.0
    if s_in + N <= cap or N <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (2.0 * (tau * kvtok - bk) + cap - s_in) / N))


def trailing_peak(hist, e, H):
    """prereg §4.5: max of D(t)/60 (tokens) over the unit's minutes t with arrivals in [e - H, e); 0 if none."""
    v = [d for t, d in hist.items() if e - H <= t < e]
    return max(v) / 60.0 if v else 0.0


def resize_split(budget, sized, target):
    """prereg §4.5: under budget each gets target x budget / sum(target); over budget, sized floors plus the rest in
    proportion to (target - sized)."""
    tt = sum(target)
    if tt <= budget:
        return [t * budget / tt for t in target]
    rem = budget - sum(sized)
    ex = [t - s for s, t in zip(sized, target)]
    se = sum(ex)
    return [s + (rem * x / se if se > 0 else 0.0) for s, x in zip(sized, ex)]


def spread_place(units, K, phys, hneed, card_gib):
    """prereg §4.7 SPREAD: f descending (ties cid), each to the card with the most free GPU memory (ties lowest index)
    where f and the host total fit; none -> ValueError."""
    free = [card_gib] * K
    tot = [phys["h_tot"]] * K
    homes = [-1] * len(units)
    for u in sorted(range(len(units)), key=lambda k: (-units[k]["f"], units[k]["cid"])):
        f = units[u]["f"]; t_ = hneed[u][1]
        feas = [c for c in range(K) if f <= free[c] + 1e-9 and t_ <= tot[c] + 1e-9]
        if not feas:
            raise ValueError(f"SPREAD infeasible at K={K}: unit {units[u]['cid']} fits no card")
        c = max(feas, key=lambda j: (free[j], -j))
        homes[u] = c
        free[c] -= f; tot[c] -= t_
    return homes


# ------------------------------------------------------------------------------------------------ simulator (fork of v1)
def simulate(trace, units, unit_of, plc, K, cfg, seed, burst_w=None, score_from=HELD, min0=T.MIN0,
             card_gib=T.CARD_GIB, s=None, phys=None, kvtok=None, copies=None, C=C_REF,
             mech=None, shed_tau=None, bpt=None, resize=None):
    """FORK of fabric_sim.simulate. New, all OFF by default: unit_of[m] = -1 (NO-CARD, +inf), mech pool | balloon |
    resize (needs bpt[u] bytes/token and kvtok = the floors), shed_tau (§4.6), resize = dict(H min, restart_s, thr).
    Returns (out per-model Counter of TTFT keys, metrics)."""
    phys = phys or F.PHYS0
    s = phys["s"] if s is None else s
    v2 = cfg["physics"] == "v2"
    protect = cfg["protect"]
    queue = cfg["sched"] == "queue"
    phi = cfg["phi"]
    beta = cfg["beta"]
    kv_on = kvtok is not None
    awake_boot = bool(cfg.get("all_awake"))
    bal = mech == "balloon"
    pool = mech == "pool"
    rsz = mech == "resize"
    if mech is not None:
        if mech not in ("pool", "balloon", "resize") or not kv_on or bpt is None:
            raise ValueError("an elastic mechanism needs kvtok (the floors) and bpt per unit")
        if (pool or rsz) and not awake_boot:
            raise ValueError(f"{mech} is all-awake only (prereg §4.3/§4.5)")
    if shed_tau is not None and not kv_on:
        raise ValueError("shed needs the KV model on")
    models = trace.models
    M, NU = len(models), len(units)
    kind, assign = plc[0], plc[1]
    if kind != "card":
        raise ValueError("fabric simulate supports card placements only")
    if len(assign) != NU or any(h < 0 or h >= K for h in assign):
        raise ValueError("homes must give every unit a card in [0, K)")
    cands = copies if copies is not None else [[assign[u]] for u in range(NU)]
    if len(cands) != NU or any(not cs or cs[0] != assign[u] or any(c < 0 or c >= K for c in cs)
                               for u, cs in enumerate(cands)):
        raise ValueError("copies must list every unit's card set, primary first, inside [0, K)")
    first, last, order = V.arrivals_v2(trace, seed, burst_w)
    groups = None if v2 else W.make_groups(K, cfg)
    gpu = [0.0] * K
    awake = [False] * NU
    cur = [-1] * NU
    ep_of = [None] * NU
    pend_ep = [None] * NU
    last_arr = [-1e18] * NU
    last_busy = [-1e18] * NU
    bk_kv = [0.0] * NU
    fut_kv = [dict() for _ in range(NU)]
    kv_last_min = [None] * NU
    card_units = [set() for _ in range(K)]
    for u in range(NU):
        if awake_boot:
            c = assign[u]
            gpu[c] += units[u]["f"]
            awake[u] = True; cur[u] = c; card_units[c].add(u)
        else:
            for c in cands[u]:
                gpu[c] += s
    for c in range(K):
        if gpu[c] > card_gib + 1e-9:
            raise ValueError(f"boot over-commit: card {c} holds {gpu[c]:.2f} GiB > {card_gib} GiB")
    backlog = [0.0] * K
    cap_min = C * 60.0
    rcap = F.rho_cap(phys) if kv_on else 0.0
    out = [collections.Counter() for _ in range(M)]
    st = collections.Counter()
    kvw_hist = collections.Counter()
    heap = []
    hseq = [0]
    deferred = []
    svc = []
    week = collections.Counter()
    # ---- v2 state (prereg §4)
    reach = kvtok
    Pc = None
    res_of = None
    borrow = [0.0] * NU
    borrow_min = [None] * NU
    kv_cur = None
    hist = None
    restart_at = [None] * NU
    next_epoch = None
    if pool or rsz:
        Pc = [card_gib] * K
        res_of = [[] for _ in range(K)]
        for u in range(NU):
            Pc[assign[u]] -= units[u]["w"] + OVH
            res_of[assign[u]].append(u)
        fbsum = [0.0] * K
        for u in range(NU):
            fbsum[assign[u]] += kvtok[u] * bpt[u] / G30
        if pool:
            reach = [max(kvtok[u], (Pc[assign[u]] - (fbsum[assign[u]] - kvtok[u] * bpt[u] / G30)) * G30 / bpt[u])
                     for u in range(NU)]
    if bal:
        nsl = [0] * K
        for u in range(NU):
            for c in cands[u]:
                nsl[c] += 1
        reach = [max(kvtok[u], (card_gib - units[u]["w"] - OVH - s * (nsl[assign[u]] - 1)) * G30 / bpt[u])
                 for u in range(NU)]
    if rsz:
        H = int(resize["H"])
        rs_s = float(resize["restart_s"])
        thr = float(resize["thr"])
        sized = [kvtok[u] * bpt[u] / G30 for u in range(NU)]
        kv_cur = list(kvtok)
        for c in range(K):
            us = res_of[c]
            if us:
                new = resize_split(Pc[c], [sized[u] for u in us], [sized[u] for u in us])
                for u, g in zip(us, new):
                    kv_cur[u] = g * G30 / bpt[u]
        reach = kv_cur
        hist = [dict() for _ in range(NU)]
        next_epoch = min0 + H

    def bsum_at(c, t):
        """Σ borrows in force on card c at time t: a grant made at the end of minute m holds through minute m + 1."""
        mt = min0 + int(t // 60.0)
        tot_ = 0.0
        for r in card_units[c]:
            if borrow_min[r] is not None and borrow_min[r] + 1 == mt:
                tot_ += borrow[r]
        return tot_

    def bor_at(r, t):
        mt = min0 + int(t // 60.0)
        return borrow[r] if (borrow_min[r] is not None and borrow_min[r] + 1 == mt) else 0.0

    def adv(c, t):
        if groups is not None:
            groups[c].advance(t)

    def ready_by(u, t):
        e = ep_of[u]
        if e is None:
            return True
        if e.ready is None and groups is not None:
            adv(e.card, t)
        return e.ready is not None and e.ready <= t

    def plan(u, c, t, force):
        need = units[u]["f"] - s
        bs = bsum_at(c, t) if bal else 0.0
        if gpu[c] + bs + need <= card_gib + 1e-9:
            return []
        adv(c, t)
        cand = sorted((last_arr[r], r) for r in card_units[c]
                      if ready_by(r, t) and (force or not last_arr[r] > t - protect) and not last_busy[r] > t)
        free = card_gib - gpu[c] - bs
        vict = []
        for _, r in cand:
            if free >= need - 1e-9:
                break
            vict.append(r); free += units[r]["f"] - s + (bor_at(r, t) if bal else 0.0)
        if free < need - 1e-9:
            return None
        return vict

    def h2d_done_cb(e):
        def cb(tc):
            e.ready = tc + e.extra
        return cb

    def start_h2d(e, t):
        if v2:
            return
        e.mult = 1.0
        groups[e.card].add("h", e.card, e.h2d_gb, 1.0, h2d_done_cb(e), t)

    def dem_done_cb(e):
        def cb(tc):
            e.pend_dem -= 1
            if e.pend_dem == 0:
                start_h2d(e, tc)
        return cb

    def admit(u, c, vict, t, e):
        U = units[u]
        for r in vict:
            awake[r] = False; card_units[c].discard(r); gpu[c] -= units[r]["f"] - s; cur[r] = -1
            bk_kv[r] = 0.0
            fut_kv[r].clear()
            if bal:
                borrow[r] = 0.0; borrow_min[r] = None
            st["evictions"] += 1
        busy = any(r != u for r in card_units[c])
        awake[u] = True; card_units[c].add(u); gpu[c] += U["f"] - s; cur[u] = c
        if bal and gpu[c] + bsum_at(c, t) > card_gib + 1e-6:
            raise RuntimeError(f"BALLOON invariant broken at admission: card {c} gpu + borrows > {card_gib}")
        e.card = c; e.t_adm = t
        st["wakes"] += 1
        if c != assign[u]:
            st["copy_wakes"] += 1
        if v2:
            L = len(vict) * V.SLEEP_S + V.PAGEIN_FIXED + U["w"] * GIB / V.PAGEIN_GBPS
            e.ready = t + L
        else:
            e.extra = W.FIXED_S
            e.h2d_gb = U["w"] * GIB * (beta if busy else 1.0)
            dem_gb = 0.0 if cfg["zero_demote"] else None
            e.pend_dem = len(vict)
            if not vict:
                start_h2d(e, t)
            else:
                for r in vict:
                    groups[c].add("d", c, dem_gb if dem_gb is not None else units[r]["w"] * GIB, 1.0, dem_done_cb(e), t)
        ep_of[u] = e
        svc.append(e)

    def try_admit(u, t, force, e):
        best = None
        for c in cands[u]:
            vict = plan(u, c, t, force)
            if vict is None:
                continue
            adv(c, t)
            nj = groups[c].n_h2d(c) if groups is not None else 0
            key = (len(vict), backlog[c], nj, -(card_gib - gpu[c]), c)
            if best is None or key < best[0]:
                best = (key, c, vict)
        if best is None:
            return False
        admit(u, best[1], best[2], t, e)
        return True

    def next_retry(u, t, e):
        nxt = e.t_create + W.W_MAX_S
        if t >= nxt - 1e-9:
            nxt = INF_T
        for c in cands[u]:
            for r in card_units[c]:
                if ready_by(r, t):
                    x = max(last_arr[r] + protect, last_busy[r])
                    if x > t + 1e-9:
                        nxt = min(nxt, x)
                else:
                    er = ep_of[r]
                    if er.ready is not None and er.ready > t:
                        nxt = min(nxt, er.ready)
            if bal:     # a borrow ends at a minute boundary: retry then (prereg §4.4)
                for r in card_units[c]:
                    if borrow_min[r] is not None and borrow[r] > 0.0:
                        x = (borrow_min[r] + 2 - min0) * 60.0
                        if x > t + 1e-9:
                            nxt = min(nxt, x)
        if nxt >= INF_T:
            nxt = t + 0.05
        return min(nxt, e.t_create + F.W_GIVEUP_S)

    def push(t, u):
        hseq[0] += 1
        heapq.heappush(heap, (t, hseq[0], u))

    def run_events(t):
        while heap and heap[0][0] <= t:
            te, _, u = heapq.heappop(heap)
            e = pend_ep[u]
            if e is None:
                continue
            if te > e.t_create + 86400.0:
                raise RuntimeError(f"wake of unit {u} pending > 1 day: placement invariant broken")
            if te >= e.t_create + F.W_GIVEUP_S - 1e-9:      # v1 amendment 2: card-mates never drained in time
                e.ready = math.inf
                pend_ep[u] = None
                st["abandoned_wakes"] += 1
                continue
            force = te >= e.t_create + W.W_MAX_S - 1e-9
            if try_admit(u, te, force, e):
                pend_ep[u] = None
                st["queued_admits"] += 1
                if force:
                    st["forced_admits"] += 1
            else:
                push(next_retry(u, te, e), u)

    def home_card(u):
        return cur[u] if cur[u] >= 0 else assign[u]

    def do_epoch(e):
        """prereg §4.5: re-split each card by trailing demand over [e - H, e); restart every resident of a re-split card."""
        t_e = (e - min0) * 60.0
        for c in range(K):
            us = res_of[c]
            if not us:
                continue
            old = [kv_cur[u] * bpt[u] / G30 for u in us]
            tgt = [max(sized[u], trailing_peak(hist[u], e, H) / KV_UTIL * bpt[u] / G30) for u in us]
            new = resize_split(Pc[c], [sized[u] for u in us], tgt)
            if max(abs(n - o) / max(o, 1e-12) for n, o in zip(new, old)) > thr:
                st["resplits"] += 1
                for u, g in zip(us, new):
                    kv_cur[u] = g * G30 / bpt[u]
                    ep = W.Ep(u, t_e)
                    ep.card = c; ep.ready = t_e + rs_s
                    ep_of[u] = ep
                    restart_at[u] = e
                    st["restarts"] += 1
        for u in range(NU):
            for t in [t for t in hist[u] if t < e - H]:
                del hist[u][t]

    N = len(order)
    i_pos = 0
    prev_mi = None
    while i_pos < N:
        mi = int(trace.minute[order[i_pos]])
        if prev_mi is not None and mi - prev_mi > 1:
            for c in range(K):
                backlog[c] = max(0.0, backlog[c] - cap_min * (mi - prev_mi - 1))
        prev_mi = mi
        if rsz:
            while next_epoch <= mi:
                do_epoch(next_epoch)
                next_epoch += H
        j = i_pos
        while j < N and int(trace.minute[order[j]]) == mi:
            j += 1
        Wc = [0.0] * K
        pend = []
        kvcells = []
        base = (mi - min0) * 60.0
        for p in range(i_pos, j):
            i = int(order[p])
            m = int(trace.mid[i]); n = int(trace.n[i])
            u = unit_of[m]
            t0 = base + float(first[i])
            run_events(t0)
            if u < 0:                                           # NO-CARD (prereg §2): weights do not fit this card class
                st["no_card_req"] += n
                pend.append((m, -3, n, 0.0, None, 0.0, i, u))
                continue
            if kv_on and (trace.it[i] + trace.ot[i]) / n > reach[u]:
                st["unservable_req"] += n
                pend.append((m, -2, n, 0.0, None, 0.0, i, u))
                continue
            ab = models[m][2]
            served = True
            wait_ep = None
            if awake[u]:
                if not ready_by(u, t0):
                    wait_ep = ep_of[u]
            elif pend_ep[u] is not None:
                wait_ep = pend_ep[u]
            else:
                e = W.Ep(u, t0)
                if try_admit(u, t0, False, e):
                    if not (e.ready is not None and e.ready <= t0):
                        wait_ep = e
                elif queue:
                    pend_ep[u] = e
                    st["queued"] += 1
                    push(next_retry(u, t0, e), u)
                    wait_ep = e
                else:
                    served = False
            last_arr[u] = base + float(last[i])
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C
            if not served:
                st["unserved_req"] += n
                pend.append((m, -1, n, pre, None, 0.0, i, u))
                continue
            c = home_card(u)
            wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
            Wc[c] += wm
            extra = V._burst_extra(wm, burst_w)
            pend.append((m, c, n, pre, wait_ep, extra, i, u))
            if kv_on:
                kvcells.append((u, c, n, trace.it[i] / n + phys["kappa"] * trace.ot[i] / n, pre, trace.ot[i] / n,
                                float(last[i]), len(pend) - 1))
        q = [0.0] * K
        rho = [0.0] * K
        for c in range(K):
            if kv_on:
                rho[c] = min(rcap, (backlog[c] + Wc[c]) / cap_min)
            q[c] = (backlog[c] + 0.5 * max(0.0, Wc[c] - cap_min)) / C
            backlog[c] = max(0.0, backlog[c] + Wc[c] - cap_min)
        kvw = {}
        adm = {}
        if kv_on and kvcells:
            byu = collections.defaultdict(list)
            for u, c, n, occ, pre, outr, lo, px in kvcells:
                byu[u].append((n * occ, pre + outr * (phys["tpot_q"] / (1.0 - rho[c])), lo, n, px))
            if mech is None and shed_tau is None:               # v1's block, verbatim arithmetic (pitfall 1)
                for u, items in byu.items():
                    bk = F.kv_catch_up(bk_kv[u], fut_kv[u], kvtok[u], kv_last_min[u], mi)
                    kv_last_min[u] = mi
                    D = fut_kv[u].pop(mi, 0.0)
                    for x, R, lo, n_, px in items:
                        D += F.spread(fut_kv[u], mi, x, R)
                    kvw[u], bk_kv[u] = F.kv_step(bk, D, kvtok[u])
                    for x, R, lo, n_, px in items:
                        lb = base + lo + kvw[u] + R
                        if lb > last_busy[u]:
                            last_busy[u] = lb
            else:
                ent = {}
                for u, items in byu.items():
                    k0 = kv_cur[u] if rsz else kvtok[u]
                    bk = F.kv_catch_up(bk_kv[u], fut_kv[u], k0, kv_last_min[u], mi)
                    kv_last_min[u] = mi
                    s_in = fut_kv[u].pop(mi, 0.0)
                    Nn = 0.0
                    for x, R, lo, n_, px in items:
                        Nn += x * min(R, 60.0)
                    ent[u] = (bk, s_in, Nn, k0)
                kte = {u: v[3] for u, v in ent.items()}
                if pool or bal:
                    bycard = collections.defaultdict(list)
                    for u in ent:
                        if cur[u] >= 0:
                            bycard[cur[u]].append(u)
                    for c, us in bycard.items():
                        us.sort()
                        fb = [kvtok[u] * bpt[u] / G30 for u in us]
                        nd = [(ent[u][0] + ent[u][1] + ent[u][2]) / 60.0 * bpt[u] / G30 for u in us]
                        spare = (Pc[c] - sum(fb)) if pool else (card_gib - gpu[c])
                        caps, gr = elastic_caps(fb, nd, spare)
                        if pool and sum(caps) > Pc[c] + 1e-6:
                            raise RuntimeError(f"POOL invariant broken: card {c} caps {sum(caps):.3f} > {Pc[c]:.3f}")
                        if bal and gpu[c] + sum(gr) > card_gib + 1e-6:
                            raise RuntimeError(f"BALLOON invariant broken: card {c} gpu + grants > {card_gib}")
                        for u, cp, g in zip(us, caps, gr):
                            kte[u] = cp * G30 / bpt[u]
                            if g > 0.0:
                                st["grant_gib_min"] += g
                                if mi >= score_from:
                                    st["grant_gib_min_held"] += g
                            if bal:
                                borrow[u] = g; borrow_min[u] = mi
                if rsz:
                    for u in ent:
                        if restart_at[u] == mi:
                            kte[u] = kte[u] * (60.0 - rs_s) / 60.0
                for u, (bk, s_in, Nn, k0) in ent.items():
                    a = 1.0 if shed_tau is None else shed_fraction(bk, s_in, Nn, kte[u], shed_tau)
                    D = s_in
                    for x, R, lo, n_, px in byu[u]:
                        if a < 1.0:
                            k_ = math.floor(a * n_ + 1e-9)
                            adm[px] = k_
                            if k_ == 0:
                                continue
                            x = x * k_ / n_
                        D += F.spread(fut_kv[u], mi, x, R)
                    kvw[u], bk_kv[u] = F.kv_step(bk, D, kte[u])
                    if rsz:
                        hist[u][mi] = D
                    for x, R, lo, n_, px in byu[u]:
                        if adm.get(px, n_) == 0:
                            continue
                        lb = base + lo + kvw[u] + R
                        if lb > last_busy[u]:
                            last_busy[u] = lb
        for px, (m, c, n, pre, e, extra, i, u) in enumerate(pend):
            na = adm.get(px, n)
            if na < n:
                st["shed_req"] += n - na
                if mi >= score_from:
                    st["shed_req_held"] += n - na
                    out[m][T.INF_KEY] += n - na
                    week[T.INF_KEY] += n - na
            if mi < score_from or na == 0:
                continue
            if c < 0:
                out[m][T.INF_KEY] += n
                week[T.INF_KEY] += n
                continue
            kw = kvw.get(u, 0.0)
            if kv_on:
                kvw_hist[T._key(kw)] += na
            if e is None:
                key = T._key(q[c] + extra + pre + kw)
                out[m][key] += na
                week[key] += na
            else:
                ts = T.cell_times(trace, i, first, last, seed) + base
                deferred.append((m, ts[:na] if na < n else ts, q[c], extra, pre, e, kw))
        i_pos = j
    run_events(INF_T / 10)
    if groups is not None:
        for c in range(K):
            groups[c].advance(INF_T)
    for m, ts, qc, extra, pre, e, kw in deferred:
        if e.ready is None:
            raise RuntimeError("episode never became ready (link engine bug)")
        waits = np.maximum(0.0, e.ready - ts)
        for wv in waits:
            wv = float(wv)
            if phi > 0.0:
                hid = min(phi * pre, wv)
                key = T._key(wv + qc + extra + (pre - hid) + kw)
            else:
                key = T._key(wv + qc + extra + pre + kw)
            out[m][key] += 1
            week[key] += 1
    held_svc = [e.ready - e.t_adm for e in svc if e.t_adm is not None and e.t_adm >= (score_from - min0) * 60.0]
    maxconc = 0
    if groups is not None:
        for g in {id(g): g for g in groups.values()}.values():
            if g.maxconc:
                maxconc = max(maxconc, max(g.maxconc.values()))
    met = {"stats": dict(st), "max_h2d_jobs_per_link": maxconc, "svc_p50": W._q(held_svc, 0.5),
           "svc_p99": W._q(held_svc, 0.99), "svc_n": len(held_svc), "week_n": int(sum(week.values())),
           "week_unserved": int(week.get(T.INF_KEY, 0))}
    for qq, nm in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
        met["week_" + nm] = W._jsonable(T._pct(week, qq))
        met["kvw_" + nm] = W._jsonable(T._pct(kvw_hist, qq)) if kvw_hist else None
    return out, met


# ------------------------------------------------------------------------------------------------ population (§2)
_CACHE = {}


def load_census(root):
    p = os.path.join(root, CENSUS_JSON)
    if T.sha256_file(p) != CENSUS_SHA:
        raise ValueError(f"{CENSUS_JSON} sha mismatch")
    d = os.path.join(root, DEMAND_JSON)
    if T.sha256_file(d) != DEMAND_SHA:
        raise ValueError(f"{DEMAND_JSON} sha mismatch")
    rows = {r["cid"]: r for r in json.load(open(p))["rows"]}
    klass = {r["cid"]: r["class_PRIMARY"] for r in json.load(open(d))["rows"]}
    return rows, klass


def model_from_census(mod, row, L_occ, reading):
    """prereg §2 -> ((cid, tb, ab, prec, w, f), bpt bf16 bytes/token, status). GUESSED keeps v1's sizes and bpt_of."""
    cid, tb, ab, prec, w, f = mod
    pt = row.get("params_total_R2") if reading == "R2" else row.get("params_total")
    pa = row.get("params_active_R2") if reading == "R2" else row.get("params_active")
    if row["status"].startswith("MEASURED") and pt:
        tbm = pt / 1e9
        abm = (pa or pt) / 1e9
        wm = SC.weights_gib_fp8(pt)
        kd = {a[3:]: row[a] for a in row if a.startswith("kv_")}
        return (cid, tbm, abm, "fp8", wm, wm + OVH + T.KV_GIB), SC.bpt_effective(kd, L_occ, "SW_AWARE"), "MEASURED"
    return mod, F.bpt_of(tb), "GUESSED"


def build_pop2(phys, root, cache):
    """-> dict(tr, bpt per model (bf16), status, klass, cnt, work_train). sizes 'p8' = v1's population exactly."""
    key = ("pop2", phys["sizes"], phys["reading"], phys["bpt_occ"], phys["kappa"])
    if key in _CACHE:
        return _CACHE[key]
    tr, cnt, work_train = F.build_pop("P8", root, cache)
    M = len(tr.models)
    if phys["sizes"] == "p8":
        models = list(tr.models)
        bpt = [F.bpt_of(m[1]) for m in models]
        status = ["P8"] * M
        klass = ["P8"] * M
    elif phys["sizes"] == "census":
        rows, kl = load_census(root)
        k = (tr.minute >= HELD) if phys["bpt_occ"] == "heldout" else (tr.minute < HELD)
        sn = np.bincount(tr.mid[k], weights=tr.n[k].astype(float), minlength=M)
        si = np.bincount(tr.mid[k], weights=tr.it[k], minlength=M)
        so = np.bincount(tr.mid[k], weights=tr.ot[k], minlength=M)
        L = np.where(sn > 0, (si + phys["kappa"] * so) / np.maximum(sn, 1), 0.0)
        models, bpt, status, klass = [], [], [], []
        for m, mod in enumerate(tr.models):
            mm, b, stt = model_from_census(mod, rows[mod[0]], float(L[m]), phys["reading"])
            models.append(mm); bpt.append(b); status.append(stt); klass.append(kl[mod[0]])
    else:
        raise ValueError(f"unknown sizes {phys['sizes']}")
    tr2 = T.Trace.__new__(T.Trace)
    tr2.minute, tr2.mid, tr2.n, tr2.it, tr2.ot, tr2.explicit = tr.minute, tr.mid, tr.n, tr.it, tr.ot, None
    tr2.models = models
    _CACHE[key] = dict(tr=tr2, bpt=bpt, status=status, klass=klass, cnt=cnt, work_train=work_train)
    return _CACHE[key]


def unit_kv2(units, mode, card_gib, bpt_u, floor_tok=None, conc_tok=None):
    """v1 unit_kv with per-unit bytes/token (int8 already applied) -> (kv_gib or None (NO-CARD), kvtok or None)."""
    kv, tok = [], []
    for k, U in enumerate(units):
        b = bpt_u[k]
        hi = card_gib - U["w"] - OVH
        if hi <= 0:
            kv.append(None); tok.append(None)
            continue
        if mode == "fixed1":
            g = 1.0
        elif mode == "eq":
            g = F.EQ_KV["small"] if U["tb"] <= 5.0 else F.EQ_KV["big"]
        elif mode == "floor":
            g = max(min(floor_tok[k] * b / G30, hi), min(F.KV_MIN_GIB, hi))
        elif mode == "size":
            g = max(min(max(floor_tok[k], conc_tok[k]) * b / G30, hi), min(F.KV_MIN_GIB, hi))
        else:
            raise ValueError(f"unknown kv mode {mode}")
        g = min(g, hi)
        kv.append(g)
        tok.append(g * G30 / b)
    return kv, tok


def setup2(arm, phys, root, cache):
    skey = ("setup2", arm, json.dumps(phys, sort_keys=True, default=str))
    if skey not in _CACHE:
        _CACHE[skey] = _setup2(arm, phys, root, cache)
    return _CACHE[skey]


def _setup2(arm, phys, root, cache):
    cfg, _ = arm_cfg2(arm)
    if arm in REPRO and phys["sizes"] != "p8":
        raise ValueError("V1_* reproductions run on v1's sizes: --set sizes=p8")
    pop = build_pop2(phys, root, cache)
    tr = pop["tr"]
    big = cfg["card"] == "BIG"
    card_gib = phys["big_card_gib"] if big else T.CARD_GIB
    C = C_REF * (phys["big_mult"] if big else 1.0) * phys["cref_mult"]
    models = W.build_models(tr, cfg["prec"])
    census = W.load_census(root) if cfg["units"] != "none" else None
    units_all, unit_of_all = W.build_units(models, cfg["units"], census, pop["work_train"])
    F.check_adapter_slots(units_all)
    bpt_all = [pop["bpt"][U["rep"]] / (2.0 if cfg["int8"] else 1.0) for U in units_all]
    ft = F.kv_floor_tok(tr, unit_of_all, len(units_all), phys["floor_q"])
    ct = F.kv_conc_tok(tr, unit_of_all, len(units_all), models, phys, phys["size_q"], C=C) if cfg["kv"] == "size" else None
    kv_gib_all, kvtok_all = unit_kv2(units_all, cfg["kv"], card_gib, bpt_all, ft, ct)
    keep = [k for k in range(len(units_all)) if kv_gib_all[k] is not None]
    new_of = {k: j for j, k in enumerate(keep)}
    units = [dict(units_all[k]) for k in keep]
    for U, k in zip(units, keep):
        U["f"] = U["w"] + OVH + kv_gib_all[k]
    unit_of = [new_of.get(u, -1) for u in unit_of_all]
    live = None
    if cfg["placement"] == "COACT" or cfg["copies"]:
        lkey = ("live", phys["sizes"], cfg["units"], cfg["prec"])
        if lkey not in _CACHE:
            _CACHE[lkey] = W.train_live(tr, unit_of_all, len(units_all))
        live = _CACHE[lkey][keep]
    hneed = F.host_need(units, phys, cfg["ram"], cfg["all_awake"])
    return dict(cfg=cfg, tr=tr, units=units, unit_of=unit_of, kvtok=[kvtok_all[k] for k in keep],
                kv_gib=[kv_gib_all[k] for k in keep], bpt=[bpt_all[k] for k in keep], live=live, hneed=hneed,
                card_gib=card_gib, C=C, big=big, box_g=4 if big else W.BOX_G, pop=pop,
                no_card_units=len(units_all) - len(keep))


def place_arm2(S, K, phys):
    cfg = S["cfg"]
    if cfg["placement"] == "SPREAD":
        homes = spread_place(S["units"], K, phys, S["hneed"], S["card_gib"])
        return homes, [[h] for h in homes], {}
    return F.place_arm(S, K, phys)


def k_hold(S, phys, grid):
    for K in grid:
        try:
            place_arm2(S, K, phys)
            return K
        except ValueError:
            continue
    return None


def ref_scores(pop, card, phys, burst_w=None):
    """prereg §3: dedicated reference for a card class (bf16 KV, census bytes/token). NO-CARD -> kvtok -1 (all +inf)."""
    key = ("ref", card, phys["sizes"], phys["reading"], phys["bpt_occ"], phys["arrivals"], phys["tpot_q"],
           phys["tpot_sat"], phys["kappa"], phys["cref_mult"], phys["big_card_gib"], phys["big_mult"])
    if key not in _CACHE:
        tr = pop["tr"]
        cg = phys["big_card_gib"] if card == "BIG" else T.CARD_GIB
        C = C_REF * phys["cref_mult"] * (phys["big_mult"] if card == "BIG" else 1.0)
        kt = [((cg - m[4] - OVH) * G30 / pop["bpt"][k]) if cg - m[4] - OVH > 0 else -1.0
              for k, m in enumerate(tr.models)]
        _CACHE[key] = F.dedicated(tr, kt, phys, burst_w, C=C)
    return _CACHE[key]


def _sub_scores(out, ded, labels, slos=T.SLOS):
    res = {}
    for lab in sorted(set(labels)):
        idx = [m for m, x in enumerate(labels) if x == lab]
        sc = {str(x): T.score([out[m] for m in idx], [ded[m] for m in idx], x) for x in slos}
        res[lab] = {k: {"M_frac": v["M_frac"], "meet": v["meet"], "feasible": v["slo_feasible_models"],
                        "R_ratio": v["R_ratio"]} for k, v in sc.items()}
    return res


def run_cell2(arm, K, seed, phys, root, cache, tag=""):
    S = setup2(arm, phys, root, cache)
    cfg = S["cfg"]
    bw = V.BURST_W if phys["arrivals"] == "BURST" else None
    row = {"arm": arm, "K": K, "seed": seed, "tag": tag, "phys": phys, "cfg": cfg, "n_units": len(S["units"]),
           "n_models": len(S["tr"].models), "no_card_units": S["no_card_units"]}
    kg = sorted(S["kv_gib"])
    if not kg:
        raise ValueError(f"{arm}: every unit is NO-CARD on this card class (QA r1)")
    row["kv_gib"] = {"p50": kg[len(kg) // 2], "p90": kg[int(0.9 * (len(kg) - 1))], "max": kg[-1], "sum": sum(kg)}
    t0 = time.time()
    try:
        homes, cands, info = place_arm2(S, K, phys)
    except ValueError as e:
        row.update(infeasible=str(e), elapsed_s=round(time.time() - t0, 1))
        return row
    row.update(info)
    row["ram_need_gib_per_gpu"] = round(F.ram_need_per_gpu(S["units"], cands, S["hneed"], K), 3)
    cfg_sim = dict(cfg)
    cfg_sim["h2d"], cfg_sim["d2h"], cfg_sim["protect"] = phys["h2d"], phys["d2h"], phys["protect"]
    rs = dict(H=phys["resize_h"], restart_s=phys["restart_s"], thr=phys["resize_thr"]) if cfg["mech"] == "resize" else None
    out, met = simulate(S["tr"], S["units"], S["unit_of"], ("card", homes), K, cfg_sim, seed, bw,
                        card_gib=S["card_gib"], phys=phys, kvtok=S["kvtok"], copies=cands if cfg["copies"] else None,
                        C=S["C"], mech=cfg["mech"], shed_tau=cfg["shed_tau"], bpt=S["bpt"], resize=rs)
    row.update(met)
    pop = S["pop"]
    d5, db = ref_scores(pop, "5090", phys, bw), ref_scores(pop, "BIG", phys, bw)
    row["slo_5090"] = {str(x): T.score(out, d5, x) for x in T.SLOS}
    row["slo_big"] = {str(x): T.score(out, db, x) for x in T.SLOS}
    prim = db if (S["big"] and arm not in REPRO) else d5
    row["slo"] = row["slo_big"] if prim is db else row["slo_5090"]
    row["by_status"] = _sub_scores(out, prim, pop["status"])
    row["by_class"] = _sub_scores(out, prim, pop["klass"])
    row["heldout_req"] = int(sum(sum(c.values()) for c in out))
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def k_list2(arm, phys, root, cache, ks=None, kgrid=False):
    cfg, _ = arm_cfg2(arm)
    if cfg["placement"] == "COFIT":
        S = setup2(arm, phys, root, cache)
        _, nb = F.cofit_place(S["units"], phys, S["hneed"], S["card_gib"], cfg["ram"])
        return [nb]
    if kgrid:
        return list(K_GRID_BIG if cfg["card"] == "BIG" else K_GRID_5090)
    if not ks:
        raise ValueError(f"{arm}: needs --ks or --kgrid (QA r1)")
    return list(ks)


# ------------------------------------------------------------------------------------------------ instrument checks (§8)
def check_i1(d, root, cache):
    """I1v2: v2 with v1's sizes and every v2 feature off reproduces v1's committed rows."""
    want = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            for l in open(os.path.join(d, f)):
                if l.strip():
                    r = json.loads(l)
                    if r.get("tag", "") == "" and r["arm"] in ("ALL", "BIG_ALL", "BIG_AWAKE_KV"):
                        want[(r["arm"], r["K"], r["seed"])] = r
    plan_ = [("V1_ALL", "ALL", (336, 384)), ("V1_BIG_ALL", "BIG_ALL", (56, 192)), ("V1_BIG_AWAKE_KV", "BIG_AWAKE_KV", None)]
    phys = phys_with({"sizes": "p8"})
    res = []
    for a2, a1, ks in plan_:
        for K in (ks or k_list2(a2, phys, root, cache)):
            for seed in F.SEEDS:
                r1 = want.get((a1, K, seed))
                if r1 is None:
                    res.append({"arm": a2, "K": K, "seed": seed, "ok": False, "why": "v1 row missing"})
                    continue
                r2 = run_cell2(a2, K, seed, phys, root, cache)
                if "infeasible" in r1:
                    ok = "infeasible" in r2
                else:
                    ok = ("infeasible" not in r2 and r2["stats"] == r1["stats"] and r2["slo_5090"] == r1["slo"]
                          and r2["heldout_req"] == r1["heldout_req"])
                res.append({"arm": a2, "K": K, "seed": seed, "ok": bool(ok)})
    return {"I1v2": res, "I1v2_pass": bool(res) and all(x["ok"] for x in res)}


def check_i5(root, cache):
    """I5: REF_5090 with census R1 sizes + held-out occupancy reproduces the census's honest dedicated counts."""
    phys = phys_with({"reading": "R1", "bpt_occ": "heldout"})
    pop = build_pop2(phys, root, cache)
    tr = pop["tr"]
    keep = [m for m, mod in enumerate(tr.models) if mod[4] + OVH <= T.CARD_GIB]
    d5 = ref_scores(pop, "5090", phys)
    got = {}
    for slo in (10.0, 2.0):
        sc = T.score([d5[m] for m in keep], [d5[m] for m in keep], slo)
        got[f"S{int(slo)}"] = {"eligible": sc["eligible"], "feasible_honest": sc["slo_feasible_models"]}
    return {"I5": got, "want": I5_WANT, "I5_pass": got == I5_WANT, "n_keep": len(keep)}


# ------------------------------------------------------------------------------------------------ summarize (§7)
def _rows(d):
    return F._rows(d)


def _m(r, slo):
    if "infeasible" in r:
        return None
    return r["slo"][str(slo)]["M_frac"]


def q1_verdict(ctrl, arm):
    """prereg §7 Q1 on per-seed (M) lists (None = infeasible)."""
    if arm is None or any(x is None for x in arm):
        return "INFEASIBLE-AT-TOP"
    if ctrl is None or any(x is None for x in ctrl):
        return "CONTROL-INFEASIBLE"
    if min(arm) > max(ctrl):
        return "LIFTS"
    if max(arm) < min(ctrl):
        return "LOWERS"
    return "UNATTRIBUTABLE"


def k_match(byk, seeds, th, top, step):
    """prereg §7 Q2: smallest grid K from which every K' up to the top meets th on every seed (infeasible fails);
    -> (all-seed value, [per-seed min, max]); never -> top + step."""
    ks = sorted(byk)

    def stable(pred):
        best = None
        for k in reversed(ks):
            if pred(k):
                best = k
            else:
                break
        return best if best is not None else top + step

    def ok(k, s_):
        r = byk[k].get(s_)
        return r is not None and r is not False and r >= th

    a = stable(lambda k: all(ok(k, s_) for s_ in seeds))
    per = [stable(lambda k, s_=s_: ok(k, s_)) for s_ in seeds]
    return a, [min(per), max(per)]


def summarize(d, root=None):
    rows = _rows(d)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["arm"], r.get("tag", ""))].append(r)
    res = {}
    for (arm, tag), rs in g.items():
        if arm in REPRO:
            continue
        cfg, base = arm_cfg2(arm)
        big = cfg["card"] == "BIG"
        grid = K_GRID_BIG if big else K_GRID_5090
        top, step = grid[-1], grid[1] - grid[0]
        seeds = sorted({r["seed"] for r in rs})
        one = {"rows": len(rs), "seeds": seeds, "card": cfg["card"], "base": base}
        feas_k = sorted({r["K"] for r in rs if "infeasible" not in r})
        one["K_hold"] = feas_k[0] if feas_k else None
        for slo in T.SLOS:
            byk = collections.defaultdict(dict)
            for r in rs:
                byk[r["K"]][r["seed"]] = _m(r, slo)
            o = {"M_top": [byk.get(top, {}).get(s_) for s_ in seeds], "byk": {k: [v.get(s_) for s_ in seeds]
                                                                           for k, v in sorted(byk.items())}}
            o["v1_rule"] = F.min_ks(rs, slo, top)
            o["bar_833"] = k_match(byk, seeds, V1_BAR, top, step)
            one[str(slo)] = o
        res[f"{arm}|{tag}"] = one
    out = {"arms": res, "Q1": {}, "Q1b": {}, "Q2": {}}

    def get(a, tag):
        return res.get(f"{a}|{tag}")

    tags = sorted({t for (_, t) in g})
    for tag in tags:
        for key, one in res.items():
            arm, t = key.split("|")
            if t != tag:
                continue
            big = arm.startswith("BIG_")
            base = one["base"]
            pre = "BIG_" if big else ""
            ctrl_name = pre + ("ALL" if base in ("ALL", "BAL_SZ", "BAL_LEN") or base.startswith("ALL_") or
                               base.startswith("BAL_") else "AWAKE")
            ctl = get(ctrl_name, tag)
            if ctl is None:
                continue
            grid = K_GRID_BIG if big else K_GRID_5090
            top, step = grid[-1], grid[1] - grid[0]
            for slo in T.SLOS:
                sk = str(slo)
                cm, am = ctl[sk]["M_top"], one[sk]["M_top"]
                for cn in CTRL.get(base, ()):
                    c2 = get(pre + cn, tag)
                    if c2 is None:
                        continue
                    v = q1_verdict(c2[sk]["M_top"], one[sk]["M_top"])
                    dm = (None if v in ("INFEASIBLE-AT-TOP", "CONTROL-INFEASIBLE") else
                          sum(one[sk]["M_top"]) / len(one[sk]["M_top"]) - sum(c2[sk]["M_top"]) / len(c2[sk]["M_top"]))
                    out["Q1"][f"{arm}|{tag}|vs:{pre + cn}|S{slo:g}"] = {"verdict": v, "mean_delta_M": dm,
                                                                         "arm": one[sk]["M_top"], "ctrl": c2[sk]["M_top"]}
                    kop = ctl.get("K_hold")
                    if kop is not None:
                        ca = c2[sk]["byk"].get(kop)
                        aa = one[sk]["byk"].get(kop)
                        v2_ = q1_verdict(ca, aa)
                        out["Q1b"][f"{arm}|{tag}|vs:{pre + cn}|S{slo:g}|K_op={kop}"] = {"verdict": v2_, "arm": aa,
                                                                                       "ctrl": ca}
                if cm is None or any(x is None for x in cm):
                    continue
                th = min(cm)
                byk_a = {k: dict(zip(one["seeds"], v)) for k, v in one[sk]["byk"].items()}
                byk_c = {k: dict(zip(ctl["seeds"], v)) for k, v in ctl[sk]["byk"].items()}
                ka, kar = k_match(byk_a, one["seeds"], th, top, step)
                kc, kcr = k_match(byk_c, ctl["seeds"], th, top, step)
                out["Q2"][f"{arm}|{tag}|S{slo:g}"] = {"theta": th, "ctrl": ctrl_name, "K_match": ka,
                                                      "per_seed": kar, "ctrl_K_match": kc, "ctrl_per_seed": kcr,
                                                      "delta_cards": kc - ka,
                                                      "attributable": not (kar[1] >= kcr[0] and kcr[1] >= kar[0])}
    if root is not None and os.path.exists(os.path.join(root, V.OFFERS_CSV)):
        out["Q4"] = costs2(rows, out, root)
    else:
        out["Q4"] = {"skipped": f"offers CSV not under root {root!r}"}
    return out


def costs2(rows, summ, root):
    """prereg §7 Q4 (descriptive): $/h at K_match. 5090 by RAM tier of the frozen offers; big card $1.35/GPU-h."""
    if T.sha256_file(os.path.join(root, V.OFFERS_CSV)) != F.OFFERS_SHA:
        raise ValueError("offers CSV sha mismatch")
    import csv
    offers = list(csv.DictReader(open(os.path.join(root, V.OFFERS_CSV))))
    res = {}
    for key, q in summ["Q2"].items():
        arm, tag, slo = key.split("|")
        K = q["K_match"]
        at = [r for r in rows if r["arm"] == arm and r.get("tag", "") == tag and r["K"] == K and "infeasible" not in r]
        if not at:
            res[key] = None
            continue
        if arm.startswith("BIG_"):
            usd = at[0]["phys"]["big_usd"]
        else:
            usd = V.price_tier(offers, max(r["ram_need_gib_per_gpu"] for r in at))["median"]
        res[key] = {"K": K, "usd_per_gpu_h": usd, "usd_per_h": None if usd is None else K * usd}
    return res


# ------------------------------------------------------------------------------------------------ self-test (§9)
def self_test():
    fails, n = [], 0

    def chk(name, got, want, tol=1e-6):
        nonlocal n
        n += 1
        if isinstance(want, list):
            ok = isinstance(got, list) and len(got) == len(want) and all(
                (a == b) if (not isinstance(b, float) or b == math.inf) else (a is not None and abs(a - b) <= tol)
                for a, b in zip(got, want))
        elif isinstance(want, float):
            ok = got is not None and got != math.inf and abs(got - want) <= tol
        else:
            ok = got == want
        if not ok:
            fails.append(f"FAIL {name}: got {got!r} want {want!r}")

    def vals(counter):
        o = []
        for k in sorted(counter):
            o += [math.inf if k >= T.INF_KEY else k / 100.0] * counter[k]
        return o

    def raises(name, fn, needle):
        nonlocal n
        n += 1
        try:
            fn()
        except ValueError as e:
            if needle not in str(e):
                fails.append(f"FAIL {name}: ValueError without '{needle}': {e}")
            return
        fails.append(f"FAIL {name}: no ValueError")

    phys = phys_with({"tpot_sat": 0.065})        # rho_cap 0.9 (prereg §9)
    mk = lambda cid, b, kv=1.0: (cid, b, b, "fp8") + T.footprint(b, "fp8", kv)
    sing = lambda models: W.build_units(models, "none")
    B8 = F.bpt_of(8.0)

    def run(models, reqs, homes, K, over=None, kvg=None, card_gib=T.CARD_GIB, copies=None, mech=None, shed=None,
            rsz=None, unit_of=None):
        cfg = F.arm_cfg("A0"); cfg.update(over or {})
        tr = T.Trace.explicit_cells(models, reqs)
        units, uo = sing(models)
        kvt = None if kvg is None else [g * G30 / F.bpt_of(models[k][1]) for k, g in enumerate(kvg)]
        return simulate(tr, units, unit_of or uo, ("card", homes), K, cfg, 0, score_from=0, min0=0, card_gib=card_gib,
                        phys=phys, kvtok=kvt, copies=copies, mech=mech, shed_tau=shed,
                        bpt=[F.bpt_of(m[1]) for m in models], resize=rsz)

    # GA capacity rule
    Pc = 88.0 - ((30 + OVH) + (20 + OVH) + (10 + OVH))
    chk("GA P_c", Pc, 25.54, 1e-9)
    chk("GA1", elastic_caps([10.0, 5.0], [14.0, 2.0], Pc - 15.0)[0], [14.0, 5.0], 1e-9)
    chk("GA2", elastic_caps([10.0, 5.0], [30.0, 2.0], Pc - 15.0)[0], [20.54, 5.0], 1e-9)
    chk("GA3 pool", elastic_caps([10.0, 5.0], [20.0, 13.0], Pc - 15.0)[0], [15.855555555555554, 9.684444444444445], 1e-9)
    sp_bal = 88.0 - ((30 + OVH + 10) + (20 + OVH + 5) + (10 + OVH + 3))
    chk("GA3 balloon", elastic_caps([10.0, 5.0], [20.0, 13.0], sp_bal)[0], [14.188888888888885, 8.351111111111107], 1e-9)
    # GB adversarial: growth never exceeds the card's physical free memory
    chk("GB1", elastic_caps([1.0], [101.0], 88.0 - 87.5)[1], [0.5], 1e-9)
    chk("GB2 spare clamps to 0", elastic_caps([1.0], [101.0], 88.0 - 88.3)[1], [0.0], 1e-12)
    rng = np.random.default_rng(27403)
    bad = 0
    for _ in range(2000):
        k = int(rng.integers(1, 6))
        fb = list(rng.uniform(0, 10, k)); nd = list(rng.uniform(0, 30, k)); sp = float(rng.uniform(-5, 20))
        caps, gr = elastic_caps(fb, nd, sp)
        e = [max(0.0, a - b) for b, a in zip(fb, nd)]
        if sum(gr) > max(0.0, sp) + 1e-9 or any(x < -1e-12 or x > y + 1e-9 for x, y in zip(gr, e)):
            bad += 1
    chk("GB fuzz 2,000 cases: sum grants <= spare, 0 <= g <= e", bad, 0)
    # GC shed (pure)
    a = shed_fraction(0.0, 0.0, 1800000.0, 10000.0, 10.0)
    chk("GC1 a*", a, 0.4444444444444444, 1e-12)
    k_ = math.floor(a * 30 + 1e-9)
    chk("GC1 admitted", k_, 13)
    chk("GC1 wait", F.kv_step(0.0, k_ / 30 * 1800000.0, 10000.0)[0], 9.0, 1e-9)
    chk("GC2 no excess", shed_fraction(0.0, 0.0, 600000.0, 10000.0, 10.0), 1.0, 0)
    chk("GC3 backlog alone > tau", shed_fraction(150000.0, 0.0, 1800000.0, 10000.0, 10.0), 0.0, 0)
    # GC4 shed through simulate (v1 G2 minute 0, all-awake, KV 0.2 GiB, tau 10)
    A = mk("A", 8.0, 0.2)
    reqs = [(1.0 + 0.5 * k, 0, 100, 1000) for k in range(30)]
    o, met = run([A], reqs, [0], 1, dict(all_awake=True), kvg=[0.2], card_gib=30.0, shed=10.0)
    chk("GC4 shed 3 at +inf, 27 at 9.20", vals(o[0]), [9.2] * 27 + [math.inf] * 3, 0.0051)
    chk("GC4 shed counted", met["stats"].get("shed_req", 0), 3)
    chk("GC4 conservation", met["week_n"], 30)
    # GD RESIZE pure
    chk("GD1 trailing peak excludes the epoch minute", trailing_peak({0: 60000.0, 1: 120000.0, 2: 600000.0}, 2, 2),
        2000.0, 1e-9)
    chk("GD2 under budget", resize_split(10.0, [1.0, 1.0], [6.0, 2.0]), [7.5, 2.5], 1e-9)
    chk("GD3 over budget", resize_split(10.0, [1.0, 1.0], [9.0, 5.0]), [6.333333333333333, 3.6666666666666665], 1e-9)
    # GE POOL through simulate
    A, Bm = mk("A", 8.0, 0.2), mk("B", 8.0, 0.2)
    reqs = [(1.0 + 0.5 * k, 0, 100, 1000) for k in range(30)]
    o, _ = run([A, Bm], reqs, [0, 0], 1, dict(all_awake=True), kvg=[0.2, 0.2], card_gib=30.0)
    chk("GE1 FIXED", vals(o[0]), [13.55] * 30, 0.0051)
    o, met = run([A, Bm], reqs, [0, 0], 1, dict(all_awake=True), kvg=[0.2, 0.2], card_gib=30.0, mech="pool")
    chk("GE1 POOL", vals(o[0]), [0.01] * 30, 0.0051)
    chk("GE1 POOL lent", met["stats"].get("grant_gib_min", 0.0), 0.2902555927757131 - 0.2, 1e-9)
    reqs2 = reqs + [(1.0 + 0.5 * k, 1, 100, 1000) for k in range(30)]
    o, _ = run([A, Bm], reqs2, [0, 0], 1, dict(all_awake=True), kvg=[0.2, 0.2], card_gib=19.26)
    chk("GE2 FIXED", vals(o[0]) + vals(o[1]), [17.42] * 60, 0.0051)
    o, met = run([A, Bm], reqs2, [0, 0], 1, dict(all_awake=True), kvg=[0.2, 0.2], card_gib=19.26, mech="pool")
    chk("GE2 POOL", vals(o[0]) + vals(o[1]), [7.94] * 60, 0.0051)
    chk("GE2 POOL grants = spare", met["stats"].get("grant_gib_min", 0.0), 0.1, 1e-9)
    o, _ = run([A, Bm], reqs, [0, 0], 1, dict(all_awake=True), kvg=[0.2, 0.2], card_gib=19.20, mech="pool")
    chk("GE3 POOL lends the IDLE resident's floor (built for mutant M2)", vals(o[0]), [0.01] * 30, 0.0051)
    # GG BALLOON adversarial (sleep stack, drop): A borrows the card's spare; a wake must not double-book it
    A, Bm = mk("A", 8.0, 1.0), mk("B", 8.0, 1.0)
    reqs = [(1.0 + 0.5 * k, 0, 100, 6600) for k in range(30)] + [(70.0, 1, 0, 10), (125.0, 1, 0, 10)]
    o, met = run([A, Bm], reqs, [0, 0], 1, dict(beta=1.1), kvg=[1.0, 1.0], card_gib=24.0)
    chk("GG FIXED B@70, B@125", vals(o[1]), [0.0, 0.47], 0.0051)
    o, met = run([A, Bm], reqs, [0, 0], 1, dict(beta=1.1), kvg=[1.0, 1.0], card_gib=24.0, mech="balloon")
    chk("GG BALLOON B@70 blocked, B@125 admitted", vals(o[1]), [0.47, math.inf], 0.0051)
    chk("GG BALLOON borrow = the card's spare", met["stats"].get("grant_gib_min", 0.0), 12.62, 1e-9)
    # GR RESIZE through simulate
    A, Bm = mk("A", 8.0, 1.0), mk("B", 8.0, 1.0)
    reqs = [(1.0 + 0.5 * k, 0, 1000, 2000) for k in range(30)] + [(125.0, 1, 0, 10), (200.0, 1, 0, 10)]
    rz = dict(H=2, restart_s=37.0, thr=0.25)
    o, met = run([A, Bm], reqs, [0, 0], 1, dict(all_awake=True), kvg=[1.0, 1.0], card_gib=30.0, mech="resize", rsz=rz)
    chk("GR B waits for the restart", vals(o[1]), [0.0, 32.0], 0.0051)
    chk("GR restarts both", (met["stats"].get("resplits", 0), met["stats"].get("restarts", 0)), (1, 2))
    o, _ = run([A, Bm], reqs, [0, 0], 1, dict(all_awake=True), kvg=[1.0, 1.0], card_gib=30.0)
    chk("GR FIXED", vals(o[1]), [0.0, 0.0], 0.0051)
    Pc30 = 30.0 - 2 * (8.56 + OVH)
    chk("GR boot split", resize_split(Pc30, [1.0, 1.0], [1.0, 1.0]), [5.619999999999999] * 2, 1e-9)
    # GP census population
    row = {"status": "MEASURED", "params_total_R2": 14768307200, "params_active_R2": 14768307200,
           "params_total": 14768307200, "params_active": 14768307200, "kv_sliding_layers": 0, "kv_bpt_all": 163840.0,
           "kv_bpt_global": 163840.0, "kv_bpt_sw": 0.0, "kv_window": None}
    mm, b, stt = model_from_census(("q", 8.0, 8.0, "fp8", 8.56, 10.38), row, 1000.0, "R2")
    chk("GP Qwen3-14B tb / w / bpt", [mm[1], mm[4], b], [14.7683072, 15.802088704, 163840.0], 1e-9)
    chk("GP status", stt, "MEASURED")
    sw = dict(row, kv_sliding_layers=40, kv_bpt_all=139264.0, kv_bpt_global=20480.0, kv_bpt_sw=118784.0, kv_window=1024)
    chk("GP sliding @4096", model_from_census(("g", 8.0, 8.0, "fp8", 8.56, 10.38), sw, 4096.0, "R2")[1], 50176.0, 1e-9)
    chk("GP sliding @512", model_from_census(("g", 8.0, 8.0, "fp8", 8.56, 10.38), sw, 512.0, "R2")[1], 139264.0, 1e-9)
    gs = model_from_census(("x", 8.0, 8.0, "fp8", 8.56, 10.38), {"status": "GUESSED"}, 1000.0, "R2")
    chk("GP GUESSED keeps v1's size + bpt_of", [gs[0][1], gs[1], gs[2]], [8.0, B8, "GUESSED"])
    # GS SPREAD
    us = [dict(cid=f"s{k}", f=f_) for k, f_ in enumerate([20.0, 15.0, 10.0, 8.0])]
    p0 = phys_with({"tpot_sat": 0.065})
    chk("GS homes", spread_place(us, 2, p0, [(0.0, 0.0)] * 4, 30.0), [0, 1, 1, 0])
    raises("GS infeasible", lambda: spread_place([dict(cid=f"t{k}", f=20.0) for k in range(3)], 2, p0,
                                                 [(0.0, 0.0)] * 3, 30.0), "SPREAD infeasible")
    # GQ decision rules
    ctrl = [0.80, 0.81, 0.82]
    chk("GQ1 LIFTS", q1_verdict(ctrl, [0.83, 0.85, 0.84]), "LIFTS")
    chk("GQ1 UNATTRIBUTABLE", q1_verdict(ctrl, [0.82, 0.86, 0.85]), "UNATTRIBUTABLE")
    chk("GQ1 LOWERS", q1_verdict(ctrl, [0.70, 0.75, 0.79]), "LOWERS")
    chk("GQ1 infeasible at top", q1_verdict(ctrl, [None, None, None]), "INFEASIBLE-AT-TOP")
    grid = {4: {0: 0.5, 1: 0.5, 2: 0.5}, 8: {0: 0.81, 1: 0.79, 2: 0.85}, 12: {0: 0.82, 1: 0.83, 2: 0.84},
            16: {0: 0.9, 1: 0.9, 2: 0.9}}
    chk("GQ2 K_match", k_match(grid, [0, 1, 2], 0.80, 16, 4), (12, [8, 12]))
    chk("GQ2 never", k_match({4: {0: 0.1}, 8: {0: 0.1}}, [0], 0.8, 8, 4), (12, [12, 12]))
    chk("GQ2 infeasible K fails", k_match({4: {0: 0.9}, 8: {0: None}}, [0], 0.8, 8, 4), (12, [12, 12]))
    # QA r1 guards: a K list is required; enum-valued params are validated at parse time
    raises("QA k_list2 without --ks/--kgrid", lambda: k_list2("AWAKE", phys, None, None), "needs --ks or --kgrid")
    raises("QA phys_with bad enum", lambda: phys_with({"sizes": "Census"}), "sizes must be one of")
    # GN NO-CARD: a model with unit -1 scores +inf and is conserved
    A = mk("A", 8.0, 1.0)
    o, met = run([A], [(1.0, 0, 100, 10), (2.0, 0, 100, 10)], [0], 1, dict(all_awake=True), kvg=[1.0], card_gib=30.0,
                 unit_of=[-1])
    chk("GN NO-CARD +inf", vals(o[0]), [math.inf, math.inf])
    chk("GN counted", met["stats"].get("no_card_req", 0), 2)
    # GF equivalence with v1 (every v2 feature off): random traces, KV off and on
    rng = np.random.default_rng(27404)
    ev = 0
    m8 = [mk(f"e{k}", 8.0) for k in range(4)]
    for trial in range(4):
        reqs = [(float(rng.uniform(0, 600)), int(rng.integers(0, 4)), float(rng.integers(0, 3000)),
                 float(rng.integers(0, 400))) for _ in range(60)]
        tr8 = T.Trace.explicit_cells(m8, reqs)
        units, unit_of = sing(m8)
        kvt = [float(x) * G30 / B8 for x in rng.uniform(0.05, 0.6, 4)]
        for nm, over, kv in (("L0", {}, None), ("L4", dict(sched="queue", fifo_cap=1), None),
                             ("L7", dict(zero_demote=True), None), ("A1-kv", {}, kvt),
                             ("L4-kv", dict(sched="queue", fifo_cap=1), kvt), ("awake-kv", dict(all_awake=True), kvt)):
            cf = F.arm_cfg("A0"); cf.update(over)
            cg = 30.0 if over.get("all_awake") else 15.0
            o1, m1 = F.simulate(tr8, units, unit_of, ("card", [0, 0, 1, 1]), 2, cf, 0, score_from=0, min0=0,
                                card_gib=cg, phys=phys, kvtok=kv)
            o2, m2 = simulate(tr8, units, unit_of, ("card", [0, 0, 1, 1]), 2, cf, 0, score_from=0, min0=0,
                              card_gib=cg, phys=phys, kvtok=kv)
            chk(f"GF {nm} trial {trial} Counters", o1 == o2, True)
            chk(f"GF {nm} trial {trial} metrics", m1 == m2, True)
            ev += m1["stats"].get("evictions", 0)
    chk("GF the equivalence traces exercise evictions", ev > 0, True)
    U8 = sing([mk("Q", 8.0), mk("R", 14.0)])[0]
    a1 = F.unit_kv(U8, "size", True, T.CARD_GIB, np.array([2000.0, 9000.0]), np.array([30000.0, 500.0]))
    a2 = unit_kv2(U8, "size", T.CARD_GIB, [F.bpt_of(8.0, True), F.bpt_of(14.0, True)], np.array([2000.0, 9000.0]),
                  np.array([30000.0, 500.0]))
    chk("GF unit_kv2 == v1 unit_kv", (list(a1[0]), list(a1[1])) == (list(a2[0]), list(a2[1])), True)
    for f_ in fails:
        print(f_)
    if fails:
        print(f"SELF-TEST FAIL {n - len(fails)}/{n}")
        return 1
    print(f"SELF-TEST PASS {n}/{n}")
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-i1", help="DIR of v1's committed grid rows (*.jsonl); read-only")
    ap.add_argument("--check-i5", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize")
    ap.add_argument("--arm")
    ap.add_argument("--ks")
    ap.add_argument("--kgrid", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    root = a.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        if a.self_test:
            return self_test()
        over = {}
        for kv in a.set:
            if "=" not in kv:
                raise ValueError(f"--set wants k=v, got {kv}")
            k, v = kv.split("=", 1)
            over[k] = v
        phys = phys_with(over)
        if a.check_i1:
            if not os.path.isdir(a.check_i1):
                raise ValueError(f"not a directory: {a.check_i1}")
            r = check_i1(a.check_i1, root, a.cache)
            print(json.dumps(r))
            return 0 if r["I1v2_pass"] else 1
        if a.check_i5:
            r = check_i5(root, a.cache)
            print(json.dumps(r))
            return 0 if r["I5_pass"] else 1
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"not a directory: {a.summarize}")
            print(json.dumps(summarize(a.summarize, root), default=str))
            return 0
        if a.run:
            if not a.arm or not a.out:
                raise ValueError("--run needs --arm and --out")
            cfg, _ = arm_cfg2(a.arm)
            if not a.kgrid and not a.ks and cfg["placement"] != "COFIT":
                raise ValueError("--run needs --ks or --kgrid")
            seeds = [int(x) for x in a.seeds.split(",")]
            ks = k_list2(a.arm, phys, root, a.cache, [int(x) for x in a.ks.split(",")] if a.ks else None, a.kgrid)
            with open(a.out, "a") as fh:
                for K in ks:
                    for seed in seeds:
                        row = run_cell2(a.arm, K, seed, phys, root, a.cache, a.tag)
                        fh.write(json.dumps(row, default=str) + "\n")
                        fh.flush()
            return 0
        ap.print_help()
        return 2
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
