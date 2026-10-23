#!/usr/bin/env python3
"""fabric_sim.py — the gpu-bunk FABRIC on the real tail trace. Every measured lever (co-activity homes,
no-drop wake queue, per-card sleeping copies, dedup-as-adapters, zero-copy demote, int8 KV, KV sizing) on top of two
CORRECTIONS that lacked: per-engine KV admission (the trace's long requests) and honest host RAM
(power-of-two pinning + per-process overhead). Cards for the 375-model tail at p95 TTFT <= 2 s / 10 s, held-out week.

Prereg (frozen before this file existed): (private report). On any divergence
the prereg wins. Imports (tailpool_sim), (tailpool_v2) and (wakestorm_sim)
UNCHANGED. `simulate()` here is a FORK of wakestorm's (it has no hook for a KV term or copy cards);
with every new feature off it equals the original byte for byte (self-test G8, instrument check I1).

Output contract:
  --self-test            exit 0 + "SELF-TEST PASS n/n" (hand goldens G1-G13 from the prereg); exit 1 + FAIL lines otherwise.
  --calibrate --root R   read-only: prereg §3 TPOT_sat fit on ch15 one JSON object on stdout; exit 0.
  --check --root R --cache F
                         read-only: instrument checks I1 (A0 == wakestorm poisson_L0 rows) + I4 (dedicated KV-off ==
                         tailpool_v2); one JSON object; exit 0 if both PASS, exit 1 otherwise.
  --floors --root R --cache F
                         read-only: compute + memory floors (prereg §2.10); one JSON object on stdout.
  --run --arm A [--ks 8,16 | --kgrid] [--seeds 0,1,2] [--set k=v ...] [--tag T] --out F
                         simulates and APPENDS one JSON line per (K, seed) to F (WRITES F); exit 0.
  --summarize DIR        read-only: prereg §6 decision rules over DIR/*.jsonl; one JSON object on stdout.
  --costs DIR --floors-json F
                         read-only: D3 $/h and $/M tok at each arm's min-K (frozen offers CSV); one JSON object on stdout.
  Bad input (sha mismatch, unknown arm/param, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed while building this or its parents):
  1. **Float order is load-bearing for byte-equivalence** — wakestorm keys TTFT at 0.01 s; the KV term is appended LAST
     (`... + pre + kvw`, kvw = 0.0 when off) so the A0 rows reproduce exactly (pitfall 2, same edge).
  2. **A seed of the trace is not the population** — the tail set, homes, copies, KV floors and sizing are TRAIN-only; the
     wakestorm dedup representative used 14-day work (a held-out leak) and is NOT reused (`work_train`).
  3. **`ssh host 'write && … &'` backgrounds the whole chain** — launch runners in separate calls on a CPU server.
"""
import argparse, collections, heapq, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tailpool_sim as T   # noqa: E402  (not edited)
import tailpool_v2 as V    # noqa: E402  (not edited)
import wakestorm_sim as W  # noqa: E402  (not edited)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

G30 = float(2 ** 30)
GIB = V.GIB
HELD = T.HELDOUT_MIN
OVH = T.OVERHEAD_GIB              # 0.82
INF_T = W.INF_T
KV_MIN_GIB = 0.25                 # prereg §2.3
FLOOR_Q = 0.99                    # prereg §2.3 / P11
SIZE_Q = 0.95                     # prereg §2.3 / P12
RHO_DESIGN = 0.5
KV_UTIL = 0.8
COFIT_HEAD = 0.90                 # headroom
ADAPTER_SLOTS = 16                # probe16
W_GIVEUP_S = 600.0                # amendment 2: a queued wake abandoned after 600 s (its requests score +inf)
EQ_KV = {"small": 3.23, "big": 2.88}   # kv-parity: <= 5 B engines 3.23 GiB, 8 B 2.88 GiB
K_GRID_5090 = tuple(range(8, 385, 8))
K_GRID_BIG = tuple(range(4, 193, 4))
SEEDS = (0, 1, 2)
K_REF = 128
CAL_GRID = tuple(round(0.02 + 0.01 * k, 2) for k in range(14))   # 0.02 .. 0.15
CH15 = "data/gpu_calibration"
CH15_SHA = {"inputs/sched_s1.jsonl": "ee3ae3f437d4bd0d2c10b2b706ca85e878177f98c8427e264c7dd80c07220732",
            "inputs/models.json": "f6017899ce3cbbfd46fb655b97cbd81cab348ccc3ed528471706c381f77559e8",
            "run_h1/score_h1.json": "c69cdf73077149dc190fbbe3a0e45f0be482809485bd948b7efefac761df5ac1",
            "run_h2/score_h2.json": "95b8177f79f2916b6584d702a36bede3e1d2ed800948be70200630c8bde2f3d9"}
L0_ROWS = "data/wakestorm/poisson_L0.jsonl"
L0_SHA = "94e85a1c7077731a42b67e4d211a53280034ce6f4e4f0f2f5ef03f9995f818a3"
OFFERS_SHA = "2b9464d4113919054d034313ccc363c941b02341969ec27dc5438e5edccf970a"

# prereg §5 point values (the physics); every key is overridable with --set k=v and recorded in each row
PHYS0 = dict(tpot_q=0.0065, tpot_sat=None, kappa=0.5, cref_mult=1.0, r_pin=1.27, o_proc=5.35, s=1.0, protect=60.0,
             h2d=22.6, d2h=23.8, h_pin=78.2, h_tot=135.0, floor_q=FLOOR_Q, size_q=SIZE_Q, copy_frac=0.25, scen="P8",
             arrivals="POISSON", big_card_gib=88.0, big_mult=1.1, big_usd=1.35, kv_cal_label="")
PHYS_FLOAT = {"tpot_q", "tpot_sat", "kappa", "cref_mult", "r_pin", "o_proc", "s", "protect", "h2d", "d2h", "h_pin",
              "h_tot", "floor_q", "size_q", "copy_frac", "big_card_gib", "big_mult", "big_usd"}
PHYS_STR = {"scen", "arrivals", "kv_cal_label"}

# arm flags on top of wakestorm's BASE (prereg §4). kv: None (off, A0) | fixed1 | floor | size | eq
LEV = dict(kv=None, ram=False, copies=False, int8=False, all_awake=False, card="5090")
ALL = dict(kv="size", ram=True, placement="COACT", sched="queue", fifo_cap=1, copies=True, units="dup",
           zero_demote=True, int8=True)
ARMS = {
    "A0": {},
    "A1": dict(kv="floor"),
    "A1_FIXED": dict(kv="fixed1"),
    "A2": dict(kv="floor", ram=True),
    "C1": dict(kv="size", ram=True),
    "C2": dict(kv="size", ram=True, placement="COACT"),
    "C3": dict(kv="size", ram=True, placement="COACT", sched="queue", fifo_cap=1),
    "C4": dict(kv="size", ram=True, placement="COACT", sched="queue", fifo_cap=1, copies=True),
    "C5": dict(kv="size", ram=True, placement="COACT", sched="queue", fifo_cap=1, copies=True, units="dup"),
    "C6": dict(kv="size", ram=True, placement="COACT", sched="queue", fifo_cap=1, copies=True, units="dup",
               zero_demote=True),
    "ALL": dict(ALL),
    "LOO_C1": dict(ALL, kv="floor"),
    "LOO_C2": dict(ALL, placement="BLIND"),
    "LOO_C3": dict(ALL, sched="drop", fifo_cap=None),
    "LOO_C4": dict(ALL, copies=False),
    "LOO_C5": dict(ALL, units="none"),
    "LOO_C6": dict(ALL, zero_demote=False),
    "LOO_C7": dict(ALL, int8=False),
    "MEASURED_ONLY": dict(kv="floor", ram=True, placement="COACT", sched="queue", fifo_cap=1, copies=True, units="dup"),
    "ALL_LIN": dict(ALL, units="lin"),
    "COFIT_5090": dict(kv="size", int8=True, ram=True, placement="COFIT", all_awake=True),
    "BIG_AWAKE_EQ": dict(kv="eq", ram=True, placement="COFIT", all_awake=True, card="BIG"),
    "BIG_AWAKE_KV": dict(kv="size", int8=True, ram=True, placement="COFIT", all_awake=True, card="BIG"),
    "BIG_ALL": dict(ALL, card="BIG"),
}
CHAIN = ("A0", "A1", "A2", "C1", "C2", "C3", "C4", "C5", "C6", "ALL")
CHAIN_LEVER = {"C1": "KVSIZE", "C2": "COACT", "C3": "QUEUE", "C4": "COPIES", "C5": "DEDUP", "C6": "ZC", "ALL": "INT8"}
LOO = {"KVSIZE": "LOO_C1", "COACT": "LOO_C2", "QUEUE": "LOO_C3", "COPIES": "LOO_C4", "DEDUP": "LOO_C5", "ZC": "LOO_C6",
       "INT8": "LOO_C7"}


def arm_cfg(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    c = dict(W.BASE)
    c.update(LEV)
    c.update(ARMS[arm])
    return c


def phys_with(over=None):
    p = dict(PHYS0)
    for k, v in (over or {}).items():
        if k not in PHYS0:
            raise ValueError(f"unknown param {k}")
        p[k] = v if k in PHYS_STR else float(v)
    return p


# ------------------------------------------------------------------------------------------------ KV physics (§2.2)
def bpt_of(tb, int8=False):
    """KV bytes per token (bf16) by size class (prereg §5, ASSUMED mapping of size -> Qwen3 config); int8 halves it."""
    if tb <= 2.5:
        b = 114688.0
    elif tb <= 9.5:
        b = 147456.0
    elif tb <= 16.0:
        b = 163840.0
    else:
        b = 262144.0
    return b / 2.0 if int8 else b


def rho_cap(phys):
    if not phys.get("tpot_sat"):
        raise ValueError("tpot_sat is not set (run --calibrate first, pass --set tpot_sat=X)")
    return max(0.0, 1.0 - phys["tpot_q"] / phys["tpot_sat"])


def tpot_of(rho, phys):
    """prereg §2.2: TPOT_c = TPOT_q / (1 - min(rho, rho_cap))."""
    return phys["tpot_q"] / (1.0 - min(rho, rho_cap(phys)))


def kv_step(bk, D, kvtok):
    """One minute of an engine's KV queue (token-seconds). -> (wait s, new backlog)."""
    cap = 60.0 * kvtok
    return (bk + 0.5 * max(0.0, D - cap)) / kvtok, max(0.0, bk + D - cap)


def kv_drain(bk, kvtok, idle_min):
    return max(0.0, bk - 60.0 * kvtok * idle_min) if idle_min > 0 else bk


def spread(fut, mi, x, R):
    """Amendment 1: x token-occupancy held for R s from the start of minute mi. Returns minute mi's token-seconds and adds
    the spill to fut[mi + k] (k >= 1): minute k carries x * clip(R - 60k, 0, 60)."""
    first = x * min(R, 60.0)
    rem = R - 60.0
    k = 1
    while rem > 0.0:
        fut[mi + k] = fut.get(mi + k, 0.0) + x * min(rem, 60.0)
        rem -= 60.0
        k += 1
    return first


def kv_catch_up(bk, fut, kvtok, last_min, mi):
    """Bring an engine's KV backlog from its last active minute to minute mi: each carried-in spill minute is served
    against one minute of capacity; pure idle minutes drain."""
    if last_min is None:
        return bk
    cap = 60.0 * kvtok
    tp = last_min
    for k in sorted(x for x in fut if x < mi):
        bk = kv_drain(bk, kvtok, k - tp - 1)
        bk = max(0.0, bk + fut.pop(k) - cap)
        tp = k
    return kv_drain(bk, kvtok, mi - tp - 1)


# ------------------------------------------------------------------------------------------------ simulator (fork)
def simulate(trace, units, unit_of, plc, K, cfg, seed, burst_w=None, score_from=HELD, min0=T.MIN0,
             card_gib=T.CARD_GIB, s=None, phys=None, kvtok=None, copies=None, C=C_REF):
    """FORK of wakestorm_sim.simulate (kind='card' only; no shells/multipath/storm windows). New, all OFF by
    default: kvtok[u] (KV admission §2.2 + unservable + drain protection), copies[u] (card list, primary first, §2.6),
    cfg['all_awake'] (COFIT boot). Returns (out per-model Counter of TTFT keys, metrics)."""
    phys = phys or PHYS0
    s = phys["s"] if s is None else s
    v2 = cfg["physics"] == "v2"
    protect = cfg["protect"]
    queue = cfg["sched"] == "queue"
    phi = cfg["phi"]
    beta = cfg["beta"]
    kv_on = kvtok is not None
    awake_boot = bool(cfg.get("all_awake"))
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
    rcap = rho_cap(phys) if kv_on else 0.0
    out = [collections.Counter() for _ in range(M)]
    st = collections.Counter()
    kvw_hist = collections.Counter()
    heap = []
    hseq = [0]
    deferred = []
    svc = []
    week = collections.Counter()

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
        if gpu[c] + need <= card_gib + 1e-9:
            return []
        adv(c, t)
        cand = sorted((last_arr[r], r) for r in card_units[c]
                      if ready_by(r, t) and (force or not last_arr[r] > t - protect) and not last_busy[r] > t)
        free = card_gib - gpu[c]
        vict = []
        for _, r in cand:
            if free >= need - 1e-9:
                break
            vict.append(r); free += units[r]["f"] - s
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
            st["evictions"] += 1
        busy = any(r != u for r in card_units[c])
        awake[u] = True; card_units[c].add(u); gpu[c] += U["f"] - s; cur[u] = c
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
        if nxt >= INF_T:
            nxt = t + 0.05
        return min(nxt, e.t_create + W_GIVEUP_S)

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
            if te >= e.t_create + W_GIVEUP_S - 1e-9:      # amendment 2: card-mates never drained in time
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
            if kv_on and (trace.it[i] + trace.ot[i]) / n > kvtok[u]:
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
                                float(last[i])))
        q = [0.0] * K
        rho = [0.0] * K
        for c in range(K):
            if kv_on:
                rho[c] = min(rcap, (backlog[c] + Wc[c]) / cap_min)
            q[c] = (backlog[c] + 0.5 * max(0.0, Wc[c] - cap_min)) / C
            backlog[c] = max(0.0, backlog[c] + Wc[c] - cap_min)
        kvw = {}
        if kv_on and kvcells:
            byu = collections.defaultdict(list)
            for u, c, n, occ, pre, outr, lo in kvcells:
                byu[u].append((n * occ, pre + outr * (phys["tpot_q"] / (1.0 - rho[c])), lo))
            for u, items in byu.items():
                bk = kv_catch_up(bk_kv[u], fut_kv[u], kvtok[u], kv_last_min[u], mi)
                kv_last_min[u] = mi
                D = fut_kv[u].pop(mi, 0.0)
                for x, R, lo in items:
                    D += spread(fut_kv[u], mi, x, R)
                kvw[u], bk_kv[u] = kv_step(bk, D, kvtok[u])
                for x, R, lo in items:
                    lb = base + lo + kvw[u] + R
                    if lb > last_busy[u]:
                        last_busy[u] = lb
        if mi >= score_from:
            for m, c, n, pre, e, extra, i, u in pend:
                if c < 0:
                    out[m][T.INF_KEY] += n
                    week[T.INF_KEY] += n
                    continue
                kw = kvw.get(u, 0.0)
                if kv_on:
                    kvw_hist[T._key(kw)] += n
                if e is None:
                    key = T._key(q[c] + extra + pre + kw)
                    out[m][key] += n
                    week[key] += n
                else:
                    ts = T.cell_times(trace, i, first, last, seed) + base
                    deferred.append((m, ts, q[c], extra, pre, e, kw))
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


# ------------------------------------------------------------------------------------------------ dedicated (§2.9)
def dedicated(trace, kvtok_ded=None, phys=None, burst_w=None, score_from_min=HELD, C=C_REF):
    """One always-awake card per model. kvtok_ded None = dedicated (I4); else §2.2 KV on its own card."""
    phys = phys or PHYS0
    M = len(trace.models)
    out = [collections.Counter() for _ in range(M)]
    backlog = [0.0] * M
    bk = [0.0] * M
    fut = [dict() for _ in range(M)]
    kv_last = [None] * M
    last_min = [None] * M
    cap_min = C * 60.0
    rcap = rho_cap(phys) if kvtok_ded is not None else 0.0
    for i in range(len(trace.n)):
        m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        ab = trace.models[m][2]
        if kvtok_ded is not None and (trace.it[i] + trace.ot[i]) / n > kvtok_ded[m]:
            if mi >= score_from_min:
                out[m][T.INF_KEY] += n
            continue
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        if last_min[m] is not None:
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        last_min[m] = mi
        kw = 0.0
        if kvtok_ded is not None:
            bk[m] = kv_catch_up(bk[m], fut[m], kvtok_ded[m], kv_last[m], mi)
            kv_last[m] = mi
            rho = min(rcap, (backlog[m] + Wm) / cap_min)
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C
            R = pre + (trace.ot[i] / n) * tpot_of(rho, phys)
            occ = trace.it[i] / n + phys["kappa"] * trace.ot[i] / n
            D = fut[m].pop(mi, 0.0) + spread(fut[m], mi, n * occ, R)
            kw, bk[m] = kv_step(bk[m], D, kvtok_ded[m])
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / C + V._burst_extra(Wm, burst_w)
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        if mi >= score_from_min:
            out[m][T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C + kw)] += n
    return out


# ------------------------------------------------------------------------------------------------ KV per unit (§2.3)
def _train_cells(trace, unit_of, held=HELD):
    k = trace.minute < held
    return (np.asarray(unit_of)[trace.mid[k]], trace.minute[k], trace.n[k].astype(float), trace.it[k], trace.ot[k],
            trace.mid[k])


def kv_floor_tok(trace, unit_of, n_units, q=FLOOR_Q, held=HELD):
    """TRAIN request-weighted quantile (nearest rank) of cell-mean (it+ot)/n per unit; q >= 1 -> max."""
    u, _, n, it, ot, _ = _train_cells(trace, unit_of, held)
    L = (it + ot) / n
    out = np.zeros(n_units)
    o = np.lexsort((L, u))
    u, L, n = u[o], L[o], n[o]
    bounds = np.flatnonzero(np.diff(u)) + 1
    for a, b in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(u)]))):
        if a >= b:
            continue
        if q >= 1.0:
            out[u[a]] = L[b - 1]
            continue
        cw = np.cumsum(n[a:b])
        idx = int(np.searchsorted(cw, q * cw[-1] - 1e-9))
        out[u[a]] = L[a + min(idx, b - a - 1)]
    return out


def kv_conc_tok(trace, unit_of, n_units, models, phys, q=SIZE_Q, held=HELD, C=C_REF):
    """q-quantile (nearest rank) over a unit's active TRAIN minutes of D(t)/60/0.8 at TPOT_design, no queue."""
    u, mi, n, it, ot, mid = _train_cells(trace, unit_of, held)
    ab = np.array([m[2] for m in models])[mid]
    td = phys["tpot_q"] / (1.0 - RHO_DESIGN)
    pre = (it / n) * (ab / ACTIVE_REF_B) / C
    R = pre + (ot / n) * td
    x = n * (it / n + phys["kappa"] * ot / n)
    key = u.astype(np.int64) * 10 ** 7 + mi.astype(np.int64)
    keys = [key]
    vals = [x * np.minimum(R, 60.0)]          # amendment 1: minute 0's portion
    sk, sv = [], []
    for j in np.flatnonzero(R > 60.0):          # the spill of residences longer than a minute
        fut = {}
        spread(fut, 0, float(x[j]), float(R[j]))
        for k, v in fut.items():
            sk.append(int(key[j]) + k); sv.append(v)
    if sk:
        keys.append(np.array(sk, np.int64)); vals.append(np.array(sv))
    uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
    Dm = np.bincount(inv, weights=np.concatenate(vals))
    uu = (uk // 10 ** 7).astype(np.int64)
    out = np.zeros(n_units)
    o = np.lexsort((Dm, uu))
    uu, Dm = uu[o], Dm[o]
    bounds = np.flatnonzero(np.diff(uu)) + 1
    for a, b in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(uu)]))):
        if a >= b:
            continue
        k = max(0, min(b - a - 1, math.ceil(q * (b - a)) - 1))
        out[uu[a]] = Dm[a + k] / 60.0 / KV_UTIL
    return out


def unit_kv(units, mode, int8, card_gib, floor_tok=None, conc_tok=None):
    """-> (kv_gib list, kvtok list). Modes per prereg §2.3; KV clamped to [0.25, card_gib - w - 0.82]."""
    kv, tok = [], []
    for k, U in enumerate(units):
        b = bpt_of(U["tb"], int8)
        hi = card_gib - U["w"] - OVH
        if hi <= 0:
            raise ValueError(f"unit {U['cid']} weights {U['w']:.2f} GiB leave no KV room on a {card_gib} GiB card")
        if mode == "fixed1":
            g = 1.0
        elif mode == "eq":
            g = EQ_KV["small"] if U["tb"] <= 5.0 else EQ_KV["big"]
        elif mode == "floor":
            g = max(min(floor_tok[k] * b / G30, hi), min(KV_MIN_GIB, hi))
        elif mode == "size":
            g = max(min(max(floor_tok[k], conc_tok[k]) * b / G30, hi), min(KV_MIN_GIB, hi))
        else:
            raise ValueError(f"unknown kv mode {mode}")
        g = min(g, hi)
        kv.append(g)
        tok.append(g * G30 / b)
    return kv, tok


# ------------------------------------------------------------------------------------------------ placement (§2.4-2.6)
def host_need(units, phys, honest, all_awake):
    """per unit (pinned GiB, total GiB). honest off -> (w, w) (sum w <= H)."""
    out = []
    for U in units:
        if not honest:
            out.append((U["w"], U["w"]))
        elif all_awake:
            out.append((0.0, phys["o_proc"]))
        else:
            p = phys["r_pin"] * U["w"]
            out.append((p, p + phys["o_proc"]))
    return out


def blind_place(units, K, phys, hneed, honest, card_gib, s):
    """ BLIND (V.assign_homes) + the §2.4 total-RAM check when honest."""
    NU = len(units)
    H, Ht = phys["h_pin"], phys["h_tot"]
    ram = [H] * K; tot = [Ht] * K; n_h = [0] * K; fmax = [0.0] * K
    homes = [-1] * NU
    for u in sorted(range(NU), key=lambda k: units[k]["cid"]):
        p, t_ = hneed[u]
        f = units[u]["f"]
        if p > H + 1e-9 or (honest and t_ > Ht + 1e-9):
            raise ValueError(f"host RAM cap exceeded: unit {units[u]['cid']} weights {p:.2f} GiB pinned")

        def ram_ok(c):
            return ram[c] >= p - 1e-9 and (not honest or t_ <= tot[c] + 1e-9)

        feas = [c for c in range(K) if ram_ok(c) and n_h[c] * s + max(fmax[c], f) <= card_gib + 1e-9]
        if not feas:
            if any(ram_ok(c) for c in range(K)):
                raise ValueError(f"sleeper GPU cap exceeded at K={K}: unit {units[u]['cid']} fits no card's GPU budget")
            raise ValueError(f"host RAM cap exceeded at K={K}: unit {units[u]['cid']} fits no card (pinned or total)")
        c = max(feas, key=lambda j: (ram[j], -j))
        homes[u] = c
        ram[c] -= p; tot[c] -= t_; n_h[c] += 1; fmax[c] = max(fmax[c], f)
    return homes


def coact_place(units, K, phys, hneed, honest, live, card_gib, s):
    """wakestorm L1 COACT (W.coact_assign, mode 'card') + the §2.4 checks."""
    NU = len(units)
    H, Ht = phys["h_pin"], phys["h_tot"]
    ram = np.full(K, float(H)); tot = np.full(K, float(Ht))
    n_h = np.zeros(K, np.int64); fmax = np.zeros(K)
    L = np.zeros((K, live.shape[1]), np.int16)
    homes = [-1] * NU
    cnt = live.sum(axis=1)
    for u in sorted(range(NU), key=lambda m: (-int(cnt[m]), units[m]["cid"])):
        p, t_ = hneed[u]
        f = units[u]["f"]
        if p > float(H) + 1e-9 or (honest and t_ > Ht + 1e-9):
            raise ValueError(f"host RAM cap exceeded: unit {units[u]['cid']} weights {p:.2f} GiB pinned")
        feas = [c for c in range(K) if ram[c] >= p - 1e-9 and (not honest or tot[c] >= t_ - 1e-9)
                and n_h[c] * s + max(fmax[c], f) <= card_gib + 1e-9]
        if not feas:
            raise ValueError(f"COACT infeasible at K={K}: unit {units[u]['cid']} fits no card")
        lm = live[u].astype(np.int16)
        best = None
        for c in feas:
            so = W.card_slots(n_h[c], fmax[c], card_gib, s)
            sn = W.card_slots(n_h[c] + 1, max(fmax[c], f), card_gib, s)
            Lc = L[c]
            over_old = int(np.maximum(0, Lc.astype(np.int32) - so).sum()) if n_h[c] else 0
            over_new = int(np.maximum(0, (Lc + lm).astype(np.int32) - sn).sum())
            key = (over_new - over_old, -ram[c], c)
            if best is None or key < best[0]:
                best = (key, c)
        c = best[1]
        homes[u] = c
        ram[c] -= p; tot[c] -= t_; n_h[c] += 1; fmax[c] = max(fmax[c], f); L[c] += lm
    return homes


def copies_place(units, homes, K, phys, hneed, honest, live, card_gib, s, box_g, frac):
    """§2.6: the top `frac` of units by TRAIN live minutes get ONE extra sleeping copy inside the primary's box."""
    NU = len(units)
    H, Ht = phys["h_pin"], phys["h_tot"]
    ram = np.full(K, float(H)); tot = np.full(K, float(Ht))
    n_h = np.zeros(K, np.int64); fmax = np.zeros(K)
    L = np.zeros((K, live.shape[1]), np.int16)
    for u in range(NU):
        c = homes[u]
        ram[c] -= hneed[u][0]; tot[c] -= hneed[u][1]; n_h[c] += 1; fmax[c] = max(fmax[c], units[u]["f"])
        L[c] += live[u].astype(np.int16)
    cnt = live.sum(axis=1)
    nsel = int(math.ceil(frac * NU - 1e-9))
    sel = sorted(range(NU), key=lambda m: (-int(cnt[m]), units[m]["cid"]))[:nsel]
    out = [[homes[u]] for u in range(NU)]
    made = 0
    for u in sel:
        p, t_ = hneed[u]
        f = units[u]["f"]
        b = homes[u] // box_g
        lm = live[u].astype(np.int16)
        best = None
        for c in range(b * box_g, min(K, b * box_g + box_g)):
            if c == homes[u]:
                continue
            if not (ram[c] >= p - 1e-9 and (not honest or tot[c] >= t_ - 1e-9)
                    and n_h[c] * s + max(fmax[c], f) <= card_gib + 1e-9):
                continue
            so = W.card_slots(n_h[c], fmax[c], card_gib, s)
            sn = W.card_slots(n_h[c] + 1, max(fmax[c], f), card_gib, s)
            over_old = int(np.maximum(0, L[c].astype(np.int32) - so).sum()) if n_h[c] else 0
            over_new = int(np.maximum(0, (L[c] + lm).astype(np.int32) - sn).sum())
            key = (over_new - over_old, -ram[c], c)
            if best is None or key < best[0]:
                best = (key, c)
        if best is None:
            continue
        c = best[1]
        out[u].append(c)
        made += 1
        ram[c] -= p; tot[c] -= t_; n_h[c] += 1; fmax[c] = max(fmax[c], f); L[c] += lm
    return out, {"copies_selected": nsel, "copies_made": made}


def cofit_place(units, phys, hneed, card_gib, honest):
    """§2.5 COFIT: first-fit decreasing on f (ties cid); a bin takes u if sum f <= 0.90 card and total RAM fits."""
    bins = []        # [sum_f, sum_total]
    homes = [-1] * len(units)
    for u in sorted(range(len(units)), key=lambda k: (-units[k]["f"], units[k]["cid"])):
        f = units[u]["f"]; t_ = hneed[u][1]
        # amendment 2: the 0.90 headroom is for co-residents; a unit above it opens a bin of its own (no co-resident can
        # join: sum f + any f > 0.90 card). Only a unit larger than the whole card raises.
        if f > card_gib + 1e-9:
            raise ValueError(f"COFIT: unit {units[u]['cid']} f {f:.2f} > card {card_gib}")
        for k, bn in enumerate(bins):
            if bn[0] + f <= COFIT_HEAD * card_gib + 1e-9 and (not honest or bn[1] + t_ <= phys["h_tot"] + 1e-9):
                bn[0] += f; bn[1] += t_; homes[u] = k
                break
        else:
            bins.append([f, t_]); homes[u] = len(bins) - 1
    return homes, len(bins)


def ram_need_per_gpu(units, cands, hneed, K):
    """max over cards of max(sum total, sum pinned / 0.6) (prereg §5 price rule)."""
    pin = [0.0] * K; tot = [0.0] * K
    for u, cs in enumerate(cands):
        for c in cs:
            pin[c] += hneed[u][0]; tot[c] += hneed[u][1]
    return max(max(t_, p / 0.6) for p, t_ in zip(pin, tot)) if K else 0.0


# ------------------------------------------------------------------------------------------------ setup + runs
_CACHE = {}


def check_adapter_slots(units):
    """§2.7: a merged engine carries at most ADAPTER_SLOTS members (probe16)."""
    for U in units:
        if len(U["members"]) > ADAPTER_SLOTS:
            raise ValueError(f"merged unit {U['cid']} has {len(U['members'])} > {ADAPTER_SLOTS} adapter slots")


def build_pop(scen, root, cache):
    key = ("pop", scen)
    if key not in _CACHE:
        tr, _, _, cnt = V.build(scen, root, cache)
        k = tr.minute < HELD
        wt = np.bincount(tr.mid[k], weights=(tr.it[k] + tr.ot[k]), minlength=len(tr.models))
        work_train = {tr.models[m][0]: float(wt[m]) for m in range(len(tr.models))}
        _CACHE[key] = (tr, cnt, work_train)
    return _CACHE[key]


def setup(arm, phys, root, cache):
    """-> dict with the arm's trace, units, unit_of, placement inputs, kv arrays (None for A0). Cached per (arm, phys)."""
    skey = ("setup", arm, json.dumps(phys, sort_keys=True, default=str))
    if skey not in _CACHE:
        _CACHE[skey] = _setup(arm, phys, root, cache)
    return _CACHE[skey]


def _setup(arm, phys, root, cache):
    cfg = arm_cfg(arm)
    tr, cnt, work_train = build_pop(phys["scen"], root, cache)
    big = cfg["card"] == "BIG"
    card_gib = phys["big_card_gib"] if big else T.CARD_GIB
    C = C_REF * (phys["big_mult"] if big else 1.0) * phys["cref_mult"]
    models = W.build_models(tr, cfg["prec"])
    census = W.load_census(root) if cfg["units"] != "none" else None
    units, unit_of = W.build_units(models, cfg["units"], census, work_train)
    check_adapter_slots(units)
    tr2 = T.Trace.__new__(T.Trace)
    tr2.minute, tr2.mid, tr2.n, tr2.it, tr2.ot, tr2.explicit = tr.minute, tr.mid, tr.n, tr.it, tr.ot, None
    tr2.models = models
    kvtok = None
    kv_gib = None
    if cfg["kv"] is not None:
        ft = kv_floor_tok(tr2, unit_of, len(units), phys["floor_q"])
        ct = kv_conc_tok(tr2, unit_of, len(units), models, phys, phys["size_q"], C=C) if cfg["kv"] == "size" else None
        kv_gib, kvtok = unit_kv(units, cfg["kv"], cfg["int8"], card_gib, ft, ct)
        for U, g in zip(units, kv_gib):
            U["f"] = U["w"] + OVH + g
    live = None
    if cfg["placement"] == "COACT" or cfg["copies"]:
        key = ("live", phys["scen"], cfg["units"], cfg["prec"])
        if key not in _CACHE:
            _CACHE[key] = W.train_live(tr2, unit_of, len(units))
        live = _CACHE[key]
    hneed = host_need(units, phys, cfg["ram"], cfg["all_awake"])
    return dict(cfg=cfg, tr=tr2, units=units, unit_of=unit_of, kvtok=kvtok, kv_gib=kv_gib, live=live, hneed=hneed,
                card_gib=card_gib, C=C, big=big, box_g=4 if big else W.BOX_G, cnt=cnt)


def place_arm(S, K, phys):
    cfg = S["cfg"]
    if cfg["placement"] == "COFIT":
        homes, nb = cofit_place(S["units"], phys, S["hneed"], S["card_gib"], cfg["ram"])
        if nb != K:
            raise ValueError(f"COFIT packs into {nb} cards, not {K}")
        return homes, [[h] for h in homes], {}
    if cfg["placement"] == "COACT":
        homes = coact_place(S["units"], K, phys, S["hneed"], cfg["ram"], S["live"], S["card_gib"], phys["s"])
    else:
        homes = blind_place(S["units"], K, phys, S["hneed"], cfg["ram"], S["card_gib"], phys["s"])
    if cfg["copies"]:
        cands, info = copies_place(S["units"], homes, K, phys, S["hneed"], cfg["ram"], S["live"], S["card_gib"],
                                   phys["s"], S["box_g"], phys["copy_frac"])
        return homes, cands, info
    return homes, [[h] for h in homes], {}


def dedicated_ref(arm, phys, root, cache):
    """A0 -> dedicated; every other arm -> the honest 5090 dedicated (§2.9) at the run's physics."""
    cfg = arm_cfg(arm)
    tr, _, _ = build_pop(phys["scen"], root, cache)
    bw = V.BURST_W if phys["arrivals"] == "BURST" else None
    if cfg["kv"] is None:
        key = ("ded0", phys["scen"], phys["arrivals"])
        if key not in _CACHE:
            _CACHE[key] = V.simulate_dedicated_v2(tr, bw)
        return _CACHE[key]
    key = ("dedkv", phys["scen"], phys["arrivals"], phys["tpot_q"], phys["tpot_sat"], phys["kappa"], phys["cref_mult"])
    if key not in _CACHE:
        kt = [(T.CARD_GIB - m[4] - OVH) * G30 / bpt_of(m[1]) for m in tr.models]
        _CACHE[key] = dedicated(tr, kt, phys, bw, C=C_REF * phys["cref_mult"])
    return _CACHE[key]


def run_cell(arm, K, seed, phys, root, cache, tag=""):
    S = setup(arm, phys, root, cache)
    cfg = S["cfg"]
    bw = V.BURST_W if phys["arrivals"] == "BURST" else None
    ded = dedicated_ref(arm, phys, root, cache)
    row = {"arm": arm, "K": K, "seed": seed, "tag": tag, "phys": phys, "cfg": cfg, "n_units": len(S["units"]),
           "n_models": len(S["tr"].models)}
    if S["kv_gib"] is not None:
        kg = sorted(S["kv_gib"])
        row["kv_gib"] = {"p50": kg[len(kg) // 2], "p90": kg[int(0.9 * (len(kg) - 1))], "max": kg[-1], "sum": sum(kg)}
    t0 = time.time()
    try:
        homes, cands, info = place_arm(S, K, phys)
    except ValueError as e:
        row.update(infeasible=str(e), elapsed_s=round(time.time() - t0, 1))
        return row
    row.update(info)
    row["ram_need_gib_per_gpu"] = round(ram_need_per_gpu(S["units"], cands, S["hneed"], K), 3)
    cfg_sim = dict(cfg)
    cfg_sim["h2d"], cfg_sim["d2h"], cfg_sim["protect"] = phys["h2d"], phys["d2h"], phys["protect"]
    out, met = simulate(S["tr"], S["units"], S["unit_of"], ("card", homes), K, cfg_sim, seed, bw,
                        card_gib=S["card_gib"], phys=phys, kvtok=S["kvtok"],
                        copies=cands if cfg["copies"] else None, C=S["C"])
    row.update(met)
    row["slo"] = {str(x): T.score(out, ded, x) for x in T.SLOS}
    row["heldout_req"] = int(sum(sum(c.values()) for c in out))
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def k_list(arm, phys, root, cache, ks=None, kgrid=False):
    cfg = arm_cfg(arm)
    if cfg["placement"] == "COFIT":
        S = setup(arm, phys, root, cache)
        _, nb = cofit_place(S["units"], phys, S["hneed"], S["card_gib"], cfg["ram"])
        return [nb]
    if kgrid:
        return list(K_GRID_BIG if cfg["card"] == "BIG" else K_GRID_5090)
    return list(ks)


# ------------------------------------------------------------------------------------------------ calibration (§3)
def calibrate(root):
    for rel, sha in CH15_SHA.items():
        got = T.sha256_file(os.path.join(root, CH15, rel))
        if got != sha:
            raise ValueError(f"{rel} sha mismatch: {got}")
    mj = json.load(open(os.path.join(root, CH15, "inputs/models.json")))
    cls = {m["name"]: m["class"] for m in mj["models"]}
    ab_of = {k: v["active_b"] for k, v in mj["classes"].items()}
    names = [f"m{k:02d}" for k in range(1, 11)]
    rows = [json.loads(l) for l in open(os.path.join(root, CH15, "inputs/sched_s1.jsonl")) if l.strip()]
    rows = [r for r in rows if r["model"] in names]
    meas = {}
    for h in ("h1", "h2"):
        d = json.load(open(os.path.join(root, CH15, f"run_{h}/score_{h}.json")))
        pm = d["blocks"]["S_KMEM"]["cells"]["1"]["per_model"]
        meas[h] = {m: (math.inf if pm[m]["p95"] in (None, float("inf")) or pm[m]["fail"] else pm[m]["p95"]) for m in names}
    res = []
    for ts in CAL_GRID:
        pred = cal_predict(rows, cls, ab_of, ts)
        agree = 0
        for h in ("h1", "h2"):
            for m in names:
                for bar in (2.0, 10.0):
                    agree += int((pred[m] <= bar) == (meas[h][m] <= bar))
        errs = []
        for m in names:
            fin = [meas[h][m] for h in ("h1", "h2") if meas[h][m] != math.inf]
            mm = sum(fin) / len(fin)
            errs.append(abs(math.log10(max(pred[m], 1e-3) / mm)))
        res.append({"tpot_sat": ts, "agree": agree, "med_log_err": statistics.median(errs),
                    "pred": {m: round(pred[m], 3) for m in names}})
    best = sorted(res, key=lambda r: (-r["agree"], r["med_log_err"], r["tpot_sat"]))[0]
    return {"grid": res, "chosen": best["tpot_sat"], "agree": best["agree"], "of": 40,
            "label": "" if best["agree"] >= 36 else "UNCALIBRATED", "measured": meas}


def cal_predict(rows, cls, ab_of, tpot_sat):
    """per model nearest-rank p95 of (kvw + pre) with TPOT fixed = tpot_sat, minute cells (prereg §3)."""
    cells = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: r["t"]):
        k = (r["model"], int(r["t"] // 60.0))
        c = cells.setdefault(k, [0, 0.0, 0.0])
        c[0] += 1; c[1] += r["in"]; c[2] += r["out"]
    per = collections.defaultdict(list)
    for m in sorted({k[0] for k in cells}):
        g = EQ_KV["big"] if cls[m] == "d8b" else EQ_KV["small"]
        kt = g * G30 / bpt_of(ab_of[cls[m]])
        bk = 0.0
        last = None
        fut = {}
        ab = ab_of[cls[m]]
        for mi in sorted(k[1] for k in cells if k[0] == m):
            n, si, so = cells[(m, mi)]
            bk = kv_catch_up(bk, fut, kt, last, mi)
            last = mi
            pre = (si / n) * (ab / ACTIVE_REF_B) / C_REF
            D = fut.pop(mi, 0.0) + spread(fut, mi, n * (si / n + PHYS0["kappa"] * so / n), pre + (so / n) * tpot_sat)
            kw, bk = kv_step(bk, D, kt)
            per[m] += [kw + pre] * n
    return {m: sorted(v)[max(0, math.ceil(0.95 * len(v)) - 1)] for m, v in per.items()}


# ------------------------------------------------------------------------------------------------ checks + floors
def check(root, cache):
    if T.sha256_file(os.path.join(root, L0_ROWS)) != L0_SHA:
        raise ValueError("poisson_L0.jsonl sha mismatch")
    with open(os.path.join(root, L0_ROWS)) as fh:
        ref = {(r["K"], r["seed"]): r for r in map(json.loads, fh)}
    phys = phys_with({})
    i1 = []
    for (K, seed), r in sorted(ref.items()):
        mine = run_cell("A0", K, seed, phys, root, cache)
        if "infeasible" in r:
            ok = "infeasible" in mine
        else:
            ok = ("infeasible" not in mine and mine["stats"] == r["stats"] and mine["slo"] == r["slo"]
                  and mine["heldout_req"] == r["heldout_req"])
        i1.append({"K": K, "seed": seed, "ok": ok})
    tr, _, _ = build_pop("P8", root, cache)
    a = dedicated(tr, None)
    b = V.simulate_dedicated_v2(tr, None)
    i4 = all(x == y for x, y in zip(a, b)) and len(a) == len(b)
    return {"I1": i1, "I1_pass": all(x["ok"] for x in i1), "I4_pass": bool(i4), "n_models": len(a)}


def floors(root, cache, phys):
    tr, cnt, _ = build_pop(phys["scen"], root, cache)
    k = tr.minute >= HELD
    ab = np.array([m[2] for m in tr.models])
    wk = (tr.it[k] + tr.ot[k]) * (ab[tr.mid[k]] / ACTIVE_REF_B)
    tot = np.bincount((tr.minute[k] - HELD).astype(np.int64), weights=wk, minlength=7 * 1440)
    cards = tot / (C_REF * phys["cref_mult"] * 60.0)
    out = {"compute_floor_heldout": {"mean": float(cards.mean()), "p95": float(np.percentile(cards, 95)),
                                     "p99": float(np.percentile(cards, 99)), "max": float(cards.max())},
           "cited_2680": "7-12", "heldout_tail_tokens": float((tr.it[k] + tr.ot[k]).sum()),
           "heldout_tail_req": int(tr.n[k].sum()), "pop": cnt}
    mem = {}
    for arm in ("A0", "A1", "C1", "ALL"):
        S = setup(arm, phys, root, cache)
        mem[arm] = {"sum_f_over_card": sum(U["f"] for U in S["units"]) / S["card_gib"],
                    "sum_w_over_card": sum(U["w"] for U in S["units"]) / S["card_gib"], "units": len(S["units"])}
    out["memory_floor"] = mem
    return out


# ------------------------------------------------------------------------------------------------ summarize (§6)
def _rows(d):
    """All rows of DIR/*.jsonl; a repeated (arm, tag, K, seed) keeps the LAST row read (QA r1, gemini-3.5-flash-lite:
    --run appends, so a manual re-run must not double-count a cell)."""
    keep = collections.OrderedDict()
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            with open(os.path.join(d, f)) as fh:
                for l in fh:
                    if l.strip():
                        r = json.loads(l)
                        keep[(r["arm"], r.get("tag", ""), r["K"], r["seed"])] = r
    return list(keep.values())


def _meets(r, slo, thr_m):
    if "infeasible" in r:
        return False
    sc = r["slo"][str(slo)]
    return sc["M_frac"] is not None and sc["R_ratio"] is not None and sc["M_frac"] >= thr_m and sc["R_ratio"] >= 0.99


def min_ks(rows, slo, top):
    """-> dict nominal/resolved/stable over all seeds + per-seed stable [min, max]; None -> top + step marker."""
    byk = collections.defaultdict(dict)
    for r in rows:
        byk[r["K"]][r["seed"]] = r
    ks = sorted(byk)
    step = ks[1] - ks[0] if len(ks) > 1 else 0
    over = (top + step) if step else top + 1
    seeds = sorted({r["seed"] for r in rows})

    def all_seeds(k, thr):
        return len(byk[k]) == len(seeds) and all(_meets(byk[k][s], slo, thr) for s in seeds)

    def first(pred):
        for k in ks:
            if pred(k):
                return k
        return None

    def stable(pred):
        best = None
        for k in reversed(ks):
            if pred(k):
                best = k
            else:
                break
        return best

    nom = first(lambda k: all_seeds(k, 0.95))
    res = first(lambda k: all_seeds(k, 0.96))
    stb = stable(lambda k: all_seeds(k, 0.96))
    per = []
    for s in seeds:
        v = stable(lambda k: s in byk[k] and _meets(byk[k][s], slo, 0.96))
        per.append(v if v is not None else over)
    return {"nominal": nom if nom is not None else over, "resolved": res if res is not None else over,
            "stable": stb if stb is not None else over, "per_seed_stable": [min(per), max(per)] if per else None,
            "served": stb is not None, "over_marker": over}


def m_at(rows, slo, K):
    vals = [r["slo"][str(slo)]["M_frac"] for r in rows if r["K"] == K and "infeasible" not in r]
    feas = [r for r in rows if r["K"] == K]
    if not feas:
        return None
    if len(vals) < len(feas):
        return {"min": None, "max": None, "infeasible": True}
    return {"min": min(vals), "max": max(vals)}


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def summarize(d):
    rows = _rows(d)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["arm"], r.get("tag", ""))].append(r)
    res = {}
    for (arm, tag), rs in groups.items():
        big = arm_cfg(arm)["card"] == "BIG"
        top = (K_GRID_BIG if big else K_GRID_5090)[-1]
        one = {}
        for slo in T.SLOS:
            one[str(slo)] = min_ks(rs, slo, top)
            one[str(slo)]["M_at_128"] = m_at(rs, slo, K_REF)
            one[str(slo)]["M_at_256"] = m_at(rs, slo, 256)
        one["rows"] = len(rs)
        one["seeds"] = sorted({r["seed"] for r in rs})
        res[f"{arm}|{tag}"] = one
    out = {"arms": res}

    def get(arm, tag=""):
        return res.get(f"{arm}|{tag}")

    def delta(a, b, slo):
        """Δ = cards(a) - cards(b) (positive = b uses fewer); attributable iff per-seed ranges do not overlap."""
        A, B = get(*a), get(*b)
        if A is None or B is None:
            return None
        x, y = A[str(slo)], B[str(slo)]
        return {"delta_cards": x["stable"] - y["stable"],
                "attributable": not _overlap(x["per_seed_stable"], y["per_seed_stable"]),
                "flag": ">=" if (not x["served"] or not y["served"]) else ""}

    chain = []
    for k in range(1, len(CHAIN)):
        prev, nxt = CHAIN[k - 1], CHAIN[k]
        chain.append({"step": f"{prev}->{nxt}", "lever": CHAIN_LEVER.get(nxt, nxt),
                      "S2": delta((prev, ""), (nxt, ""), 2.0), "S10": delta((prev, ""), (nxt, ""), 10.0)})
    out["chain"] = chain
    out["loo"] = {lv: {"S2": delta((a, ""), ("ALL", ""), 2.0), "S10": delta((a, ""), ("ALL", ""), 10.0)}
                  for lv, a in LOO.items()}
    sens = collections.defaultdict(dict)
    for key, v in res.items():
        arm, tag = key.split("|")
        if arm == "ALL" and tag.startswith("P"):
            sens[tag.split("=")[0]][tag] = v
    shop = []
    pt = get("ALL", "")
    for p, vals in sorted(sens.items()):
        score, flag, rng = 0, "", {}
        for slo in ("2.0", "10.0"):
            xs = [v[slo]["stable"] for v in vals.values()] + ([pt[slo]["stable"]] if pt else [])
            score += max(xs) - min(xs)
            rng[slo] = [min(xs), max(xs)]
            if any(not v[slo]["served"] for v in vals.values()) or (pt and not pt[slo]["served"]):
                flag = ">="
        shop.append({"param": p, "score": score, "flag": flag, "range": rng, "values": sorted(vals)})
    for lv in ("KVSIZE", "ZC", "INT8"):
        l = out["loo"].get(lv) or {}
        if l.get("S2") and l.get("S10"):
            shop.append({"param": f"LEVER:{lv}", "score": l["S2"]["delta_cards"] + l["S10"]["delta_cards"],
                         "flag": l["S2"]["flag"] or l["S10"]["flag"], "range": None, "values": None})
    for r in shop:
        r["test"], r["usd"] = TEST_MAP.get(r["param"], ("design choice, not a test", None))
    shop.sort(key=lambda r: (-r["score"], r["usd"] if r["usd"] is not None else 1e9, r["param"]))
    out["shopping"] = shop
    first_gpu = next((r for r in shop if r["usd"]), None)
    out["first_gpu_test"] = first_gpu
    out["free_first"] = [r for r in shop[:shop.index(first_gpu)] if r["usd"] == 0] if first_gpu else \
        [r for r in shop if r["usd"] == 0]
    # D1 verdicts
    def verdict(a, b, slo):
        A, B = get(a), get(b)
        if A is None or B is None:
            return None
        x, y = A[str(slo)]["per_seed_stable"], B[str(slo)]["per_seed_stable"]
        if x[1] < y[0]:
            return "FEWER"
        if x[0] > y[1]:
            return "MORE"
        return "UNATTRIBUTABLE"
    out["D1"] = {str(slo): {"ALL_vs_A0": verdict("ALL", "A0", slo), "ALL_vs_A2": verdict("ALL", "A2", slo),
                            "A2_vs_A0": verdict("A2", "A0", slo)} for slo in T.SLOS}
    # big-card sweeps (B1/B2) ranges
    bs = collections.defaultdict(dict)
    for key, v in res.items():
        arm, tag = key.split("|")
        if arm in ("BIG_ALL", "BIG_AWAKE_KV") and tag.startswith("B"):
            bs[(arm, tag.split("=")[0])][tag] = v
    out["big_sweeps"] = {f"{a}|{p}": {slo: [min(v[slo]["stable"] for v in vals.values()),
                                           max(v[slo]["stable"] for v in vals.values())] for slo in ("2.0", "10.0")}
                         for (a, p), vals in bs.items()}
    # fallback (prereg §6): (M) at K = 256 for the chain / LOO when ALL is not served <= grid top
    out["fallback_M256"] = {}
    for key, v in res.items():
        a, t = key.split("|")
        if t == "":
            out["fallback_M256"][a] = {slo: v[slo]["M_at_256"] for slo in ("2.0", "10.0")}
    return out


# prereg §7 test map (param -> (test, est. $)); None = design choice / trace property, not a GPU test
TEST_MAP = {"P1": ("T-KVP 5090 KV-pressure probe", 4.0), "P2": ("T-HOST2 second-host repeat", 1.0),
            "P3": ("T-KVP 5090 KV-pressure probe", 4.0), "P4": ("T-2710b zero-copy GPU probe + RSS", 1.5),
            "P5": ("T-2710b zero-copy GPU probe + RSS", 1.5), "P6": ("T-DWELL arm E", 2.0),
            "P7": ("T-HOST2 second-host repeat", 1.0), "P8": ("T-HOST2 second-host repeat", 1.0),
            "P9": ("T-CREF compute calibration, 2nd host", 1.0), "P10": ("T-SIZE config.json reads (data)", 0.0),
            "P11": ("design choice, not a test", None), "P12": ("design choice, not a test", None),
            "P13": ("design choice, not a test", None), "P14": ("trace property, not a test", None),
            "LEVER:KVSIZE": ("T-KVP + a burst-quantile sizing prereg (T-2709 spent, sibling rule WORSE)", 4.0),
            "LEVER:ZC": ("T-2710b zero-copy GPU probe + RSS", 1.5), "LEVER:INT8": ("T-2707 int8 KV quality", 2.0)}


def costs(d, root, floors_json):
    """D3: $/h and $/M tok at each arm's min-K_stable (5090 by RAM tier from the frozen offers; big card fixed $)."""
    if T.sha256_file(os.path.join(root, V.OFFERS_CSV)) != OFFERS_SHA:
        raise ValueError("offers CSV sha mismatch")
    import csv
    offers = list(csv.DictReader(open(os.path.join(root, V.OFFERS_CSV))))
    fl = json.load(open(floors_json))
    tok = fl["heldout_tail_tokens"]
    rows = _rows(d)
    summ = summarize(d)["arms"]
    out = {}
    for key, v in summ.items():
        arm, tag = key.split("|")
        rs = [r for r in rows if r["arm"] == arm and r.get("tag", "") == tag]
        big = arm_cfg(arm)["card"] == "BIG"
        one = {}
        for slo in ("2.0", "10.0"):
            K = v[slo]["stable"]
            at = [r for r in rs if r["K"] == K and "infeasible" not in r]
            if not v[slo]["served"] or not at:
                one[slo] = None
                continue
            if big:
                usd = at[0]["phys"]["big_usd"]; tier = {"n": None, "median": usd}
            else:
                need = max(r["ram_need_gib_per_gpu"] for r in at)
                tier = V.price_tier(offers, need)
                usd = tier["median"]
            one[slo] = {"K": K, "usd_per_gpu_h": usd, "tier": tier, "usd_per_h": None if usd is None else K * usd,
                        "usd_per_mtok": None if usd is None else V.usd_per_mtok(K, usd, tok)}
        out[key] = one
    return out


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

    phys = phys_with({"tpot_sat": 0.065})        # rho_cap = 1 - 0.0065/0.065 = 0.9 (prereg §9)
    mk = lambda cid, b, kv=1.0: (cid, b, b, "fp8") + T.footprint(b, "fp8", kv)
    sing = lambda models: W.build_units(models, "none")

    def run(models, reqs, homes, K, over=None, kvg=None, card_gib=T.CARD_GIB, copies=None, ph=None):
        cfg = arm_cfg("A0"); cfg.update(over or {})
        tr = T.Trace.explicit_cells(models, reqs)
        units, unit_of = sing(models)
        kvt = None if kvg is None else [g * G30 / bpt_of(models[k][1]) for k, g in enumerate(kvg)]
        return simulate(tr, units, unit_of, ("card", homes), K, cfg, 0, score_from=0, min0=0, card_gib=card_gib,
                        phys=ph or phys, kvtok=kvt, copies=copies)

    W8 = 8.56 * GIB / 22.6
    # G13 TPOT formula
    chk("G13 tpot(0.95) capped", tpot_of(0.95, phys), 0.065, 1e-12)
    chk("G13 tpot(0.5)", tpot_of(0.5, phys), 0.013, 1e-12)
    # amendment 1: spread of a residence over minutes
    fut = {}
    chk("A1 spread minute 0", spread(fut, 0, 100.0, 150.0), 6000.0, 1e-9)
    chk("A1 spread spill", fut, {1: 6000.0, 2: 3000.0})
    chk("A1 catch-up", kv_catch_up(0.0, {1: 500.0, 3: 900.0}, 10.0, 0, 4), 300.0, 1e-9)
    chk("A1 catch-up idle drain", kv_catch_up(0.0, {1: 500.0, 3: 900.0}, 10.0, 0, 5), 0.0, 1e-9)
    # G11 int8
    chk("G11 bf16 tokens/GiB", G30 / bpt_of(8.0), 7281.777777777777, 1e-6)
    chk("G11 int8 tokens/GiB", G30 / bpt_of(8.0, True), 14563.555555555555, 1e-6)
    # G1 light load, KV inert
    A = mk("A", 8.0)
    o, met = run([A], [(10.0, 0, 1000, 100), (20.0, 0, 1000, 100)], [0], 1, kvg=[1.0])
    chk("G1 TTFTs", vals(o[0]), [0.14, 0.56], 0.0051)
    chk("G1 no unserved", met["stats"].get("unserved_req", 0), 0)
    # G2 KV binds (dedicated): minutes 0, 1, 3; n 30 each, in 100, out 1000; KV 0.2 GiB
    B = mk("B", 8.0)
    reqs = []
    for mi in (0, 1, 3):
        reqs += [(mi * 60.0 + 1.0 + 0.5 * k, 0, 100, 1000) for k in range(30)]
    tr = T.Trace.explicit_cells([B], reqs)
    kt = 0.2 * G30 / bpt_of(8.0)
    od = dedicated(tr, [kt], phys, score_from_min=0)
    chk("G2 KV waits", vals(od[0]), [13.55] * 30 + [13.55] * 30 + [40.63] * 30, 0.0051)
    o0 = dedicated(tr, None, score_from_min=0)
    chk("G2 KV-off dedicated = tailpool", o0 == V.simulate_dedicated_v2(tr, None, score_from_min=0), True)
    chk("G2 kv_step minute 0", kv_step(0.0, 126814.60352099422, kt)[0], 13.53833891635696, 1e-6)
    # G3 unservable (mean length 2000 > 1456.36)
    tr3 = T.Trace.explicit_cells([B], [(1.0 + k, 0, 1000, 1000) for k in range(30)])
    chk("G3 unservable dedicated", vals(dedicated(tr3, [kt], phys, score_from_min=0)[0]), [math.inf] * 30)
    o, met = run([B], [(1.0 + k, 0, 1000, 1000) for k in range(30)], [0], 1, kvg=[0.2])
    chk("G3 unservable pool", vals(o[0]), [math.inf] * 30)
    chk("G3 counted", met["stats"].get("unservable_req", 0), 30)
    chk("G3 no wake", met["stats"].get("wakes", 0), 0)
    # G4 over-commit fails loudly
    p0 = phys_with({"tpot_sat": 0.065})
    u20 = lambda k: dict(cid=f"u{k:02d}", rep=k, members=[k], tb=8.0, prec="fp8", w=8.56, f=20.5, shape=(8.0, "fp8"))
    ok8 = blind_place([u20(k) for k in range(8)], 1, p0, [(8.56, 8.56)] * 8, False, T.CARD_GIB, 1.0)
    chk("G4a 8 units fit", ok8, [0] * 8)
    raises("G4a 9 units", lambda: blind_place([u20(k) for k in range(9)], 1, p0, [(8.56, 8.56)] * 9, False,
                                              T.CARD_GIB, 1.0), "sleeper GPU")
    p1 = phys_with({"tpot_sat": 0.065, "r_pin": 1.42})
    u8 = lambda k: dict(cid=f"v{k:02d}", rep=k, members=[k], tb=8.0, prec="fp8", w=8.56, f=10.38, shape=(8.0, "fp8"))
    hn = host_need([u8(k) for k in range(7)], p1, True, False)
    chk("G4b pinned per unit", hn[0][0], 12.1552, 1e-9)
    blind_place([u8(k) for k in range(6)], 1, p1, hn[:6], True, T.CARD_GIB, 1.0)
    raises("G4b 7 pinned", lambda: blind_place([u8(k) for k in range(7)], 1, p1, hn, True, T.CARD_GIB, 1.0), "host RAM")
    p2 = phys_with({"tpot_sat": 0.065, "r_pin": 1.0})
    u3 = lambda k: dict(cid=f"w{k:02d}", rep=k, members=[k], tb=3.0, prec="fp8", w=3.21, f=5.03, shape=(3.0, "fp8"))
    hn3 = host_need([u3(k) for k in range(16)], p2, True, False)
    chk("G4c total per unit", hn3[0][1], 8.56, 1e-9)
    blind_place([u3(k) for k in range(15)], 1, p2, hn3[:15], True, T.CARD_GIB, 1.0)
    raises("G4c 16 total", lambda: blind_place([u3(k) for k in range(16)], 1, p2, hn3, True, T.CARD_GIB, 1.0),
           "host RAM")
    live16 = np.zeros((16, 5), bool)
    coact_place([u3(k) for k in range(15)], 1, p2, hn3[:15], True, live16[:15], T.CARD_GIB, 1.0)
    raises("G4e COACT 16 total", lambda: coact_place([u3(k) for k in range(16)], 1, p2, hn3, True, live16, T.CARD_GIB,
                                                     1.0), "COACT infeasible")
    m30 = [mk(f"x{k:02d}", 3.0) for k in range(30)]
    raises("G4d boot over-commit", lambda: run(m30, [(1.0, 0, 0, 0)], [0] * 30, 1), "boot over-commit")
    # G6b copies stay inside the primary's box (box of 2, K = 4): u0 (most live) home 0, u2 home 1, u1 home 2
    u6 = [u8(0), u8(1), u8(2)]
    live6 = np.array([[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0]], bool)
    cp, info = copies_place(u6, [0, 2, 1], 4, p0, host_need(u6, p0, False, False), False, live6, T.CARD_GIB, 1.0, 2, 0.3)
    chk("G6b copy inside the box", cp, [[0, 1], [2], [1]])
    chk("G6b one copy made", info, {"copies_selected": 1, "copies_made": 1})
    # G16 (amendment 2) COFIT: f 28.0 > 0.9 x 28.22 sits alone; 10 + 10 share; the third 10 opens a bin
    uc = [dict(cid=f"c{k}", f=f_) for k, f_ in enumerate([28.0, 10.0, 10.0, 10.0])]
    chk("G16 COFIT singleton", cofit_place(uc, p0, [(0.0, 0.0)] * 4, T.CARD_GIB, False), ([0, 1, 1, 2], 3))
    raises("G16 COFIT larger than a card", lambda: cofit_place([dict(cid="z", f=28.5)], p0, [(0.0, 0.0)], T.CARD_GIB,
                                                               False), "> card")
    # G17 decision rule (§1): grid 8..32; all seeds meet at 16 and 32, seed 2 fails at 24 -> nominal 16, resolved 16,
    # stable 32 (24 breaks the run from the top); per-seed stable [16, 32]; served
    def row(K, s_, M):
        return {"K": K, "seed": s_, "slo": {"2.0": {"M_frac": M, "R_ratio": 1.0}}}
    rr = [row(8, s_, 0.5) for s_ in (0, 1, 2)] + [row(16, s_, 0.97) for s_ in (0, 1, 2)] + \
         [row(24, 0, 0.97), row(24, 1, 0.97), row(24, 2, 0.90)] + [row(32, s_, 0.99) for s_ in (0, 1, 2)]
    mk_ = min_ks(rr, 2.0, 32)
    chk("G17 nominal/resolved/stable", [mk_["nominal"], mk_["resolved"], mk_["stable"]], [16, 16, 32])
    chk("G17 per-seed stable spread", mk_["per_seed_stable"], [16, 32])
    chk("G17 not served -> top + step", min_ks([row(8, 0, 0.5), row(16, 0, 0.5)], 2.0, 16)["stable"], 24)
    # G18 (QA r1) a repeated (arm, tag, K, seed) row is read once, the last one
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "x.jsonl"), "w") as fh:
            fh.write(json.dumps({"arm": "A0", "tag": "", "K": 8, "seed": 0, "v": 1}) + "\n")
            fh.write(json.dumps({"arm": "A0", "tag": "", "K": 8, "seed": 0, "v": 2}) + "\n")
        rr8 = _rows(td)
    chk("G18 dedupe keeps the last row", [r["v"] for r in rr8], [2])
    # G14 adapter slots: 16 members ok, 17 raise
    chk("G14 16 members ok", check_adapter_slots([dict(cid="g", members=list(range(16)))]), None)
    raises("G14 17 members", lambda: check_adapter_slots([dict(cid="g", members=list(range(17)))]), "adapter slots")
    # G5 drain protection: card 22.0; A (KV 2 GiB) in 100 out 12000 at 0; B at 70 in 100 out 10
    A2, B2 = mk("A", 8.0, 2.0), mk("B", 8.0, 2.0)
    reqs = [(0.0, 0, 100, 12000), (70.0, 1, 100, 10)]
    o, met = run([A2, B2], reqs, [0, 0], 1, dict(beta=1.0), kvg=[2.0, 2.0], card_gib=22.0)
    chk("G5 drop: B unserved by drain", vals(o[1]), [math.inf])
    o, met = run([A2, B2], reqs, [0, 0], 1, dict(beta=1.0, sched="queue", fifo_cap=1), kvg=[2.0, 2.0], card_gib=22.0)
    chk("G5 queue: B after drain", vals(o[1]), [11.06], 0.0051)
    A6, B6 = mk("A", 8.0, 6.0), mk("B", 8.0, 6.0)
    reqs = [(0.0, 0, 100, 40000), (70.0, 1, 100, 10)]
    o, met = run([A6, B6], reqs, [0, 0], 1, dict(beta=1.0, sched="queue", fifo_cap=1), kvg=[6.0, 6.0], card_gib=30.0)
    chk("G5b forced still waits for drain", vals(o[1]), [217.14], 0.0051)
    chk("G5b forced admit counted", met["stats"].get("forced_admits", 0), 1)
    # G15 (amendment 2): A keeps long generations running every 30 s until 600 s, so B's queued wake at 70 s never gets
    # room and is abandoned at 670 s (+inf); B again at 700 s wakes after A's drain (680.24) -> 0.83
    reqs = [(30.0 * k, 0, 100, 12000) for k in range(21)] + [(70.0, 1, 100, 10), (700.0, 1, 100, 10)]
    o, met = run([A6, B6], reqs, [0, 0], 1, dict(beta=1.0, sched="queue", fifo_cap=1), kvg=[6.0, 6.0], card_gib=30.0)
    chk("G15 starved wake abandoned, later wake served", vals(o[1]), [0.83, math.inf], 0.0051)
    chk("G15 abandoned counted", met["stats"].get("abandoned_wakes", 0), 1)
    # G6 copies: box of 2, card 21; A copies {0,1}; B home 0 (heavy minute 0), C home 1; A wakes at 65 s
    capmin = C_REF * 60.0
    tinB = 2 * capmin * ACTIVE_REF_B / 8.0
    mA, mB, mC = mk("A", 8.0), mk("B", 8.0), mk("C", 8.0)
    reqs = [(0.5 + 0.01 * k, 1, tinB / 500.0, 0) for k in range(500)] + [(1.0, 2, 0, 0), (65.0, 0, 0, 0)]
    o, met = run([mA, mB, mC], reqs, [0, 0, 1], 2, dict(beta=1.1), kvg=[1.0, 1.0, 1.0], card_gib=21.0,
                 copies=[[0, 1], [0], [1]])
    chk("G6 A wakes on the copy card without backlog", vals(o[0]), [0.47], 0.0051)
    chk("G6 copy wake counted", met["stats"].get("copy_wakes", 0), 1)
    # G7 KV floor + sizing (TRAIN only; held-out cell must not move them)
    m7 = [mk("Q", 8.0)]
    reqs = [(HELD * 0 + 10.0 + k, 0, 1000, 1000) for k in range(10)] + [(130.0, 0, 50000, 0)]
    tr7 = T.Trace.explicit_cells(m7, reqs + [(400 * 60.0 + 1.0, 0, 200000, 0)])
    ft = kv_floor_tok(tr7, [0], 1, 0.99, held=300)
    chk("G7 floor p99", float(ft[0]), 50000.0, 1e-6)
    chk("G7 floor GiB", float(ft[0]) * bpt_of(8.0) / G30, 6.866455078125, 1e-9)
    ct = kv_conc_tok(tr7, [0], 1, m7, phys, 0.95, held=300)
    chk("G7 conc q95", float(ct[0]), 7158.145439437606, 1e-6)
    tr7b = T.Trace.explicit_cells(m7, [(10.0 + 0.1 * k, 0, 1000, 1000) for k in range(100)])
    chk("G7 conc n100", float(kv_conc_tok(tr7b, [0], 1, m7, phys, 0.95, held=300)[0]), 41054.48872636626, 1e-4)
    U7 = W.build_units(m7, "none")[0]
    g, _ = unit_kv(U7, "size", False, T.CARD_GIB, np.array([2000.0]), np.array([200000.0]))
    chk("G7 clamp", g[0], 18.84, 1e-9)
    g, _ = unit_kv(U7, "size", False, T.CARD_GIB, np.array([50000.0]), np.array([7158.145439437606]))
    chk("G7 floor wins", g[0], 6.866455078125, 1e-9)
    # G8 equivalence with wakestorm.simulate (features off) on random explicit traces
    rng = np.random.default_rng(2721)
    ev8 = 0
    m8 = [mk(f"e{k}", 8.0) for k in range(4)]
    for trial in range(4):
        reqs = [(float(rng.uniform(0, 600)), int(rng.integers(0, 4)), float(rng.integers(0, 3000)),
                 float(rng.integers(0, 400))) for _ in range(60)]
        tr8 = T.Trace.explicit_cells(m8, reqs)
        units, unit_of = sing(m8)
        for nm, over in (("L0", {}), ("L4", dict(sched="queue", fifo_cap=1)), ("L7", dict(zero_demote=True))):
            cw = dict(W.BASE); cw.update(over)
            ow, mw = W.simulate(tr8, units, unit_of, ("card", [0, 0, 1, 1]), 2, cw, 0, score_from=0, min0=0,
                                storm={0}, storm_hour={0}, card_gib=15.0)
            cf = arm_cfg("A0"); cf.update(over)
            of, mf = simulate(tr8, units, unit_of, ("card", [0, 0, 1, 1]), 2, cf, 0, score_from=0, min0=0,
                              card_gib=15.0, phys=phys)
            chk(f"G8 {nm} trial {trial} Counters", ow == of, True)
            chk(f"G8 {nm} trial {trial} stats", mw["stats"] == mf["stats"], True)
            ev8 += mw["stats"].get("evictions", 0)
    chk("G8 the equivalence traces exercise evictions", ev8 > 0, True)
    # G9 calibration p95 (TPOT fixed 0.05)
    rows9 = [{"t": mi * 60.0 + 1.0 + 0.5 * k, "model": "m01", "in": 100, "out": 1000} for mi in (0, 1) for k in range(30)]
    p9 = cal_predict(rows9, {"m01": "s4b"}, {"s4b": 8.0}, 0.05)
    # s4b uses EQ 3.23 GiB in calibration; G9 checks the arithmetic at KV 0.2 GiB via kv_step directly
    k9 = 0.2 * G30 / bpt_of(8.0)
    R9 = (100 * (8.0 / ACTIVE_REF_B) / C_REF) + 1000 * 0.05
    a, bk = kv_step(0.0, 30 * 600 * R9, k9)
    b, _ = kv_step(bk, 30 * 600 * R9, k9)
    chk("G9 kvw minute 0", a, 279.07541158895424, 1e-6)
    chk("G9 kvw minute 1", b, 837.2262347668627, 1e-6)
    chk("G9 p95 nearest rank", T._key(sorted([a + R9 - 1000 * 0.05] * 30 + [b + R9 - 1000 * 0.05] * 30)[56]) / 100.0,
        837.24, 0.0051)
    chk("G9 cal_predict runs", p9["m01"] > 0.0, True)
    # G10 placement equivalence
    rng = np.random.default_rng(10)
    m10 = [mk(f"p{k:03d}", float(rng.choice([3.0, 8.0, 14.0]))) for k in range(40)]
    u10, _ = sing(m10)
    for K in (8, 12, 20):
        try:
            a = V.assign_homes(m10, [0.0] * 40, K, 78.2, 1.0, mode="BLIND")
        except ValueError:
            a = "ERR"
        try:
            b = blind_place(u10, K, phys, host_need(u10, phys, False, False), False, T.CARD_GIB, 1.0)
        except ValueError:
            b = "ERR"
        chk(f"G10 BLIND K={K}", a == b, True)
    live = rng.random((40, 200)) < 0.2
    for K in (8, 12):
        try:
            a = W.coact_assign(W.utuples(u10), live, K, 78.2)
        except ValueError:
            a = "ERR"
        try:
            b = coact_place(u10, K, phys, host_need(u10, phys, False, False), False, live, T.CARD_GIB, 1.0)
        except ValueError:
            b = "ERR"
        chk(f"G10 COACT K={K}", a == b, True)
    # G12 all-awake boot: no wake, TTFT = queue + prefill
    cfgA = arm_cfg("A0"); cfgA["all_awake"] = True
    tr12 = T.Trace.explicit_cells([A], [(10.0, 0, 1000, 100)])
    u12, uo12 = sing([A])
    o, met = simulate(tr12, u12, uo12, ("card", [0]), 1, cfgA, 0, score_from=0, min0=0, phys=phys)
    chk("G12 all-awake no wake", met["stats"].get("wakes", 0), 0)
    chk("G12 all-awake TTFT", vals(o[0]), [0.14], 0.0051)
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
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--floors", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize")
    ap.add_argument("--costs", help="DIR: D3 $/h and $/M tok per arm (needs --floors-json); read-only")
    ap.add_argument("--floors-json")
    ap.add_argument("--arm")
    ap.add_argument("--ks")
    ap.add_argument("--kgrid", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
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
        if a.calibrate:
            print(json.dumps(calibrate(a.root)))
            return 0
        if a.check:
            r = check(a.root, a.cache)
            print(json.dumps(r))
            return 0 if r["I1_pass"] and r["I4_pass"] else 1
        if a.floors:
            print(json.dumps(floors(a.root, a.cache, phys)))
            return 0
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"not a directory: {a.summarize}")
            print(json.dumps(summarize(a.summarize), default=str))
            return 0
        if a.costs:
            if not os.path.isdir(a.costs) or not a.floors_json:
                raise ValueError("--costs needs a directory and --floors-json")
            print(json.dumps(costs(a.costs, a.root, a.floors_json), default=str))
            return 0
        if a.run:
            if not a.arm or not a.out:
                raise ValueError("--run needs --arm and --out")
            arm_cfg(a.arm)
            if not a.kgrid and not a.ks and arm_cfg(a.arm)["placement"] != "COFIT":
                raise ValueError("--run needs --ks or --kgrid")
            seeds = [int(x) for x in a.seeds.split(",")]
            ks = k_list(a.arm, phys, a.root, a.cache, [int(x) for x in a.ks.split(",")] if a.ks else None, a.kgrid)
            with open(a.out, "a") as fh:
                for K in ks:
                    for seed in seeds:
                        row = run_cell(a.arm, K, seed, phys, a.root, a.cache, a.tag)
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
