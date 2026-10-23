#!/usr/bin/env python3
"""wakestorm_sim.py — WAKE STORMS on a multi-GPU tail pool. A per-PCIe-link bandwidth model (processor
sharing per direction, optional per-link FIFO cap, optional shared uplinks) under pre-staged pool, with the
levers L1-L7 as arms, replayed on the real Chutes held-out week plus a storm window and a SYNC probe.

Prereg (frozen before this file existed): (private report). On any divergence
the prereg wins. Imports (scripts/tailpool_sim.py) and (scripts/tailpool_v2.py) UNCHANGED —
nothing here edits them.

Output contract:
  --self-test            exit 0 + "SELF-TEST PASS n/n" on the hand-computed goldens G1-G12; exit 1 + FAIL lines otherwise.
  --describe             read-only: storm set (top-100 R(t) minutes), STORM_HOUR, L2 unit counts; one JSON object on stdout.
  --run --arm A [--ks 48,96] [--seeds 0,1,2] [--arrivals POISSON|BURST] [--variant base|switch|x8|pcie5|eta03|box|fifo]
        --out F          simulates and APPENDS one JSON line per (K, seed) to F (WRITES F); exit 0.
  --sync --arm A --out F APPENDS one SYNC-probe JSON line (K=96, box 0) to F (WRITES F); exit 0.
  --summarize DIR        read-only: prereg §5/§6 verdicts over DIR/*.jsonl; one JSON object on stdout.
  --capacity --g G --cls FP8|INT4|DELTA --dram DESKTOP|WORKSTATION|SERVER --lever L --out F
                         amendment 2: max sustained wakes/s per box at p99 <= 1 s / 2 s; APPENDS one JSON line (WRITES F).
  --trace-demand         amendment 2 §3(a): read-only trace burst demand (D1, R5), JSON on stdout.
  --bursts --arm A --ks .. --seeds .. --out F
                         amendment 2 §3(b): the pool's wake admissions per sliding window; APPENDS JSON lines (WRITES F).
  Bad input (sha mismatch, unknown arm/variant, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed while building):
  1. **A wake's ready time under processor sharing is unknown when it starts** — later wakes on the same link slow it. So
     waits are scored DEFERRED (resolved after every link drains), never at the minute's end.
  2. **L0_V2 must equal byte for byte** — float addition order matters at the 0.01 s key edge; the scorer keeps
     `float(wv) + q + extra + pre` order for every arm without L5.
"""
import argparse, collections, heapq, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tailpool_sim as T  # noqa: E402  (not edited)
import tailpool_v2 as V   # noqa: E402  (not edited)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

GIB = V.GIB
H2D_GBPS = 22.6            # page-in fit, PCIe 4 x16
D2H_GBPS = 23.8            # later-sleep fit: 0.10/0.40/0.70 s for 2.67/9.45/16.95 GB
FIXED_S = 0.02             # page-in fixed term
BETA_BUSY = 1.10           # D2: 0.474 / 0.430 s with a decode-busy neighbour
PROTECT_S = T.PROTECT_S    # 60 s,
W_MAX_S = 60.0             # L4 forced admission
PHI = 0.97                 # L5: 1 - 1/36 layers
DELTA_GB_PER_B = 0.0176    # L2_LIN: rank-16 = 71 MB / 8,045 MB bf16 = 0.0088 x 2 bytes/param
SHELLS_PER_CARD = 3        # L6
SHAPE_SWITCH_S = 4.0       # L6: warm graph capture (n=1)
BOX_G = 8
STORM_N = 100
STORM_IDLE_MIN = 5
H_GIB = V.H_ARMS["H84"]
S_GIB = V.SLEEPER_GIB
K_GRID = (48, 64, 96, 128, 192)
K_PRIMARY = 96
SEEDS = (0, 1, 2)
HELD = T.HELDOUT_MIN
HELDOUT_REQ = 3687246
CENSUS = "data/lineage_census/census.json"
CENSUS_SHA = "8778a4ce6e34c2e8d18739e5bcf10d54808e353c03346db1a6b3e3d04adde6af"
V2_ROWS = "data/tailpool_v2/pool_BLIND.jsonl"
INF_T = 1e15

BASE = dict(physics="link", placement="BLIND", units="none", prec="fp8", sched="drop", fifo_cap=None,
            protect=PROTECT_S, phi=0.0, anycard=False, multipath=False, zero_demote=False, box_coact=False,
            topo="direct", h2d=H2D_GBPS, d2h=D2H_GBPS, eta=0.0, beta=BETA_BUSY)
ARMS = {
    "L0_V2": dict(physics="v2"),
    "L0": {},
    "L1": dict(placement="COACT"),
    "L2_DUP": dict(units="dup"),
    "L2_LIN": dict(units="lin"),
    "L3": dict(prec="int4"),
    "L4": dict(sched="queue", fifo_cap=1),
    "L4_D2": dict(sched="queue", fifo_cap=1, protect=2.0),
    "L5": dict(phi=PHI),
    "L6": dict(anycard=True),
    "L6M": dict(anycard=True, multipath=True),
    "L7": dict(zero_demote=True),
    "S_NOPATCH": dict(placement="COACT", units="dup", sched="queue", fifo_cap=1),
    "S_LOSSLESS": dict(placement="COACT", units="dup", sched="queue", fifo_cap=1, phi=PHI, zero_demote=True),
    "S_ALL": dict(placement="COACT", units="lin", sched="queue", fifo_cap=1, phi=PHI, zero_demote=True, prec="int4",
                  anycard=True, box_coact=True),
}
LEVERS = ("L1", "L2_DUP", "L2_LIN", "L3", "L4", "L4_D2", "L5", "L6", "L6M", "L7")
STACKS = ("S_NOPATCH", "S_LOSSLESS", "S_ALL")
LOSSY = {"L2_LIN", "L3", "S_ALL"}
STACK_PARTS = {"S_NOPATCH": ("L1", "L2_DUP", "L4"), "S_LOSSLESS": ("L1", "L2_DUP", "L4", "L5", "L7"),
               "S_ALL": ("L1", "L2_LIN", "L3", "L4", "L5", "L6", "L7")}
VARIANTS = {"base": {}, "switch": dict(topo="switch"), "x8": dict(h2d=11.3, d2h=11.9),
            "pcie5": dict(h2d=44.0, d2h=46.3), "eta03": dict(eta=0.3), "box": dict(topo="box"),
            "fifo": dict(same_card_fifo=True)}
SENS_ARMS = ("L0", "L4", "L6", "S_LOSSLESS", "S_ALL")
SENS_VARIANTS = ("switch", "x8", "pcie5", "eta03")
# prereg §5 GPU-arm rule inputs: lever -> (load-bearing premise, measured?, covered by ?)
PREMISES = {
    "L1": ("TRAIN co-activity predicts held-out co-activity (a trace property, tested by this sim)", True, False),
    "L2_DUP": ("validated/byte-identical duplicates serve identically (20/20)", True, False),
    "L2_LIN": ("rank-16 sibling deltas keep quality (Affine row; re-run in step 2)", False, False),
    "L3": ("int4 quality vs fp8 (LOSSY under the quality bar)", False, False),
    "L4": ("same-card wakes can be serialized by a router without loss (D1 measures same-card pairs)", False, True),
    "L4_D2": ("a 2 s dwell does not flap (arm E measures dwell 1.5 s)", False, True),
    "L5": ("vLLM can start prefill on layers already paged in (no implementation exists)", False, False),
    "L6": ("weights swap into a live engine shell on any card of the box at page-in speed (C measures replicas, not shells)",
           False, False),
    "L6M": ("GPU P2P relay of H2D copies (unsupported on RTX 5090; datacenter GPUs only)", False, False),
    "L7": ("keeping the pinned host copy on wake makes a demote ~free (vLLM frees it, cumem.py:356)", False, False),
}


def cfg_for(arm, variant="base"):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    c = dict(BASE)
    c.update(ARMS[arm])
    v = dict(VARIANTS[variant])
    if v.pop("same_card_fifo", False):
        c["fifo_cap"] = 1
    c.update(v)
    return c


# ------------------------------------------------------------------------------------------------ link engine
class Group:
    """One uplink group: its links (cards) x two directions ('h' = H2D page-in, 'd' = D2H demote). Processor sharing per
    link with e(n) = 1/(1+eta(n-1)); 'h' may carry a per-link FIFO cap; a finite uplink U is shared max-min per job.
    Lazy discrete-event: advance(t) completes every job due by t in time order; callbacks may add jobs at that time."""

    def __init__(self, cards, h2d, d2h, uplink_h, uplink_d, eta, cap):
        self.now = 0.0
        self.C = {"h": h2d, "d": d2h}
        self.U = {"h": uplink_h, "d": uplink_d}
        self.cap = {"h": cap, "d": None}
        self.eta = eta
        self.q = {}
        for c in cards:
            self.q[("h", c)] = []
            self.q[("d", c)] = []
        self.jobs = {}
        self.seq = 0
        self.maxconc = collections.Counter()
        self._busy = False          # re-entrancy guard: callbacks run inside advance() and may add() jobs

    def n_h2d(self, c):
        return len(self.q[("h", c)])

    def add(self, d, c, gb, mult, cb, t):
        self.advance(t)             # a no-op inside a callback (advance() is re-entrancy guarded; now == t there)
        if gb <= 0.0:
            cb(t)
            return
        self.seq += 1
        j = self.seq
        self.jobs[j] = [d, c, gb, mult, cb]
        self.q[(d, c)].append(j)
        if d == "h":
            self.maxconc[c] = max(self.maxconc[c], len(self.q[(d, c)]))

    def _rates(self):
        rates = {}
        for d in ("h", "d"):
            per = {}
            cap = self.cap[d]
            for (dd, c), lst in self.q.items():
                if dd != d or not lst:
                    continue
                act = lst[:cap] if cap else lst
                n = len(act)
                share = self.C[d] / (1.0 + self.eta * (n - 1)) / n
                for j in act:
                    per[j] = share
            U = self.U[d]
            if per and U is not None and sum(per.values()) > U + 1e-12:
                items = sorted(per.items(), key=lambda x: (x[1], x[0]))
                rem, k = U, len(items)
                for idx, (j, sh) in enumerate(items):
                    fair = rem / (k - idx)
                    if sh <= fair:
                        per[j] = sh
                        rem -= sh
                    else:
                        for jj, _ in items[idx:]:
                            per[jj] = fair
                        break
            for j, r in per.items():
                rates[j] = r * self.jobs[j][3]
        return rates

    def advance(self, t):
        if self._busy:
            return
        self._busy = True
        try:
            self._advance(t)
        finally:
            self._busy = False

    def _advance(self, t):
        while True:
            rates = self._rates()
            if not rates:
                self.now = max(self.now, t)
                return
            dt, jmin = min((self.jobs[j][2] / r, j) for j, r in rates.items())
            if self.now + dt <= t + 1e-12:
                tc = self.now + dt
                for j, r in rates.items():
                    self.jobs[j][2] -= r * dt
                self.now = tc
                done = sorted(j for j in rates if j == jmin or self.jobs[j][2] <= 1e-9)
                for j in done:
                    d, c, _, _, cb = self.jobs.pop(j)
                    self.q[(d, c)].remove(j)
                    cb(tc)
            else:
                dd = t - self.now
                for j, r in rates.items():
                    self.jobs[j][2] -= r * dd
                self.now = t
                return


def make_groups(K, cfg):
    """card -> Group, by topology. direct: one link per uplink; switch: pairs share one x16; box: 8 cards share one."""
    size = {"direct": 1, "switch": 2, "box": BOX_G}[cfg["topo"]]
    out = {}
    for g0 in range(0, K, size):
        cards = list(range(g0, min(K, g0 + size)))
        U_h = None if size == 1 else cfg["h2d"]
        U_d = None if size == 1 else cfg["d2h"]
        grp = Group(cards, cfg["h2d"], cfg["d2h"], U_h, U_d, cfg["eta"], cfg["fifo_cap"])
        for c in cards:
            out[c] = grp
    return out


def multipath_mult(r):
    return 1.0 + 0.48 * min(r, 4) + 0.22 * max(0, min(r, 7) - 4)


# ------------------------------------------------------------------------------------------------ units / placement
def build_models(tr, prec):
    if prec == "fp8":
        return list(tr.models)
    return [(cid, tb, ab, prec) + T.footprint(tb, prec) for cid, tb, ab, p, w, f in tr.models]


def load_census(root):
    p = os.path.join(root, CENSUS)
    got = T.sha256_file(p)
    if got != CENSUS_SHA:
        raise ValueError(f"census sha mismatch: {got} != {CENSUS_SHA}")
    return json.load(open(p))


def build_units(models, mode, census=None, work=None):
    """-> (units, unit_of). unit = dict(cid, rep, members, tb, prec, w, f, shape). mode 'none': unit u == model u."""
    M = len(models)
    groups = []
    if mode != "none":
        sets = census["dup_sets_strict"] if mode == "dup" else census["lineage_sets"]
        idx = {m[0]: k for k, m in enumerate(models)}
        seen = set()
        for g in sets:
            mem = sorted({idx[c] for rp in g for c in census["repo_cids"].get(rp, []) if c in idx})
            mem = [k for k in mem if k not in seen]
            if len(mem) >= 2:
                seen.update(mem)
                groups.append(mem)
    rep_of = {}
    for mem in groups:
        rep = max(mem, key=lambda k: ((work or {}).get(models[k][0], 0.0), models[k][0]))
        for k in mem:
            rep_of[k] = (rep, mem)
    units, unit_of = [], [-1] * M
    for k in range(M):
        if k in rep_of and rep_of[k][0] != k:
            continue
        rep, mem = rep_of.get(k, (k, [k]))
        cid, tb, ab, prec, w, f = models[rep]
        if mode == "lin" and len(mem) > 1:
            dg = sum(DELTA_GB_PER_B * models[o][1] for o in mem if o != rep) / GIB
            w, f = w + dg, f + dg
        u = len(units)
        units.append(dict(cid=cid, rep=rep, members=list(mem), tb=tb, prec=prec, w=w, f=f, shape=(tb, prec)))
        for o in mem:
            unit_of[o] = u
    return units, unit_of


def utuples(units):
    return [(U["cid"], U["tb"], U["tb"], U["prec"], U["w"], U["f"]) for U in units]


def train_live(tr, unit_of, n_units, min0=T.MIN0, held=HELD):
    """bool [n_units, held-min0]: a unit is live in minute t if any member has a request in minute t or t-1 (TRAIN)."""
    Tn = held - min0
    live = np.zeros((n_units, Tn), bool)
    k = tr.minute < held
    u = np.asarray(unit_of)[tr.mid[k]]
    t = (tr.minute[k] - min0).astype(np.int64)
    live[u, t] = True
    t1 = t + 1
    ok = t1 < Tn
    live[u[ok], t1[ok]] = True
    return live


def card_slots(n, fmax, card_gib=T.CARD_GIB, s=S_GIB):
    if n <= 0 or fmax <= s:
        return 10 ** 6
    return max(0, int(math.floor((card_gib - n * s) / (fmax - s) + 1e-9)))


def shell_slots(fmax, card_gib=T.CARD_GIB, s=S_GIB):
    return min(SHELLS_PER_CARD, max(0, int(math.floor((card_gib - SHELLS_PER_CARD * s) / (fmax - s) + 1e-9))))


def coact_assign(ut, live, nbins, H, mode="card", card_gib=T.CARD_GIB, s=S_GIB, box_g=BOX_G):
    """L1 COACT (prereg §3): greedy over units by TRAIN live-minutes desc (ties cid); each to the feasible bin minimizing the
    added overflow sum_t max(0, live_bin(t) - slots); ties -> BLIND's rule (most free RAM, then lowest index)."""
    M = len(ut)
    ram = np.full(nbins, float(H) * (box_g if mode == "box" else 1))
    n_h = np.zeros(nbins, np.int64)
    fmax = np.zeros(nbins)
    L = np.zeros((nbins, live.shape[1]), np.int16)
    homes = [-1] * M
    cnt = live.sum(axis=1)
    order = sorted(range(M), key=lambda m: (-int(cnt[m]), ut[m][0]))

    def slots(n, fm):
        if mode == "box":
            return box_g * shell_slots(fm, card_gib, s)
        return card_slots(n, fm, card_gib, s)

    for m in order:
        w, f = ut[m][4], ut[m][5]
        if w > ram.max() + 1e-9 and w > float(H) * (box_g if mode == "box" else 1) + 1e-9:
            raise ValueError(f"host RAM cap exceeded: unit {ut[m][0]} weights {w:.2f} GiB")
        if mode == "box":
            feas = [c for c in range(nbins) if ram[c] >= w - 1e-9 and f <= card_gib + 1e-9]
        else:
            feas = [c for c in range(nbins) if ram[c] >= w - 1e-9 and n_h[c] * s + max(fmax[c], f) <= card_gib + 1e-9]
        if not feas:
            raise ValueError(f"COACT infeasible: unit {ut[m][0]} fits no bin")
        lm = live[m].astype(np.int16)
        best = None
        for c in feas:
            so = slots(n_h[c], fmax[c])
            sn = slots(n_h[c] + 1, max(fmax[c], f))
            Lc = L[c]
            over_old = int(np.maximum(0, Lc.astype(np.int32) - so).sum()) if n_h[c] else 0
            over_new = int(np.maximum(0, (Lc + lm).astype(np.int32) - sn).sum())
            key = (over_new - over_old, -ram[c], c)
            if best is None or key < best[0]:
                best = (key, c)
        c = best[1]
        homes[m] = c
        ram[c] -= w; n_h[c] += 1; fmax[c] = max(fmax[c], f); L[c] += lm
    return homes


def blind_boxes(ut, nboxes, H, box_g=BOX_G, card_gib=T.CARD_GIB, s=S_GIB):
    """L6: BLIND at box level — cid order, box with the most free RAM (box RAM = box_g x H), ties lowest index."""
    ram = [float(H) * box_g] * nboxes
    out = [-1] * len(ut)
    for m in sorted(range(len(ut)), key=lambda k: ut[k][0]):
        w, f = ut[m][4], ut[m][5]
        if f > card_gib + 1e-9:
            raise ValueError(f"unit {ut[m][0]} does not fit a card")
        feas = [b for b in range(nboxes) if ram[b] >= w - 1e-9]
        if not feas:
            raise ValueError(f"box host RAM cap exceeded: unit {ut[m][0]}")
        b = max(feas, key=lambda j: (ram[j], -j))
        out[m] = b
        ram[b] -= w
    return out


def pin_cards(units, box_of, nboxes, box_g=BOX_G, card_gib=T.CARD_GIB, s=S_GIB):
    """Amendment 1: a unit with f > card - (shells-1)*s is PINNED to the highest-index free card of its box
    (one shell, card-bound); further pinned units take the next-highest cards, sharing the last one if the box runs out.
    Returns pin[u] (absolute card or -1). A box with no unpinned card left and a floating unit = ValueError (INFEASIBLE)."""
    lim = card_gib - (SHELLS_PER_CARD - 1) * s
    pin = [-1] * len(units)
    for b in range(nboxes):
        big = sorted((u for u in range(len(units)) if box_of[u] == b and units[u]["f"] > lim + 1e-9),
                     key=lambda u: units[u]["cid"])
        for k, u in enumerate(big):
            pin[u] = b * box_g + max(0, box_g - 1 - k)
        npinned = len({pin[u] for u in big})
        floaters = [u for u in range(len(units)) if box_of[u] == b and pin[u] < 0]
        if floaters and npinned >= box_g:
            raise ValueError(f"box {b}: no unpinned card left for {len(floaters)} floating units")
    return pin


def place(units, K, cfg, live=None, box_g=BOX_G):
    ut = utuples(units)
    if cfg["anycard"]:
        nb = K // box_g
        if nb * box_g != K:
            raise ValueError(f"K={K} not a multiple of box size {box_g}")
        if cfg["box_coact"]:
            box_of = coact_assign(ut, live, nb, H_GIB, mode="box", box_g=box_g)
        else:
            box_of = blind_boxes(ut, nb, H_GIB, box_g)
        return ("box", box_of, pin_cards(units, box_of, nb, box_g))
    if cfg["placement"] == "COACT":
        return ("card", coact_assign(ut, live, K, H_GIB))
    return ("card", V.assign_homes(ut, [0.0] * len(ut), K, H_GIB, S_GIB, mode="BLIND"))


# ------------------------------------------------------------------------------------------------ storm windows
def storm_sets(tr, held=HELD, nmin=10080, top=STORM_N, idle=STORM_IDLE_MIN):
    """prereg §1: R(t) = # tail models with a request in minute t and none in t-idle..t-1; STORM = top-`top` held-out
    minutes by R (ties earliest); STORM_HOUR = the 60-minute window with max sum R (ties earliest)."""
    mid = np.asarray(tr.mid, np.int64)
    mi = np.asarray(tr.minute, np.int64)
    o = np.lexsort((mi, mid))
    m_s, t_s = mid[o], mi[o]
    prev = np.full(len(o), -10 ** 9, np.int64)
    if len(o) > 1:
        prev[1:] = t_s[:-1]
        newm = np.ones(len(o), bool)
        newm[1:] = m_s[1:] != m_s[:-1]
        prev[newm] = -10 ** 9
    react = (t_s - prev) > idle
    hk = (t_s >= held) & (t_s < held + nmin)
    R = np.bincount((t_s[react & hk] - held).astype(np.int64), minlength=nmin)[:nmin]
    tops = sorted(range(nmin), key=lambda k: (-int(R[k]), k))[:top]
    cs = np.concatenate(([0], np.cumsum(R)))
    win = cs[60:] - cs[:-60] if nmin >= 60 else np.array([cs[-1]])
    h0 = int(np.argmax(win))
    return {held + k for k in tops}, set(range(held + h0, held + h0 + min(60, nmin))), R


# ------------------------------------------------------------------------------------------------ simulator
class Ep:
    __slots__ = ("u", "card", "t_create", "t_adm", "ready", "pend_dem", "h2d_gb", "extra", "mult", "shell")

    def __init__(self, u, t):
        self.u = u; self.card = -1; self.t_create = t; self.t_adm = None; self.ready = None
        self.pend_dem = 0; self.h2d_gb = 0.0; self.extra = 0.0; self.mult = 1.0; self.shell = None


def simulate(trace, units, unit_of, plc, K, cfg, seed, burst_w=None, score_from=HELD, min0=T.MIN0,
             storm=frozenset(), storm_hour=frozenset(), card_gib=T.CARD_GIB, s=S_GIB, box_g=BOX_G, collect=None):
    """Returns (out per-model Counter of TTFT keys over scored cells, metrics dict). Deterministic given seed.
    collect (amendment 2 --bursts): a list that receives (admission time s, card) per wake; changes nothing scored."""
    v2 = cfg["physics"] == "v2"
    protect = cfg["protect"]
    queue = cfg["sched"] == "queue"
    phi = cfg["phi"]
    beta = cfg["beta"]
    anycard = cfg["anycard"]
    models = trace.models
    M, NU = len(models), len(units)
    kind, assign = plc[0], plc[1]
    pin = plc[2] if len(plc) > 2 else [-1] * len(units)
    pinned_cards = {c for c in pin if c >= 0}
    if kind == "card":
        if len(assign) != NU or any(h < 0 or h >= K for h in assign):
            raise ValueError("homes must give every unit a card in [0, K)")
    else:
        if len(assign) != NU or any(b < 0 or b >= K // box_g for b in assign):
            raise ValueError("boxes must give every unit a box in [0, K/box_g)")
    first, last, order = V.arrivals_v2(trace, seed, burst_w)
    groups = None if v2 else make_groups(K, cfg)
    gpu = [0.0] * K
    shells = None
    if kind == "card":
        for u in range(NU):
            gpu[assign[u]] += s
    else:
        shells = []
        dom = {}
        for b in range(K // box_g):
            mem = [units[u]["shape"] for u in range(NU) if assign[u] == b]
            dom[b] = collections.Counter(mem).most_common(1)[0][0] if mem else None
        pin_shape = {pin[u]: units[u]["shape"] for u in range(NU) if pin[u] >= 0}
        for c in range(K):
            if c in pinned_cards:
                gpu[c] = s
                shells.append([[pin_shape[c], None]])
            else:
                gpu[c] = SHELLS_PER_CARD * s
                shells.append([[dom[c // box_g], None] for _ in range(SHELLS_PER_CARD)])
    awake = [False] * NU          # occupies GPU memory (awake or waking)
    cur = [-1] * NU               # card it occupies
    ep_of = [None] * NU           # last admitted episode
    pend_ep = [None] * NU         # pending-admission episode (L4)
    last_arr = [-1e18] * NU
    card_units = [set() for _ in range(K)]
    backlog = [0.0] * K
    cap_min = C_REF * 60.0
    out = [collections.Counter() for _ in range(M)]
    st = collections.Counter()
    heap = []
    hseq = [0]
    deferred = []
    svc = []
    agg = {k: collections.Counter() for k in ("storm_all", "storm_wa", "week_all", "week_wa", "hour_all", "hour_wa")}

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
        """-> victims list or None (no room even after legal evictions)."""
        need = units[u]["f"] - s
        if kind == "box":
            free_sh = sum(1 for sh in shells[c] if sh[1] is None)
        else:
            free_sh = 1
        if gpu[c] + need <= card_gib + 1e-9 and free_sh >= 1:
            return []
        adv(c, t)
        cand = sorted((last_arr[r], r) for r in card_units[c]
                      if ready_by(r, t) and (force or not last_arr[r] > t - protect))
        free = card_gib - gpu[c]
        vict = []
        for _, r in cand:
            if free >= need - 1e-9 and free_sh >= 1:
                break
            vict.append(r); free += units[r]["f"] - s; free_sh += 1
        if free < need - 1e-9 or free_sh < 1:
            return None
        return vict

    def h2d_done_cb(e):
        def cb(tc):
            e.ready = tc + e.extra
        return cb

    def start_h2d(e, t):
        c = e.card
        if v2:
            return
        mult = 1.0
        if cfg["multipath"]:
            b = c // box_g
            idle = sum(1 for cc in range(b * box_g, b * box_g + box_g) if cc != c and groups[cc].n_h2d(cc) == 0)
            mult = multipath_mult(idle)
        e.mult = mult
        groups[c].add("h", c, e.h2d_gb, mult, h2d_done_cb(e), t)

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
            st["evictions"] += 1
            if kind == "box":
                for sh in shells[c]:
                    if sh[1] == r:
                        sh[1] = None
        extra_shape = 0.0
        if kind == "box":
            free = [sh for sh in shells[c] if sh[1] is None]
            match = [sh for sh in free if sh[0] == U["shape"]]
            sh = match[0] if match else free[0]
            if sh[0] != U["shape"]:
                extra_shape = SHAPE_SWITCH_S
                st["shape_switches"] += 1
                sh[0] = U["shape"]
            sh[1] = u
        busy = any(r != u for r in card_units[c])
        awake[u] = True; card_units[c].add(u); gpu[c] += U["f"] - s; cur[u] = c
        e.card = c; e.t_adm = t
        st["wakes"] += 1
        if v2:
            L = len(vict) * V.SLEEP_S + V.PAGEIN_FIXED + U["w"] * GIB / V.PAGEIN_GBPS
            e.ready = t + L
        else:
            e.extra = FIXED_S + extra_shape
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
        if collect is not None:
            collect.append((t, c))

    def box_cands(u):
        if kind == "card":
            return [assign[u]]
        if pin[u] >= 0:
            return [pin[u]]
        b = assign[u]
        return [c for c in range(b * box_g, b * box_g + box_g) if c not in pinned_cards]

    def try_admit(u, t, force, e):
        cands = box_cands(u)
        best = None
        for c in cands:
            vict = plan(u, c, t, force)
            if vict is None:
                continue
            adv(c, t)   # the lazy link engine must be current before its job count is read
            nj = groups[c].n_h2d(c) if groups is not None else 0
            key = (len(vict), nj, -(card_gib - gpu[c]), c)
            if best is None or key < best[0]:
                best = (key, c, vict)
        if best is None:
            return False
        admit(u, best[1], best[2], t, e)
        return True

    def next_retry(u, t, e):
        cands = box_cands(u)
        nxt = e.t_create + W_MAX_S
        if t >= nxt - 1e-9:
            nxt = INF_T
        for c in cands:
            for r in card_units[c]:
                if ready_by(r, t):
                    x = last_arr[r] + protect
                    if x > t + 1e-9:
                        nxt = min(nxt, x)
                else:
                    er = ep_of[r]
                    if er.ready is not None and er.ready > t:
                        nxt = min(nxt, er.ready)
        if nxt >= INF_T:
            nxt = t + 0.05
        return nxt

    def push(t, u):
        hseq[0] += 1
        heapq.heappush(heap, (t, hseq[0], u))

    def run_events(t):
        while heap and heap[0][0] <= t:
            te, _, u = heapq.heappop(heap)
            e = pend_ep[u]
            if e is None:
                continue
            if te > e.t_create + 86400.0:   # review: an invariant break must error, never livelock
                raise RuntimeError(f"wake of unit {u} pending > 1 day: placement invariant broken")
            force = te >= e.t_create + W_MAX_S - 1e-9
            if try_admit(u, te, force, e):
                pend_ep[u] = None
                st["queued_admits"] += 1
                if force:
                    st["forced_admits"] += 1
            else:
                push(next_retry(u, te, e), u)

    def home_card(u):
        if kind == "card":
            return assign[u]
        if cur[u] >= 0:
            return cur[u]
        return max(box_cands(u), key=lambda c: (card_gib - gpu[c], -c))

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
            u = unit_of[m]
            t0 = base + float(first[i])
            run_events(t0)
            ab = models[m][2]
            served = True
            wait_ep = None
            if awake[u]:
                if not ready_by(u, t0):
                    wait_ep = ep_of[u]
            elif pend_ep[u] is not None:
                wait_ep = pend_ep[u]
            else:
                e = Ep(u, t0)
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
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF
            if not served:
                st["unserved_req"] += n
                pend.append((m, -1, n, pre, None, 0.0, i))
                continue
            c = home_card(u)
            wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
            W[c] += wm
            extra = V._burst_extra(wm, burst_w)
            pend.append((m, c, n, pre, wait_ep, extra, i))
        q = [0.0] * K
        for c in range(K):
            q[c] = (backlog[c] + 0.5 * max(0.0, W[c] - cap_min)) / C_REF
            backlog[c] = max(0.0, backlog[c] + W[c] - cap_min)
        if mi >= score_from:
            in_storm = mi in storm
            in_hour = mi in storm_hour
            for m, c, n, pre, e, extra, i in pend:
                if c < 0:
                    out[m][T.INF_KEY] += n
                    for k in ("week_all", "week_wa") + (("storm_all", "storm_wa") if in_storm else ()) + \
                            (("hour_all", "hour_wa") if in_hour else ()):
                        agg[k][T.INF_KEY] += n
                    continue
                if e is None:
                    key = T._key(q[c] + extra + pre)
                    out[m][key] += n
                    agg["week_all"][key] += n
                    if in_storm:
                        agg["storm_all"][key] += n
                    if in_hour:
                        agg["hour_all"][key] += n
                else:
                    ts = T.cell_times(trace, i, first, last, seed) + base
                    deferred.append((m, ts, q[c], extra, pre, e, in_storm, in_hour))
        i_pos = j
    run_events(INF_T / 10)
    if groups is not None:
        for c in range(K):
            groups[c].advance(INF_T)
    for m, ts, qc, extra, pre, e, in_storm, in_hour in deferred:
        if e.ready is None:
            raise RuntimeError("episode never became ready (link engine bug)")
        waits = np.maximum(0.0, e.ready - ts)
        for wv, tsv in zip(waits, ts):
            wv = float(wv)
            if phi > 0.0:
                hid = min(phi * pre, wv)
                key = T._key(wv + qc + extra + (pre - hid))
            else:
                key = T._key(wv + qc + extra + pre)
            out[m][key] += 1
            wa = e.ready > tsv
            agg["week_all"][key] += 1
            if wa:
                agg["week_wa"][key] += 1
            if in_storm:
                agg["storm_all"][key] += 1
                if wa:
                    agg["storm_wa"][key] += 1
            if in_hour:
                agg["hour_all"][key] += 1
                if wa:
                    agg["hour_wa"][key] += 1
    held_svc = [e.ready - e.t_adm for e in svc if e.t_adm is not None and e.t_adm >= (score_from - min0) * 60.0]
    held_q = [e.t_adm - e.t_create for e in svc if e.t_adm is not None and e.t_adm >= (score_from - min0) * 60.0]
    maxconc = 0
    if groups is not None:
        for g in {id(g): g for g in groups.values()}.values():
            if g.maxconc:
                maxconc = max(maxconc, max(g.maxconc.values()))
    met = {"stats": dict(st), "max_h2d_jobs_per_link": maxconc,
           "svc_p50": _q(held_svc, 0.5), "svc_p99": _q(held_svc, 0.99), "svc_n": len(held_svc),
           "adm_wait_p99": _q(held_q, 0.99), "adm_wait_max": max(held_q) if held_q else None}
    for k, cn in agg.items():
        met[k + "_n"] = int(sum(cn.values()))
        met[k + "_unserved"] = int(cn.get(T.INF_KEY, 0))
        for qq, nm in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
            met[f"{k}_{nm}"] = _jsonable(T._pct(cn, qq))
        met[k + "_max"] = _jsonable(max(cn) / 100.0 if cn and max(cn) < T.INF_KEY else (math.inf if cn else None))
    return out, met


def _q(xs, qq):
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, math.ceil(qq * len(ys)) - 1))]


def _jsonable(v):
    if v is None:
        return None
    return "inf" if v == math.inf else v


def _num(v):
    return math.inf if v == "inf" else v


# ------------------------------------------------------------------------------------------------ data + runs
_CACHE = {}


def data(root, cache):
    if "tr" not in _CACHE:
        tr, units_u, meta, cnt = V.build("P8", root, cache)
        pm = json.load(open(os.path.join(root, T.TRACE_DIR, "derived/chutes_w1_analysis.json")))["per_model"]
        _CACHE["tr"] = tr
        _CACHE["cnt"] = cnt
        _CACHE["work"] = {c: r.get("work_tok", 0.0) for c, r in pm.items()}
        _CACHE["storm"] = storm_sets(tr)
    return _CACHE["tr"], _CACHE["cnt"], _CACHE["work"], _CACHE["storm"]


def setup(arm, variant, root, cache):
    cfg = cfg_for(arm, variant)
    tr, cnt, work, storm = data(root, cache)
    models = build_models(tr, cfg["prec"])
    census = load_census(root) if cfg["units"] != "none" else None
    units, unit_of = build_units(models, cfg["units"], census, work)
    tr2 = T.Trace.__new__(T.Trace)
    tr2.minute, tr2.mid, tr2.n, tr2.it, tr2.ot, tr2.explicit = tr.minute, tr.mid, tr.n, tr.it, tr.ot, None
    tr2.models = models
    live = None
    if cfg["placement"] == "COACT" or cfg["box_coact"]:
        key = ("live", cfg["units"], cfg["prec"])
        if key not in _CACHE:
            _CACHE[key] = train_live(tr2, unit_of, len(units))
        live = _CACHE[key]
    return cfg, tr2, units, unit_of, live, storm, cnt


def run_cell(arm, K, seed, arrivals, variant, root, cache):
    cfg, tr, units, unit_of, live, (storm, hour, R), cnt = setup(arm, variant, root, cache)
    bw = V.BURST_W if arrivals == "BURST" else None
    dkey = ("ded", arrivals)
    if dkey not in _CACHE:
        _CACHE[dkey] = V.simulate_dedicated_v2(_CACHE["tr"], bw)
    ded = _CACHE[dkey]
    row = {"arm": arm, "K": K, "seed": seed, "arrivals": arrivals, "variant": variant, "cfg": cfg,
           "n_units": len(units), "n_models": len(tr.models)}
    t = time.time()
    try:
        plc = place(units, K, cfg, live)
    except ValueError as e:
        row.update(infeasible=str(e), elapsed_s=round(time.time() - t, 1))
        return row
    out, met = simulate(tr, units, unit_of, plc, K, cfg, seed, bw, storm=storm, storm_hour=hour)
    row.update(met)
    row["slo"] = {str(x): T.score(out, ded, x) for x in T.SLOS}
    row["heldout_req"] = int(sum(sum(c.values()) for c in out))
    row["elapsed_s"] = round(time.time() - t, 1)
    return row


def sync_probe(arm, root, cache, K=K_PRIMARY, box=0, variant="base"):
    """prereg §4 SYNC: everything asleep; every engine homed on box `box` gets one 1-token request at the same instant."""
    cfg, tr, units, unit_of, live, _, _ = setup(arm, variant, root, cache)
    plc = place(units, K, cfg, live)
    kind, assign = plc[0], plc[1]
    cards = set(range(box * BOX_G, box * BOX_G + BOX_G))
    if kind == "card":
        sel = [u for u in range(len(units)) if assign[u] in cards]
        sub_plc = (kind, [assign[u] - box * BOX_G for u in sel])
    else:
        sel = [u for u in range(len(units)) if assign[u] == box]
        sub_plc = (kind, [0 for _ in sel], [plc[2][u] - box * BOX_G if plc[2][u] >= 0 else -1 for u in sel])
    sub_models = []
    for u in sel:
        U = units[u]
        cid, tb, ab, prec, _, _ = tr.models[U["rep"]]
        sub_models.append((cid, tb, ab, prec, U["w"], U["f"]))
    sub_units = [dict(cid=m[0], rep=k, members=[k], tb=m[1], prec=m[3], w=m[4], f=m[5], shape=units[u]["shape"])
                 for k, (m, u) in enumerate(zip(sub_models, sel))]
    reqs = [(1.0, k, 1.0, 0.0) for k in range(len(sel))]
    etr = T.Trace.explicit_cells(sub_models, reqs)
    out, met = simulate(etr, sub_units, list(range(len(sel))), sub_plc, BOX_G, cfg, 0,
                        score_from=0, min0=0, storm={0}, storm_hour={0})
    vals = []
    for cn in out:
        for k, v in cn.items():
            vals += [math.inf if k >= T.INF_KEY else k / 100.0] * v
    vals.sort()
    fin = [x for x in vals if x != math.inf]
    return {"arm": arm, "variant": variant, "K": K, "box": box, "engines": len(sel), "requests": len(vals),
            "unserved": len(vals) - len(fin), "p50": _jsonable(_q(vals, 0.5)), "p99": _jsonable(_q(vals, 0.99)),
            "max": _jsonable(vals[-1] if vals else None), "last_finite": fin[-1] if fin else None,
            "wakes": met["stats"].get("wakes", 0), "max_h2d_jobs_per_link": met["max_h2d_jobs_per_link"]}


# ------------------------------------------------------------------------------------------------ summarize (prereg §5/§6)
def _comp(r):
    """Composite C = (p99 storm WTFT [THIN -> max], unserved share of storm WA)."""
    n = r.get("storm_wa_n", 0) or 0
    p = _num(r.get("storm_wa_p99")) if n >= 20 else _num(r.get("storm_wa_max"))
    if p is None:
        p = 0.0
    us = (r.get("storm_wa_unserved", 0) / n) if n else 0.0
    return (p, us)


def _verdict(lev, base):
    """prereg §5: BETTER iff worst seed of lev beats best seed of base on C AND guard; WORSE iff best of lev worse than
    worst of base; else UNRESOLVED."""
    cl = [_comp(r) for r in lev]
    cb = [_comp(r) for r in base]
    gl = max(_num(r.get("storm_all_p99")) for r in lev)
    gb = max(_num(r.get("storm_all_p99")) for r in base)
    guard = gl <= 1.05 * gb if gb != math.inf else True
    if max(cl) < min(cb):
        return ("BETTER" if guard else "BETTER-ON-WAKES-ONLY"), guard
    if min(cl) > max(cb):
        return "WORSE", guard
    return "UNRESOLVED", guard


def summarize(d, root):
    rows = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            rows += T._rows(os.path.join(d, f))
    trace_rows = [r for r in rows if "seed" in r and "infeasible" not in r]
    out = {"n_rows": len(rows), "instrument": {}, "primary": {}, "verdicts": {}, "ranking": [], "stacks": {},
           "min_k": {}, "secondary": {}, "sensitivity": {}, "sync": {}, "gpu_arm": None}
    # I1: L0_V2 vs rows
    v2rows = {(r["K"], r["seed"]): r for r in T._rows(os.path.join(root, V2_ROWS))}
    i1 = []
    for r in trace_rows:
        if r["arm"] == "L0_V2" and r["arrivals"] == "POISSON" and r["variant"] == "base" and r["K"] in (96, 128):
            ref = v2rows.get((r["K"], r["seed"]))
            same = ref is not None and all(r["stats"].get(k, 0) == ref["stats"].get(k, 0)
                                           for k in ("wakes", "evictions", "unserved_req")) and all(
                r["slo"][s]["M_frac"] == ref["slo"][s]["M_frac"] and r["slo"][s]["R_ratio"] == ref["slo"][s]["R_ratio"]
                for s in ref["slo"])
            i1.append({"K": r["K"], "seed": r["seed"], "match": bool(same)})
    out["instrument"]["I1"] = {"cells": i1, "pass": bool(i1) and len(i1) == 6 and all(x["match"] for x in i1)}
    cons = sorted({r["heldout_req"] for r in trace_rows})
    storm_n = collections.defaultdict(set)
    for r in trace_rows:
        storm_n[(r["seed"], r["arrivals"])].add(r["storm_all_n"])
    out["instrument"]["I2"] = {"heldout_req_values": cons, "storm_n_per_seed": {str(k): sorted(v) for k, v in storm_n.items()},
                               "pass": cons == [HELDOUT_REQ] and all(len(v) == 1 for v in storm_n.values())}

    def cell(arm, K, arrivals="POISSON", variant="base"):
        return sorted((r for r in trace_rows if r["arm"] == arm and r["K"] == K and r["arrivals"] == arrivals
                       and r["variant"] == variant), key=lambda r: r["seed"])

    l0 = cell("L0", K_PRIMARY)
    kp = K_PRIMARY
    if l0 and any((r.get("storm_wa_n") or 0) < 20 for r in l0):
        kp = 64
        l0 = cell("L0", kp)
    out["primary"] = {"K": kp, "fallback_used": kp != K_PRIMARY,
                      "L0": [{"seed": r["seed"], "C": _comp(r), "storm_wa_n": r["storm_wa_n"],
                              "storm_all_p99": r["storm_all_p99"]} for r in l0]}

    def brief(rs):
        return {"seeds": [r["seed"] for r in rs], "C": [list(_comp(r)) for r in rs],
                "storm_wa_n": [r["storm_wa_n"] for r in rs], "storm_wa_p99": [r["storm_wa_p99"] for r in rs],
                "storm_all_p99": [r["storm_all_p99"] for r in rs], "week_all_p95": [r["week_all_p95"] for r in rs],
                "week_wa_share": [round(r["week_wa_n"] / r["week_all_n"], 6) if r["week_all_n"] else None for r in rs],
                "storm_wa_share": [round(r["storm_wa_n"] / r["storm_all_n"], 6) if r["storm_all_n"] else None for r in rs],
                "wakes": [r["stats"].get("wakes", 0) for r in rs], "unserved": [r["stats"].get("unserved_req", 0) for r in rs],
                "svc_p99": [r["svc_p99"] for r in rs], "max_h2d_jobs_per_link": [r["max_h2d_jobs_per_link"] for r in rs],
                "M_2": [round(r["slo"]["2.0"]["M_frac"], 4) for r in rs],
                "M_10": [round(r["slo"]["10.0"]["M_frac"], 4) for r in rs]}

    for arm in ("L0_V2", "L0") + LEVERS + STACKS:
        rs = cell(arm, kp)
        inf = [r for r in rows if r.get("arm") == arm and r.get("K") == kp and r.get("arrivals") == "POISSON"
               and r.get("variant") == "base" and "infeasible" in r]
        if not rs:
            out["verdicts"][arm] = {"verdict": "NOT RUN" if not inf else "INFEASIBLE", "detail": inf[0]["infeasible"] if inf else None}
            continue
        v = {"brief": brief(rs)}
        if arm not in ("L0", "L0_V2") and len(rs) == 3 and len(l0) == 3:
            v["verdict"], v["guard_ok"] = _verdict(rs, l0)
        out["verdicts"][arm] = v
    better = []
    for arm in LEVERS:
        v = out["verdicts"].get(arm, {})
        if v.get("verdict") == "BETTER":
            rs = cell(arm, kp)
            cs = sorted(_comp(r) for r in rs)
            better.append((cs[1], arm, cs[0], cs[-1]))
    better.sort()
    l0med = sorted(_comp(r) for r in l0)[1] if len(l0) == 3 else None
    rank = []
    for med, arm, lo, hi in better:
        mag = None
        if l0med and med[0] > 0:
            mag = ("inf -> %.2f s" % med[0]) if l0med[0] == math.inf else round(l0med[0] / med[0], 2)
        rank.append({"arm": arm, "median_C": list(med), "seed_range": [list(lo), list(hi)], "magnitude": mag,
                     "lossy": arm in LOSSY})
    for a in range(len(rank)):
        rank[a]["co_ranked_with"] = [rank[b]["arm"] for b in range(len(rank)) if b != a and not (
            tuple(rank[a]["seed_range"][1]) < tuple(rank[b]["seed_range"][0]) or
            tuple(rank[b]["seed_range"][1]) < tuple(rank[a]["seed_range"][0]))]
    out["ranking"] = rank
    for sname in STACKS:
        v = out["verdicts"].get(sname, {})
        sres = {"verdict_vs_L0": v.get("verdict"), "lossy": sname in LOSSY}
        if v.get("verdict") == "BETTER":
            parts = [p for p in STACK_PARTS[sname] if out["verdicts"].get(p, {}).get("verdict") == "BETTER"]
            if parts:
                bestp = min(parts, key=lambda p: sorted(_comp(r) for r in cell(p, kp))[1])
                sres["best_part"] = bestp
                sres["vs_best_part"] = _verdict(cell(sname, kp), cell(bestp, kp))[0]
                sres["adds_value"] = sres["vs_best_part"] == "BETTER"
        out["stacks"][sname] = sres
    okst = [s for s in STACKS if out["stacks"][s].get("verdict_vs_L0") == "BETTER"]
    med = lambda a: sorted(_comp(r) for r in cell(a, kp))[1]
    out["best_stack"] = min(okst, key=med) if okst else None
    lossless = [s for s in okst if s not in LOSSY]
    out["best_lossless_stack"] = min(lossless, key=med) if lossless else None
    # min-K (cards)
    for arm in ("L0_V2", "L0") + LEVERS + STACKS:
        res = {}
        for slo in ("2.0", "10.0"):
            nom = resv = None
            for K in K_GRID:
                rs = cell(arm, K)
                if len(rs) < 3:
                    continue
                if nom is None and all(r["slo"][slo]["M_frac"] is not None and r["slo"][slo]["M_frac"] >= 0.95
                                       and r["slo"][slo]["R_ratio"] >= 0.99 for r in rs):
                    nom = K
                if resv is None and all(r["slo"][slo]["M_frac"] is not None and r["slo"][slo]["M_frac"] >= 0.96
                                        and r["slo"][slo]["R_ratio"] >= 0.99 for r in rs):
                    resv = K
            res[slo] = {"nominal": nom, "resolved": resv,
                        "verdict": "NOT SERVED <= 192" if nom is None else ("RESOLVED" if nom == resv else "UNRESOLVED (on-bar)")}
        out["min_k"][arm] = res
    for arm in ("L0",) + LEVERS + STACKS:
        sec = {}
        for K in (64, 128):
            rs = cell(arm, K)
            if rs:
                sec[f"K{K}"] = brief(rs)
                if arm != "L0" and len(rs) == 3 and len(cell("L0", K)) == 3:
                    sec[f"K{K}"]["verdict"] = _verdict(rs, cell("L0", K))[0]
        for K in (64, 96, 128):
            rs = cell(arm, K, "BURST")
            if rs:
                sec[f"BURST_K{K}"] = brief(rs)
                if arm != "L0" and len(rs) == 3 and len(cell("L0", K, "BURST")) == 3:
                    sec[f"BURST_K{K}"]["verdict"] = _verdict(rs, cell("L0", K, "BURST"))[0]
        out["secondary"][arm] = sec
    for arm in SENS_ARMS:
        for var in SENS_VARIANTS:
            rs = cell(arm, K_PRIMARY, "POISSON", var)
            if rs:
                b = brief(rs)
                l0v = cell("L0", K_PRIMARY, "POISSON", var)
                if arm != "L0" and len(rs) == 3 and len(l0v) == 3:
                    b["verdict_vs_L0_same_variant"] = _verdict(rs, l0v)[0]
                out["sensitivity"][f"{arm}|{var}"] = b
    for r in rows:
        if "box" in r and "engines" in r:
            out["sync"][f'{r["arm"]}|{r.get("variant", "base")}'] = r
    cands = []
    for arm in LEVERS:
        v = out["verdicts"].get(arm, {})
        prem, measured, by2699 = PREMISES[arm]
        if v.get("verdict") == "BETTER" and not measured and not by2699:
            cands.append((sorted(_comp(r) for r in cell(arm, kp))[1], arm, prem))
    cands.sort()
    out["gpu_arm"] = ({"arm": cands[0][1], "premise": cands[0][2], "median_C": list(cands[0][0]),
                       "others": [c[1] for c in cands[1:]]} if cands else None)
    out["premises"] = {a: {"premise": p, "measured": m, "covered_by_2699": c} for a, (p, m, c) in PREMISES.items()}
    out["fail_clause"] = "NO LEVER RESOLVED" if not any(r["arm"] for r in rank) else None
    return out


# ------------------------------------------------------------------------------------------------ storm CAPACITY (amendment 2)
# Amendment 2: how many wakes/s can a box SUSTAIN under a p99
# wake-to-first-token target, and which levers raise that ceiling. Box-level discrete-event model, independent of the trace.
CAP_GB = {"FP8": 8.0 * 1.07 * GIB, "INT4": 8.0 * 0.60 * GIB, "DELTA": DELTA_GB_PER_B * 8.0}
CAP_DEMOTE = {"FP8": True, "INT4": True, "DELTA": False}     # dropping an adapter moves no bytes
CAP_SLOTS = {"FP8": 2, "INT4": 4, "DELTA": 16}
DRAM = {"DESKTOP": (1, 0.75 * 2 * 5600 * 8 / 1000.0), "WORKSTATION": (1, 0.75 * 8 * 4800 * 8 / 1000.0),
        "SERVER": (2, 0.75 * 12 * 4800 * 8 / 1000.0)}
XGMI_GBPS = 0.75 * 256.0
NVL_GBPS = 300.0
PREFILL_S = 1000.0 * 8.0 / ACTIVE_REF_B / C_REF            # 0.137 s: a 1,000-token prompt on an 8 B at C_REF
CAP_TARGETS = (1.0, 2.0)
CAP_G = (1, 2, 4, 8)
CAP_WARM, CAP_N = 2000, 20000
CAP_LEVERS = {
    "BASE": {}, "L7": dict(zero_demote=True), "L5": dict(phi=PHI), "PCIE5": dict(h2d=44.0, d2h=46.3),
    "ANY2": dict(copies=2), "ANYG": dict(copies=-1), "P2P_NVL": dict(p2p="nvl", rho=0.5),
    "P2P_PCIE": dict(p2p="pcie", rho=0.5), "HOT": dict(hot=True), "NUMA_BAD": dict(numa_bad=True),
    "STACK_LOSSLESS": dict(zero_demote=True, phi=PHI, copies=2),
    "STACK_LOSSLESS_HOT": dict(zero_demote=True, phi=PHI, copies=2, hot=True),
    "STACK_LOSSLESS_NUMA_BAD": dict(zero_demote=True, phi=PHI, copies=2, numa_bad=True),
    # amendment 2b: the wake scheduler — one job in service per link direction, FIFO
    "SCHED": dict(fifo=True),
    "STACK_LOSSLESS_SCHED": dict(fifo=True, zero_demote=True, phi=PHI, copies=2),
    "STACK_LOSSLESS_SCHED_HOT": dict(fifo=True, zero_demote=True, phi=PHI, copies=2, hot=True),
    "STACK_LOSSLESS_SCHED_NUMA_BAD": dict(fifo=True, zero_demote=True, phi=PHI, copies=2, numa_bad=True),
}


def cap_cfg(G, cls, dram, lever="BASE", **over):
    if cls not in CAP_GB or dram not in DRAM or lever not in CAP_LEVERS:
        raise ValueError(f"unknown capacity cell {G}/{cls}/{dram}/{lever}")
    if not isinstance(G, int) or G < 1:   # review: G <= 0 would divide by zero downstream
        raise ValueError(f"GPUs per box must be an int >= 1, got {G!r}")
    nodes, bw = DRAM[dram]
    c = dict(G=G, cls=cls, dram=dram, lever=lever, h2d=H2D_GBPS, d2h=D2H_GBPS, beta=BETA_BUSY, phi=0.0,
             zero_demote=False, copies=1, p2p=None, rho=0.0, hot=False, numa_bad=False, fifo=False, nodes=min(nodes, G),
             dram_gbps=bw, slots=CAP_SLOTS[cls], payload=CAP_GB[cls], demote=CAP_GB[cls] if CAP_DEMOTE[cls] else 0.0,
             prefill=PREFILL_S, fixed=FIXED_S)
    c.update(CAP_LEVERS[lever])
    c.update(over)
    if c["copies"] == -1:
        c["copies"] = G
    c["copies"] = max(1, min(c["copies"], G))
    if c["zero_demote"]:
        c["demote"] = 0.0
    return c


def _node(cfg, card):
    return 0 if (cfg["numa_bad"] or cfg["nodes"] <= 1) else (card * cfg["nodes"]) // cfg["G"]


def cap_simulate(cfg, times, cards, coins, srcs, copysets=None):
    """One box. times[i] = arrival (s), cards[i] = home card, coins[i] in [0,1) for P2P, srcs[i] = P2P source card,
    copysets[i] = candidate cards (any-card,: each is its own sleeping copy). Returns the WTFT list in arrival order.
    Resources: ('h',c) / ('d',c) PCIe directions; ('dram',n) shared by H2D reads + D2H writes of node n; ('xgmi',)
    for remote-node traffic under NUMA_BAD; ('nvl',c) NVLink ingress. Max-min fair progressive filling per event."""
    G = cfg["G"]
    n = len(times)
    wtft = [None] * n
    inflight = [0] * G
    queues = [collections.deque() for _ in range(G)]
    jobs = {}               # jid -> [wake, phase, remaining, resources, t_phase_start]
    lq = collections.defaultdict(list)    # SCHED: primary link-direction -> FIFO of job ids (one in service)
    nxt = [0]
    t = 0.0
    i_arr = 0
    pagein_start = {}

    def res_for(w, phase, c):
        if phase == "d":
            r = [("d", c), ("dram", _node(cfg, c))]
        elif cfg["p2p"] and coins[w] < cfg["rho"]:
            r = [("nvl", c)] if cfg["p2p"] == "nvl" else [("h", c), ("d", srcs[w])]
        else:
            r = [("h", c), ("dram", _node(cfg, c))]
        if cfg["numa_bad"] and any(x[0] == "dram" for x in r) and c >= G // 2 and G > 1:
            r.append(("xgmi",))
        return tuple(r)

    def capacity(r):
        if r[0] == "h":
            return cfg["h2d"]
        if r[0] == "d":
            return cfg["d2h"]
        if r[0] == "dram":
            return cfg["dram_gbps"]
        if r[0] == "xgmi":
            return XGMI_GBPS
        return NVL_GBPS

    def primary(res):
        return next((r for r in res if r[0] in ("h", "d")), None)

    def rates():
        if cfg["fifo"]:
            act = {j for j, jb in jobs.items() if primary(jb[3]) is None or lq[primary(jb[3])][0] == j}
        else:
            act = set(jobs)
        users = collections.defaultdict(list)
        for j in act:
            for r in jobs[j][3]:
                users[r].append(j)
        rem = {r: capacity(r) for r in users}
        rate = {j: 0.0 for j in jobs}
        live = set(act)
        while live:
            inc = min(rem[r] / sum(1 for j in us if j in live) for r, us in users.items() if any(j in live for j in us))
            for j in live:
                rate[j] += inc
            for r, us in users.items():
                k = sum(1 for j in us if j in live)
                rem[r] -= inc * k
            sat = {r for r in users if rem[r] <= 1e-9 * capacity(r) and any(j in live for j in users[r])}
            frz = {j for j in live if any(r in sat for r in jobs[j][3])}
            if not frz:
                break
            live -= frz
        return rate

    def add_job(w, phase, c, gb, now):
        nxt[0] += 1
        jobs[nxt[0]] = [w, phase, gb, res_for(w, phase, c), now]
        pr = primary(jobs[nxt[0]][3])
        if pr is not None:
            lq[pr].append(nxt[0])
        if phase == "h":
            pagein_start[w] = now

    def start(w, c, now):
        inflight[c] += 1
        if cfg["demote"] > 0.0:
            add_job(w, "d", c, cfg["demote"], now)
        else:
            add_job(w, "h", c, cfg["payload"] * cfg["beta"], now)

    def finish(j, now):
        w, phase, _, res, _ = jobs.pop(j)
        pr = primary(res)
        if pr is not None:
            lq[pr].remove(j)
        c = next(r[1] for r in res if r[0] in ("h", "nvl")) if phase == "h" else next(r[1] for r in res if r[0] == "d")
        if phase == "d":
            add_job(w, "h", c, cfg["payload"] * cfg["beta"], now)
            return
        pagein_start.pop(w)
        # L5: prefill can overlap only bytes actually arriving; a FIFO wait for the link hides nothing. Conservative bound:
        # the uncontended page-in time (a contended copy takes longer, so this never over-credits the overlap).
        hidden = cfg["phi"] * min(cfg["prefill"], cfg["payload"] * cfg["beta"] / cfg["h2d"])
        wtft[w] = now + cfg["fixed"] + (cfg["prefill"] - hidden) - times[w]
        inflight[c] -= 1
        if queues[c]:
            start(queues[c].popleft(), c, now)

    wake_card = [None] * n
    while i_arr < n or jobs:
        rt = rates() if jobs else {}
        tc, jmin = (math.inf, None)
        if rt:
            tc, jmin = min(((t + jobs[j][2] / r) if r > 0 else math.inf, j) for j, r in rt.items())
        ta = times[i_arr] if i_arr < n else math.inf
        tn = min(tc, ta)
        if tn == math.inf:
            raise RuntimeError("capacity sim stalled (a job with zero rate)")
        for j, r in rt.items():
            jobs[j][2] -= r * (tn - t)
        t = tn
        if tc <= ta:
            done = sorted(j for j in list(jobs) if j == jmin or jobs[j][2] <= 1e-12)
            for j in done:
                if j in jobs:
                    finish(j, t)
        else:
            w = i_arr
            i_arr += 1
            cand = copysets[w] if copysets is not None else [cards[w]]
            c = min(cand, key=lambda x: (inflight[x] + len(queues[x]), x))
            wake_card[w] = c
            if inflight[c] < cfg["slots"]:
                start(w, c, t)
            else:
                queues[c].append(w)
    return wtft


def cap_streams(cfg, seed, n=CAP_WARM + CAP_N):
    """Common random numbers per seed: unit exponentials, home cards (uniform or HOT), P2P coins + sources, copy sets."""
    G = cfg["G"]
    rng = np.random.default_rng([seed, 2705])
    unit = np.cumsum(rng.exponential(1.0, n))
    r2 = np.random.default_rng([seed, 2705, 2])
    if cfg["hot"] and G > 1:
        hot = r2.random(n) < 0.5
        other = r2.integers(1, G, n)
        cards = np.where(hot, 0, other)
    else:
        cards = r2.integers(0, G, n)
    r3 = np.random.default_rng([seed, 2705, 3])
    coins = r3.random(n)
    srcs = np.array([(c + 1 + int(x)) % G for c, x in zip(cards, r3.integers(0, max(1, G - 1), n))]) if G > 1 else cards
    copysets = None
    if cfg["copies"] > 1:
        r4 = np.random.default_rng([seed, 2705, 4])
        copysets = []
        for c in cards:
            others = [x for x in range(G) if x != c]
            pick = list(r4.choice(others, size=cfg["copies"] - 1, replace=False)) if others else []
            copysets.append(sorted([int(c)] + [int(x) for x in pick]))
    return unit, [int(c) for c in cards], list(coins), [int(s) for s in srcs], copysets


def cap_p99(cfg, lam, seed, streams=None):
    unit, cards, coins, srcs, copysets = streams or cap_streams(cfg, seed)
    times = list(unit / lam)
    w = cap_simulate(cfg, times, cards, coins, srcs, copysets)[CAP_WARM:]
    return _q(w, 0.99)


def cap_bound(cfg):
    """λ_bw: min over resources of capacity / per-unit-λ load (uncontended service for slots). Returns (λ, binding)."""
    G = cfg["G"]
    share = [0.5 if (cfg["hot"] and G > 1 and c == 0) else ((0.5 / (G - 1)) if (cfg["hot"] and G > 1) else 1.0 / G)
             for c in range(G)]
    if cfg["copies"] > 1:
        share = [1.0 / G] * G
    h = cfg["payload"] * cfg["beta"]
    frac_dram = 1.0 - (cfg["rho"] if cfg["p2p"] else 0.0)
    loads = collections.Counter()
    for c in range(G):
        loads[("h", c)] += share[c] * h * (1.0 if cfg["p2p"] != "nvl" else (1.0 - cfg["rho"]))
        loads[("d", c)] += share[c] * cfg["demote"]
        nd = _node(cfg, c)
        loads[("dram", nd)] += share[c] * (h * frac_dram + cfg["demote"])
        svc = cfg["demote"] / cfg["d2h"] + h / cfg["h2d"]
        loads[("slot", c)] += share[c] * svc / cfg["slots"]
    caps = {"h": cfg["h2d"], "d": cfg["d2h"], "dram": cfg["dram_gbps"], "slot": 1.0}
    best = (math.inf, None)
    for r, ld in loads.items():
        if ld > 0:
            v = caps[r[0]] / ld
            if v < best[0]:
                best = (v, r[0])
    return best


def cap_search(cfg, target, seeds=SEEDS, rel=0.01):
    """prereg amendment 2 §3: max λ (wakes/s per box) with p99 WTFT <= target, per seed by bisection (CRN)."""
    lam_bw, binding = cap_bound(cfg)
    out = []
    for sd in seeds:
        st = cap_streams(cfg, sd)
        lo_l = lam_bw * 1e-4
        if cap_p99(cfg, lo_l, sd, st) > target:
            out.append(0.0)
            continue
        lo, hi = lo_l, lam_bw
        if cap_p99(cfg, hi, sd, st) <= target:
            out.append(hi)
            continue
        while (hi - lo) > rel * lo:
            mid = 0.5 * (lo + hi)
            if cap_p99(cfg, mid, sd, st) <= target:
                lo = mid
            else:
                hi = mid
        out.append(lo)
    mn = min(out)
    verdict = "INFEASIBLE" if mn == 0.0 else ("RESOLVED" if (max(out) - mn) <= 0.10 * mn else "ON-BAR")
    return {"lambda_min": round(mn, 3), "lambda_seeds": [round(x, 3) for x in out], "verdict": verdict,
            "lambda_bw": round(lam_bw, 3), "binding": binding}


def flash_k(cfg, target, kmax=256):
    """Descriptive (added after amendment 2, before any capacity number): the most SIMULTANEOUS wakes one box absorbs with
    every WTFT <= target, homes round-robin over the cards (copy sets: the next cards round-robin)."""
    G = cfg["G"]
    best = 0
    for k in range(1, kmax + 1):
        cards = [i % G for i in range(k)]
        copysets = ([sorted({(c + j) % G for j in range(cfg["copies"])}) for c in cards] if cfg["copies"] > 1 else None)
        coins = [(i % 2) * 0.99 for i in range(k)]        # alternate replica / no replica at rho = 0.5
        srcs = [(c + 1) % G for c in cards]
        w = cap_simulate(cfg, [0.0] * k, cards, coins, srcs, copysets)
        if max(w) <= target + 1e-12:
            best = k
        else:
            break
    return best


def cap_cell(G, cls, dram, lever):
    cfg = cap_cfg(G, cls, dram, lever)
    row = {"G": G, "cls": cls, "dram": dram, "lever": lever, "dram_gbps": round(cfg["dram_gbps"], 2),
           "single_wtft": round(cap_simulate(cfg, [0.0], [0], [0.99], [1 % G])[0], 6),
           "host_ram_copies": cfg["copies"], "gpu_residual_gib_per_model": cfg["copies"] * S_GIB}
    t0 = time.time()
    for tg in CAP_TARGETS:
        row[f"p99_le_{tg:g}s"] = cap_search(cfg, tg)
        row[f"flash_k_{tg:g}s"] = flash_k(cfg, tg)
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def trace_demand(root, cache):
    """prereg amendment 2 §3(a): D1(t) = tail models active in minute t and not in t-1; R5(t) = re-activations after
    >= 5 idle minutes (prereg §1). Held-out week. Read-only."""
    tr, cnt, work, (storm, hour, R) = data(root, cache)
    mid = np.asarray(tr.mid, np.int64)
    mi = np.asarray(tr.minute, np.int64)
    o = np.lexsort((mi, mid))
    m_s, t_s = mid[o], mi[o]
    prev = np.full(len(o), -10 ** 9, np.int64)
    prev[1:] = t_s[:-1]
    newm = np.ones(len(o), bool)
    newm[1:] = m_s[1:] != m_s[:-1]
    prev[newm] = -10 ** 9
    d1 = (t_s - prev) > 1
    hk = (t_s >= HELD) & (t_s < HELD + 10080)
    D1 = np.bincount((t_s[d1 & hk] - HELD), minlength=10080)[:10080]

    def st(a):
        return {"max": int(a.max()), "p999": float(np.percentile(a, 99.9)), "p99": float(np.percentile(a, 99)),
                "mean": round(float(a.mean()), 3), "max_per_s_over_60s": round(a.max() / 60.0, 3),
                "max_per_s_in_6s_burst": round(a.max() / 6.0, 3)}
    return {"D1_wakes_if_awake_one_minute": st(D1), "R5_reactivations": st(R)}


def burst_windows(times, windows=(1.0, 10.0, 60.0)):
    ts = sorted(times)
    out = {}
    for wdw in windows:
        best, j = 0, 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > wdw:
                j += 1
            best = max(best, i - j + 1)
        out[f"{wdw:g}s"] = best
    return out


def run_bursts(arm, K, seed, root, cache):
    """prereg amendment 2 §3(b): the pool's own wake admissions in the held-out week, per box of 8 cards and fleet."""
    cfg, tr, units, unit_of, live, (storm, hour, R), cnt = setup(arm, "base", root, cache)
    plc = place(units, K, cfg, live)
    adm = []
    simulate(tr, units, unit_of, plc, K, cfg, seed, None, storm=storm, storm_hour=hour, collect=adm)
    t_held = (HELD - T.MIN0) * 60.0
    adm = [(t, c) for t, c in adm if t >= t_held]
    per_box = collections.defaultdict(list)
    for t, c in adm:
        per_box[c // BOX_G].append(t)
    boxes = {b: burst_windows(ts) for b, ts in per_box.items()}
    worst = {w: max((v[w] for v in boxes.values()), default=0) for w in ("1s", "10s", "60s")}
    return {"arm": arm, "K": K, "seed": seed, "heldout_wakes": len(adm), "fleet": burst_windows([t for t, _ in adm]),
            "worst_box": worst, "boxes": len(boxes)}


# ------------------------------------------------------------------------------------------------ self-test
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

    mk = lambda cid, b, prec="fp8": (cid, b, b, prec) + T.footprint(b, prec)
    sing = lambda models: build_units(models, "none")

    def run(models, reqs, homes, K, cfg_over, units=None, unit_of=None, kind="card", card_gib=T.CARD_GIB, box_g=BOX_G,
            pins=None):
        cfg = dict(BASE); cfg.update(cfg_over)
        tr = T.Trace.explicit_cells(models, reqs)
        if units is None:
            units, unit_of = sing(models)
        plc = (kind, homes) if pins is None else (kind, homes, pins)
        return simulate(tr, units, unit_of, plc, K, cfg, 0, score_from=0, min0=0, storm={0, 1, 2},
                        storm_hour={0, 1, 2}, card_gib=card_gib, box_g=box_g)

    W8 = 8.56 * GIB / 22.6           # 0.406692 s copy of an 8B fp8 at 22.6 GB/s
    big = 1000.0
    m3 = [mk("A", 8.0), mk("B", 8.0), mk("D", 8.0)]
    # G1 single wake
    o, met = run(m3[:1], [(1.0, 0, 0, 0)], [0], 1, dict(beta=1.0))
    chk("G1 single wake", vals(o[0]), [0.43], 0.0051)
    chk("G1 conservation", sum(len(vals(c)) for c in o), 1)
    # G2 three simultaneous wakes on one link: PS vs FIFO; law last >= N*single - (N-1)*fixed
    reqs = [(1.0, k, 0, 0) for k in range(3)]
    o, met = run(m3, reqs, [0, 0, 0], 1, dict(beta=1.0), card_gib=big)
    ps = sorted(x for c in o for x in vals(c))
    chk("G2 PS all at 3x", ps, [0.02 + 3 * W8] * 3, 0.0051)
    o, met = run(m3, reqs, [0, 0, 0], 1, dict(beta=1.0, fifo_cap=1), card_gib=big)
    ff = sorted(x for c in o for x in vals(c))
    chk("G2 FIFO staggered", ff, [0.02 + W8, 0.02 + 2 * W8, 0.02 + 3 * W8], 0.0051)
    single = 0.02 + W8
    chk("G2 law PS", max(ps) >= 3 * single - 2 * 0.02 - 0.0051, True)
    chk("G2 law FIFO", max(ff) >= 3 * single - 2 * 0.02 - 0.0051, True)
    chk("G2 max concurrent H2D", met["max_h2d_jobs_per_link"], 3)
    # G3 two wakes on two cards: direct overlap vs switch-shared uplink
    o, met = run(m3[:2], [(1.0, 0, 0, 0), (1.0, 1, 0, 0)], [0, 1], 2, dict(beta=1.0))
    chk("G3 direct", sorted(x for c in o for x in vals(c)), [0.02 + W8] * 2, 0.0051)
    o, met = run(m3[:2], [(1.0, 0, 0, 0), (1.0, 1, 0, 0)], [0, 1], 2, dict(beta=1.0, topo="switch"))
    chk("G3 switch", sorted(x for c in o for x in vals(c)), [0.02 + 2 * W8] * 2, 0.0051)
    # G4 no drop: A@0, B@5 awake+protected, D@10 -> L0 unserved, L4 admitted at 60
    reqs = [(0.0, 0, 0, 0), (5.0, 1, 0, 0), (10.0, 2, 0, 0)]
    o, met = run(m3, reqs, [0, 0, 0], 1, {})
    chk("G4 L0 D unserved", vals(o[2]), [math.inf])
    chk("G4 L0 B beta wake", vals(o[1]), [round(0.02 + 1.1 * W8, 2)], 0.0051)
    o, met = run(m3, reqs, [0, 0, 0], 1, dict(sched="queue", fifo_cap=1))
    dem = 8.56 * GIB / 23.8
    chk("G4 L4 D admitted at 60", vals(o[2]), [round(60.0 + dem + 1.1 * W8 + 0.02 - 10.0, 2)], 0.0051)
    chk("G4 L4 no unserved", met["stats"].get("unserved_req", 0), 0)
    chk("G4 L4 conservation", sum(len(vals(c)) for c in o), 3)
    o, met = run(m3, reqs, [0, 0, 0], 1, dict(sched="queue", fifo_cap=1, zero_demote=True))
    chk("G4 L4+L7", vals(o[2]), [round(60.0 + 1.1 * W8 + 0.02 - 10.0, 2)], 0.0051)
    # G5 forced admission at arrival + 60 s when card-mates keep protection alive
    reqs = [(0.0, 0, 0, 0), (5.0, 1, 0, 0), (10.0, 2, 0, 0), (30.0, 0, 0, 0), (35.0, 1, 0, 0), (60.0, 0, 0, 0),
            (65.0, 1, 0, 0)]
    o, met = run(m3, reqs, [0, 0, 0], 1, dict(sched="queue", fifo_cap=1))
    chk("G5 forced at 70", vals(o[2]), [round(70.0 + dem + 1.1 * W8 + 0.02 - 10.0, 2)], 0.0051)
    chk("G5 forced count", met["stats"].get("forced_admits", 0), 1)
    # G6 L5 streaming hides min(phi*pre, wait)
    it6 = 0.1 * C_REF * ACTIVE_REF_B / 8.0
    o, met = run(m3[:1], [(1.0, 0, it6, 0)], [0], 1, dict(beta=1.0))
    chk("G6 no stream", vals(o[0]), [round(0.02 + W8 + 0.1, 2)], 0.0051)
    o, met = run(m3[:1], [(1.0, 0, it6, 0)], [0], 1, dict(beta=1.0, phi=PHI))
    chk("G6 stream", vals(o[0]), [round(0.02 + W8 + 0.1 - 0.097, 2)], 0.0051)
    # G7 beta next to an awake engine
    o, met = run(m3[:2], [(1.0, 0, 0, 0), (10.0, 1, 0, 0)], [0, 0], 1, {})
    chk("G7 beta", vals(o[1]), [round(0.02 + 1.1 * W8, 2)], 0.0051)
    # G8 L0_V2 == tailpool_v2.simulate_prestaged (explicit + Poisson traces)
    m5 = [mk("A", 8.0), mk("B", 8.0), mk("D", 8.0), mk("E", 3.0), mk("F", 8.0)]
    reqs = [(0, 0, 50, 10), (3, 1, 20, 5), (7, 2, 40, 0), (61, 3, 10, 10), (70, 4, 10, 0), (100, 0, 5, 5),
            (130, 2, 1, 1), (135, 1, 9, 9), (200, 4, 3, 3), (260, 3, 2, 2), (262, 0, 7, 7)]
    tr = T.Trace.explicit_cells(m5, reqs)
    homes = [0, 0, 0, 1, 1]
    ref, rst = V.simulate_prestaged(tr, homes, 2, 0, card_gib=21.0, score_from_min=0, min0=0)
    u5, uo5 = sing(m5)
    cfg = dict(BASE); cfg.update(physics="v2")
    got, gm = simulate(tr, u5, uo5, ("card", homes), 2, cfg, 0, score_from=0, min0=0, card_gib=21.0)
    chk("G8 explicit counters", [dict(x) for x in got], [dict(x) for x in ref])
    chk("G8 explicit stats", {k: gm["stats"].get(k, 0) for k in ("wakes", "evictions", "unserved_req")},
        {k: rst.get(k, 0) for k in ("wakes", "evictions", "unserved_req")})
    tr = T.Trace([0, 0, 0, 1, 1, 2, 3, 3], [0, 1, 2, 0, 3, 4, 1, 2], [5, 3, 1, 40, 2, 7, 9, 1],
                 [100, 50, 20, 900, 30, 70, 80, 10], [50, 10, 5, 300, 9, 20, 30, 2], m5)
    for sd in (0, 7):
        ref, rst = V.simulate_prestaged(tr, homes, 2, sd, card_gib=21.0, score_from_min=0, min0=0)
        got, gm = simulate(tr, u5, uo5, ("card", homes), 2, cfg, sd, score_from=0, min0=0, card_gib=21.0)
        chk(f"G8 poisson seed {sd}", [dict(x) for x in got], [dict(x) for x in ref])
    # G9 COACT separates co-live pairs (a,c) and (b,d); BLIND does not
    m4 = [mk(x, 3.0) for x in "abcd"]
    u4, _ = sing(m4)
    live = np.zeros((4, 400), bool)
    live[0, 0:101] = live[2, 0:101] = True
    live[1, 200:301] = live[3, 200:301] = True
    chk("G9 COACT", coact_assign(utuples(u4), live, 2, 1000.0, card_gib=7.0), [0, 1, 1, 0])
    chk("G9 BLIND contrast", V.assign_homes(utuples(u4), [0.0] * 4, 2, 1000.0, S_GIB, card_gib=7.0, mode="BLIND"),
        [0, 1, 0, 1])
    # G10 duplicates share one unit: one sleeper, one wake
    md = [mk("A", 8.0), mk("A2", 8.0)]
    census = {"dup_sets_strict": [["r1", "r2"]], "repo_cids": {"r1": ["A"], "r2": ["A2"]}}
    ud, uod = build_units(md, "dup", census, {"A": 5.0, "A2": 1.0})
    chk("G10 one unit", (len(ud), uod), (1, [0, 0]))
    o, met = run(md, [(1.0, 0, 0, 0), (5.0, 1, 0, 0)], [0], 1, dict(beta=1.0), units=ud, unit_of=uod)
    chk("G10 one wake", met["stats"].get("wakes", 0), 1)
    chk("G10 second served awake", vals(o[1]), [0.0])
    cl = {"lineage_sets": [["r1", "r2"]], "repo_cids": {"r1": ["A"], "r2": ["A2"]}}
    ul, _ = build_units(md, "lin", cl, {"A": 5.0, "A2": 1.0})
    chk("G10 lin delta bytes", ul[0]["w"] - md[0][4], DELTA_GB_PER_B * 8.0 / GIB, 1e-9)
    # G11 L6 any card in the box serves what card binding cannot
    reqs = [(0.0, 0, 0, 0), (5.0, 1, 0, 0), (10.0, 2, 0, 0)]
    o, met = run(m3, reqs, [0, 0, 0], 2, {})
    chk("G11 L0 card-bound unserved", vals(o[2]), [math.inf])
    o, met = run(m3, reqs, [0, 0, 0], 2, dict(anycard=True), kind="box", box_g=2)
    chk("G11 L6 served", met["stats"].get("unserved_req", 0), 0)
    chk("G11 L6 D wake (beta: A awake on its card)", vals(o[2]), [round(0.02 + 1.1 * W8, 2)], 0.0051)
    # G13 amendment 1: a unit too big for a 3-shell card is pinned to the box's highest card (one shell)
    mb = [mk("a", 8.0), mk("b", 24.0)]
    ub, _ = sing(mb)
    pins = pin_cards(ub, [0, 0], 1, box_g=2)
    chk("G13 pin", pins, [-1, 1])
    o, met = run(mb, [(1.0, 0, 0, 0), (1.0, 1, 0, 0)], [0, 0], 2, dict(anycard=True, beta=1.0), kind="box", box_g=2,
                 pins=pins)
    chk("G13 big wakes on its pinned card", vals(o[1]), [round(0.02 + 24.0 * 1.07 * GIB / 22.6, 2)], 0.0051)
    chk("G13 floater on the other card", vals(o[0]), [round(0.02 + W8, 2)], 0.0051)
    chk("G13 no unserved", met["stats"].get("unserved_req", 0), 0)
    # G14 link-engine re-entrancy: two demotes finish at the same instant and each chains a page-in (FIFO cap 1);
    # grid run 1 crashed here with KeyError (a nested advance completed a job the outer loop then popped again)
    g = Group([0], 10.0, 10.0, None, None, 0.0, 1)
    fin = []
    chain = lambda tc: g.add("h", 0, 10.0, 1.0, lambda t2: fin.append(round(t2, 9)), tc)
    g.add("d", 0, 5.0, 1.0, chain, 0.0)
    g.add("d", 0, 5.0, 1.0, chain, 0.0)
    try:
        g.advance(10.0)
        chk("G14 chained completions", fin, [2.0, 3.0], 1e-9)
    except KeyError as e:
        chk("G14 chained completions (no KeyError)", repr(e), "no exception")
    # ---- amendment 2 capacity goldens (hand-computed in (private report) §4)
    def capw(cfg, times, cards, coins=None, srcs=None, copysets=None):
        k = len(times)
        return cap_simulate(cfg, times, cards, coins or [0.99] * k, srcs or [(c + 1) % cfg["G"] for c in cards], copysets)
    c1 = cap_cfg(1, "FP8", "SERVER")
    chk("C1 single FP8 BASE", capw(c1, [0.0], [0]), [0.990983], 2e-6)
    chk("C1 L7", capw(cap_cfg(1, "FP8", "SERVER", "L7"), [0.0], [0]), [0.604797], 2e-6)
    chk("C1 L5", capw(cap_cfg(1, "FP8", "SERVER", "L5"), [0.0], [0]), [0.857670], 2e-6)
    chk("C1 L7+L5", capw(cap_cfg(1, "FP8", "SERVER", "L7", phi=PHI), [0.0], [0]), [0.471484], 2e-6)
    c2 = cap_cfg(4, "FP8", "DESKTOP", "L7")
    chk("C2 DRAM binds (desktop, 4 cards)", capw(c2, [0.0] * 4, [0, 1, 2, 3]), [0.759243] * 4, 2e-6)
    c3 = cap_cfg(1, "INT4", "SERVER", "L7")
    w3 = capw(c3, [0.0] * 3, [0, 0, 0])
    chk("C3 three INT4 on one link", w3, [0.910005] * 3, 2e-6)
    single3 = capw(c3, [0.0], [0])[0]
    chk("C3 law last >= N*single - (N-1)*(fixed+prefill)",
        max(w3) >= 3 * single3 - 2 * (FIXED_S + PREFILL_S) - 1e-9, True)
    c4 = cap_cfg(1, "FP8", "SERVER", "L7")
    chk("C4 slot cap A=2", sorted(capw(c4, [0.0] * 3, [0, 0, 0])), [1.052158, 1.052158, 1.499519], 2e-6)
    c5 = cap_cfg(2, "FP8", "SERVER", "L7", copies=2)
    chk("C5 any-card copies split", capw(c5, [0.0, 0.0], [0, 0], copysets=[[0, 1], [0, 1]]), [0.604797] * 2, 2e-6)
    c6 = cap_cfg(1, "FP8", "SERVER", "L7", p2p="nvl", rho=1.0, dram_gbps=1.0)
    chk("C6 NVLink P2P bypasses DRAM", capw(c6, [0.0], [0], coins=[0.0]), [0.191137], 2e-6)
    c7 = cap_cfg(2, "FP8", "SERVER", "L7")
    chk("C7 NUMA-local nodes do not share", capw(c7, [0.0, 0.0], [0, 1]), [0.604797] * 2, 2e-6)
    c7b = cap_cfg(2, "FP8", "SERVER", "L7", dram_gbps=22.6)
    chk("C7b same check binds when DRAM = one link (node-local)", capw(c7b, [0.0, 0.0], [0, 1]), [0.604797] * 2, 2e-6)
    c7c = cap_cfg(2, "FP8", "SERVER", "L7", dram_gbps=22.6, numa_bad=True)
    chk("C7c NUMA_BAD puts both on node 0 DRAM", capw(c7c, [0.0, 0.0], [0, 1]),
        [round(2 * 1.1 * CAP_GB["FP8"] / 22.6 + 0.02 + PREFILL_S, 6)] * 2, 2e-6)
    c8 = cap_cfg(1, "FP8", "SERVER")
    chk("C8 BASE lock-step convoy", capw(c8, [0.0, 0.0], [0, 0]), [1.824530] * 2, 2e-6)
    chk("C8 SCHED pipelines", capw(cap_cfg(1, "FP8", "SERVER", "SCHED"), [0.0, 0.0], [0, 0]), [0.990983, 1.438344], 2e-6)
    # G12 storm rule
    tm = [mk("x", 3.0), mk("y", 3.0), mk("z", 3.0)]
    # y@108 follows y@103 by exactly 5 minutes: 103 is inside t-5..t-1, so it is NOT a re-activation (boundary case)
    tr = T.Trace([100, 104, 110, 111, 103, 108, 105, 106], [0, 0, 0, 0, 1, 1, 2, 2], [1] * 8, [1] * 8, [1] * 8, tm)
    st_, hr_, R = storm_sets(tr, held=100, nmin=20, top=3, idle=5)
    chk("G12 R", [int(x) for x in R[:12]], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    chk("G12 top-3 ties earliest", sorted(st_), [100, 103, 105])
    # determinism
    r1 = run(m3, [(0.0, 0, 5, 5), (5.0, 1, 5, 5), (10.0, 2, 5, 5)], [0, 0, 0], 1, dict(sched="queue", fifo_cap=1))
    r2 = run(m3, [(0.0, 0, 5, 5), (5.0, 1, 5, 5), (10.0, 2, 5, 5)], [0, 0, 0], 1, dict(sched="queue", fifo_cap=1))
    chk("determinism", json.dumps([dict(x) for x in r1[0]]) == json.dumps([dict(x) for x in r2[0]]), True)
    for f in fails:
        print(f)
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n - len(fails)}/{n}")
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--ks", default=",".join(map(str, K_GRID)))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arrivals", default="POISSON", choices=["POISSON", "BURST"])
    ap.add_argument("--variant", default="base")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/tailpool/bins_w1.npz"))
    ap.add_argument("--out", help="append JSON lines here (WRITES)")
    ap.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--capacity", action="store_true", help="amendment 2: one capacity cell -> APPENDS a JSON line to --out")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--cls", default="FP8")
    ap.add_argument("--dram", default="SERVER")
    ap.add_argument("--lever", default="BASE")
    ap.add_argument("--trace-demand", action="store_true", help="amendment 2 §3(a), read-only JSON on stdout")
    ap.add_argument("--bursts", action="store_true", help="amendment 2 §3(b): APPENDS one JSON line per (K, seed) to --out")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    try:
        if a.summarize:
            print(json.dumps(summarize(a.summarize, a.root), indent=1, default=str))
            return 0
        if a.trace_demand:
            print(json.dumps(trace_demand(a.root, a.cache), indent=1))
            return 0
        if a.capacity:
            if not a.out:
                ap.print_help()
                return 2
            r = cap_cell(a.g, a.cls, a.dram, a.lever)
            line = json.dumps(r, sort_keys=True, default=str)
            with open(a.out, "a") as sink:
                print(line, flush=True); sink.write(line + "\n")
            return 0
        if a.bursts:
            if not a.out or not a.arm:
                ap.print_help()
                return 2
            cfg_for(a.arm)
            with open(a.out, "a") as sink:
                for K in [int(x) for x in a.ks.split(",")]:
                    for sd in [int(x) for x in a.seeds.split(",")]:
                        line = json.dumps(run_bursts(a.arm, K, sd, a.root, a.cache), sort_keys=True, default=str)
                        print(line, flush=True); sink.write(line + "\n"); sink.flush()
            return 0
        if a.describe:
            tr, cnt, work, (storm, hour, R) = data(a.root, a.cache)
            census = load_census(a.root)
            d = {"tail_counts": cnt, "storm_minutes": sorted(storm), "storm_R_min": int(min(R[m - HELD] for m in storm)),
                 "storm_R_max": int(R.max()), "R_mean": float(R.mean()), "storm_hour": [min(hour), max(hour)],
                 "storm_hour_sumR": int(sum(R[m - HELD] for m in hour))}
            for mode in ("dup", "lin"):
                units, unit_of = build_units(list(tr.models), mode, census, work)
                d[f"units_{mode}"] = {"units": len(units), "multi": sum(1 for U in units if len(U["members"]) > 1),
                                      "models_in_multi": sum(len(U["members"]) for U in units if len(U["members"]) > 1)}
            print(json.dumps(d, indent=1))
            return 0
        if not a.out or not a.arm or not (a.run or a.sync):
            ap.print_help()
            return 2
        cfg_for(a.arm, a.variant)
        with open(a.out, "a") as sink:
            if a.sync:
                r = sync_probe(a.arm, a.root, a.cache, variant=a.variant)
                line = json.dumps(r, sort_keys=True, default=str)   # QA(gemini-3.5-flash-lite): parity with --run
                print(line, flush=True); sink.write(line + "\n"); sink.flush()
                return 0
            for K in [int(x) for x in a.ks.split(",")]:
                for sd in [int(x) for x in a.seeds.split(",")]:
                    r = run_cell(a.arm, K, sd, a.arrivals, a.variant, a.root, a.cache)
                    line = json.dumps(r, sort_keys=True, default=str)
                    print(line, flush=True); sink.write(line + "\n"); sink.flush()
    except (ValueError, OSError, KeyError) as e:
        print(f"ERROR: {e!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
