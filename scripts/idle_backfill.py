#!/usr/bin/env python3
"""idle_backfill.py — put the fleet's idle to work. Two levers on fixed fleet (N H100 SXM 80 GB, the
Kimi-K2-shaped anchor + as much of the census as fits): (a) SLEEP-ZERO — models with no traffic in a trailing TRAIN window sleep in
host RAM on a few shared COOL cards and wake on demand (v2's sleep path), so the GPUs they held can host models with traffic;
(b) BACKFILL — idle GPU-minutes are sold as preemptible batch work that yields to live traffic (credited only on idle minutes).

Prereg (frozen before this file existed): (private report). On any divergence the
prereg wins. Imports `cloud_sim` (P / BS / BT admission, references, scoring) and v1 / v2 UNCHANGED. Forks:
(1) the busy-matrix / in-flight helpers (equivalence: I-REUSE), (2) `evaluate` with a COOL pool and the backfill block (I-REUSE on
P / BS, I-EMPTY on PZ / BZ with an empty COOL class), (3) the Fleet fit test with COOL cards (ZFleet; I-EMPTY).
Arms: P · BS · BT · PZ = P + SLEEP-ZERO · BZ = BS + SLEEP-ZERO · BTZ = BT + SLEEP-ZERO. BACKFILL is computed inside every row.

Output contract:
  --self-test [--root R] [--cache F]   exit 0 + "SELF-TEST PASS n/n"; exit 1 + FAIL lines (goldens GB1-GB8 + CLI goldens).
  --admit-spec FILE                     read-only: a toy admission spec (JSON) through the real ZFleet; one JSON object; exit 0.
  --backfill-spec FILE                  read-only: a toy busy series (or cells) through the real backfill code; one JSON object.
  --plan --arm A --fmt F --N n --root R --cache C [--set k=v]   read-only: anchor sizing + the admission walk; one JSON object.
  --check REUSE|EMPTY|LIVE --root R --cache C                   read-only instrument checks (prereg §7); exit 0 PASS, 1 FAIL.
  --run --arm A --fmt F --N n [--set k=v ...] [--tag T] --root R --cache C --out F
                                        WRITES (appends) one JSON line per n-grid point to F; exit 0 (anchor not placeable: exit 3).
  --summarize DIR                       read-only: prereg §4 over DIR/*.jsonl; one JSON object on stdout.
  Bad input (unknown arm/param, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed earlier and here):
  1. **A unit index of -1 wraps numpy** — `wakestorm.train_live` indexes live[unit_of[mid]] with no mask; with the compacted v2 unit_of
     (MG models = -1) it would write MG cells into the LAST unit's row. `unit_live` here masks first (I-LIVE).
  2. **`ssh host '... &'` holds the session open** — launch with `ssh -n` + `setsid -f` from a shipped script.
  3. **A COOL window of 0 minutes is an EMPTY window, which would make every model "silent"** — `cool_class` treats W <= 0 as "no
     model is COOL" (the I-EMPTY control), never as "all models are COOL".
"""
import argparse, collections, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import node_sim as NS        # noqa: E402  (not edited)
import fabric_sim as F       # noqa: E402  (v1, not edited)
import fabric_sim_v2 as V2   # noqa: E402  (v2, not edited)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
SLOS = CS.SLOS
SEEDS = CS.SEEDS
N_GRID = CS.N_GRID
GIVEUP_S = CS.GIVEUP_S
EXTRA0 = dict(CS.EXTRA0, win_min=1440.0, h2d=NS.H2D_PCIE, d2h=NS.D2H_PCIE)   # prereg §1.2: 1-day window, H100 wake bandwidth
Z_KEYS = ("win_min", "h2d", "d2h")
ARM2 = {"P": ("DED", "DED", False), "BS": ("SPREAD", "SHARE", False), "BT": ("SPREAD", "DED", False),
        "PZ": ("DED", "DED", True), "BZ": ("SPREAD", "SHARE", True), "BTZ": ("SPREAD", "DED", True)}
BASE_OF = {"PZ": "P", "BZ": "BS", "BTZ": "BT"}
WB_GRID = (1, 5, 15)                 # minutes a GPU sits idle before batch starts (prereg §1.3)
LAM_GRID = (0.0, 30.0, 60.0)         # s of batch work lost per PREEMPTED run
P_GRID = (0.05, 0.79, 1.6)           # s of extra TTFT a live request pays when it preempts batch
SIGMA_S = 1.6                        # s setup per started run
POINT = dict(wb=1, lam=30.0, p=0.79)
PRICES = (0.79, 1.0, 2.0, 3.19, 3.99)   # $ per recovered GPU-hour (point 2.00 = 50 % batch discount on $3.99 on-demand;
#                                        0.79 Nebius dynamic-spot floor, 3.19 DeepInfra 20 % batch discount; amendment 1)
VARIANTS = ("CONS", "POINT")         # CONS = IDLE-W busy (deciding), POINT = point busy
PZ_ARM = "PZCOOL"                    # PZ's COOL engines: model-level (no dedup), bf16 KV sized like kv="size", v2 sleep path
V2.BASE_ARMS.setdefault(PZ_ARM, dict(F.ALL, units="none", int8=False, copies=False))   # runtime registration, file not edited


# ------------------------------------------------------------------------------------------------ busy (fork of §5 helpers)
def busy_matrix2(t0, last, s, model, n_models, n_min, ext=None):
    """bool (n_models, n_min): cell i marks minutes t0 .. t0 + floor((last + s [+ ext[model]]) / 60) busy (== CS.busy_matrix)."""
    t0 = np.asarray(t0, np.int64)
    add = np.asarray(last, float) + np.asarray(s, float)
    if ext is not None:
        add = add + np.asarray(ext, float)[np.asarray(model)]
    end = np.minimum(t0 + np.floor(add / 60.0).astype(np.int64), n_min - 1)
    D = np.zeros((n_models, n_min + 1), np.int32)
    np.add.at(D, (np.asarray(model), t0), 1)
    np.add.at(D, (np.asarray(model), end + 1), -1)
    return np.cumsum(D, axis=1)[:, :n_min] > 0


def inflight2(t0, last, s, n, n_min):
    """requests in flight per minute (== CS.inflight)."""
    t0 = np.asarray(t0, np.int64)
    end = np.minimum(t0 + np.floor((np.asarray(last, float) + np.asarray(s, float)) / 60.0).astype(np.int64), n_min - 1)
    D = np.zeros(n_min + 1, np.int64)
    np.add.at(D, t0, np.asarray(n, np.int64))
    np.add.at(D, end + 1, -np.asarray(n, np.int64))
    return np.cumsum(D)[:n_min]


def replica_busy(cnt, r):
    """prereg §3: anchor replica j is busy iff the in-flight count > j (so 16 x sum_j busy_j == 16 x min(r, count))."""
    cnt = np.asarray(cnt)
    return [cnt > j for j in range(r)]


def variant_rows(v, B, Bw, acnt, acnt_w, r):
    """prereg §0.2: CONS (deciding) = the queue-inclusive IDLE-W busy matrix + the extended anchor in-flight; POINT =."""
    if v == "CONS":
        return Bw, replica_busy(acnt_w, r)
    return B, replica_busy(acnt, r)


# ------------------------------------------------------------------------------------------------ backfill (prereg §3)
def runs_of(idle):
    """maximal runs of True -> (starts, ends), inclusive."""
    x = np.concatenate(([0], np.asarray(idle, np.int8), [0]))
    d = np.diff(x)
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1


def slot_backfill(busy, wb, sig_min, lam_min, day_len):
    """one GPU's busy row -> (per-day credited minutes [nd], hit minutes, credited-minute mask). A run of idle minutes [a, b] starts
    batch at a + wb; credit = max(0, minutes - sigma - lambda x [preempted]) (overhead never above the minutes); preempted iff a busy
    minute follows inside the week; the overhead is charged to the day the batch STARTS (carried forward if that day is short)."""
    busy = np.asarray(busy, bool)
    n_min = len(busy)
    nd = n_min // day_len
    a, b = runs_of(~busy)
    s0 = a + wb
    m = b - s0 + 1
    ok = m > 0
    b, s0, m = b[ok], s0[ok], m[ok]
    pre = b + 1 < n_min
    o = np.minimum(m.astype(float), sig_min + lam_min * pre)
    D = np.arange(nd)
    lo = np.maximum(s0[:, None], D[None, :] * day_len)
    hi = np.minimum(b[:, None], (D[None, :] + 1) * day_len - 1)
    C = np.maximum(0, hi - lo + 1).astype(float)
    sd = s0 // day_len
    C[np.arange(len(s0)), sd] -= o
    for d in range(nd - 1):
        neg = np.minimum(C[:, d], 0.0)
        C[:, d] -= neg
        C[:, d + 1] += neg
    mask = np.zeros(n_min + 1, np.int32)
    np.add.at(mask, s0, 1)
    np.add.at(mask, b + 1, -1)
    return C.sum(axis=0), b[pre] + 1, np.cumsum(mask)[:n_min] > 0


def room_sizes(homes, w, bpt, card, s):
    """POST-HOC ROOM challenger: each COOL engine wakes with its card minus the other sleepers' charges as f (one awake per card);
    KV tokens = (f - w - OVH) x 2^30 / bytes per token."""
    nc = collections.Counter(homes)
    f = [card - (nc[h] - 1) * s for h in homes]
    return f, [(f[j] - w[j] - F.OVH) * NS.G30 / bpt[j] for j in range(len(homes))]


def bf_verdict(frac, safe):
    """prereg §4(b). A None fraction (a replicate with no idle minute) reads as 0 (QA r1)."""
    frac = [0.0 if x is None else x for x in frac]
    if max(frac) < 0.2:
        return "NEGLIGIBLE"
    if min(frac) >= 0.2:
        if not all(safe):
            return "SLO-COST"
        return "RECOVERS-MOST" if min(frac) >= 0.5 else "RECOVERS-SOME"
    return "UNRESOLVED"


def guaranteed(c_low, c_band, n, h, eligible):
    """prereg §3 SLO-SAFE: worst-case placement of h hit requests (+p each) on the band (S - p, S]."""
    keep = c_low + max(0, c_band - h)
    return keep >= math.ceil(0.95 * n) if eligible else keep == n


def band_counts(week, days, keys):
    """per LODO replicate (week - day d): n and the count of requests with TTFT key <= k for every k in keys."""
    ks = sorted(week)
    karr = np.array(ks, np.int64)
    idx = {k: j for j, k in enumerate(ks)}
    Wv = np.array([week[k] for k in ks], np.int64)
    Dm = np.zeros((NDAYS, len(ks)), np.int64)
    for d in range(NDAYS):
        for k, v in days[d].items():
            if v:
                Dm[d, idx[k]] = v
    R = Wv[None, :] - Dm
    out = {"n": R.sum(axis=1)}
    for k in keys:
        out[k] = R[:, karr <= k].sum(axis=1) if len(ks) else np.zeros(NDAYS, np.int64)
    return out


def week_gpu_hours(credit_min, days=NDAYS - 1):
    """prereg §3: credited GPU-minutes of a (days)-day replicate -> GPU-hours per week."""
    return credit_min * 7.0 / days / 60.0


# ------------------------------------------------------------------------------------------------ SLEEP-ZERO classification
def cool_class(tr, groups_of, held, win_min):
    """prereg §2: group g (a unit's members, or one model) is COOL iff it got 0 requests in [held - win, held). W <= 0 -> none."""
    if win_min <= 0:
        return set()
    sel = (np.asarray(tr.minute) < held) & (np.asarray(tr.minute) >= held - win_min)
    cnt = np.bincount(np.asarray(tr.mid)[sel], weights=np.asarray(tr.n)[sel].astype(float), minlength=len(tr.models))
    return {g for g, mem in enumerate(groups_of) if mem and sum(cnt[m] for m in mem) == 0}


def unit_live(tr, unit_of, n_units, held=HELD, min0=T.MIN0):
    """wakestorm.train_live with the -1 mask (pitfall 1): bool [n_units, held - min0], live in t if a member requests in t or t-1."""
    Tn = held - min0
    live = np.zeros((n_units, Tn), bool)
    uo = np.asarray(unit_of)
    k = (tr.minute < held) & (uo[tr.mid] >= 0)
    u = uo[tr.mid[k]]
    t = (tr.minute[k] - min0).astype(np.int64)
    live[u, t] = True
    t1 = t + 1
    ok = t1 < Tn
    live[u[ok], t1[ok]] = True
    return live


# ------------------------------------------------------------------------------------------------ admission with COOL cards
class ZFleet(CS.Fleet):
    """ Fleet + a COOL pool: cool = dict(units {uid: dict(cid, f, w, pin, tot)}, ids set, live {uid: bool row}, card, s, h_pin,
    h_tot) or None (then every method equals CS.Fleet's). K_cool starts at 0 and never shrinks (prereg §2)."""

    def __init__(self, N, tail_mode, giant_mode, anchor_G, anchor_r, units, gp, cool=None):
        super().__init__(N, tail_mode, giant_mode, anchor_G, anchor_r, units, gp)
        self.cool = cool
        self.cool_loaded = []
        self.k_cool = 0

    def k_tail(self):
        if self.tm == "DED":
            return self.N - self.group_gpus() - self.ded - self.k_cool
        return self.N - self.group_gpus() - self.k_cool

    def cool_place(self, uids, kc):
        c = self.cool
        info = [dict(c["units"][u]) for u in uids]
        hneed = [(x["pin"], x["tot"]) for x in info]
        live = np.array([c["live"][u] for u in uids])
        return F.coact_place(info, kc, dict(h_pin=c["h_pin"], h_tot=c["h_tot"]), hneed, True, live, c["card"], c["s"])

    def try_tail(self, cid, uid):
        if self.cool is not None and uid in self.cool["ids"]:
            return self.try_cool(cid, uid)
        if self.tm == "DED":
            if self.k_tail() >= 1 and self.nodes_ok(self.group_list(), self.ded + self.k_cool + 1):
                self.ded += 1
                self.admitted.append(cid)
                return True
            return False
        if uid in self.loaded:                       # a sibling rides its unit's engine at 0 GiB
            self.admitted.append(cid)
            return True
        K = self.k_tail()
        try:
            if not self.nodes_ok(self.group_list(), K + self.k_cool):
                return False
            self.place(self.loaded + [uid], K)
        except ValueError:
            return False
        self.loaded.append(uid)
        self.admitted.append(cid)
        return True

    def try_cool(self, cid, uid):
        if uid in self.cool_loaded:
            self.admitted.append(cid)
            return True
        for kc in sorted({max(1, self.k_cool), self.k_cool + 1}):
            if self.tm == "DED":
                if self.N - self.group_gpus() - self.ded - kc < 0 or not self.nodes_ok(self.group_list(), self.ded + kc):
                    continue
            else:
                Kh = self.N - self.group_gpus() - kc
                if Kh < 0 or not self.nodes_ok(self.group_list(), Kh + kc):
                    continue
                if self.loaded:
                    try:
                        self.place(self.loaded, Kh)
                    except ValueError:
                        continue
            try:
                self.cool_place(self.cool_loaded + [uid], kc)
            except ValueError:
                continue
            self.k_cool = kc
            self.cool_loaded.append(uid)
            self.admitted.append(cid)
            return True
        return False

    def try_giant(self, cid, G, W_, rho):
        if self.gm == "SHARE":
            for g in self.groups:
                if g["G"] == G and self.join_ok(g, G, W_, rho):
                    g["members"].append(cid); g["W"].append(W_); g["rho"] += rho
                    self.admitted.append(cid)
                    return True
        key = f"g{len(self.groups):03d}"
        new = (key, G)
        if self.group_gpus() + G + self.k_cool > self.N:
            return False
        try:
            if self.tm == "DED":
                if (self.N - self.group_gpus() - G - self.ded - self.k_cool < 0
                        or not self.nodes_ok(self.group_list(new), self.ded + self.k_cool)):
                    return False
            else:
                K = self.N - self.group_gpus() - G - self.k_cool
                if not self.nodes_ok(self.group_list(new), K + self.k_cool):
                    return False
                if self.loaded:
                    self.place(self.loaded, K, groups=self.group_list(new))
        except ValueError:
            return False
        if self.gm == "SHARE":                       # a giant alone: host RAM / KV floor only (§4.2)
            if not self.fits_group(G, [W_]):
                return False
        self.groups.append(dict(key=key, G=G, members=[cid], W=[W_], rho=rho))
        self.admitted.append(cid)
        return True

    def snapshot(self):
        s = super().snapshot()
        s.update(cool_loaded=list(self.cool_loaded), k_cool=self.k_cool)
        return s


def admit_spec(spec):
    """Toy admission through the real ZFleet (CLI golden GB6). spec = dict(N, anchor_G, anchor_r, bh, h_tot_card, cool=dict(h_pin,
    h_tot, s, r_pin, o_proc), arms, candidates [dict(id, kind tail, unit, f, w, cls hot|cool, host)])."""
    res = {}
    cz = spec["cool"]
    for arm in spec.get("arms", ["P", "BS", "PZ", "BZ"]):
        tm, gm, zz = ARM2[arm]
        units, cu, live = {}, {}, {}
        for c in spec["candidates"]:
            units.setdefault(c["unit"], dict(cid=c["unit"], f=float(c["f"]), w=float(c["w"]), host=float(c.get("host", cz["o_proc"]))))
            if c.get("cls") == "cool":
                p = cz["r_pin"] * float(c["w"])
                cu[c["unit"]] = dict(cid=c["unit"], f=float(c["f"]), w=float(c["w"]), pin=p, tot=p + cz["o_proc"])
                live[c["unit"]] = np.zeros(10, bool)
        gp = dict(bh=float(spec["bh"]), h_tot=float(spec["h_tot_card"]), node_ram=1e9, r_pin=1.0, o_g=0.0, s_g=0.0, rho_cap=1e9)
        cool = dict(units=cu, ids=set(cu), live=live, card=float(spec["bh"]), s=cz["s"], h_pin=cz["h_pin"], h_tot=cz["h_tot"]) \
            if zz else None
        fl = ZFleet(spec["N"], tm, gm, spec["anchor_G"], spec["anchor_r"], units, gp, cool)
        cands = [dict(cid=c["id"], kind="tail", uid=c["unit"]) for c in spec["candidates"]]
        CS.walk(fl, cands, [len(cands)])
        res[arm] = dict(admitted=fl.admitted, turned_away=[c["id"] for c in spec["candidates"] if c["id"] not in fl.admitted],
                        k_cool=fl.k_cool, cool_loaded=fl.cool_loaded)
    return res


def backfill_spec(spec):
    """Toy backfill through the real code (CLI goldens GB1 / GB2). spec = dict(n_min, day_len, wb, sigma_s, lam_s, and busy [minutes]
    or cells with optional ext (s per cell model 0))."""
    n_min, dl = int(spec["n_min"]), int(spec["day_len"])
    if "cells" in spec:
        cl = spec["cells"]
        ext = [float(spec["ext"])] if "ext" in spec else None
        B = busy_matrix2([c[0] for c in cl], [c[1] for c in cl], [c[2] for c in cl], [0] * len(cl), 1, n_min, ext=ext)[0]
    else:
        B = np.zeros(n_min, bool)
        B[list(spec["busy"])] = True
    per, hits, mask = slot_backfill(B, int(spec["wb"]), float(spec["sigma_s"]) / 60.0, float(spec["lam_s"]) / 60.0, dl)
    return dict(busy=[int(x) for x in np.nonzero(B)[0]], credit=round(float(per.sum()), 9), per_day=[round(float(x), 9) for x in per],
                hits=[int(x) for x in hits], credited_minutes=[int(x) for x in np.nonzero(mask)[0]])


# ------------------------------------------------------------------------------------------------ the world
class ZWorld:
    """ World + the COOL classes, PZ's model-level COOL setup and the dense held-out request matrix."""

    def __init__(self, fmt, over, extra, root, cache):
        self.extra = dict(extra)
        self.wd = wd = CS.make_world(fmt, over, extra, root, cache)
        cx = wd.cx
        tr, S, M = cx["tr"], cx["S"], cx["M"]
        self.root, self.cache = root, cache
        self.win = float(extra["win_min"])
        self.h2d, self.d2h = float(extra["h2d"]), float(extra["d2h"])
        self.cool_room = float(extra.get("cool_room", 0.0))   # POST-HOC challenger (never deciding): wake with the card's room
        uo = S["unit_of"]
        tail = wd.tail
        groups_bs = [[] for _ in range(len(S["units"]))]
        for m in tail:
            groups_bs[uo[m]].append(m)
        self.cool_bs = cool_class(tr, groups_bs, HELD, self.win)
        mod_groups = [[m] if m in set(tail) else [] for m in range(M)]
        self.cool_models = cool_class(tr, mod_groups, HELD, self.win)
        self._spz = None
        self._live_bs = None
        self._bcache = {}
        k = (tr.minute >= HELD) & (tr.minute < HELD + WEEK_MIN)
        self.R = np.zeros((M, WEEK_MIN), np.int64)
        np.add.at(self.R, (tr.mid[k], tr.minute[k] - HELD), tr.n[k].astype(np.int64))
        self.h_pin, self.h_tot = cx["tph"]["h_pin"], cx["tph"]["h_tot"]
        self.r_pin, self.o_proc = cx["tph"]["r_pin"], cx["tph"]["o_proc"]

    def spz(self):
        if self._spz is None:
            cx = self.wd.cx
            S2 = V2.setup2("BIG_" + PZ_ARM, cx["tph"], self.root, self.cache)
            if len(S2["tr"].models) != len(cx["tr"].models) or S2["tr"].models != cx["tr"].models:
                raise RuntimeError("PZ setup trace differs from the world's trace")
            self._spz = S2
        return self._spz

    def live_bs(self):
        if self._live_bs is None:
            S = self.wd.cx["S"]
            self._live_bs = unit_live(self.wd.cx["tr"], S["unit_of"], len(S["units"]))
        return self._live_bs

    def cool_spec(self, arm):
        """-> (cool dict for ZFleet, candidate list) for a Z arm; (None, candidates) otherwise."""
        wd = self.wd
        tm, gm, zz = ARM2[arm]
        if not zz:
            return None, wd.cands
        if tm == "SPREAD":
            S = wd.cx["S"]
            live = self.live_bs()
            ids = set(self.cool_bs)
            cu = {}
            for u in ids:
                U = S["units"][u]
                p = self.r_pin * U["w"]
                cu[u] = dict(cid=U["cid"], f=U["f"], w=U["w"], pin=p, tot=p + self.o_proc)
            return (dict(units=cu, ids=ids, live={u: live[u] for u in ids}, card=wd.bh, s=wd.tph["s"], h_pin=self.h_pin,
                         h_tot=self.h_tot), wd.cands)
        S2 = self.spz()
        live = S2["live"]
        cu, ids, cands = {}, set(), []
        for c in wd.cands:
            m = c["m"]
            if c["kind"] == "tail" and m in self.cool_models and S2["unit_of"][m] >= 0:
                u = S2["unit_of"][m]
                key = ("pz", u)
                U = S2["units"][u]
                cu[key] = dict(cid=U["cid"], f=U["f"], w=U["w"], pin=S2["hneed"][u][0], tot=S2["hneed"][u][1])
                ids.add(key)
                cands.append(dict(c, uid=key))
            else:
                cands.append(c)
        return (dict(units=cu, ids=ids, live={k_: live[k_[1]] for k_ in ids}, card=wd.bh, s=wd.tph["s"], h_pin=self.h_pin,
                     h_tot=self.h_tot), cands)

    def busy(self, seed, ext=None):
        """bool (M, WEEK_MIN) with the fork (== CS World.busy at beta 0); the point matrix is cached per seed."""
        if ext is None and seed in self._bcache:
            return self._bcache[seed]
        B = self._busy(seed, ext)
        if ext is None:
            self._bcache[seed] = B
        return B

    def _busy(self, seed, ext):
        wd = self.wd
        tr = wd.cx["tr"]
        k, last = wd.arrivals(seed)
        mid = tr.mid[k]
        isg = np.zeros(wd.cx["M"], bool)
        isg[list(wd.mg)] = True
        s = np.where(isg[mid], wd.service("giant"), wd.service("tail", 0.0))
        return busy_matrix2(tr.minute[k] - HELD, last, s, mid, wd.cx["M"], WEEK_MIN, ext=ext)

    def anchor_cnt(self, seed, ext=0.0):
        wd = self.wd
        tr = wd.cx["tr"]
        k, last = wd.arrivals(seed)
        sel = tr.mid[k] == wd.am
        s = wd.service("giant")[sel] + ext
        return inflight2(tr.minute[k][sel] - HELD, last[sel], s, tr.n[k][sel], WEEK_MIN)


def plan2(Z, arm, N):
    wd = Z.wd
    G = wd.cx["tab"][NS.ANCHOR]["G"]
    if G > N:
        raise CS.AnchorError(f"ANCHOR-NOT-PLACEABLE: the anchor needs {G} GPUs, the fleet has {N}")
    rmin = wd.anchor_r_min()
    flag = None
    if rmin is None or rmin > N // G:
        flag = "ANCHOR-NOT-SERVABLE"
    elif rmin * G == N:
        flag = "DEGENERATE"
    if flag:
        return dict(flag=flag, r_min=rmin, r=None if flag == "ANCHOR-NOT-SERVABLE" else rmin), None
    tm, gm, zz = ARM2[arm]
    units = {u: dict(cid=x["cid"], f=x["f"], w=x["w"], host=x["host"]) for u, x in enumerate(wd.uinfo)}
    cool, cands = Z.cool_spec(arm)
    fl = ZFleet(N, tm, gm, G, rmin, units, wd.gp, cool)
    snaps = CS.walk(fl, cands, N_GRID)
    return dict(flag=None, r_min=rmin, r=rmin), snaps


# ------------------------------------------------------------------------------------------------ COOL runner
def run_cool(Z, S, units_idx, homes, Kc, admitted, seed, f_over=None, kt_over=None):
    """v2 sleep path (cfg ALL, copies off, topology direct, h2d / d2h per card, protect 60 s) for the COOL units on Kc cards; only
    ADMITTED members are served. Mirrors run_dgroups at G = 1 with the tail physics."""
    wd = Z.wd
    cx = wd.cx
    tr, M = cx["tr"], cx["M"]
    cfg, _ = V2.arm_cfg2("BIG_ALL")
    cfg = dict(cfg, copies=False, topo="direct", h2d=Z.h2d, d2h=Z.d2h)
    cfg["protect"] = cx["tph"]["protect"]
    sub = [S["units"][u] if f_over is None else dict(S["units"][u], f=f_over[j]) for j, u in enumerate(units_idx)]
    kts = [S["kvtok"][u] for u in units_idx] if kt_over is None else list(kt_over)
    new_of = {u: j for j, u in enumerate(units_idx)}
    unit_of = [new_of.get(S["unit_of"][m], -1) if (S["unit_of"][m] >= 0 and m in admitted) else -1 for m in range(M)]
    keep_m = [m for m in range(M) if unit_of[m] >= 0]
    st = NS.relabel_days(NS.subtrace(tr, keep_m), M)
    out, met = V2.simulate(st, sub, unit_of * (1 + NDAYS), ("card", homes), Kc, cfg, seed, None, card_gib=wd.bh,
                           phys=cx["tph"], kvtok=kts, copies=None, C=wd.Ch, mech=None,
                           shed_tau=None, bpt=[S["bpt"][u] for u in units_idx], resize=None)
    w, dy = NS.split_days(out, M)
    return {m: w[m] for m in keep_m}, {m: [dy[d][m] for d in range(NDAYS)] for m in keep_m}, met


# ------------------------------------------------------------------------------------------------ one fleet state -> one row
def _bf_keys():
    for v in VARIANTS:
        for wb in WB_GRID:
            for lam in LAM_GRID:
                yield v, wb, lam


def evaluate2(Z, arm, snap, N, r, beta=0.0, seeds=SEEDS, bf=True):
    """FORK of CS.evaluate (arms P / BS / BT and their Z versions; no NODE arm) + the COOL pool + the backfill block."""
    wd = Z.wd
    cx = wd.cx
    tr, M, tab, S = cx["tr"], cx["M"], cx["tab"], cx["S"]
    tm, gm, zz = ARM2[arm]
    adm = [cx["cid_m"][c] for c in snap["admitted"]]
    tail_adm = [m for m in adm if m not in wd.mg]
    mg_adm = [m for m in adm if m in wd.mg]
    kc = snap.get("k_cool", 0)
    cool_loaded = snap.get("cool_loaded", [])
    S_c = (Z.spz() if tm == "DED" else S) if cool_loaded else None
    cool_u = [u[1] if isinstance(u, tuple) else u for u in cool_loaded]
    cool_set_u = set(cool_u)
    # a tail model is COOL iff its unit in the COOL setup (PZ: one unit per model; BZ: BS's own units) was admitted via the COOL
    # pool; HOT units live in snap["loaded"] / snap["ded"], disjoint by construction
    cool_m = [m for m in tail_adm if S_c is not None and S_c["unit_of"][m] >= 0 and S_c["unit_of"][m] in cool_set_u]
    cset = set(cool_m)
    hot_m = [m for m in tail_adm if m not in cset]
    hosted = [wd.am] + adm
    t0 = time.time()
    out_w, out_d = {}, {}
    tail_slots, tail_w = [], []
    tail_stats = {}
    tail_from_ref = tm == "DED"
    K = N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"]) - (kc if tm == "SPREAD" else 0)
    if tm == "DED":
        tail_slots = [(1, [m]) for m in hot_m]
        tail_w = [tr.models[m][4] for m in hot_m]
    elif snap["loaded"]:
        loaded = snap["loaded"]
        aset = set(hot_m)
        info = [dict(wd.uinfo[u]) for u in loaded]
        homes = NS.hetero_spread(info, [[wd.bh, wd.tph["h_tot"]] for _ in range(K)])
        wk, dy, met = CS.run_tail(cx, loaded, homes, K, wd.bh, wd.Ch, wd.tph, aset)
        out_w.update(wk); out_d.update(dy)
        NS._stats_add(tail_stats, met)
        on = collections.defaultdict(list)
        for m in hot_m:
            on[homes[loaded.index(S["unit_of"][m])]].append(m)
        tail_slots = [(1, on[k]) for k in range(K)]
        wsum = collections.Counter()
        for j, u in enumerate(loaded):
            wsum[homes[j]] += wd.uinfo[u]["w"]
        tail_w = [wsum[k] for k in range(K)]
    else:
        tail_slots = [(1, []) for _ in range(K)]
        tail_w = [0.0] * K
    # COOL pool (homes re-derived exactly as the admission's last successful placement)
    cool_slots, cool_w, cool_runs, cool_homes = [], [], {}, None
    if cool_loaded:
        cool, _ = Z.cool_spec(arm)
        info = [dict(cool["units"][u]) for u in cool_loaded]
        hneed = [(x["pin"], x["tot"]) for x in info]
        live = np.array([cool["live"][u] for u in cool_loaded])
        cool_homes = F.coact_place(info, kc, dict(h_pin=Z.h_pin, h_tot=Z.h_tot), hneed, True, live, wd.bh, wd.tph["s"])
        on = collections.defaultdict(list)
        for m in cool_m:
            on[cool_homes[cool_u.index(S_c["unit_of"][m])]].append(m)
        cool_slots = [(1, on[k]) for k in range(kc)]
        wsum = collections.Counter()
        for j, u in enumerate(cool_u):
            wsum[cool_homes[j]] += S_c["units"][u]["w"]
        cool_w = [wsum[k] for k in range(kc)]
        f_over = kt_over = None
        if Z.cool_room > 0:                          # POST-HOC ROOM: KV = the card minus its sleepers (one awake per card)
            f_over, kt_over = room_sizes(cool_homes, [S_c["units"][u]["w"] for u in cool_u], [S_c["bpt"][u] for u in cool_u],
                                         wd.bh, wd.tph["s"])
            if any(f_over[j] + 1e-9 < S_c["units"][u]["f"] for j, u in enumerate(cool_u)):
                raise RuntimeError("ROOM: a room f below the sized f")
        for s_ in seeds:
            cool_runs[s_] = run_cool(Z, S_c, cool_u, cool_homes, kc, cset, s_, f_over, kt_over)
    # giants
    if gm == "DED":
        gslots = [(tab[tr.models[m][0]]["G"], [m]) for m in mg_adm]
        g_w = [tab[tr.models[m][0]]["W"] for m in mg_adm]
        d_runs = {s_: ({}, {}, {}) for s_ in seeds}
    else:
        gslots = [(g["G"], [cx["cid_m"][c] for c in g["members"]]) for g in snap["groups"]]
        g_w = [sum(g["W"]) for g in snap["groups"]]
        grp = [(g["G"], g["members"]) for g in snap["groups"]]
        d_runs = {s_: (CS.run_dgroups(cx, wd.np, grp, s_) if grp else ({}, {}, {})) for s_ in seeds}
    allm = list(range(M))
    rst = {m: wd.ref_stats(m, r) for m in allm}
    ost_common = {wd.am: rst[wd.am]}
    cnt_common = {wd.am: wd.anchor_out(r)}
    for m in hot_m:
        if tail_from_ref:
            ost_common[m] = rst[m]
            cnt_common[m] = (wd.ref_w[m], [wd.ref_d[x][m] for x in range(NDAYS)])
        else:
            ost_common[m] = CS.cstats(out_w[m], out_d[m])
            cnt_common[m] = (out_w[m], out_d[m])
    if gm == "DED":
        for m in mg_adm:
            ost_common[m] = rst[m]
            cnt_common[m] = (wd.ref_w[m], [wd.ref_d[x][m] for x in range(NDAYS)])
    k_row = (N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"]) - snap["ded"] - kc
             if tm == "DED" else K)
    row = dict(arm=arm, N=N, r_anchor=r, beta=beta, hosted=len(hosted), admitted=[c[:8] for c in snap["admitted"]],
               n_tail=len(tail_adm), n_mg=len(mg_adm), loaded_units=len(snap["loaded"]), ded_tail=snap["ded"],
               groups=[dict(G=g["G"], members=[c[:8] for c in g["members"]], rho=round(g["rho"], 6)) for g in snap["groups"]],
               k_tail=k_row, tail_stats=tail_stats)
    away = [m for m in allm if m not in set(hosted)]
    wk_tot = float(wd.nH.sum())
    row["turned_away"] = dict(n=len(away), share_week=float(sum(wd.nH[m] for m in away)) / wk_tot,
                              n_mg=sum(1 for m in away if m in wd.mg),
                              top10=[(tr.models[m][0][:8], int(wd.nH[m]), "mg" if m in wd.mg else "tail")
                                     for m in sorted(away, key=lambda x: (-wd.nH[x], tr.models[x][0]))[:10]])
    slo_res = {str(x): collections.defaultdict(list) for x in SLOS}
    idle, idle_w, idle_a, zero_g, alloc_l = [], [], [], [], []
    rd = wd.req_day()
    d_stats = {}
    anc_g = snap["anchor"][0][1] if snap["anchor"] else tab[NS.ANCHOR]["G"]
    cool_out = dict(k_cool=kc, n_cool=len(cool_m), cool_held_req=int(sum(wd.nH[m] for m in cool_m)), stats={},
                    meet={str(x): [] for x in SLOS}, elig_feas={str(x): [] for x in SLOS}, max_p95_up=[])
    if Z.cool_room > 0 and cool_loaded:
        cool_out["room_kvtok"] = [round(min(kt_over)), round(max(kt_over))]
        cool_out["sized_kvtok"] = [round(min(S_c["kvtok"][u] for u in cool_u)), round(max(S_c["kvtok"][u] for u in cool_u))]
    bfo = {f"{v}|{wb}|{int(lam)}": collections.defaultdict(list) for v, wb, lam in _bf_keys()}
    slo_b = {f"{v}|{wb}|{p}|{x}": collections.defaultdict(list) for v in VARIANTS for wb in WB_GRID for p in P_GRID for x in SLOS}
    keys_needed = sorted({T._key(x - p) for x in SLOS for p in P_GRID} | {T._key(x) for x in SLOS})
    for s_ in seeds:
        ost = dict(ost_common)
        cnt = dict(cnt_common)
        gw, gd, gst = d_runs[s_]
        d_stats[str(s_)] = gst
        for m in mg_adm:
            if m not in ost:
                ost[m] = CS.cstats(gw[m], gd[m])
                cnt[m] = (gw[m], gd[m])
        if cool_loaded:
            cw, cd, cmet = cool_runs[s_]
            cool_out["stats"][str(s_)] = cmet["stats"]
            for m in cool_m:
                ost[m] = CS.cstats(cw[m], cd[m])
                cnt[m] = (cw[m], cd[m])
        for x in SLOS:
            sc = CS.score_reps(hosted, ost, rst, allm, x)
            for k_, v in sc.items():
                slo_res[str(x)][k_].extend(v)
            if cool_loaded:
                for d in range(NDAYS):
                    ef = [m for m in cool_m if rst[m]["n"][d] >= 20 and rst[m]["p95"][d] <= x]
                    cool_out["elig_feas"][str(x)].append(len(ef))
                    cool_out["meet"][str(x)].append(sum(1 for m in ef if ost[m]["p95"][d] <= x))
        if cool_loaded:
            for d in range(NDAYS):
                ups = [ost[m]["p95"][d] - rst[m]["p95"][d] for m in cool_m
                       if rst[m]["n"][d] >= 20 and ost[m]["p95"][d] is not None and rst[m]["p95"][d] is not None
                       and math.isfinite(ost[m]["p95"][d]) and math.isfinite(rst[m]["p95"][d])]
                cool_out["max_p95_up"].append(round(max(ups), 2) if ups else None)
        # idle (§5, on the forked busy matrices)
        slots = tail_slots + cool_slots + gslots
        B = Z.busy(s_)
        acnt = Z.anchor_cnt(s_)
        i_n, alloc, i_a = CS.fleet_idle(B, slots, acnt, r, anc_g, N, WEEK_MIN, DAY_MIN)
        idle.extend(i_n)
        alloc_l.append(alloc)
        idle_a.extend(i_a)
        ext = np.zeros(M)
        for m in hosted:
            p = ost[m]["week_p95"]
            ext[m] = 0.0 if p is None else min(p, GIVEUP_S)
        Bw = Z.busy(s_, ext=ext)
        acnt_w = Z.anchor_cnt(s_, ext=float(ext[wd.am]))
        pdw = CS.slot_busy_by_day(Bw, slots, acnt_w, r, anc_g, WEEK_MIN, DAY_MIN)
        idle_w.extend(CS.idle_reps(pdw, N, DAY_MIN))
        wk_req = rd.sum(axis=0)
        for d in range(NDAYS):
            rep_req = wk_req - rd[d]
            zero_g.append(int(sum(g for g, ms in slots if ms and sum(int(rep_req[m]) for m in ms) == 0)))
        if not bf:
            continue
        # ---- BACKFILL (prereg §3)
        pdb = CS.slot_busy_by_day(B, slots, acnt, r, anc_g, WEEK_MIN, DAY_MIN)
        idle_min = [float(N * DAY_MIN * (NDAYS - 1) - (pdb.sum() - pdb[d])) for d in range(NDAYS)]
        busy_min = {"POINT": pdb, "CONS": pdw}
        bslots = []                                    # (kind, g, models, weights GiB homed, row index)
        for (g, ms), w_ in zip(tail_slots, tail_w):
            if ms:
                bslots.append(("tail", g, ms, w_))
        for (g, ms), w_ in zip(cool_slots, cool_w):
            if ms:
                bslots.append(("cool", g, ms, w_))
        for (g, ms), w_ in zip(gslots, g_w):
            if ms:
                bslots.append(("giant", g, ms, w_))
        n_un = N - alloc
        if n_un < 0:
            raise RuntimeError(f"I-BF: allocated {alloc} > N {N}")
        anc_W = tab[NS.ANCHOR]["W"]
        tot_req = [int((wk_req - rd[d]).sum()) for d in range(NDAYS)]
        bc = {m: band_counts(cnt[m][0], cnt[m][1], keys_needed) for m in hosted if m in cnt}
        for v in VARIANTS:
            BB, reps_b = variant_rows(v, B, Bw, acnt, acnt_w, r)
            rows_ = [(k_, g, ms, w_, np.any(BB[list(ms)], axis=0)) for k_, g, ms, w_ in bslots]
            rows_ += [("anchor", anc_g, [wd.am], anc_W, rb) for rb in reps_b]
            elig = [Z.r_pin * w_ / g <= Z.h_pin + 1e-9 for _, g, _, w_, _ in rows_]
            cat_of = [[("zero_tail" if k_ == "tail" and sum(int(wk_req[m] - rd[d][m]) for m in ms) == 0 else k_)
                       for d in range(NDAYS)] for k_, g, ms, w_, _ in rows_]
            for wb in WB_GRID:
                hit_min = collections.defaultdict(list)
                creds = {lam: np.zeros((len(rows_), NDAYS)) for lam in LAM_GRID}
                for i, (k_, g, ms, w_, brow) in enumerate(rows_):
                    if not elig[i]:
                        continue
                    for lam in LAM_GRID:
                        per, hits, mask = slot_backfill(brow, wb, SIGMA_S / 60.0, lam / 60.0, DAY_MIN)
                        if (mask & brow).any():
                            raise RuntimeError(f"I-BF: credit on a busy minute ({k_}, {v}, wb {wb})")
                        if per.sum() > (~brow).sum() + 1e-6:
                            raise RuntimeError(f"I-BF: credit above the idle minutes ({k_}, {v})")
                        creds[lam][i] = per * g
                    if len(hits):
                        for m in ms:
                            hit_min[m].append(hits)
                H = np.zeros((M, NDAYS), np.int64)
                for m, hl in hit_min.items():              # union across slots (anchor replicas): a request is hit once
                    t_ = np.unique(np.concatenate(hl))
                    np.add.at(H[m], t_ // DAY_MIN, Z.R[m, t_])
                un_per, _, _ = slot_backfill(np.zeros(WEEK_MIN, bool), wb, SIGMA_S / 60.0, 0.0, DAY_MIN)
                for lam in LAM_GRID:
                    o = bfo[f"{v}|{wb}|{int(lam)}"]
                    C_ = creds[lam]
                    rs_ = C_.sum(axis=1)
                    for d in range(NDAYS):
                        col = rs_ - C_[:, d]
                        cat = collections.Counter()
                        for i in range(len(rows_)):
                            cat[cat_of[i][d]] += col[i]
                        cat["unalloc"] = n_un * float(un_per.sum() - un_per[d])
                        tot = float(sum(cat.values()))
                        v_idle = float(N * DAY_MIN * (NDAYS - 1) - (busy_min[v].sum() - busy_min[v][d]))
                        if tot > v_idle + 1e-6 or tot > idle_min[d] + 1e-6:
                            raise RuntimeError(f"I-BF: credited {tot} > {v} idle {v_idle} / point idle {idle_min[d]}")
                        o["credit"].append(round(tot, 3))
                        o["frac"].append(round(tot / idle_min[d], 6) if idle_min[d] > 0 else None)
                        for kk in ("anchor", "giant", "tail", "zero_tail", "cool", "unalloc"):
                            o[kk].append(round(float(cat.get(kk, 0.0)), 3))
                # SLO bound per p (hits depend on wb only), worst-case placement of the hits (prereg §3)
                hw = H.sum(axis=1)
                for p in P_GRID:
                    for x in SLOS:
                        o2 = slo_b[f"{v}|{wb}|{p}|{x}"]
                        klo, ks = T._key(x - p), T._key(x)
                        for d in range(NDAYS):
                            at_risk, lost, hit_n, ok_req = 0, 0, 0, 0
                            for m in hosted:
                                h = int(hw[m] - H[m, d])
                                hit_n += h
                                if m not in bc:
                                    continue
                                b_ = bc[m]
                                n_ = int(b_["n"][d])
                                c_s = int(b_[ks][d])
                                c_low = int(b_[klo][d])
                                band = c_s - c_low
                                ok_req += c_s
                                lost += min(h, band)
                                rn = rst[m]["n"][d]
                                if rn >= 20:
                                    base = rst[m]["p95"][d] <= x and ost[m]["p95"][d] <= x
                                else:
                                    base = c_s == n_
                                if base and not guaranteed(c_low, band, n_, h, rn >= 20):
                                    at_risk += 1
                            o2["safe"].append(at_risk == 0)
                            o2["at_risk"].append(at_risk)
                            o2["rs_lb"].append(round((ok_req - lost) / tot_req[d], 6) if tot_req[d] else None)
                            o2["hit_share"].append(round(hit_n / tot_req[d], 6) if tot_req[d] else None)
    row["slo"] = {k: dict(v) for k, v in slo_res.items()}
    row.update(idle=idle, idle_w=idle_w, idle_alloc=idle_a, zero_traffic_gpus=zero_g, allocated=alloc_l[0],
               unallocated=N - alloc_l[0], d_stats=d_stats, reps=[[s_, d] for s_ in seeds for d in range(NDAYS)])
    if zz:
        row["cool"] = cool_out
    if bf:
        row["bf"] = {k: dict(v) for k, v in bfo.items()}
        row["bf_slo"] = {k: dict(v) for k, v in slo_b.items()}
    row["sim_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------------ jobs
def run_job(arm, fmt, N, over, extra, tag, root, cache, out, seeds=SEEDS):
    if arm not in ARM2:
        raise ValueError(f"unknown arm {arm}")
    Z = ZWorld(fmt, over, extra, root, cache)
    wd = Z.wd
    beta = wd.np["beta"]                         # a label for these arms (only BC reads it); recorded as does
    base = dict(arm=arm, fmt=fmt, N=N, tag=tag, nphys=wd.np, extra=extra)
    with open(out, "a") as fh:
        try:
            info, snaps = plan2(Z, arm, N)
        except CS.AnchorError as e:
            fh.write(json.dumps(dict(base, flag="ANCHOR-NOT-PLACEABLE", error=str(e))) + "\n")
            raise
        if info["flag"]:
            fh.write(json.dumps(dict(base, flag=info["flag"], r_min=info["r_min"])) + "\n")
            return 0
        prev, prev_len = None, None
        for n in N_GRID:
            sn = snaps[n]
            t0 = time.time()
            if prev is not None and len(sn["admitted"]) == prev_len:
                row = dict(prev, n=n, reused_from=prev["n"], elapsed_s=0.0)
            else:
                row = dict(base, n=n, r_min=info["r_min"])
                row.update(evaluate2(Z, arm, sn, N, info["r"], beta, seeds))
                row.update(base)
                row["elapsed_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            prev, prev_len = row, len(sn["admitted"])
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks (§7)
CS_KEYS_SKIP = {"sim_s", "elapsed_s"}


def _same(a, b):
    diff = [k for k in a if k not in CS_KEYS_SKIP and a.get(k) != b.get(k)]
    return diff


def check(which, root, cache):
    if which == "LIVE":
        Z = ZWorld("int4", {}, dict(EXTRA0), root, cache)
        S2 = Z.spz()
        mine = unit_live(Z.wd.cx["tr"], S2["unit_of"], len(S2["units"]))
        ok_pz = bool(np.array_equal(mine, S2["live"]))
        S = Z.wd.cx["S"]
        lb = Z.live_bs()
        mg_rows = [S["unit_of"][m] for m in Z.wd.mg]
        return {"I_LIVE": dict(pz_equal=ok_pz, bs_units=int(lb.shape[0]), mg_unit_of=sorted(set(mg_rows))),
                "I_LIVE_pass": bool(ok_pz and set(mg_rows) == {-1})}
    if which == "REUSE":
        Z = ZWorld("int4", {}, dict(EXTRA0), root, cache)
        wd = Z.wd
        res, ok = {}, True
        for arm in ("P", "BS"):
            info, snaps = CS.plan(wd, arm, 128)
            info2, snaps2 = plan2(Z, arm, 128)
            for n in (0, 64, 136, 144):
                same_adm = snaps[n]["admitted"] == snaps2[n]["admitted"]
                a = CS.evaluate(wd, arm, snaps[n], 128, info["r"], wd.np["beta"])
                b = evaluate2(Z, arm, snaps2[n], 128, info2["r"], wd.np["beta"], bf=False)
                diff = _same(a, b)
                res[f"{arm}_{n}"] = dict(admitted_equal=same_adm, diff=diff)
                ok = ok and same_adm and not diff
        return {"I_REUSE": res, "I_REUSE_pass": bool(ok)}
    if which == "EMPTY":
        ex = dict(EXTRA0, win_min=0.0)
        Z = ZWorld("int4", {}, ex, root, cache)
        wd = Z.wd
        res, ok = {}, True
        for zarm, barm in (("PZ", "P"), ("BZ", "BS")):
            info, snaps = CS.plan(wd, barm, 128)
            info2, snaps2 = plan2(Z, zarm, 128)
            n = 136
            same_adm = snaps[n]["admitted"] == snaps2[n]["admitted"] and snaps2[n]["k_cool"] == 0
            a = CS.evaluate(wd, barm, snaps[n], 128, info["r"], wd.np["beta"])
            b = evaluate2(Z, zarm, snaps2[n], 128, info2["r"], wd.np["beta"], bf=False)
            diff = [k for k in _same(a, b) if k != "arm"]
            res[zarm] = dict(admitted_equal=same_adm, cool_models=len(Z.cool_models), cool_units=len(Z.cool_bs), diff=diff)
            ok = ok and same_adm and not diff and not Z.cool_models and not Z.cool_bs
        return {"I_EMPTY": res, "I_EMPTY_pass": bool(ok)}
    raise ValueError(f"unknown check {which}")


# ------------------------------------------------------------------------------------------------ summarize (§4)
def _rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def summarize(d):
    rows = CS._rows(d)
    seen = set()
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    flags = {}
    for r in rows:
        key = (r["fmt"], r["N"], r.get("tag", ""), json.dumps(r["nphys"], sort_keys=True), json.dumps(r["extra"], sort_keys=True))
        k2 = key + (r["arm"], r.get("n"))
        if k2 in seen:
            raise ValueError(f"duplicate row {r['arm']} {r['fmt']} N={r['N']} n={r.get('n')} tag={r.get('tag')}")
        seen.add(k2)
        if r.get("flag"):
            flags.setdefault(key, {})[r["arm"]] = dict(flag=r["flag"], r_min=r.get("r_min"))
            continue
        by[key][r["arm"]][r.get("n", 0)] = r
    cells = {}
    for key in sorted(set(by) | set(flags)):
        fmt, N, tag, nt, ex = key
        one = dict(fmt=fmt, N=N, tag=tag, extra=json.loads(ex), flags=flags.get(key, {}), arms={})
        for arm, rr in sorted(by.get(key, {}).items()):
            a = {}
            for x in SLOS:
                s = str(x)
                pm = {n: all(r_["slo"][s]["pass"]) for n, r_ in rr.items()}
                nm = CS.stable_n(pm)
                nreps = len(next(iter(rr.values()))["slo"][s]["pass"])
                per = [CS.stable_n({n: r_["slo"][s]["pass"][i] for n, r_ in rr.items()}) for i in range(nreps)]
                if nm is None:
                    a[s] = dict(n_match=None, note="n = 0 fails (anchor only)")
                    continue
                r0 = rr[nm]
                sl = r0["slo"][s]
                e = dict(n_match=nm, n_match_reps=_rng(per), hosted=r0["hosted"], n_tail=r0["n_tail"], n_mg=r0["n_mg"],
                         at=sl["at"], at_rng=_rng(sl["at"]), RS=sl["RS"], RS_rng=_rng(sl["RS"]), M_rng=_rng(sl["M"]),
                         idle=r0["idle"], idle_rng=_rng(r0["idle"]), idle_w_rng=_rng(r0["idle_w"]),
                         allocated=r0["allocated"], unallocated=r0["unallocated"], zero_gpu_rng=_rng(r0["zero_traffic_gpus"]),
                         turned_away=r0["turned_away"], k_tail=r0["k_tail"], r_anchor=r0["r_anchor"], groups=r0["groups"])
                if "cool" in r0:
                    c = r0["cool"]
                    e["cool"] = dict(k_cool=c["k_cool"], n_cool=c["n_cool"], cool_held_req=c["cool_held_req"],
                                     meet_rng=_rng(c["meet"][s]), elig_feas_rng=_rng(c["elig_feas"][s]),
                                     max_p95_up_rng=_rng(c["max_p95_up"]),
                                     wakes=[st.get("wakes") for st in c["stats"].values()],
                                     evictions=[st.get("evictions") for st in c["stats"].values()],
                                     abandoned=[st.get("abandoned_wakes") for st in c["stats"].values()],
                                     queued=[st.get("queued") for st in c["stats"].values()])
                if "bf" in r0:
                    bfx = {}
                    for kk, o in r0["bf"].items():
                        bfx[kk] = dict(frac_rng=_rng(o["frac"]), credit_rng=_rng(o["credit"]),
                                       gpu_h_week_rng=[round(week_gpu_hours(q), 1) for q in _rng(o["credit"])],
                                       split_mean={c_: round(float(np.mean(o[c_])), 1) for c_ in
                                                   ("anchor", "giant", "tail", "zero_tail", "cool", "unalloc")})
                    sb = {}
                    for kk, o in r0["bf_slo"].items():
                        if kk.endswith("|" + s):
                            sb[kk] = dict(safe_all=all(o["safe"]), at_risk_max=max(o["at_risk"]), rs_lb_rng=_rng(o["rs_lb"]),
                                          hit_share_rng=_rng(o["hit_share"]))
                    pk = f"CONS|{POINT['wb']}|{int(POINT['lam'])}"
                    sk = f"CONS|{POINT['wb']}|{POINT['p']}|{s}"
                    fr = r0["bf"][pk]["frac"]
                    sf = r0["bf_slo"][sk]["safe"]
                    gh = [week_gpu_hours(q) for q in r0["bf"][pk]["credit"]]
                    e["bf"] = dict(verdict=bf_verdict(fr, sf), frac_rng=_rng(fr), safe_all=all(sf),
                                   gpu_h_week_rng=[round(min(gh), 1), round(max(gh), 1)],
                                   usd_week={str(pr): [round(min(gh) * pr), round(max(gh) * pr)] for pr in PRICES},
                                   at_risk_max=max(r0["bf_slo"][sk]["at_risk"]), knobs=bfx, slo_knobs=sb)
                a[s] = e
            one["arms"][arm] = a
        cells[key] = one
    # verdicts: every arm vs P (rules) and every Z arm vs its base (prereg §4a); base from the 'main' cell if absent
    for key, one in cells.items():
        fmt, N, tag, nt, ex = key
        exd = json.loads(ex)
        one["verdicts"] = {}

        def find(arm):
            if arm in one["arms"]:
                return one["arms"][arm]
            for k2, o2 in cells.items():
                e2 = json.loads(k2[4])
                if (k2[0], k2[1], k2[3]) == (fmt, N, nt) and k2[2] == "main" and abs(e2["rho_cap"] - exd["rho_cap"]) < 1e-12 \
                        and arm in o2["arms"]:
                    return o2["arms"][arm]
            return None
        for arm, a in one["arms"].items():
            v = {}
            for vs in ("P", BASE_OF.get(arm)):
                if vs is None or vs == arm:
                    continue
                b = find(vs)
                if b is None:
                    continue
                vv = {}
                for x in SLOS:
                    s = str(x)
                    if a[s].get("n_match") is None or b[s].get("n_match") is None:
                        vv[s] = dict(note="no catalogue")
                        continue
                    cat = CS.verdict_hi(a[s]["at"], b[s]["at"])
                    idl = CS.verdict_idle(a[s]["idle"], b[s]["idle"])
                    rs = CS.verdict_rs(a[s]["RS"], b[s]["RS"])
                    if vs == BASE_OF.get(arm):
                        head = ("SLEEP-ZERO HOSTS MORE" if cat == "MORE" and rs != "LOWERS" else
                                "NO-GAIN" if cat != "MORE" and rs != "LIFTS" else f"MIXED (catalogue {cat}, requests {rs})")
                    else:
                        head = ("GPU-BUNK LIFTS" if cat == "MORE" and idl == "LOWER-IDLE" and rs != "LOWERS" else
                                "NO-LIFT" if cat != "MORE" and idl != "LOWER-IDLE" else
                                f"MIXED (catalogue {cat}, idle {idl}, requests {rs})")
                    vv[s] = dict(catalogue=cat, idle=idl, requests=rs, headline=head,
                                 d_at=_rng([p - q for p, q in zip(a[s]["at"], b[s]["at"])]),
                                 d_idle=_rng([p - q for p, q in zip(a[s]["idle"], b[s]["idle"])]),
                                 d_RS=_rng([p - q for p, q in zip(a[s]["RS"], b[s]["RS"])]))
                v[f"vs_{vs}"] = vv
            one["verdicts"][arm] = v
    return [dict(c, key=list(k[:3])) for k, c in cells.items()]


# ------------------------------------------------------------------------------------------------ self-test (prereg §8)
GB6_SPEC = dict(N=24, anchor_G=16, anchor_r=1, bh=50.0, h_tot_card=100.0,
                cool=dict(h_pin=60.0, h_tot=100.0, s=1.0, r_pin=1.0, o_proc=5.0), arms=["P", "BS", "PZ", "BZ"],
                candidates=[dict(id=f"h{i}", kind="tail", unit=f"uh{i}", f=45.0, w=10.0, cls="hot") for i in range(1, 7)]
                + [dict(id=f"c{i}", kind="tail", unit=f"uc{i}", f=30.0, w=8.0, cls="cool") for i in range(1, 10)]
                + [dict(id="h7", kind="tail", unit="uh7", f=45.0, w=10.0, cls="hot")])
GB1_SPEC = dict(n_min=20, day_len=10, wb=1, sigma_s=6.0, lam_s=30.0, busy=[3, 4, 10, 18])
GB2_SPEC = dict(n_min=12, day_len=12, wb=0, sigma_s=0.0, lam_s=0.0, cells=[[5, 50.0, 20.0]])


class _Tr:
    def __init__(self, minute, mid, n, nm):
        self.minute, self.mid, self.n, self.models = np.array(minute), np.array(mid), np.array(n), [None] * nm


def self_test(root=None, cache=None):
    ok, fails, n = True, [], 0

    def chk(name, cond, detail=""):
        nonlocal ok, n
        n += 1
        if not cond:
            ok = False
            fails.append(f"FAIL {name} {detail}")

    def near(a, b, tol=1e-9):
        return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))

    # GB1 runs
    r1 = backfill_spec(GB1_SPEC)
    chk("GB1 total credit 10.2", near(r1["credit"], 10.2, 1e-9), r1)
    chk("GB1 hits [3, 10, 18]", r1["hits"] == [3, 10, 18], r1["hits"])
    chk("GB1 per day 4.8 / 5.4", near(r1["per_day"][0], 4.8, 1e-9) and near(r1["per_day"][1], 5.4, 1e-9), r1["per_day"])
    idle_d = [8.0, 8.0]
    chk("GB1 fraction 0.6375 + LODO 0.675 / 0.6", near(r1["credit"] / 16.0, 0.6375, 1e-12)
        and near(r1["per_day"][1] / idle_d[1], 0.675, 1e-12) and near(r1["per_day"][0] / idle_d[0], 0.6, 1e-12))
    r1b = backfill_spec(dict(GB1_SPEC, busy=list(range(0, 7)) + list(range(14, 20))))   # the run [7,13] alone
    chk("GB1b straddle: day0 1.4, day1 4.0", near(r1b["per_day"][0], 1.4, 1e-9) and near(r1b["per_day"][1], 4.0, 1e-9),
        r1b["per_day"])
    r1b2 = backfill_spec(dict(GB1_SPEC, busy=[6, 14]))   # + run [0,5] (4.4, day0) + unpreempted [15,19] (3.9, day1), by hand
    chk("GB1b whole series busy {6,14}: day0 5.8, day1 7.9", near(r1b2["per_day"][0], 5.8, 1e-9)
        and near(r1b2["per_day"][1], 7.9, 1e-9), r1b2["per_day"])
    r1c = backfill_spec(dict(GB1_SPEC, busy=[3, 4, 10]))
    run_c = r1c["credit"] - 1.4 - 3.4
    chk("GB1c unpreempted run never charged lambda (7.9)", near(run_c, 7.9, 1e-9), r1c)
    # GB2 adversarial
    r2 = backfill_spec(GB2_SPEC)
    chk("GB2 point busy {5,6}", r2["busy"] == [5, 6], r2["busy"])
    chk("GB2 point credit 10 at W_b 0", near(r2["credit"], 10.0, 1e-9), r2)
    chk("GB2 minutes 5, 6 never credited (point)", 5 not in r2["credited_minutes"] and 6 not in r2["credited_minutes"],
        r2["credited_minutes"])
    r2c = backfill_spec(dict(GB2_SPEC, ext=60.0))
    chk("GB2 CONS busy {5,6,7}", r2c["busy"] == [5, 6, 7], r2c["busy"])
    chk("GB2 CONS credit 9, minute 7 not credited", near(r2c["credit"], 9.0, 1e-9) and 7 not in r2c["credited_minutes"]
        and 5 not in r2c["credited_minutes"] and 6 not in r2c["credited_minutes"], r2c)
    # GB2 through the evaluator's own variant switch: CONS must read the extended matrix (and the extended anchor count)
    Bp = busy_matrix2([5], [50.0], [20.0], [0], 1, 12)
    Bx = busy_matrix2([5], [50.0], [20.0], [0], 1, 12, ext=[60.0])
    vc, rc_ = variant_rows("CONS", Bp, Bx, [0] * 12, [1] * 12, 1)
    vp, rp_ = variant_rows("POINT", Bp, Bx, [0] * 12, [1] * 12, 1)
    chk("GB2 variant switch: CONS credit 9 / POINT 10 at W_b 0", near(slot_backfill(vc[0], 0, 0, 0, 12)[0].sum(), 9.0, 1e-12)
        and near(slot_backfill(vp[0], 0, 0, 0, 12)[0].sum(), 10.0, 1e-12) and bool(rc_[0].all()) and not rp_[0].any())
    # property: credited minutes never busy, on a random toy
    rng = np.random.default_rng(7)
    bad = 0
    for _ in range(50):
        b = rng.random(300) < 0.3
        for wb in (0, 1, 5):
            per, hits, mask = slot_backfill(b, wb, 0.05, 0.5, 100)
            bad += int((mask & b).any()) + int(per.sum() > (~b).sum() + 1e-9) + int(any(b[h] == 0 for h in hits))
    chk("GB2 property: no credit on a busy minute, hits are busy minutes", bad == 0, bad)
    # GB3 anchor replicas
    cnt = [0, 2, 1, 0, 0, 3, 0, 0]
    rb = replica_busy(cnt, 2)
    chk("GB3 replica busy sets", list(np.nonzero(rb[0])[0]) == [1, 2, 5] and list(np.nonzero(rb[1])[0]) == [1, 5])
    chk("GB3 16 x sum busy == 16 x sum min(r, count) == 80", 16 * sum(int(x.sum()) for x in rb) == 80
        == 16 * int(np.minimum(np.array(cnt), 2).sum()))
    cr = sum(slot_backfill(x, 1, 0.0, 0.0, 8)[0].sum() for x in rb)
    chk("GB3 backfill 5 replica-minutes = 80 GPU-min", near(cr * 16, 80.0, 1e-12), cr)
    # GB4 SLO bound
    chk("GB4 h<=1 SAFE, h=2 AT-RISK", guaranteed(90, 6, 100, 1, True) and not guaranteed(90, 6, 100, 2, True)
        and guaranteed(90, 6, 100, 0, True))
    chk("GB4 n<20: h=0 SAFE, h=1 AT-RISK", guaranteed(8, 2, 10, 0, False) and not guaranteed(8, 2, 10, 1, False))
    wk = collections.Counter({T._key(0.5): 90, T._key(9.5): 6, T._key(12.0): 4})
    bcn = band_counts(wk, [collections.Counter() for _ in range(NDAYS)], [T._key(9.2), T._key(10.0)])
    chk("GB4 band counts from a Counter", int(bcn["n"][0]) == 100 and int(bcn[T._key(9.2)][0]) == 90
        and int(bcn[T._key(10.0)][0]) == 96, bcn)
    # GB5 classification
    H0 = 10000
    trc = _Tr([H0 - 1500, H0 - 100, H0 - 1440, H0 + 5], [0, 1, 2, 1], [3, 2, 1, 4], 3)
    cc = cool_class(trc, [[0], [1], [2], [0, 1]], H0, 1440)
    chk("GB5 a COOL, b HOT, c (exactly H-1440) HOT, unit {a,b} HOT", cc == {0}, cc)
    chk("GB5 W <= 0 -> no COOL class", cool_class(trc, [[0], [1], [2]], H0, 0) == set())
    # GB6 admission
    r6 = admit_spec(GB6_SPEC)
    hs = [f"h{i}" for i in range(1, 7)]
    chk("GB6 BS", r6["BS"]["admitted"] == hs + ["c1", "c2"], r6["BS"])
    chk("GB6 BZ 15, k_cool 2", r6["BZ"]["admitted"] == hs + [f"c{i}" for i in range(1, 10)] and r6["BZ"]["k_cool"] == 2,
        r6["BZ"])
    chk("GB6 P", r6["P"]["admitted"] == hs + ["c1", "c2"], r6["P"])
    chk("GB6 PZ 15, k_cool 2", r6["PZ"]["admitted"] == hs + [f"c{i}" for i in range(1, 10)] and r6["PZ"]["k_cool"] == 2,
        r6["PZ"])
    chk("GB6 h7 turned away everywhere", all("h7" in r6[a]["turned_away"] for a in r6))
    # GB6b: with no COOL candidates a Z arm equals its base (ZFleet == CS.Fleet)
    spec_hot = dict(GB6_SPEC, candidates=[dict(c, cls="hot") for c in GB6_SPEC["candidates"]])
    r6b = admit_spec(spec_hot)
    chk("GB6b empty COOL class: PZ == P, BZ == BS", r6b["PZ"]["admitted"] == r6b["P"]["admitted"]
        and r6b["BZ"]["admitted"] == r6b["BS"]["admitted"] and r6b["BZ"]["k_cool"] == 0)
    cs_res = CS.admit_spec(dict(N=24, anchor_G=16, anchor_r=1, bh=50.0, arms=["P", "BS"],
                                candidates=[dict(id=c["id"], kind="tail", unit=c["unit"], f=c["f"]) for c in GB6_SPEC["candidates"]]))
    chk("GB6b ZFleet without COOL == Fleet", cs_res["P"]["admitted"] == r6b["P"]["admitted"]
        and cs_res["BS"]["admitted"] == r6b["BS"]["admitted"], (cs_res, r6b))
    # ROOM helper (post-hoc challenger): homes [0, 0, 1], card 50, s 1 -> f [49, 49, 50]; KV tokens from (f - w - OVH)
    fr_, kt_ = room_sizes([0, 0, 1], [8.0, 4.0, 8.0], [2.0 ** 17] * 3, 50.0, 1.0)
    chk("ROOM sizes", fr_ == [49.0, 49.0, 50.0] and near(kt_[1], (49.0 - 4.0 - F.OVH) * 2.0 ** 13, 1e-12), (fr_, kt_))
    chk("GB7 None fraction reads as 0", bf_verdict([None, 0.1], [True, True]) == "NEGLIGIBLE")
    # GB7 verdicts
    for fr, sf, want in (([0.6, 0.55], [True, True], "RECOVERS-MOST"), ([0.6, 0.45], [True, True], "RECOVERS-SOME"),
                         ([0.6, 0.55], [True, False], "SLO-COST"), ([0.1, 0.15], [True, True], "NEGLIGIBLE"),
                         ([0.1, 0.3], [True, True], "UNRESOLVED")):
        chk(f"GB7 {want}", bf_verdict(fr, sf) == want, bf_verdict(fr, sf))
    # GB8 dollars
    chk("GB8 1000 GPU-min / 6 days -> 19.444444 GPU-h/week, $38.8889", near(week_gpu_hours(1000.0), 19.444444444444443, 1e-12)
        and near(week_gpu_hours(1000.0) * 2.0, 38.888888888888886, 1e-12))
    # busy fork == busy (toy)
    t0 = [2, 3, 7, 0, 4, 5, 9]
    la = [50, 40, 10, 30, 59, 0, 55]
    ss = [5, 30, 125, 20, 2, 0.5, 10]
    mo = [0, 0, 0, 1, 1, 2, 2]
    chk("busy fork == CS.busy_matrix (toy, with ext)", np.array_equal(busy_matrix2(t0, la, ss, mo, 3, 10, ext=[0, 61, 5]),
                                                                   CS.busy_matrix(t0, la, ss, mo, 3, 10, ext=[0, 61, 5])))
    chk("inflight fork == CS.inflight (toy)", np.array_equal(inflight2([1, 2], [10, 59], [20, 30], [3, 1], 10),
                                                           CS.inflight([1, 2], [10, 59], [20, 30], [3, 1], 10)))
    # CLI goldens
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f6 = os.path.join(td, "gb6.json")
        with open(f6, "w") as fh:
            json.dump(GB6_SPEC, fh)
        f1 = os.path.join(td, "gb1.json")
        with open(f1, "w") as fh:
            json.dump(GB1_SPEC, fh)
        f2 = os.path.join(td, "gb2.json")
        with open(f2, "w") as fh:
            json.dump(dict(GB2_SPEC, ext=60.0), fh)
        outs = {}
        for nm, args in (("gb6", ["--admit-spec", f6]), ("gb1", ["--backfill-spec", f1]), ("gb2", ["--backfill-spec", f2])):
            try:
                p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args, capture_output=True, text=True, timeout=120)
                outs[nm] = (p.returncode, json.loads(p.stdout) if p.returncode == 0 else {})
            except (subprocess.TimeoutExpired, ValueError):
                outs[nm] = (124, {})
        rc6, j6 = outs["gb6"]
        chk("CLI GB6", rc6 == 0 and j6.get("BZ", {}).get("k_cool") == 2 and len(j6.get("PZ", {}).get("admitted", [])) == 15
            and j6.get("BS", {}).get("turned_away") == [f"c{i}" for i in range(3, 10)] + ["h7"], outs["gb6"])
        rc1, j1 = outs["gb1"]
        chk("CLI GB1", rc1 == 0 and near(j1.get("credit"), 10.2, 1e-9) and j1.get("hits") == [3, 10, 18], outs["gb1"])
        rc2, j2 = outs["gb2"]
        chk("CLI GB2 (CONS)", rc2 == 0 and j2.get("busy") == [5, 6, 7] and 7 not in j2.get("credited_minutes", [7]), outs["gb2"])
    for f in fails:
        print(f)
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'} {n - len(fails)}/{n}")
    return ok


# ------------------------------------------------------------------------------------------------ CLI
def _parse_sets(sets):
    over, extra = {}, dict(EXTRA0)
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set needs k=v, got {s}")
        k, v = s.split("=", 1)
        if k in EXTRA0 or k == "cool_room":
            extra[k] = float(v)
        elif k in NS.NPHYS0 and k != "fmt":
            over[k] = v
        else:
            raise ValueError(f"unknown param {k}")
    return over, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--admit-spec")
    ap.add_argument("--backfill-spec")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", choices=("REUSE", "EMPTY", "LIVE"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--fmt", choices=NS.FMTS)
    ap.add_argument("--N", type=int)
    ap.add_argument("--set", action="append")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    a = ap.parse_args()
    try:
        if a.self_test:
            return 0 if self_test(a.root, a.cache) else 1
        if a.admit_spec:
            with open(a.admit_spec) as fh:
                print(json.dumps(admit_spec(json.load(fh))))
            return 0
        if a.backfill_spec:
            with open(a.backfill_spec) as fh:
                print(json.dumps(backfill_spec(json.load(fh))))
            return 0
        if a.plan:
            if not (a.arm and a.fmt and a.N is not None and a.cache):
                raise ValueError("--plan needs --arm, --fmt, --N and --cache")
            if a.arm not in ARM2:
                raise ValueError(f"unknown arm {a.arm}")
            over, extra = _parse_sets(a.set)
            Z = ZWorld(a.fmt, over, extra, a.root, a.cache)
            try:
                info, snaps = plan2(Z, a.arm, a.N)
            except CS.AnchorError as e:
                print(f"{e}", file=sys.stderr)
                return 3
            out = dict(info, arm=a.arm, fmt=a.fmt, N=a.N, cool_models=len(Z.cool_models), cool_units=len(Z.cool_bs))
            if snaps:
                sn = snaps[max(N_GRID)]
                out.update(admitted=len(sn["admitted"]), loaded_units=len(sn["loaded"]), ded=sn["ded"], k_cool=sn["k_cool"],
                           cool_loaded=len(sn["cool_loaded"]), per_n={n: len(snaps[n]["admitted"]) for n in N_GRID},
                           k_cool_per_n={n: snaps[n]["k_cool"] for n in N_GRID})
            print(json.dumps(out))
            return 0
        if a.check:
            if not a.cache:
                raise ValueError("--check needs --cache")
            res = check(a.check, a.root, a.cache)
            print(json.dumps(res))
            return 0 if all(v for k, v in res.items() if k.endswith("_pass")) else 1
        if a.run:
            if not (a.arm and a.fmt and a.N is not None and a.out and a.cache):
                raise ValueError("--run needs --arm, --fmt, --N, --out and --cache")
            if a.arm not in ARM2:
                raise ValueError(f"unknown arm {a.arm}")
            over, extra = _parse_sets(a.set)
            try:
                return run_job(a.arm, a.fmt, a.N, over, extra, a.tag, a.root, a.cache, a.out)
            except CS.AnchorError as e:
                print(f"{e}", file=sys.stderr)
                return 3
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"--summarize needs a directory, got {a.summarize}")
            print(json.dumps(summarize(a.summarize)))
            return 0
        ap.print_usage()
        return 2
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
