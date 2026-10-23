#!/usr/bin/env python3
"""cloud_sim.py — the gpu-bunk fabric from a cloud's side. A fixed fleet of N H100 SXM 80 GB (8-GPU nodes) must carry
the Kimi-K2-shaped anchor; every other census model competes for the rest under ONE admission rule. Per arm: % of GPU-time idle,
the catalogue hosted at the frozen SLOs (who is turned away), requests served at SLO.

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. Imports `node_sim` and v1/v2 UNCHANGED; forks only (a) run_group restricted to admitted models, (b) run_d with
explicit homes, (c) dedicated_by_day with anchor replicas — each with an equivalence check (prereg §0.1, §8).
Arms (prereg §3): P provider default (dedicated, whole-node giants) · BS gpu-bunk stock (STATIC tail on shared cards + sleeping
time-shared giants) · BC BS with node-scope KV pooling (beta) · BT / PD attribution (tail lever alone / giant lever alone).

Output contract:
  --self-test [--root R] [--cache F]   exit 0 + "SELF-TEST PASS n/n"; exit 1 + FAIL lines. GC6 runs only when --cache exists.
  --admit-spec FILE                     read-only: a toy admission spec (JSON) through the real Fleet code; one JSON object on
                                        stdout; exit 0, or exit 3 + "ANCHOR-NOT-PLACEABLE" on stderr when the anchor cannot fit.
  --plan --arm A --fmt F --N n --root R --cache C [--set k=v]
                                        read-only: anchor sizing + the admission at the full ranking (no serving simulation).
  --check NODE|REP|D|SEED --root R --cache C
                                        read-only instrument checks (prereg §8); one JSON object; exit 0 PASS, 1 FAIL.
  --run --arm A --fmt F --N n [--set k=v ...] [--tag T] --root R --cache C --out F
                                        WRITES (appends) one JSON line per n-grid point to F; exit 0. An anchor that cannot be
                                        placed exits 3 after writing one flagged row; not-servable / degenerate write one flagged row.
  --summarize DIR                       read-only: prereg §6 over DIR/*.jsonl; one JSON object on stdout.
  Bad input (unknown arm/param, missing file, sha mismatch) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed earlier, inherited here):
  1. **A unit index of -1 wraps numpy** — MG models are NO-CARD units in v2; arrays here are indexed by model, never by unit.
  2. **`ssh host '... &'` holds the session open** — launch the grid from a shipped script with setsid.
  3. **The seed is blind for all-awake arms** — replicates are held-out days x seeds; only sleep runs and idle carry vary.
"""
import argparse, collections, copy, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_sim as NS        # noqa: E402  (not edited)
import fabric_sim as F       # noqa: E402  (v1, not edited)
import fabric_sim_v2 as V2   # noqa: E402  (v2, not edited)
import tailpool_sim as T     # noqa: E402  (not edited)
import wakestorm_sim as W    # noqa: E402  (not edited)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

HELD = NS.HELD
NDAYS = NS.NDAYS
DAY_MIN = NS.DAY_MIN
WEEK_MIN = NDAYS * DAY_MIN
TRAIN_S = 7 * 86400.0                 # TRAIN window days 300-306
KV_MIN = NS.KV_MIN
OVH = NS.OVH
SLOS = T.SLOS
SEEDS = (0, 1, 2)
N_MAIN = (64, 128, 256)
N_DESC = (384, 512)
N_GRID = tuple(range(0, 369, 8)) + (374,)   # prereg §4.4: candidates considered (374 non-anchor models)
THETA = 1.0                          # prereg §4.4: P's own (M), 1 by construction
GIVEUP_S = F.W_GIVEUP_S              # 600 s, IDLE-W cap (prereg §5)
ANCHOR_G_DEFAULT = 16
EXTRA0 = dict(rho_cap=F.RHO_DESIGN)  # prereg §1: v1's design load, 0.5
ARM_MODES = {"P": ("DED", "DED"), "BS": ("SPREAD", "SHARE"), "BC": ("NODE", "SHARE"), "BT": ("SPREAD", "DED"),
             "PD": ("DED", "SHARE")}
O_PROC = F.PHYS0["o_proc"]           # 5.35 GiB host overhead per process (run_d)


class AnchorError(ValueError):
    """The fleet cannot hold even one anchor group (prereg §3.3): never silently dropped."""


# ------------------------------------------------------------------------------------------------ idle (prereg §5)
def busy_matrix(t0, last, s, model, n_models, n_min, ext=None):
    """bool (n_models, n_min): cell i marks minutes t0 .. t0 + floor((last + s [+ ext[model]]) / 60) busy, clipped to n_min."""
    t0 = np.asarray(t0, np.int64)
    add = np.asarray(last, float) + np.asarray(s, float)
    if ext is not None:
        add = add + np.asarray(ext, float)[np.asarray(model)]
    end = np.minimum(t0 + np.floor(add / 60.0).astype(np.int64), n_min - 1)
    D = np.zeros((n_models, n_min + 1), np.int32)
    np.add.at(D, (np.asarray(model), t0), 1)
    np.add.at(D, (np.asarray(model), end + 1), -1)
    return np.cumsum(D, axis=1)[:, :n_min] > 0


def inflight(t0, last, s, n, n_min):
    """requests in flight per minute (each cell's n requests over its busy span)."""
    t0 = np.asarray(t0, np.int64)
    end = np.minimum(t0 + np.floor((np.asarray(last, float) + np.asarray(s, float)) / 60.0).astype(np.int64), n_min - 1)
    D = np.zeros(n_min + 1, np.int64)
    np.add.at(D, t0, np.asarray(n, np.int64))
    np.add.at(D, end + 1, -np.asarray(n, np.int64))
    return np.cumsum(D)[:n_min]


def slot_busy_by_day(B, slots, anc_cnt, anc_r, anc_g, n_min, day_len):
    """busy GPU-minutes per day: slots = [(gpus, [model rows of B])]; anchor busy = anc_g x min(anc_r, in flight)."""
    nd = n_min // day_len
    tot = np.zeros(nd, np.int64)
    for g, ms in slots:
        if not ms:
            continue
        b = np.any(B[list(ms)], axis=0)
        tot += g * b.reshape(nd, day_len).sum(axis=1)
    if anc_cnt is not None and anc_r:
        tot += anc_g * np.minimum(anc_cnt, anc_r).reshape(nd, day_len).sum(axis=1)
    return tot


def fleet_idle(B, slots, anc_cnt, anc_r, anc_g, N, n_min, day_len):
    """prereg §5: -> (IDLE per LODO replicate over all N GPUs, ALLOCATED GPUs, IDLE per replicate over the allocated ones)."""
    pdb = slot_busy_by_day(B, slots, anc_cnt, anc_r, anc_g, n_min, day_len)
    alloc = anc_g * anc_r + sum(g for g, ms in slots if ms)
    return idle_reps(pdb, N, day_len), alloc, idle_reps(pdb, alloc, day_len)


def idle_reps(per_day, N, day_len):
    """LODO: replicate d drops day d. IDLE = 1 - busy / (N x day_len x (days - 1)); unallocated GPUs stay in N."""
    nd = len(per_day)
    t = int(per_day.sum())
    return [1.0 - (t - int(per_day[d])) / float(N * day_len * (nd - 1)) for d in range(nd)]


# ------------------------------------------------------------------------------------------------ replicas (prereg §3.3)
def dedicated_rep(trace, kvtok_ded, phys, C, reps, score_from_min=HELD):
    """FORK of node_sim.dedicated_by_day with r identical replicas per model: each cell splits evenly, so the fluid queue
    equals one engine with r x C compute and r x kvtok KV, while each request's prefill runs on one replica (C) and a request
    must fit one replica's KV. reps = 1 is dedicated_by_day exactly (I-REP)."""
    M = len(trace.models)
    out = [collections.Counter() for _ in range(M)]
    day = [[collections.Counter() for _ in range(M)] for _ in range(NDAYS)]
    Cs = list(C) if isinstance(C, (list, tuple, np.ndarray)) else [C] * M
    Rs = list(reps) if isinstance(reps, (list, tuple, np.ndarray)) else [reps] * M
    backlog = [0.0] * M
    bk = [0.0] * M
    fut = [dict() for _ in range(M)]
    kv_last = [None] * M
    last_min = [None] * M
    rcap = F.rho_cap(phys) if kvtok_ded is not None else 0.0
    for i in range(len(trace.n)):
        m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        ab = trace.models[m][2]
        Cm = Cs[m]
        r = Rs[m]
        Cq = Cm * r
        cap_min = Cq * 60.0
        if kvtok_ded is not None and (trace.it[i] + trace.ot[i]) / n > kvtok_ded[m]:
            if mi >= score_from_min:
                out[m][T.INF_KEY] += n
                day[(mi - HELD) // DAY_MIN][m][T.INF_KEY] += n
            continue
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        if last_min[m] is not None:
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        last_min[m] = mi
        kw = 0.0
        if kvtok_ded is not None:
            kvr = kvtok_ded[m] * r
            bk[m] = F.kv_catch_up(bk[m], fut[m], kvr, kv_last[m], mi)
            kv_last[m] = mi
            rho = min(rcap, (backlog[m] + Wm) / cap_min)
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm
            R = pre + (trace.ot[i] / n) * F.tpot_of(rho, phys)
            occ = trace.it[i] / n + phys["kappa"] * trace.ot[i] / n
            D = fut[m].pop(mi, 0.0) + F.spread(fut[m], mi, n * occ, R)
            kw, bk[m] = F.kv_step(bk[m], D, kvr)
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / Cq
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        if mi >= score_from_min:
            k_ = T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm + kw)
            out[m][k_] += n
            day[(mi - HELD) // DAY_MIN][m][k_] += n
    return out, day


# ------------------------------------------------------------------------------------------------ scoring helpers
def cstats(week, days):
    """per LODO replicate d: (n, p95 s | None | inf, {slo: requests <= slo}) of week - day[d], nearest rank exactly as
    tailpool_sim._pct / score; plus the week's own (n, p95)."""
    keys = sorted(week)
    K = len(keys)
    idx = {k: j for j, k in enumerate(keys)}
    Wv = np.array([week[k] for k in keys], np.int64)
    Dm = np.zeros((NDAYS, K), np.int64)
    for d in range(NDAYS):
        for k, v in days[d].items():
            if v:
                Dm[d, idx[k]] = v
    R = Wv[None, :] - Dm
    if (R < 0).any():
        raise RuntimeError("a day Counter exceeds its week Counter")
    n = R.sum(axis=1)
    karr = np.array(keys, np.int64)
    cum = np.cumsum(R, axis=1)

    def pct(cm, tot):
        if tot == 0:
            return None
        need = math.ceil(0.95 * int(tot))
        j = int(np.searchsorted(cm, need, side="left"))
        k = int(karr[j])
        return math.inf if k >= T.INF_KEY else k / 100.0
    p95 = [pct(cum[d], n[d]) for d in range(NDAYS)]
    ok = {str(x): [int(v) for v in R[:, karr <= T._key(x)].sum(axis=1)] if K else [0] * NDAYS for x in SLOS}
    wk_n = int(Wv.sum())
    return dict(n=[int(v) for v in n], p95=p95, ok=ok, week_n=wk_n, week_p95=pct(np.cumsum(Wv), wk_n) if K else None)


def score_reps(hosted, ost, rst, all_models, slo):
    """prereg §2/§5 per replicate: M (tailpool score over HOSTED models), HOSTED-AT-SLO, RS, R. ost/rst: model -> cstats."""
    s = str(slo)
    res = dict(M=[], at=[], RS=[], R=[], elig=[], feas=[], meet=[])
    for d in range(NDAYS):
        elig = feas = meet = at = 0
        req_ok = req_feas = 0
        tot = 0
        for m in all_models:
            tot += rst[m]["n"][d]
        for m in hosted:
            o, r = ost[m], rst[m]
            if o["n"][d] != r["n"][d]:
                raise RuntimeError(f"I-CONS: model {m} out {o['n'][d]} != ref {r['n'][d]} requests on replicate {d}")
            req_ok += o["ok"][s][d]
            req_feas += r["ok"][s][d]
            if r["n"][d] < 20:
                if o["ok"][s][d] == o["n"][d]:
                    at += 1
                continue
            elig += 1
            if r["p95"][d] > slo:
                continue
            feas += 1
            if o["p95"][d] <= slo:
                meet += 1
                at += 1
        res["M"].append(meet / feas if feas else None)
        res["at"].append(at)
        res["RS"].append(req_ok / tot if tot else None)
        res["R"].append(req_ok / req_feas if req_feas else None)
        res["elig"].append(elig); res["feas"].append(feas); res["meet"].append(meet)
    res["pass"] = [m_ is None or m_ >= THETA - 1e-12 for m_ in res["M"]]
    return res


def verdict_hi(x, p):
    """prereg §6 (higher is better): MORE iff min x > max p; FEWER iff max x < min p."""
    if min(x) > max(p):
        return "MORE"
    if max(x) < min(p):
        return "FEWER"
    return "UNRESOLVED"


def verdict_idle(x, p):
    if max(x) < min(p):
        return "LOWER-IDLE"
    if min(x) > max(p):
        return "HIGHER-IDLE"
    return "UNRESOLVED"


def verdict_rs(x, p):
    if min(x) > max(p):
        return "LIFTS"
    if max(x) < min(p):
        return "LOWERS"
    return "UNATTRIBUTABLE"


def stable_n(passmap):
    """prereg §4.4: largest grid n such that every grid n' <= n passes; None if n = 0 fails."""
    best = None
    for n in sorted(passmap):
        if passmap[n]:
            best = n
        else:
            break
    return best


def density(train_req, w_gib):
    """prereg §4.1: TRAIN requests per GiB of weights."""
    return float(train_req) / float(w_gib)


# ------------------------------------------------------------------------------------------------ admission (prereg §4)
def tail_nodes(groups, K):
    """NODE: per node with >= 1 tail GPU: (node index, [slot positions]) in pack_nodes order."""
    out = []
    for ni, nd in enumerate(NS.pack_nodes(groups, K)):
        pos = [j for j, s in enumerate(nd) if s[0] == "full"]
        if pos:
            out.append((ni, pos))
    return out


class Fleet:
    """Admission state of one arm on N GPUs. tail_mode DED | SPREAD | NODE, giant_mode DED | SHARE. units: uid -> dict(cid, f,
    w, host). gp: dict(bh, h_tot, node_ram, r_pin, o_g, s_g, rho_cap). No serving simulation happens here."""

    def __init__(self, N, tail_mode, giant_mode, anchor_G, anchor_r, units, gp):
        if N % 8:
            raise ValueError(f"N must be a whole number of 8-GPU nodes, got {N}")
        if anchor_G > N:
            raise AnchorError(f"ANCHOR-NOT-PLACEABLE: the anchor needs {anchor_G} GPUs, the fleet has {N}")
        if anchor_r < 1 or anchor_G * anchor_r > N:
            raise AnchorError(f"ANCHOR-NOT-PLACEABLE: {anchor_r} x {anchor_G} GPUs > {N}")
        self.N, self.tm, self.gm, self.units, self.gp = N, tail_mode, giant_mode, units, gp
        self.anchor = [(f"anchor{i:02d}", anchor_G) for i in range(anchor_r)]
        self.groups = []
        self.loaded = []
        self.ded = 0
        self.admitted = []

    # -- layout
    def group_list(self, extra=None):
        gl = list(self.anchor) + [(g["key"], g["G"]) for g in self.groups]
        return gl + ([extra] if extra else [])

    def group_gpus(self):
        return sum(G for _, G in self.group_list())

    def k_tail(self):
        """tail cards: SPREAD / NODE = every GPU not in a group; DED = free GPUs."""
        return self.N - self.group_gpus() - (self.ded if self.tm == "DED" else 0)

    def nodes_ok(self, groups, k_full):
        return len(NS.pack_nodes(groups, k_full)) <= self.N // 8

    def place(self, uids, K, groups=None):
        """-> homes per uid (SPREAD: card index; NODE: (node index, slot position)); ValueError if infeasible."""
        info = [dict(self.units[u]) for u in uids]
        if not uids:
            return []
        if K <= 0:
            raise ValueError("no tail cards")
        if self.tm == "SPREAD":
            return NS.hetero_spread(info, [[self.gp["bh"], self.gp["h_tot"]] for _ in range(K)])
        tn = tail_nodes(groups if groups is not None else self.group_list(), K)
        nodes = [dict(free=len(pos) * self.gp["bh"], gpus=[[self.gp["bh"], self.gp["h_tot"]] for _ in pos]) for _, pos in tn]
        hm = NS.node_place(info, nodes)
        return [(tn[n][0], tn[n][1][g]) for n, g in hm]

    # -- candidates
    def try_tail(self, cid, uid):
        if self.tm == "DED":
            if self.N - self.group_gpus() - self.ded >= 1 and self.nodes_ok(self.group_list(), self.ded + 1):
                self.ded += 1
                self.admitted.append(cid)
                return True
            return False
        if uid in self.loaded:                       # a sibling rides its unit's engine at 0 GiB
            self.admitted.append(cid)
            return True
        K = self.k_tail()
        try:
            if not self.nodes_ok(self.group_list(), K):
                return False
            self.place(self.loaded + [uid], K)
        except ValueError:
            return False
        self.loaded.append(uid)
        self.admitted.append(cid)
        return True

    def fits_group(self, G, Ws):
        """prereg §3.2 / §4.3 (b)-(c): the group's host RAM holds every homed giant and each keeps >= G x 0.25 GiB of KV."""
        gp = self.gp
        host = sum(w * gp["r_pin"] + G * O_PROC for w in Ws)
        if host > NS.d_host_cap(G, dict(node_ram=gp["node_ram"])) + 1e-9:
            return False
        s = gp["s_g"] * G
        for w in Ws:
            if G * gp["bh"] - w - G * gp["o_g"] - s * (len(Ws) - 1) < KV_MIN * G - 1e-9:
                return False
        return True

    def join_ok(self, g, G, W_, rho):
        """prereg §4.3: joining needs (a) the rho cap AND (b)-(c)."""
        if g["rho"] + rho > self.gp["rho_cap"] + 1e-12:
            return False
        return self.fits_group(G, g["W"] + [W_])

    def try_giant(self, cid, G, W_, rho):
        if self.gm == "SHARE":
            for g in self.groups:
                if g["G"] == G and self.join_ok(g, G, W_, rho):
                    g["members"].append(cid); g["W"].append(W_); g["rho"] += rho
                    self.admitted.append(cid)
                    return True
        key = f"g{len(self.groups):03d}"
        new = (key, G)
        if self.group_gpus() + G > self.N:
            return False
        try:
            if self.tm == "DED":
                if self.N - self.group_gpus() - G - self.ded < 0 or not self.nodes_ok(self.group_list(new), self.ded):
                    return False
            else:
                K = self.N - self.group_gpus() - G
                if not self.nodes_ok(self.group_list(new), K):
                    return False
                if self.loaded:
                    self.place(self.loaded, K, groups=self.group_list(new))
        except ValueError:
            return False
        if self.gm == "SHARE":                       # a giant alone: host RAM / KV floor only; the rho cap governs JOINING (§4.2)
            if not self.fits_group(G, [W_]):
                return False
        self.groups.append(dict(key=key, G=G, members=[cid], W=[W_], rho=rho))
        self.admitted.append(cid)
        return True

    def snapshot(self):
        return dict(admitted=list(self.admitted), loaded=list(self.loaded), ded=self.ded,
                    groups=[dict(g, members=list(g["members"]), W=list(g["W"])) for g in self.groups],
                    anchor=list(self.anchor))


def walk(fleet, cands, grid):
    """prereg §4.2 greedy with skip over the ranking; snapshots after the first n candidates for every grid n."""
    snaps = {}
    want = sorted(set(grid))
    if 0 in want:
        snaps[0] = fleet.snapshot()
    for i, c in enumerate(cands):
        if c["kind"] == "tail":
            fleet.try_tail(c["cid"], c["uid"])
        else:
            fleet.try_giant(c["cid"], c["G"], c["W"], c["rho"])
        if i + 1 in want:
            snaps[i + 1] = fleet.snapshot()
    for n in want:                                   # a grid point past the end of the ranking = the full walk
        snaps.setdefault(n, fleet.snapshot())
    return snaps


def admit_spec(spec):
    """Toy admission through the real Fleet (CLI golden): spec = dict(N, anchor_G, anchor_r, bh, arms [..], candidates [dict(id,
    kind tail|giant, unit, f, w, G, W, rho, held)], held_anchor)."""
    res = {}
    held = {c["id"]: c.get("held", 0) for c in spec["candidates"]}
    tot = spec.get("held_anchor", 0) + sum(held.values())
    for arm in spec.get("arms", ["P", "BS"]):
        tm, gm = ARM_MODES[arm]
        units = {}
        for c in spec["candidates"]:
            if c["kind"] == "tail":
                units.setdefault(c["unit"], dict(cid=c["unit"], f=float(c["f"]), w=float(c.get("w", c["f"])), host=0.0))
        gp = dict(bh=float(spec["bh"]), h_tot=1e9, node_ram=1e9, r_pin=1.0, o_g=0.0, s_g=0.0, rho_cap=1e9)
        fl = Fleet(spec["N"], tm, gm, spec["anchor_G"], spec["anchor_r"], units, gp)
        cands = [dict(cid=c["id"], kind=c["kind"], uid=c.get("unit"), G=c.get("G"), W=float(c.get("W", 0.0)),
                      rho=float(c.get("rho", 0.0))) for c in spec["candidates"]]
        walk(fl, cands, [len(cands)])
        away = [c["id"] for c in spec["candidates"] if c["id"] not in fl.admitted]
        res[arm] = dict(admitted=fl.admitted, turned_away=away, free=fl.k_tail() if tm == "DED" else None,
                        k_tail=fl.k_tail(), groups=[(g["key"], g["G"], g["members"]) for g in fl.groups],
                        turned_away_share=(sum(held[a] for a in away) / tot) if tot else None)
    return res


# ------------------------------------------------------------------------------------------------ the world (one format x physics)
class World:
    def __init__(self, np_, extra, root, cache):
        self.np, self.extra, self.root, self.cache = np_, extra, root, cache
        cx = self.cx = NS.ctx(root, cache, np_, "STATIC")
        tr, M, tab, S = cx["tr"], cx["M"], cx["tab"], cx["S"]
        self.tph = cx["tph"]
        self.bh = NS.b_h(np_)
        self.Ch = C_REF * self.tph["cref_mult"] * self.tph["big_mult"]
        self.am = cx["cid_m"][NS.ANCHOR]
        self.mg = set(cx["mg"])
        self.tail = [m for m in range(M) if S["unit_of"][m] >= 0]
        self.uinfo = NS.tail_units_info(cx)
        held = tr.minute >= HELD
        self.nT = np.bincount(tr.mid[~held], weights=tr.n[~held].astype(float), minlength=M)
        self.nH = np.bincount(tr.mid[held], weights=tr.n[held].astype(float), minlength=M)
        if int(self.nH.sum()) != 3687246:
            raise RuntimeError(f"held-out requests {int(self.nH.sum())} != 3,687,246")
        self.gp = dict(bh=self.bh, h_tot=self.tph["h_tot"], node_ram=np_["node_ram"], r_pin=np_["r_pin"], o_g=np_["o_g"],
                       s_g=np_["s_g"], rho_cap=extra["rho_cap"])
        self.rho = {}
        for cid, d in tab.items():
            if cid == NS.ANCHOR or not d["placeable"]:
                continue
            m = cx["cid_m"][cid]
            k = (tr.mid == m) & ~held
            work = float(((tr.it[k] + tr.ot[k]) * (tr.models[m][2] / ACTIVE_REF_B)).sum())
            self.rho[cid] = work / (TRAIN_S * self.cg(d["G"]))
        self.cands = self.ranking()
        rw, rd = NS.full_ref(cx, np_)
        self.ref_w, self.ref_d = rw, rd             # anchor entries overridden per r (anchor_ref)
        self._rst = {}
        self._anc = {}
        self._busy = {}
        self._arr = {}

    def cg(self, G):
        return G * C_REF * self.np["m_h"] * self.np["e_par"]

    def ranking(self):
        cx = self.cx
        tr, tab, S = cx["tr"], cx["tab"], cx["S"]
        out = []
        for m in range(cx["M"]):
            cid = tr.models[m][0]
            if m == self.am:
                continue
            if m in self.mg:
                d = tab[cid]
                if not d["placeable"]:
                    continue
                out.append(dict(cid=cid, m=m, kind="giant", G=d["G"], W=d["W"], rho=self.rho[cid],
                                dens=density(self.nT[m], d["W"])))
            else:
                out.append(dict(cid=cid, m=m, kind="tail", uid=S["unit_of"][m], w=tr.models[m][4],
                                dens=density(self.nT[m], tr.models[m][4])))
        out.sort(key=lambda c: (-c["dens"], c["cid"]))
        return out

    # -- anchor (prereg §3.3)
    def anchor_out(self, r):
        if r not in self._anc:
            cx, np_ = self.cx, self.np
            d = cx["tab"][NS.ANCHOR]
            m = self.am
            kt = [-1.0] * cx["M"]
            kt[m] = NS.kv_tokens(d["G"], d["L0"], cx["S"]["pop"]["bpt"][m], d["dup"])
            Cs = [1.0] * cx["M"]
            Cs[m] = self.cg(d["G"])
            gp = NS.giant_phys(np_, self.tph)
            w, dy = dedicated_rep(NS.subtrace(cx["tr"], [m]), kt, gp, Cs, r)
            self._anc[r] = (w[m], [dy[x][m] for x in range(NDAYS)])
        return self._anc[r]

    def anchor_r_min(self, r_top=32):
        """smallest r whose replica engine holds p95 <= 10 s on all 7 LODO replicates (None if none <= r_top)."""
        for r in range(1, r_top + 1):
            st = cstats(*self.anchor_out(r))
            if all(p is not None and p <= 10.0 for p in st["p95"]):
                return r
        return None

    def anchor_load(self):
        """premise / GC6: per-minute rho of the anchor's held-out work on one point-physics replica."""
        cx = self.cx
        tr = cx["tr"]
        m = self.am
        k = (tr.mid == m) & (tr.minute >= HELD)
        per = np.zeros(WEEK_MIN)
        per[tr.minute[k] - HELD] = (tr.it[k] + tr.ot[k]) * (tr.models[m][2] / ACTIVE_REF_B)
        rho = per / (self.cg(cx["tab"][NS.ANCHOR]["G"]) * 60.0)
        cs = np.cumsum(np.concatenate([[0.0], per]))
        h60 = (cs[60:] - cs[:-60]) / 60.0 / (self.cg(cx["tab"][NS.ANCHOR]["G"]) * 60.0)
        return dict(max_min=float(rho.max()), max_60=float(h60.max()), mean=float(rho.mean()))

    # -- references and per-replicate stats
    def ref_stats(self, m, r):
        key = (m, r if m == self.am else 0)
        if key not in self._rst:
            if m == self.am:
                self._rst[key] = cstats(*self.anchor_out(r))
            else:
                self._rst[key] = cstats(self.ref_w[m], [self.ref_d[x][m] for x in range(NDAYS)])
        return self._rst[key]

    # -- busy (prereg §5)
    def arrivals(self, seed):
        if seed not in self._arr:
            tr = self.cx["tr"]
            _, last, _ = T.arrivals(tr, seed)
            k = np.nonzero((tr.minute >= HELD) & (tr.minute < HELD + WEEK_MIN))[0]
            self._arr[seed] = (k, last[k])
        return self._arr[seed]

    def service(self, kind, beta=0.0):
        """unloaded service seconds per held-out cell for an engine kind: 'tail' (one H100, alone), 'giant' (C_g of the model's
        own G, giant TPOT; the anchor per replica)."""
        cx = self.cx
        tr = cx["tr"]
        k, _ = self.arrivals(0)
        mid = tr.mid[k]
        n = tr.n[k].astype(float)
        ab = np.array([mm[2] for mm in tr.models])[mid]
        if kind == "tail":
            tpot = self.tph["tpot_q"] * (1.0 + beta)
            C = np.full(len(k), self.Ch)
        else:
            tpot = self.np["g_tpot"]
            Gm = np.zeros(cx["M"])
            for cid, d in cx["tab"].items():
                if d["placeable"]:
                    Gm[cx["cid_m"][cid]] = d["G"]
            C = np.where(Gm[mid] > 0, Gm[mid] * C_REF * self.np["m_h"] * self.np["e_par"], self.Ch)
        return (tr.it[k] / n) * (ab / ACTIVE_REF_B) / C + (tr.ot[k] / n) * tpot

    def busy(self, seed, beta=0.0, ext=None):
        """bool (M, WEEK_MIN): tail models on their H100 engine (decode x (1 + beta)), MG models on their giant engine."""
        key = (seed, beta) if ext is None else None
        if key is not None and key in self._busy:
            return self._busy[key]
        cx = self.cx
        tr = cx["tr"]
        k, last = self.arrivals(seed)
        mid = tr.mid[k]
        isg = np.zeros(cx["M"], bool)
        isg[list(self.mg)] = True
        s = np.where(isg[mid], self.service("giant"), self.service("tail", beta))
        B = busy_matrix(tr.minute[k] - HELD, last, s, mid, cx["M"], WEEK_MIN, ext=ext)
        if key is not None:
            self._busy[key] = B
        return B

    def anchor_cnt(self, seed, ext=0.0):
        cx = self.cx
        tr = cx["tr"]
        k, last = self.arrivals(seed)
        sel = tr.mid[k] == self.am
        s = self.service("giant")[sel] + ext
        return inflight(tr.minute[k][sel] - HELD, last[sel], s, tr.n[k][sel], WEEK_MIN)

    def req_day(self):
        if "_rd" not in self.__dict__:
            tr = self.cx["tr"]
            held = (tr.minute >= HELD)
            d = (tr.minute[held] - HELD) // DAY_MIN
            self._rd = np.zeros((NDAYS, self.cx["M"]), np.int64)
            np.add.at(self._rd, (d, tr.mid[held]), tr.n[held].astype(np.int64))
        return self._rd


# ------------------------------------------------------------------------------------------------ forks of runners
def run_tail(cx, units_idx, homes, K, card_gib, C, phys, admitted):
    """FORK of node_sim.run_group: only ADMITTED models' cells are simulated (a loaded unit's non-admitted siblings are
    not served). With every tail model admitted it equals run_group(models_extra=()) exactly (I-NODE)."""
    S, tr, M = cx["S"], cx["tr"], cx["M"]
    cfg = dict(S["cfg"])
    cfg["h2d"], cfg["d2h"], cfg["protect"] = phys["h2d"], phys["d2h"], phys["protect"]
    sub_units = [S["units"][u] for u in units_idx]
    new_of = {u: j for j, u in enumerate(units_idx)}
    unit_of = [new_of.get(S["unit_of"][m], -1) if (S["unit_of"][m] >= 0 and m in admitted) else -1 for m in range(M)]
    keep_m = [m for m in range(M) if unit_of[m] >= 0]
    st = NS.relabel_days(NS.subtrace(tr, keep_m), M)
    uo8 = unit_of * (1 + NDAYS)
    rs = dict(H=phys["resize_h"], restart_s=phys["restart_s"], thr=phys["resize_thr"]) if cfg["mech"] == "resize" else None
    out, met = V2.simulate(st, sub_units, uo8, ("card", homes), K, cfg, 0, None, card_gib=card_gib, phys=phys,
                           kvtok=[S["kvtok"][u] for u in units_idx], copies=None, C=C, mech=cfg["mech"],
                           shed_tau=cfg["shed_tau"], bpt=[S["bpt"][u] for u in units_idx], resize=rs)
    w, dy = NS.split_days(out, M)
    return {m: w[m] for m in keep_m}, {m: [dy[d][m] for d in range(NDAYS)] for m in keep_m}, met


def run_dgroups(cx, np_, groups, seed):
    """FORK of node_sim.run_d with explicit homes: groups = [(G, [cid ...]) ...]; one v2 sleep-path simulate per G-class.
    -> (week {m: Counter}, day {m: [7 Counters]}, stats). Host RAM / KV floors re-asserted (prereg §4.3, §8)."""
    tab, tr, M, S = cx["tab"], cx["tr"], cx["M"], cx["S"]
    bh = NS.b_h(np_)
    byG = collections.defaultdict(list)
    for G, mem in groups:
        byG[G].append(list(mem))
    ow, od, stats = {}, {}, {}
    for G, gs in sorted(byG.items()):
        cls = [c for g in gs for c in g]
        homes = [i for i, g in enumerate(gs) for _ in g]
        Kc = len(gs)
        card = G * bh
        s = np_["s_g"] * G
        nh = collections.Counter(homes)
        units, kt, bpt = [], [], []
        host = collections.Counter()
        for k, cid in enumerate(cls):
            d = tab[cid]
            m = cx["cid_m"][cid]
            kv = card - d["W"] - G * np_["o_g"] - s * (nh[homes[k]] - 1)
            if kv < KV_MIN * G - 1e-9:
                raise RuntimeError(f"D KV floor: {cid[:8]} {kv:.2f} GiB")
            host[homes[k]] += d["W"] * np_["r_pin"] + G * O_PROC
            b = S["pop"]["bpt"][m] * d["dup"]
            units.append(dict(cid=cid, rep=m, members=[m], tb=tr.models[m][1], prec=np_["fmt"], w=d["W"],
                              f=d["W"] + G * np_["o_g"] + kv, shape=(tr.models[m][1], np_["fmt"])))
            kt.append(kv * NS.G30 / b)
            bpt.append(b)
        for g, h in host.items():
            if h > NS.d_host_cap(G, np_) + 1e-9:
                raise RuntimeError(f"D host RAM: group {g} {h:.1f} GiB")
        cfg, _ = V2.arm_cfg2("ALL")
        cfg = dict(cfg, copies=False, topo="direct", h2d=G * NS.H2D_PCIE, d2h=G * NS.D2H_PCIE)
        cfg["protect"] = cx["tph"]["protect"]
        gp = NS.giant_phys(np_, cx["tph"])
        unit_of = [-1] * M
        for k, cid in enumerate(cls):
            unit_of[cx["cid_m"][cid]] = k
        keep = [cx["cid_m"][c] for c in cls]
        st = NS.relabel_days(NS.subtrace(tr, keep), M)
        out, met = V2.simulate(st, units, unit_of * (1 + NDAYS), ("card", homes), Kc, cfg, seed, None, card_gib=card, s=s,
                               phys=gp, kvtok=kt, copies=None, C=G * C_REF * np_["m_h"] * np_["e_par"], mech=None,
                               shed_tau=None, bpt=bpt, resize=None)
        w, dy = NS.split_days(out, M)
        for m in keep:
            ow[m] = w[m]
            od[m] = [dy[d][m] for d in range(NDAYS)]
        stats[str(G)] = met["stats"]
    return ow, od, stats


# ------------------------------------------------------------------------------------------------ one fleet state -> one row
def evaluate(wd, arm, snap, N, r, beta, seeds=SEEDS):
    cx = wd.cx
    tr, M, tab, S = cx["tr"], cx["M"], cx["tab"], cx["S"]
    tm, gm = ARM_MODES[arm]
    adm = [cx["cid_m"][c] for c in snap["admitted"]]
    tail_adm = [m for m in adm if m not in wd.mg]
    mg_adm = [m for m in adm if m in wd.mg]
    hosted = [wd.am] + adm
    t0 = time.time()
    out_w, out_d = {}, {}
    tail_slots = []
    tail_stats = {}
    tail_from_ref = tm == "DED"
    if tm == "DED":
        tail_slots = [(1, [m]) for m in tail_adm]
    elif snap["loaded"]:
        K = N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"])
        groups = list(snap["anchor"]) + [(g["key"], g["G"]) for g in snap["groups"]]
        loaded = snap["loaded"]
        aset = set(tail_adm)
        info = [dict(wd.uinfo[u]) for u in loaded]
        if tm == "SPREAD":
            homes = NS.hetero_spread(info, [[wd.bh, wd.tph["h_tot"]] for _ in range(K)])
            wk, dy, met = run_tail(cx, loaded, homes, K, wd.bh, wd.Ch, wd.tph, aset)
            out_w.update(wk); out_d.update(dy)
            NS._stats_add(tail_stats, met)
            on = collections.defaultdict(list)
            for m in tail_adm:
                on[homes[loaded.index(S["unit_of"][m])]].append(m)
            tail_slots = [(1, on[k]) for k in range(K)]
        else:                                        # NODE (arm C on the tail's own GPUs)
            tn = tail_nodes(groups, K)
            nodes = [dict(free=len(pos) * wd.bh, gpus=[[wd.bh, wd.tph["h_tot"]] for _ in pos]) for _, pos in tn]
            hm = NS.node_place(info, nodes)
            pc = NS.tail_phys(wd.np, "STATIC", {"tpot_q": wd.tph["tpot_q"] * (1 + beta),
                                                "tpot_sat": wd.tph["tpot_sat"] * (1 + beta)})
            bykey = collections.defaultdict(list)
            for j, (_, pos) in enumerate(tn):
                bykey[(round(len(pos) * wd.bh, 9), round(len(pos) * wd.Ch, 6))].append(j)
            for (cg, cn), js in sorted(bykey.items()):
                idx = {j: x for x, j in enumerate(js)}
                us = [u for u, (n_, _) in enumerate(hm) if n_ in idx]
                if not us:
                    continue
                wk, dy, met = run_tail(cx, [loaded[u] for u in us], [idx[hm[u][0]] for u in us], len(js), cg, cn, pc, aset)
                out_w.update(wk); out_d.update(dy)
                NS._stats_add(tail_stats, met)
            on = collections.defaultdict(list)
            for m in tail_adm:
                u = loaded.index(S["unit_of"][m])
                on[(tn[hm[u][0]][0], tn[hm[u][0]][1][hm[u][1]])].append(m)
            tail_slots = [(1, on[(ni, p)]) for ni, pos in tn for p in pos]
    else:
        K = N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"])
        tail_slots = [(1, []) for _ in range(K)]
    # giants
    if gm == "DED":
        gslots = [(tab[tr.models[m][0]]["G"], [m]) for m in mg_adm]
        d_runs = {s_: ({}, {}, {}) for s_ in seeds}
    else:
        gslots = [(g["G"], [cx["cid_m"][c] for c in g["members"]]) for g in snap["groups"]]
        grp = [(g["G"], g["members"]) for g in snap["groups"]]
        d_runs = {s_: (run_dgroups(cx, wd.np, grp, s_) if grp else ({}, {}, {})) for s_ in seeds}
    # stats per model (per seed where the sleep path differs)
    allm = list(range(M))
    rst = {m: wd.ref_stats(m, r) for m in allm}
    ost_common = {wd.am: rst[wd.am]}
    for m in tail_adm:
        ost_common[m] = rst[m] if tail_from_ref else cstats(out_w[m], out_d[m])
    if gm == "DED":
        for m in mg_adm:
            ost_common[m] = rst[m]
    row = dict(arm=arm, N=N, r_anchor=r, beta=beta, hosted=len(hosted), admitted=[c[:8] for c in snap["admitted"]],
               n_tail=len(tail_adm), n_mg=len(mg_adm), loaded_units=len(snap["loaded"]), ded_tail=snap["ded"],
               groups=[dict(G=g["G"], members=[c[:8] for c in g["members"]], rho=round(g["rho"], 6)) for g in snap["groups"]],
               k_tail=(N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"]) - snap["ded"]
                       if tm == "DED" else N - sum(G for _, G in snap["anchor"]) - sum(g["G"] for g in snap["groups"])),
               tail_stats=tail_stats)
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
    anc_g = snap["anchor"][0][1] if snap["anchor"] else tab[NS.ANCHOR]["G"]   # QA r1: the anchor's own G, never a constant
    for s_ in seeds:
        ost = dict(ost_common)
        gw, gd, gst = d_runs[s_]
        d_stats[str(s_)] = gst
        for m in mg_adm:
            if m not in ost:
                ost[m] = cstats(gw[m], gd[m])
        for x in SLOS:
            sc = score_reps(hosted, ost, rst, allm, x)
            for k_, v in sc.items():
                slo_res[str(x)][k_].extend(v)
        # idle
        slots = tail_slots + gslots
        B = wd.busy(s_, beta if tm == "NODE" else 0.0)
        acnt = wd.anchor_cnt(s_)
        i_n, alloc, i_a = fleet_idle(B, slots, acnt, r, anc_g, N, WEEK_MIN, DAY_MIN)
        idle.extend(i_n)
        alloc_l.append(alloc)
        idle_a.extend(i_a)
        ext = np.zeros(M)
        for m in hosted:
            p = ost[m]["week_p95"]
            ext[m] = 0.0 if p is None else min(p, GIVEUP_S)
        Bw = wd.busy(s_, beta if tm == "NODE" else 0.0, ext=ext)
        pdw = slot_busy_by_day(Bw, slots, wd.anchor_cnt(s_, ext=float(ext[wd.am])), r, anc_g, WEEK_MIN, DAY_MIN)
        idle_w.extend(idle_reps(pdw, N, DAY_MIN))
        wk_req = rd.sum(axis=0)
        for d in range(NDAYS):
            rep_req = wk_req - rd[d]
            zero_g.append(int(sum(g for g, ms in slots if ms and sum(int(rep_req[m]) for m in ms) == 0)))
    row["slo"] = {k: dict(v) for k, v in slo_res.items()}
    row.update(idle=idle, idle_w=idle_w, idle_alloc=idle_a, zero_traffic_gpus=zero_g, allocated=alloc_l[0],
               unallocated=N - alloc_l[0], d_stats=d_stats, reps=[[s_, d] for s_ in seeds for d in range(NDAYS)])
    row["sim_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------------ jobs
def make_world(fmt, over, extra, root, cache):
    over = dict(over)
    over["fmt"] = fmt
    np_ = NS.nphys_with(over)
    return World(np_, extra, root, cache)


def plan(wd, arm, N):
    """anchor r_N + the admission walk (no serving simulation)."""
    G = wd.cx["tab"][NS.ANCHOR]["G"]
    if G > N:
        raise AnchorError(f"ANCHOR-NOT-PLACEABLE: the anchor needs {G} GPUs, the fleet has {N}")
    rmin = wd.anchor_r_min()
    flag = None
    if rmin is None or rmin > N // G:
        flag = "ANCHOR-NOT-SERVABLE"
    elif rmin * G == N:
        flag = "DEGENERATE"
    if flag:
        return dict(flag=flag, r_min=rmin, r=None if flag == "ANCHOR-NOT-SERVABLE" else rmin), None
    tm, gm = ARM_MODES[arm]
    units = {u: dict(cid=x["cid"], f=x["f"], w=x["w"], host=x["host"]) for u, x in enumerate(wd.uinfo)}
    fl = Fleet(N, tm, gm, G, rmin, units, wd.gp)
    snaps = walk(fl, wd.cands, N_GRID)
    return dict(flag=None, r_min=rmin, r=rmin), snaps


def run_job(arm, fmt, N, over, extra, tag, root, cache, out, seeds=SEEDS):
    if arm not in ARM_MODES:
        raise ValueError(f"unknown arm {arm}")
    wd = make_world(fmt, over, extra, root, cache)
    beta = wd.np["beta"]
    base = dict(arm=arm, fmt=fmt, N=N, tag=tag, nphys=wd.np, extra=extra)
    with open(out, "a") as fh:
        try:
            info, snaps = plan(wd, arm, N)
        except AnchorError as e:
            fh.write(json.dumps(dict(base, flag="ANCHOR-NOT-PLACEABLE", error=str(e))) + "\n")
            raise
        if info["flag"]:
            row = dict(base, flag=info["flag"], r_min=info["r_min"])
            if info["flag"] == "DEGENERATE":
                fl = Fleet(N, *ARM_MODES[arm], wd.cx["tab"][NS.ANCHOR]["G"], info["r"], {}, wd.gp)
                row.update(evaluate(wd, arm, fl.snapshot(), N, info["r"], beta, seeds), n=0)
                row.update(base, flag="DEGENERATE")
            fh.write(json.dumps(row) + "\n"); fh.flush()
            return 0
        prev, prev_len = None, None
        for n in N_GRID:
            sn = snaps[n]
            t0 = time.time()
            if prev is not None and len(sn["admitted"]) == prev_len:
                row = dict(prev, n=n, reused_from=prev["n"], elapsed_s=0.0)
            else:
                row = dict(base, n=n, r_min=info["r_min"])
                row.update(evaluate(wd, arm, sn, N, info["r"], beta, seeds))
                row.update(base)
                row["elapsed_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            prev, prev_len = row, len(sn["admitted"])
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks (§8)
def check(which, root, cache):
    wd = make_world("int4", {}, dict(EXTRA0), root, cache)
    cx = wd.cx
    if which == "NODE":
        K = 128
        loaded = list(range(len(wd.uinfo)))
        homes = NS.hetero_spread([dict(x) for x in wd.uinfo], [[wd.bh, wd.tph["h_tot"]] for _ in range(K)])
        a_w, a_d, a_m = run_tail(cx, loaded, homes, K, wd.bh, wd.Ch, wd.tph, set(wd.tail))
        b_w, b_d, b_m = NS.run_group(cx, loaded, homes, K, wd.bh, wd.Ch, wd.tph)
        ok = (set(a_w) == set(b_w) and all(a_w[m] == b_w[m] and a_d[m] == b_d[m] for m in a_w)
              and a_m["stats"] == b_m["stats"])
        return {"I_NODE": dict(models=len(a_w), stats_equal=a_m["stats"] == b_m["stats"]), "I_NODE_pass": bool(ok)}
    if which == "REP":
        w1, d1 = wd.anchor_out(1)
        gw, gd = NS.mg_ref(cx, wd.np)
        m = wd.am
        ok = w1 == gw[m] and all(d1[x] == gd[m][x] for x in range(NDAYS))
        return {"I_REP": dict(n=sum(w1.values()), p95=W._jsonable(T._pct(w1, 0.95))), "I_REP_pass": bool(ok)}
    if which == "D":
        G = 4
        cls = NS.d_classes(cx)[G]
        n_c = len(cls)
        ref = NS.run_d(G, n_c, 0, wd.np, root, cache)
        homes = [k % n_c for k in range(n_c)]
        groups = [(G, [c for k, c in enumerate(cls) if homes[k] == g]) for g in range(n_c)]
        ow, od, st = run_dgroups(cx, wd.np, groups, 0)
        mine = {c[:8]: W._jsonable(T._pct(ow[cx["cid_m"][c]], 0.95)) for c in cls}
        ok = mine == ref["p95"] and st[str(G)] == ref["stats"]
        return {"I_D": dict(members=n_c, mine=mine, ref=ref["p95"]), "I_D_pass": bool(ok)}
    if which == "SEED":
        info, snaps = plan(wd, "BT", 128)          # all-awake arm (STATIC tail + dedicated giants + anchor)
        sn = snaps[N_GRID[len(N_GRID) // 2]]
        r = info["r"]
        a = evaluate(wd, "BT", sn, 128, r, wd.np["beta"], seeds=(0,))
        b = evaluate(wd, "BT", sn, 128, r, wd.np["beta"], seeds=(1,))
        ok = a["slo"] == b["slo"] and a["tail_stats"] == b["tail_stats"]
        return {"I_SEED": dict(n=len(sn["admitted"]), slo_equal=a["slo"] == b["slo"]), "I_SEED_pass": bool(ok)}
    raise ValueError(f"unknown check {which}")


# ------------------------------------------------------------------------------------------------ summarize (§6)
def _rows(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            out += [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
    return out


def _rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def summarize(d):
    rows = _rows(d)
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
            if r["flag"] != "DEGENERATE":
                continue
        by[key][r["arm"]][r.get("n", 0)] = r
    res = []
    for key in sorted(set(by) | set(flags)):
        fmt, N, tag, nt, ex = key
        one = dict(fmt=fmt, N=N, tag=tag, nphys=json.loads(nt), extra=json.loads(ex), flags=flags.get(key, {}), arms={},
                   verdicts={})
        arms = by.get(key, {})
        for arm, rr in sorted(arms.items()):
            a = {}
            for x in SLOS:
                s = str(x)
                pm = {n: all(r_["slo"][s]["pass"]) for n, r_ in rr.items()}
                nm = stable_n(pm)
                nreps = len(next(iter(rr.values()))["slo"][s]["pass"])
                per = [stable_n({n: r_["slo"][s]["pass"][i] for n, r_ in rr.items()}) for i in range(nreps)]
                if nm is None:
                    a[s] = dict(n_match=None, note="n = 0 fails (anchor only)")
                    continue
                r0 = rr[nm]
                sl = r0["slo"][s]
                a[s] = dict(n_match=nm, n_match_reps=_rng(per), hosted=r0["hosted"], n_tail=r0["n_tail"], n_mg=r0["n_mg"],
                            at=sl["at"], at_rng=_rng(sl["at"]), RS=sl["RS"], RS_rng=_rng(sl["RS"]), M_rng=_rng(sl["M"]),
                            idle=r0["idle"], idle_rng=_rng(r0["idle"]), idle_w_rng=_rng(r0["idle_w"]),
                            idle_alloc_rng=_rng(r0["idle_alloc"]), allocated=r0["allocated"],
                            unallocated=r0["unallocated"], zero_gpu_rng=_rng(r0["zero_traffic_gpus"]),
                            turned_away=r0["turned_away"], groups=r0["groups"], k_tail=r0["k_tail"],
                            r_anchor=r0["r_anchor"], loaded_units=r0["loaded_units"])
            one["arms"][arm] = a
        P = one["arms"].get("P")
        for arm, a in one["arms"].items():
            if arm == "P" or not P:
                continue
            v = {}
            for x in SLOS:
                s = str(x)
                if a[s].get("n_match") is None or P[s].get("n_match") is None:
                    v[s] = dict(note="no catalogue")
                    continue
                cat = verdict_hi(a[s]["at"], P[s]["at"])
                idl = verdict_idle(a[s]["idle"], P[s]["idle"])
                rs = verdict_rs(a[s]["RS"], P[s]["RS"])
                pd_at = [x_ - y_ for x_, y_ in zip(a[s]["at"], P[s]["at"])]
                pd_idle = [x_ - y_ for x_, y_ in zip(a[s]["idle"], P[s]["idle"])]
                pd_rs = [x_ - y_ for x_, y_ in zip(a[s]["RS"], P[s]["RS"])]
                if cat == "MORE" and idl == "LOWER-IDLE" and rs != "LOWERS":
                    head = "GPU-BUNK LIFTS"
                elif cat != "MORE" and idl != "LOWER-IDLE":
                    head = "NO-LIFT"
                else:
                    head = f"MIXED (catalogue {cat}, idle {idl}, requests {rs})"
                v[s] = dict(catalogue=cat, idle=idl, requests=rs, headline=head, d_at=_rng(pd_at), d_idle=_rng(pd_idle),
                            d_RS=_rng(pd_rs))
            one["verdicts"][arm] = v
        res.append(one)
    return res


# ------------------------------------------------------------------------------------------------ self-test (§9)
GC2_SPEC = dict(N=24, anchor_G=16, anchor_r=1, bh=71.24, arms=["P", "BS"], held_anchor=800,
                candidates=[dict(id="c1", kind="tail", unit="u1", f=20.0, held=100),
                            dict(id="c2", kind="giant", G=8, W=300.0, rho=0.1, held=50),
                            dict(id="c3", kind="tail", unit="u2", f=40.0, held=30),
                            dict(id="c4", kind="tail", unit="u1", f=20.0, held=20),
                            dict(id="c5", kind="tail", unit="u3", f=70.0, held=0)])


def self_test(root=None, cache=None):
    ok, fails, n, skipped = True, [], 0, []

    def chk(name, cond, detail=""):
        nonlocal ok, n
        n += 1
        if not cond:
            ok = False
            fails.append(f"FAIL {name} {detail}")

    def near(a, b, tol=1e-9):
        return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))

    # GC1 idle toy (golden_calc.out)
    cells = dict(a=[(2, 50, 5), (3, 40, 30), (7, 10, 125)], b=[(0, 30, 20), (4, 59, 2)], c=[(5, 0, 0.5), (9, 55, 10)])
    names = ["a", "b", "c"]
    t0 = [t for k in names for t, _, _ in cells[k]]
    la = [l for k in names for _, l, _ in cells[k]]
    ss = [s for k in names for _, _, s in cells[k]]
    mo = [names.index(k) for k in names for _ in cells[k]]
    B = busy_matrix(t0, la, ss, mo, 3, 10)
    chk("GC1 busy_a", list(np.nonzero(B[0])[0]) == [2, 3, 4, 7, 8, 9], np.nonzero(B[0])[0])
    chk("GC1 busy_b", list(np.nonzero(B[1])[0]) == [0, 4, 5])
    chk("GC1 busy_c", list(np.nonzero(B[2])[0]) == [5, 9])
    pdb = slot_busy_by_day(B, [(1, [0]), (1, [1, 2])], None, 0, 1, 10, 10)
    chk("GC1 idle 3 GPUs (unallocated GPU stays in N)", near(1.0 - pdb.sum() / 30.0, 0.6666666666666667, 1e-12)
        and near(idle_reps(slot_busy_by_day(B, [(1, [0]), (1, [1, 2])], None, 0, 1, 10, 5), 3, 5)[0], 1 - 5 / 15.0, 1e-12))
    cnt = inflight([1, 2], [10, 59], [20, 30], [3, 1], 10)
    chk("GC1 replica in-flight", list(cnt) == [0, 3, 1, 1, 0, 0, 0, 0, 0, 0], list(cnt))
    pd5 = slot_busy_by_day(B, [(1, [0]), (1, [1, 2])], cnt, 2, 1, 10, 5)
    chk("GC1 idle 5 GPUs", near(1.0 - pd5.sum() / 50.0, 0.72, 1e-12), pd5)
    ir = idle_reps(pd5, 5, 5)
    chk("GC1 LODO drop day 0 / 1", near(ir[0], 0.8, 1e-12) and near(ir[1], 0.64, 1e-12), ir)
    fi, fa, fia = fleet_idle(B, [(1, [0]), (1, [1, 2]), (1, [])], None, 0, 1, 3, 10, 5)
    chk("GC1 fleet_idle keeps the unallocated GPU in N", near(fi[0], 1 - 5 / 15.0, 1e-12) and near(fi[1], 1 - 5 / 15.0, 1e-12)
        and fa == 2 and near(fia[0], 0.5, 1e-12), (fi, fa, fia))
    fi5, fa5, _ = fleet_idle(B, [(1, [0]), (1, [1, 2])], cnt, 2, 1, 5, 10, 5)
    chk("GC1 fleet_idle with replicas", near(fi5[0], 0.8, 1e-12) and near(fi5[1], 0.64, 1e-12) and fa5 == 4, (fi5, fa5))
    # GC2 admission (through the same Fleet code the CLI uses)
    res = admit_spec(GC2_SPEC)
    chk("GC2 P admitted", res["P"]["admitted"] == ["c1", "c3", "c4", "c5"], res["P"])
    chk("GC2 P turned away + free", res["P"]["turned_away"] == ["c2"] and res["P"]["free"] == 4)
    chk("GC2 turned-away share", near(res["P"]["turned_away_share"], 0.05, 1e-12))
    chk("GC2 BS admitted", res["BS"]["admitted"] == ["c1", "c3", "c4", "c5"] and res["BS"]["turned_away"] == ["c2"], res["BS"])
    r32 = admit_spec(dict(GC2_SPEC, N=32))
    chk("GC2 N32 P free 4, BS opens c2", r32["P"]["free"] == 4 and r32["P"]["turned_away"] == []
        and r32["BS"]["turned_away"] == [] and r32["BS"]["k_tail"] == 8, r32)
    # GC2b: a sibling rides its unit at 0 GiB (kills "sibling charged again"); P has no dedup
    spec_b = dict(N=24, anchor_G=16, anchor_r=1, bh=50.0, arms=["P", "BS"],
                  candidates=[dict(id=f"t{i}", kind="tail", unit=f"u{i}", f=45.0) for i in range(1, 9)]
                  + [dict(id="t9", kind="tail", unit="u1", f=45.0)])
    rb = admit_spec(spec_b)
    chk("GC2b BS sibling rides", rb["BS"]["turned_away"] == [] and len(rb["BS"]["admitted"]) == 9, rb["BS"])
    chk("GC2b P turns the sibling away", rb["P"]["turned_away"] == ["t9"], rb["P"])
    # GC2c: skip and continue (an unplaceable tail unit is skipped, the next one admitted)
    rc = admit_spec(dict(N=24, anchor_G=16, anchor_r=1, bh=71.24, arms=["BS"],
                         candidates=[dict(id="x", kind="tail", unit="ux", f=80.0), dict(id="y", kind="tail", unit="uy", f=10.0)]))
    chk("GC2c skip and continue", rc["BS"]["admitted"] == ["y"] and rc["BS"]["turned_away"] == ["x"], rc)
    # GC2d: SHARE homes a quiet giant into an existing group (rho cap, host, KV)
    fl = Fleet(32, "SPREAD", "SHARE", 16, 1, {}, dict(bh=71.24, h_tot=1e9, node_ram=1e9, r_pin=1.0, o_g=6.0, s_g=1.0,
                                                    rho_cap=0.5))
    a1 = fl.try_giant("g1", 4, 130.0, 0.3)
    a2 = fl.try_giant("g2", 4, 130.0, 0.1)
    a3 = fl.try_giant("g3", 4, 130.0, 0.2)
    chk("GC2d homing", a1 and a2 and a3 and [g["members"] for g in fl.groups] == [["g1", "g2"], ["g3"]], fl.groups)
    # GC2e (post-grid fix, T3): a giant whose own TRAIN rho exceeds the cap still opens its own group (prereg §4.2)
    fl2 = Fleet(32, "SPREAD", "SHARE", 16, 1, {}, dict(bh=71.24, h_tot=1e9, node_ram=1e9, r_pin=1.0, o_g=6.0, s_g=1.0,
                                                     rho_cap=0.1))
    b1 = fl2.try_giant("h1", 4, 130.0, 0.3)
    b2 = fl2.try_giant("h2", 4, 130.0, 0.05)
    chk("GC2e lone giant above the cap opens a group; a joiner still obeys the cap", b1 and b2
        and [g["members"] for g in fl2.groups] == [["h1"], ["h2"]], fl2.groups)
    # GC3 adversarial: the anchor never silently dropped
    try:
        Fleet(8, "DED", "DED", 16, 1, {}, dict(bh=71.24))
        chk("GC3 N8 raises", False, "no error")
    except AnchorError as e:
        chk("GC3 N8 raises ANCHOR-NOT-PLACEABLE", "ANCHOR-NOT-PLACEABLE" in str(e))
    try:
        Fleet(32, "DED", "DED", 16, 3, {}, dict(bh=71.24))
        chk("GC3 r beyond N raises", False)
    except AnchorError:
        chk("GC3 r beyond N raises", True)
    # GC4 replicas (fluid identity)
    models = [("x", 3.1, 3.1, "fp8", 1.0, 1.0)]
    p = dict(F.PHYS0, tpot_sat=0.06)
    for name, it, ot, want1, want2 in (("light", 2000.0, 200.0, 1.0, 1.0), ("heavy", 170000.0, 10000.0, 145.0, 100.0)):
        tr = T.Trace([HELD], [0], [2], [it], [ot], models)
        w1, _ = dedicated_rep(tr, None, p, [1000.0], 1)
        w2, _ = dedicated_rep(tr, None, p, [1000.0], 2)
        chk(f"GC4 {name} r1", near(T._pct(w1[0], 0.95), want1, 1e-9), T._pct(w1[0], 0.95))
        chk(f"GC4 {name} r2", near(T._pct(w2[0], 0.95), want2, 1e-9), T._pct(w2[0], 0.95))
    rng = np.random.default_rng(11)
    cells_ = sorted({(HELD + int(rng.integers(0, 3 * DAY_MIN)), 0) for _ in range(300)})
    mins = [c[0] for c in cells_]
    nn = [2 * int(rng.integers(10, 41)) for _ in mins]           # cell TOTALS: n requests of 1-4k in / 0.5-2k out tokens
    its = [float(n * int(rng.integers(1000, 4000))) for n in nn]
    ots = [float(n * int(rng.integers(500, 2000))) for n in nn]
    mods = [("y", 64.0, 6.2, "fp8", 1.0, 1.0)]
    full = T.Trace(mins, [0] * len(mins), nn, its, ots, mods)
    half = T.Trace(mins, [0] * len(mins), [x // 2 for x in nn], [x / 2 for x in its], [x / 2 for x in ots], mods)
    pk = dict(F.PHYS0, tpot_sat=0.06, kappa=0.5)
    wf, df = dedicated_rep(full, [8000.0], pk, [30000.0], 2)          # one replica's KV binds (max request 5,879 tokens)
    wh, dh = dedicated_rep(half, [8000.0], pk, [30000.0], 1)
    wn, _ = dedicated_rep(full, [1e12], pk, [30000.0], 2)
    chk("GC4 property case is KV-bound somewhere", wf[0] != wn[0])
    ka = sorted(wf[0].elements())
    kb = sorted(collections.Counter({k: 2 * v for k, v in wh[0].items()}).elements())
    chk("GC4 property r2 == 2 x half-trace r1", len(ka) == len(kb) and all(abs(x - y) <= 1 for x, y in zip(ka, kb)),
        (len(ka), len(kb)))
    w_ref, d_ref = NS.dedicated_by_day(full, [8000.0], pk, [30000.0])
    w_r1, d_r1 = dedicated_rep(full, [8000.0], pk, [30000.0], 1)
    chk("GC4 r1 == dedicated_by_day", w_ref[0] == w_r1[0] and all(d_ref[d][0] == d_r1[d][0] for d in range(NDAYS)))
    # GC5 density on census numbers
    if root and os.path.exists(os.path.join(root, V2.CENSUS_JSON)):
        rows, _ = V2.load_census(root)
        byp = {c[:8]: r_ for c, r_ in rows.items()}
        for cid, fmt, want in (("aa605b6d", "int4", 334.72481517437086), ("ae206d64", "int4", 197.7380133058041),
                               ("aa605b6d", "fp8", 203.15913520373476)):
            r_ = byp[cid]
            Wg = NS.mg_weights(r_, fmt)[0]
            chk(f"GC5 {cid} {fmt}", near(density(r_["train_req"], Wg), want, 1e-3), density(r_["train_req"], Wg))
    else:
        chk("GC5 census present under --root", False, root)
    # GC6 anchor load (real trace) — only with the cache
    if root and cache and os.path.exists(cache):
        wd = make_world("int4", {}, dict(EXTRA0), root, cache)
        ld = wd.anchor_load()
        chk("GC6 anchor load", near(ld["max_min"], 2.8014574233092224, 1e-9) and near(ld["max_60"], 1.4556591669757342, 1e-9)
            and near(ld["mean"], 0.28102342753004494, 1e-9), ld)
        chk("GC6 ranking excludes anchor, 374 candidates", len(wd.cands) == 374 and all(c["cid"] != NS.ANCHOR for c in wd.cands))
        top = {c["cid"][:8]: c["dens"] for c in wd.cands if c["kind"] == "giant"}
        chk("GC6 ranking density matches GC5 (trace TRAIN counts)", near(top["aa605b6d"], 334.72481517437086, 1e-3), top.get("aa605b6d"))
    else:
        skipped.append("GC6 (no --cache)")
    # GC7 decision rules
    chk("GC7 MORE", verdict_hi([120, 125], [90, 90]) == "MORE")
    chk("GC7 UNRESOLVED", verdict_hi([88, 95], [90, 90]) == "UNRESOLVED")
    chk("GC7 FEWER", verdict_hi([80, 85], [90, 90]) == "FEWER")
    chk("GC7 LOWER-IDLE", verdict_idle([0.40, 0.45], [0.60, 0.70]) == "LOWER-IDLE")
    chk("GC7 idle UNRESOLVED", verdict_idle([0.40, 0.62], [0.60, 0.70]) == "UNRESOLVED")
    chk("GC7 HIGHER-IDLE", verdict_idle([0.8, 0.9], [0.6, 0.7]) == "HIGHER-IDLE")
    chk("GC7 RS", verdict_rs([0.9], [0.8]) == "LIFTS" and verdict_rs([0.7], [0.8]) == "LOWERS"
        and verdict_rs([0.79, 0.81], [0.8]) == "UNATTRIBUTABLE")
    chk("GC7 stable n_match", stable_n({0: True, 8: True, 16: False, 24: True}) == 8)
    # score_reps mirrors tailpool score's M on hosted models
    k1, k5, k20 = T._key(1.0), T._key(5.0), T._key(20.0)
    wkO = [collections.Counter({k1: 30}), collections.Counter({k20: 30}), collections.Counter({k5: 5})]
    wkR = [collections.Counter({k1: 30}), collections.Counter({k1: 30}), collections.Counter({k1: 5})]
    dy0 = [collections.Counter() for _ in range(3)]
    st_o = {m: cstats(wkO[m], [dy0[m]] * NDAYS) for m in range(3)}
    st_r = {m: cstats(wkR[m], [dy0[m]] * NDAYS) for m in range(3)}
    sc = score_reps([0, 1, 2], st_o, st_r, [0, 1, 2], 10.0)
    ts = T.score(wkO, wkR, 10.0)
    chk("score_reps M == tailpool score", near(sc["M"][0], ts["M_frac"], 1e-12) and sc["at"][0] == 2, (sc["M"][0], ts["M_frac"]))
    chk("score_reps RS", near(sc["RS"][0], 35 / 65.0, 1e-12), sc["RS"][0])
    # CLI golden: GC2 + GC3 through the real entry point
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f2 = os.path.join(td, "gc2.json")
        with open(f2, "w") as fh:                   # QA r1: close before the subprocess reads it
            json.dump(GC2_SPEC, fh)
        f3 = os.path.join(td, "gc3.json")
        with open(f3, "w") as fh:
            json.dump(dict(GC2_SPEC, N=8), fh)
        try:
            p2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--admit-spec", f2], capture_output=True, text=True,
                                timeout=120)
            p3 = subprocess.run([sys.executable, os.path.abspath(__file__), "--admit-spec", f3], capture_output=True, text=True,
                                timeout=120)
        except subprocess.TimeoutExpired:
            p2 = p3 = subprocess.CompletedProcess([], 124, "", "timeout")
        try:
            j2 = json.loads(p2.stdout)
        except ValueError:
            j2 = {}
        chk("CLI GC2 rc 0 + lists", p2.returncode == 0 and j2.get("P", {}).get("turned_away") == ["c2"]
            and j2.get("BS", {}).get("admitted") == ["c1", "c3", "c4", "c5"], (p2.returncode, p2.stderr[-200:]))
        chk("CLI GC3 exit 3 ANCHOR-NOT-PLACEABLE", p3.returncode == 3 and "ANCHOR-NOT-PLACEABLE" in p3.stderr, (p3.returncode,
                                                                                                                  p3.stderr[-200:]))
    for f in fails:
        print(f)
    tail = f" (skipped: {', '.join(skipped)})" if skipped else ""
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'} {n - len(fails)}/{n}{tail}")
    return ok


# ------------------------------------------------------------------------------------------------ CLI
def _parse_sets(sets):
    over, extra = {}, dict(EXTRA0)
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set needs k=v, got {s}")
        k, v = s.split("=", 1)
        if k in EXTRA0:
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
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check", choices=("NODE", "REP", "D", "SEED"))
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
            try:
                print(json.dumps(admit_spec(json.load(open(a.admit_spec)))))
            except AnchorError as e:
                print(f"{e}", file=sys.stderr)
                return 3
            return 0
        if a.plan:
            if not (a.arm and a.fmt and a.N is not None and a.cache):
                raise ValueError("--plan needs --arm, --fmt, --N and --cache")
            over, extra = _parse_sets(a.set)
            wd = make_world(a.fmt, over, extra, a.root, a.cache)
            try:
                info, snaps = plan(wd, a.arm, a.N)
            except AnchorError as e:
                print(f"{e}", file=sys.stderr)
                return 3
            out = dict(info, arm=a.arm, fmt=a.fmt, N=a.N)
            if snaps:
                sn = snaps[max(N_GRID)]
                out.update(admitted=len(sn["admitted"]), loaded_units=len(sn["loaded"]), ded=sn["ded"],
                           groups=[(g["G"], [c[:8] for c in g["members"]]) for g in sn["groups"]],
                           per_n={n: len(snaps[n]["admitted"]) for n in N_GRID})
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
            if a.arm not in ARM_MODES:
                raise ValueError(f"unknown arm {a.arm}")
            over, extra = _parse_sets(a.set)
            try:
                return run_job(a.arm, a.fmt, a.N, over, extra, a.tag, a.root, a.cache, a.out)
            except AnchorError as e:
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
