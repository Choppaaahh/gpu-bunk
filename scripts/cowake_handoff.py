#!/usr/bin/env python3
"""cowake_handoff.py — a CO-WAKE HANDOFF as the deciding design for the 97-model 2 s door (cloud sim v6, cell D1,
NVLink neighbour tier with re-homing), on a SUB-MINUTE in-flight window built from v6's per-request arrival offsets; plus a co-wake
synchronization null. FORK BY IMPORT: no landed sim is edited. wrapper (`rehoming`) supplies the hook
machinery, the state engine `rh_simulate`, the summarizer and the guard; supplies the page-site forks and `nb_page`.

Prereg (frozen before this file existed): (private report). On any divergence the
prereg wins.

What this file adds (prereg §3): (1) `cw_engine` = `rh_simulate` with every copy recorded (park pushes, re-home pushes, the
eviction order per reclaim, the copy chain each owner wake pulls through); I-NEST-CW proves it equals `rh_simulate` on 300 toys and on
every real schedule. (2) A sub-minute timing table (v6 build_windows' own T0 / A rule), reclaim instants and serial drains. (3) A race
page per owner unit at the arm's p (SUB / LW / CW, WAIT or NO-WAIT, TRUE or WORST alignment). (4) Because a race page depends on p, the
v6 windows' page dicts are REFRESHED per p in a `queue_pass` wrapper (v6 builds them once per policy); v5's `run_variant` is wrapped to
hand the hook p and the seed's offsets. With the old rules (RH-Q / A1 / PLC-RH) every page is unchanged: I-FIX vs committed rows.

Output contract:
  --self-test                                  exit 0 + "SELF-TEST PASS n/n" (goldens G1-G12); exit 1 + FAIL lines.
  --run --setting S [--cell D1] [--seeds 0,1,2] [--tag T] --root R --cache C --out F [--maps M --maps-sha H]
                                               WRITES (appends) ONE JSON line to F: v6's row + "s2936" (as writes it) + "s2938"
                                               (prereg, setting, race stats per seed x policy); exit 0. A failed in-job check exits 1.
                                               A MAP-i setting REFUSES (exit 2) without a manifest whose ok is true and sha256 matches.
  --ifix-2936 NEW_FILE REF_FILE [--ignore-tag]  read-only: the NEW row vs a committed row, every field but timing keys and
                                               "s2938" (and the top-level tag with --ignore-tag); one JSON on stdout; exit 0 PASS / 1 FAIL.
  --ifix-nb NEW_DIR OFF_FILE SAME_FILE H12_FILE  read-only: I-NB-STATIC + I-NB-BRACKET + I-REPLAY over NEW_DIR; exit 0 / 1.
  --summarize-cw OUT_DIR                       read-only: per setting sub-directory setting_summary + DOOR-CW and every frozen
                                               flag, the placebo reads, I-HBM-EQ, the race stats; REFUSES duplicate rows (exit 2); one JSON.
  --cowake-null --root R --cache C --out F --ref-order ROW --ref-shuf ROW [--k 20]
                                               WRITES F (one JSON): the co-wake rate real vs the N-SHIFT / N-DAY / N-JIT nulls and the
                                               frozen class per (policy, map); exit 0; I-NULL0 failure exits 1 and writes nothing.
  Bad input = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **v6 builds its windows ONCE per policy (at the first p) and reuses their page dicts across the p grid**, while v5's run_variant
     recomputes per p; a p-dependent page without the per-p refresh makes I-RES fail on the second p (prereg §0).
  2. ** `install_rh` asserts that v5's run_variant is `rv_wrap`**; once this file wraps it, calling `install_rh` again
     fails, so `install_cw` is idempotent and is the only installer to call (pitfall inherited: the hook is looked up by NAME at call time).
  3. **`ssh host '... &'` holds the session open**: launch with `ssh -n` + `setsid -f`.
"""
import argparse, collections, hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rehoming as RH  # noqa: E402

NB = RH.NB
PREREG = "a7db8185a"
GIB, PF, R_NB = NB.GIB, NB.PAGE_FIXED, NB.R_NB
P_STAR = 0.842
H_FB = 12.4
BREAK_W = RH.BREAK_W
DAY = RH.DAY
REC_P = (0.0, 0.5, 0.75, 0.842)
DEC_ARMS = RH.DEC_ARMS
INF = math.inf
NB.STRIP_ROW = tuple(NB.STRIP_ROW) + ("s2938",)

_A1 = dict(RH._B, evict="REHOME", delta=0.25, avail="SAME", bounce=None, charge=True)
LEGACY = {"RH-Q": dict(RH.SETTINGS["RH-Q"]), "RH-Q-SHUF": dict(RH.SETTINGS["RH-Q-SHUF"]),
          "A1-RHS-Q": dict(_A1, map="ORDER"), "A1-RHS-Q-SHUF": dict(_A1, map="SHUF")}
_N = dict(RH._B, evict="REHOME", avail="SAME", bounce=None, charge=True)
NEW = {}
for _m, _sfx in (("ORDER", ""), ("SHUF", "-SHUF")):
    NEW["CW-Q" + _sfx] = dict(_N, map=_m, delta=0.25, race="CW", wait=True, timing="TRUE")
    NEW["CW-Q-NW" + _sfx] = dict(_N, map=_m, delta=0.25, race="CW", wait=False, timing="TRUE")
    NEW["CW-H" + _sfx] = dict(_N, map=_m, delta=0.5, race="CW", wait=True, timing="TRUE")
    NEW["CW-Q-WORST" + _sfx] = dict(_N, map=_m, delta=0.25, race="CW", wait=True, timing="WORST")
    NEW["CW-Q-NW-WORST" + _sfx] = dict(_N, map=_m, delta=0.25, race="CW", wait=False, timing="WORST")
    NEW["SUB-Q" + _sfx] = dict(_N, map=_m, delta=0.25, race="SUB", wait=None, timing="TRUE")
    NEW["LW-Q" + _sfx] = dict(_N, map=_m, delta=0.25, race="LW", wait=None, timing="TRUE")
K_MAPS = RH.K_MAPS


def setting_cfg(name):
    """-> (cfg, legacy?)"""
    if name in LEGACY:
        return dict(LEGACY[name]), True
    if name in NEW:
        return dict(NEW[name]), False
    m = re.fullmatch(r"PLC-(RH|CW)-(\d+)", name or "")
    if not m or int(m.group(2)) >= K_MAPS:
        raise ValueError(f"unknown setting {name}")
    if m.group(1) == "RH":
        return RH.setting_cfg(name), True
    cfg = dict(NEW["CW-Q"])
    cfg["map"] = f"MAP-{int(m.group(2))}"
    return cfg, False


def page_rate(w, g):
    return PF + w * GIB / g


# ------------------------------------------------------------------------------------------------ the state engine (prereg §3.1, §3.4)
def cw_engine(held_of, wake_of, units_of, node_of, cap, evict, n_min, avail="SAME", bounce=None, charge=False, day=DAY):
    """ rh_simulate VERBATIM in behaviour (I-NEST-CW / G7), plus records: every copy (park push / re-home push), the eviction
    order per reclaim, and the copy chain since the last park that every owner wake pulls through."""
    import numpy as np
    if avail not in ("SAME", "NEXT") or evict not in ("DROP", "REHOME"):
        raise ValueError(f"bad engine mode {avail} / {evict}")
    slots = sorted(held_of)
    ev = collections.defaultdict(lambda: ([], [], []))
    for s in slots:
        h = np.asarray(held_of[s], bool)
        prev = np.concatenate([[True], h[:-1]])
        for t in np.flatnonzero(h & ~prev):
            ev[int(t)][0].append(s)
        for t in np.flatnonzero(np.asarray(wake_of[s], bool)):
            if not h[t] or prev[t]:
                raise RuntimeError(f"I-NB-EVENTS: slot {s} wakes at {t} without becoming held")
            ev[int(t)][1].append(s)
        for t in np.flatnonzero(~h & prev):
            ev[int(t)][2].append(s)
    by_node = collections.defaultdict(list)
    for s in slots:
        by_node[node_of[s]].append(s)
    guests = {s: {} for s in slots}
    free = {s: float(cap) for s in slots}
    loc, rec, k, copies = {}, {}, collections.Counter(), []
    st = dict(evicted=0, rehomed=0, pushes=0, park_uncached=0, guest_gib_min=0.0,
              lender_gpu_min=int(sum(int((~np.asarray(held_of[s], bool)).sum()) for s in slots)))
    tot = dict(guest=0.0)
    nodes = sorted({node_of[s] for s in slots})
    ndays = -(-n_min // day)
    g_day = {n: [0.0] * ndays for n in nodes}
    c_day = {n: [0.0] * ndays for n in nodes}
    l_day = {n: [0] * ndays for n in nodes}
    for s in slots:
        pk = ~np.asarray(held_of[s], bool)
        for d in range(ndays):
            l_day[node_of[s]][d] += int(pk[d * day:(d + 1) * day].sum())
    lvl = {n: 0.0 for n in nodes}
    nre = {}
    inflight = set()
    x = dict(inflight_fallback=0, dropped_at_evict=0, nb1_dropped=0, rehome_items=[], bounce_hist=collections.Counter(), cap_ok=True)
    cps, chain, wake_chain, reclaims = [], {}, {}, []                       # records

    def integrate(a, b):
        for n in nodes:
            v = lvl[n]
            if v == 0.0:
                continue
            m = a
            while m < b:
                d = m // day
                e = min(b, (d + 1) * day)
                g_day[n][d] += v * (e - m)
                m = e

    def lender_for(s_home, w, t, exclude):
        best = None
        for L in by_node[node_of[s_home]]:
            if L in exclude or held_of[L][t] or free[L] < w - 1e-12:
                continue
            if best is None or free[L] > free[best] + 1e-12:
                best = L
        return best

    def place(L, key, w, t):
        guests[L][key] = w
        free[L] -= w
        loc[key] = L
        k[(L, t)] += 1
        copies.append((L, t, w))
        tot["guest"] += w
        st["pushes"] += 1
        lvl[node_of[L]] += w
        if free[L] < -1e-9:
            x["cap_ok"] = False
            raise RuntimeError(f"I-CAP: lender {L} over CAP at minute {t}")

    def unplace(L, key, w):
        del guests[L][key]
        free[L] += w
        tot["guest"] -= w
        lvl[node_of[L]] -= w

    def evictions(becomes, t):
        for L in sorted(becomes):
            rc = dict(L=L, t=t, items=[])
            reclaims.append(rc)
            for key, w in sorted(guests[L].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                unplace(L, key, w)
                loc[key] = None
                st["evicted"] += 1
                if evict == "REHOME":
                    if bounce is not None and nre.get(key, 0) >= bounce:
                        x["nb1_dropped"] += 1
                        x["dropped_at_evict"] += 1
                        x["bounce_hist"][nre.get(key, 0)] += 1
                        rc["items"].append((key, w, None, None))
                        chain[key] = []
                        continue
                    L2 = lender_for(L, w, t, {L, key[0]})
                    if L2 is not None:
                        place(L2, key, w, t)
                        st["rehomed"] += 1
                        nre[key] = nre.get(key, 0) + 1
                        x["rehome_items"].append((L, t, w, L2))
                        ci = len(cps)
                        prv = chain[key][-1] if chain.get(key) else None
                        cps.append(dict(kind="R", key=key, src=L, dst=L2, t=t, w=w, prev=prv, ri=len(reclaims) - 1))
                        chain.setdefault(key, []).append(ci)
                        rc["items"].append((key, w, L2, ci))
                        if avail == "NEXT":
                            inflight.add(key)
                        if charge:
                            c_day[node_of[L]][t // day] += w
                    else:
                        x["dropped_at_evict"] += 1
                        x["bounce_hist"][nre.get(key, 0)] += 1
                        rc["items"].append((key, w, None, None))
                        chain[key] = []
                else:
                    x["dropped_at_evict"] += 1
                    x["bounce_hist"][nre.get(key, 0)] += 1
                    rc["items"].append((key, w, None, None))
                    chain[key] = []
            if guests[L]:
                raise RuntimeError("I-NB-EVICT: guests left on a held lender")

    def wakes_(wakes, t):
        for s in sorted(wakes):
            for u, w in units_of[s].items():
                key = (s, u)
                L = loc.get(key)
                wake_chain[(s, t, u)] = tuple(chain.get(key, ()))
                chain[key] = []
                if L is not None and key in inflight:
                    rec[(s, t, u)] = None
                    unplace(L, key, w)
                    x["inflight_fallback"] += 1
                    x["bounce_hist"][nre.get(key, 0)] += 1
                    loc[key] = None
                    continue
                rec[(s, t, u)] = L
                if L is not None:
                    k[(L, t)] += 1
                    copies.append((L, t, w))
                    unplace(L, key, w)
                    x["bounce_hist"][nre.get(key, 0)] += 1
                loc[key] = None

    def parks_(parks, t):
        for s in sorted(parks):
            for u, w in sorted(units_of[s].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                nre[(s, u)] = 0
                L = lender_for(s, w, t, {s})
                if L is None:
                    loc[(s, u)] = None
                    st["park_uncached"] += 1
                    chain[(s, u)] = []
                    continue
                place(L, (s, u), w, t)
                cps.append(dict(kind="P", key=(s, u), src=s, dst=L, t=t, w=w, prev=None, ri=None))
                chain[(s, u)] = [len(cps) - 1]

    t_last = 0
    for t in sorted(ev):
        st["guest_gib_min"] += tot["guest"] * (t - t_last)
        integrate(t_last, t)
        t_last = t
        inflight.clear()
        becomes, wakes, parks = ev[t]
        evictions(becomes, t)
        wakes_(wakes, t)
        parks_(parks, t)
    st["guest_gib_min"] += tot["guest"] * (n_min - t_last)
    integrate(t_last, n_min)
    x.update(g_day=g_day, c_day=c_day, l_day=l_day, ndays=ndays, day=day, charge=bool(charge))
    return dict(rec=rec, k=k, copies=copies, stats=st, x=x, cps=cps, wake_chain=wake_chain, reclaims=reclaims)


def nest_ok(a, b):
    """I-NEST-CW / G7: the four core fields AND the x record equal (same_core + x)."""
    return RH.same_core(a, b) and a["x"] == b["x"]


# ------------------------------------------------------------------------------------------------ sub-minute timing (prereg §3.2-3.4)
def timing_table(pol, tail, slots, slot_cells, tr, first, pol_cache):
    """v6 build_windows' own (T0, A) per tail wake (prereg §3.2): -> {(slot, minute): (T0, {unit: first arrival})}."""
    import numpy as np
    V3 = NB._m2897()._v6().V3
    out = {}
    for s_i in tail:
        sl = slots[s_i]
        held, wake = pol_cache[(pol, s_i, False)]
        mins, ids = slot_cells[s_i]
        for t in np.flatnonzero(wake):
            te = V3.window_end(pol, held, int(t))
            a_, z_ = np.searchsorted(mins, t, "left"), np.searchsorted(mins, te, "right")
            cl = [(int(mins[q]), float(first[ids[q]]), int(tr.mid[ids[q]]), sl["unit_of"][int(tr.mid[ids[q]])], ids[q])
                  for q in range(a_, z_)]
            at = [c for c in cl if c[0] == t]
            if not at:
                raise RuntimeError(f"I-TIMING: no cell of slot {s_i} at wake minute {t}")
            T0 = min(60.0 * c[0] + c[1] for c in at)
            A = {}
            for c in cl:
                tt = 60.0 * c[0] + c[1]
                if c[3] not in A or tt < A[c[3]]:
                    A[c[3]] = tt
            out[(s_i, int(t))] = (T0, A)
    return out


def drain_items(tau, items, kl, delta):
    """§3.4 serial drain from tau: items [(w, k(X, t), inbound_end)] in eviction order -> [(s, e)]."""
    out, e_prev = [], tau
    for w, kx, inb in items:
        s = max(e_prev, inb)
        e = s + page_rate(w, R_NB * delta / max(kx, kl))
        out.append((s, e))
        e_prev = e
    return out


def drains_of(eng, timing, wake_of, delta, n_min):
    """reclaim instants, kL, and [s, e) of every copy (park pushes from 60 t; re-homes by the serial drain)."""
    k = eng["k"]
    se, tauR, kL = {}, {}, {}
    I = dict(reclaims=0, reclaim_nowake=0, chain_waits=0, n_rehome_copies=0, drain_max=0.0, drains_over_pstar=0, d60=True)
    for ci, c in enumerate(eng["cps"]):
        if c["kind"] == "P":
            se[ci] = (60.0 * c["t"], 60.0 * c["t"] + page_rate(c["w"], R_NB * delta / k[(c["dst"], c["t"])]))
    for rc in eng["reclaims"]:
        L, t = rc["L"], rc["t"]
        I["reclaims"] += 1
        if wake_of[L][t]:
            tau = timing[(L, t)][0]
        else:
            tau = 60.0 * t
            I["reclaim_nowake"] += 1
        cp = [(key, w, L2, ci) for key, w, L2, ci in rc["items"] if ci is not None]
        h = sum(1 for key, w, L2, ci in cp if wake_of[key[0]][t] or (t + 1 < n_min and wake_of[key[0]][t + 1]))
        kl = 1 + h
        tauR[(L, t)], kL[(L, t)] = tau, kl
        items = []
        for key, w, L2, ci in cp:
            prv = eng["cps"][ci]["prev"]
            inb = se[prv][1] if prv is not None and se[prv][1] > tau else -INF
            if inb > -INF:
                I["chain_waits"] += 1
            items.append((w, k[(L2, t)], inb))
        for (key, w, L2, ci), sei in zip(cp, drain_items(tau, items, kl, delta)):
            se[ci] = sei
        I["n_rehome_copies"] += len(cp)
        if cp:
            d = se[cp[-1][3]][1] - tau
            I["drain_max"] = max(I["drain_max"], d)
            I["drains_over_pstar"] += d > P_STAR
            if d >= 60.0:
                I["d60"] = False
    return se, tauR, kL, I


# ------------------------------------------------------------------------------------------------ the race page (prereg §3.5-3.6)
def race_page(race, wait, worst, w, pi, p, tau, e_j, e_m, kl, kx, delta, host_gbps=H_FB):
    """one unit whose bytes are in a re-home race. tau / e_j: the holder's reclaim instant and the end of the copy out of it; e_m: the
    final landing; kl / kx: contention at the holder / the final card. -> (page s, class, option)."""
    host = page_rate(w, host_gbps)
    nbl = page_rate(w, R_NB * delta / kl)
    nbx = page_rate(w, R_NB * delta / kx)
    if worst:
        if race != "CW":
            raise ValueError("WORST is defined for the handoff only")
        if wait:
            opt = p + nbl if tau + p < e_m else INF
            v = max(min(host, nbl), min(host, opt, (e_m - tau) + nbx), min(host, nbx))
        else:
            v = min(host, max(nbl, nbx))
        return v, "WORST", "worst"
    if pi >= e_m:
        return min(host, nbx), "LANDED", "new-home"
    if pi < tau:
        return min(host, nbl), "PRE", "holder-pre"
    land = (e_m - pi) + nbx
    if race == "SUB":
        return host, "INFLIGHT", "host"
    if race == "LW":
        return (host, "INFLIGHT", "host") if host <= land else (land, "INFLIGHT", "landing")
    if race != "CW":
        raise ValueError(f"unknown race rule {race}")
    start = max(pi, tau + p) if wait else pi
    opt = (start - pi) + nbl if start < e_j else INF
    v = min(host, opt, land)
    return v, "INFLIGHT", ("holder" if v == opt else "landing" if v == land else "host")


def wake_pages(A, T0, p, page_fn):
    """v3 wake_ready's order and start rule; page_fn(u, pi) -> page. -> ({u: page}, {u: pi})."""
    link = -INF
    pages, pis = {}, {}
    for u in sorted(A, key=lambda x: (A[x], str(x))):
        pi = max(A[u], T0 + p, link)
        pages[u] = page_fn(u, pi)
        pis[u] = pi
        link = pi + pages[u]
    return pages, pis


# ------------------------------------------------------------------------------------------------ the hook
class CWHook(RH.RHook):
    def __init__(self):
        super().__init__()
        self.cur_p = None
        self.cur_pol = None
        self.ctx = None

    def is_new(self):
        return self.cfg.get("mode") == "NB" and self.cfg.get("race") is not None

    def prepare(self, pol, slots, bs, khs, checks, pol_cache):
        if self.cfg["mode"] != "NB" or pol == "STATIC":
            return
        if not self.is_new():
            RH.RHook.prepare(self, pol, slots, bs, khs, checks, pol_cache)
            st = self.cur[pol]
            if "cw_timing" not in st:
                st["cw_timing"] = self._timing(pol, slots, pol_cache)
                st["cw_tchk"] = 0
            return
        M = NB._m2897()
        V6 = M._v6()
        V3, V4 = V6.V3, V6.V4
        for s_i, (sl, b) in enumerate(zip(slots, bs)):
            key = (pol, s_i, False)
            if key not in pol_cache:
                if not sl["host_ok"]:
                    checks["I_HOST_parked_blocked"] += 1
                    pol_cache[key] = V3.held_static(b)
                else:
                    pol_cache[key] = V4.slot_policy(pol, b, khs[s_i], False)
        tag = (id(pol_cache), pol)
        if any(x["tag"] == tag for x in self.states):
            self.cur[pol] = next(x for x in self.states if x["tag"] == tag)
            return
        nm = RH.node_map2(slots, self.cfg["map"], self.maps)
        tail = RH.tail_slots(slots)
        held_of = {i: pol_cache[(pol, i, False)][0] for i in tail}
        wake_of = {i: pol_cache[(pol, i, False)][1] for i in tail}
        units_of = {i: dict(slots[i]["units"]) for i in tail}
        node_of = {i: nm[i] for i in tail}
        n_min = len(next(iter(held_of.values())))
        cap = self.cfg["cap"] * self.bh
        c = self.cfg
        ref = RH.rh_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min, c["avail"], c["bounce"], c["charge"])
        st = cw_engine(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min, c["avail"], c["bounce"], c["charge"])
        if not nest_ok(st, ref):
            raise RuntimeError(f"I-NEST-CW: the engine != 's rh_simulate ({pol})")
        if not RH.same_core(RH.rh_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min),
                            NB.nb_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min)):
            raise RuntimeError(f"I-NEST: 's engine != 's nb_simulate ({pol})")
        for (s, t, u), ch in st["wake_chain"].items():                   # every chain ends on the card rec names
            L = st["rec"][(s, t, u)]
            if (L is None) != (len(ch) == 0) or (ch and st["cps"][ch[-1]]["dst"] != L) or (ch and st["cps"][ch[0]]["kind"] != "P"):
                raise RuntimeError(f"I-NEST-CW: wake chain of {(s, t, u)} does not end on its lender")
        x = st["x"]
        brk = [i for i in tail if any(w > BREAK_W for w in units_of[i].values())]
        hold_day = [0] * x["ndays"]
        for i in brk:
            pk = ~held_of[i].astype(bool)
            for d in range(x["ndays"]):
                hold_day[d] += int(pk[d * x["day"]:(d + 1) * x["day"]].sum())
        timing = self._timing(pol, slots, pol_cache)
        se, tauR, kL, I = drains_of(st, timing, wake_of, c["delta"], n_min)
        gaps = []
        for (s, t, u), ch in st["wake_chain"].items():
            if ch and st["cps"][ch[-1]]["kind"] == "R" and st["cps"][ch[-1]]["t"] == t and (s, t) in timing:
                cl = st["cps"][ch[-1]]
                gaps.append(timing[(s, t)][0] - tauR[(cl["src"], cl["t"])])
        st.update(tag=tag, pol=pol, ctx=sum(1 for y in self.states if y["pol"] == pol), pol_cache=pol_cache, units_of=units_of,
                  node_map=nm, nested=True, breakers=brk, hold_day=hold_day, cw_timing=timing, cw_tchk=0,
                  cw=dict(se=se, tauR=tauR, kL=kL, I=I, gaps=sorted(gaps), cache={}, stats={str(q): collections.Counter()
                                                                                              for q in REC_P},
                          worst_page={str(q): 0.0 for q in REC_P}, worst_break={str(q): 0.0 for q in REC_P},
                          hold={str(q): -INF for q in REC_P}, inflight_break_day={str(q): [0] * x["ndays"] for q in REC_P}))
        self.states.append(st)
        self.cur[pol] = st

    def _timing(self, pol, slots, pol_cache):
        if self.ctx is None:
            raise RuntimeError("I-TIMING: no timing context (run_variant was not wrapped)")
        return timing_table(pol, RH.tail_slots(slots), slots, self.ctx["slot_cells"], self.ctx["tr"], self.ctx["first"], pol_cache)

    def __call__(self, pol, s, t, u, sl, gb, site):
        if not self.is_new() or sl["kind"] != "tail":
            return RH.RHook.__call__(self, pol, s, t, u, sl, gb, site)
        V3 = NB._m2897()._v6().V3
        w = sl["units"][u]
        if abs(gb - self.host) > 1e-12:
            raise RuntimeError(f"I-NB-HOOK: tail gb {gb} != the setting's host rate {self.host}")
        host = V3.page_in(w, gb)
        st = self.cur.get(pol)
        if st is None:
            raise RuntimeError(f"I-NB-HOOK: no neighbour state for {pol}")
        if self.cur_p is None:
            raise RuntimeError("I-NB-HOOK: no p for a race page")
        v = self.wake_page(st, s, int(t), u, float(self.cur_p))
        self.calls[id(st)][site].add((s, int(t), u))
        if not (PF - 1e-12 <= v <= host + 1e-12):
            self.bad.append((pol, s, int(t), str(u), v, host))
            raise RuntimeError(f"I-NB-HOOK: page {v} outside [0.02, host {host}]")
        return v

    def wake_page(self, st, s, t, u, p):
        cw = st["cw"]
        ck = (s, t, p)
        if ck not in cw["cache"]:
            T0, A = st["cw_timing"][(s, t)]
            rec_ = str(p) if p in REC_P else None

            def fn(uu, pi):
                return self.unit_page(st, s, t, uu, pi, p, rec_)
            cw["cache"][ck] = wake_pages(A, T0, p, fn)[0]
        pg = cw["cache"][ck]
        if u not in pg:
            raise RuntimeError(f"I-TIMING: unit {u} of slot {s} minute {t} is not in the wake's A")
        return pg[u]

    def unit_page(self, st, s, t, u, pi, p, rec_):
        c, cw = self.cfg, st["cw"]
        w = st["units_of"][s][u]
        X = st["rec"][(s, t, u)]
        std = NB.nb_page(st, s, t, u, w, H_FB, c["delta"])                # the SAME-state page (host when X is None)
        ch = st["wake_chain"][(s, t, u)]
        if X is None:
            return std
        se = cw["se"]
        last = st["cps"][ch[-1]]
        racing = last["kind"] == "R" and last["t"] in (t, t - 1)
        if not racing:
            if last["kind"] == "P" and pi < se[ch[-1]][1]:
                if rec_ is not None:
                    cw["stats"][rec_]["PARK-RACE"] += 1
                return page_rate(w, H_FB)
            return std
        kx = st["k"][(X, t)]
        e_m = se[ch[-1]][1]
        j = next((i for i, ci in enumerate(ch) if pi < se[ci][1]), None)
        worst = c["timing"] == "WORST"
        if j == 0 and not worst:
            v, cls, opt = page_rate(w, H_FB), "PARK-RACE", "host"
        else:
            hc = st["cps"][ch[j if (j is not None and not worst) else -1]]
            H, th = hc["src"], hc["t"]
            tau, kl = cw["tauR"][(H, th)], cw["kL"][(H, th)]
            e_j = se[ch[j]][1] if (j is not None and not worst) else e_m
            v, cls, opt = race_page(c["race"], c["wait"], worst, w, pi, p, tau, e_j, e_m, kl, kx, c["delta"])
            if rec_ is not None and opt == "holder":
                start = max(pi, tau + p) if c["wait"] else pi
                cw["hold"][rec_] = max(cw["hold"][rec_], start + page_rate(w, R_NB * c["delta"] / kl) - (tau + P_STAR))
        if rec_ is not None:
            S = cw["stats"][rec_]
            S[cls] += 1
            S["opt:" + opt] += 1
            S["racing"] += 1
            cw["worst_page"][rec_] = max(cw["worst_page"][rec_], v)
            if w > BREAK_W:
                S["break:" + cls] += 1
                cw["worst_break"][rec_] = max(cw["worst_break"][rec_], v)
                if cls == "INFLIGHT" and opt == "host":
                    cw["inflight_break_day"][rec_][t // DAY] += 1
        return v

    def race_report(self):
        out = []
        for st in self.states:
            cw = st.get("cw")
            if cw is None:
                continue
            g = cw["gaps"]

            def q(f):
                return None if not g else round(g[min(len(g) - 1, int(f * len(g)))], 3)
            out.append(dict(pol=st["pol"], ctx=st["ctx"], I={k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in cw["I"].items()},
                            gaps=dict(n=len(g), q01=q(0.01), q10=q(0.1), q50=q(0.5), q90=q(0.9), q99=q(0.99),
                                      within_1s=sum(1 for v in g if abs(v) <= 1.0), within_5s=sum(1 for v in g if abs(v) <= 5.0)),
                            stats={k_: dict(v) for k_, v in cw["stats"].items()},
                            worst_page={k_: round(v, 4) for k_, v in cw["worst_page"].items()},
                            worst_break={k_: round(v, 4) for k_, v in cw["worst_break"].items()},
                            hold={k_: (None if v == -INF else round(v, 4)) for k_, v in cw["hold"].items()},
                            inflight_break_day=cw["inflight_break_day"], timing_windows_checked=st.get("cw_tchk", 0)))
        return out


HOOK3 = CWHook()
_INST = {}


def install_cw():
    """ installer once, then this file's hook on every site + the run_variant / queue_pass wrappers (pitfalls 1 / 2)."""
    if _INST:
        return _INST["V6"]
    V6 = RH.install_rh()
    V5 = V6.V5
    NB.HOOK = HOOK3
    V5._NB_PAGE = HOOK3
    V6._NB_PAGE = HOOK3
    rv0, qp0 = V5.run_variant, V6.queue_pass

    def rv_wrap2938(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache):
        HOOK3.cur_pol, HOOK3.cur_p = pol, p
        HOOK3.ctx = dict(slots=slots, slot_cells=slot_cells, tr=tr, first=first)
        return rv0(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache)

    def qp_wrap2938(windows, p):
        refresh_windows(windows, p)
        return qp0(windows, p)
    V5.run_variant = rv_wrap2938
    V6.queue_pass = qp_wrap2938
    _INST.update(V6=V6, rv0=rv0, qp0=qp0)
    if not (NB.HOOK is HOOK3 and V5._NB_PAGE is HOOK3 and V6._NB_PAGE is HOOK3 and V5.run_variant is rv_wrap2938
            and V6.queue_pass is qp_wrap2938 and rv0.__name__ == "rv_wrap"):
        raise RuntimeError("install_cw: the hook is not wired on every site")
    return V6


def refresh_windows(windows, p):
    """the per-p page refresh of v6's tail windows (pitfall 1) + I-TIMING at each state's first refresh."""
    H3 = HOOK3
    if H3.cfg.get("mode") != "NB":
        return
    if H3.cur_p != p:
        raise RuntimeError(f"I-PREF: queue_pass p {p} != the last run_variant p {H3.cur_p}")
    pol = H3.cur_pol
    st = H3.cur.get(pol)
    if st is None:
        raise RuntimeError(f"I-PREF: no state for {pol}")
    tm = st["cw_timing"]
    gb = NB._m2897()._v6().H2D_TAIL
    slots = H3.ctx["slots"]
    first_check = st.get("cw_tchk", 0) == 0
    n = 0
    for w in windows:
        if w["kind"] != "tail":
            continue
        t0 = int(w["T0"] // 60.0)
        t = next((tt for tt in (t0, t0 - 1, t0 + 1) if (w["s"], tt) in tm and tm[(w["s"], tt)][0] == w["T0"]), None)
        if t is None:
            raise RuntimeError(f"I-TIMING: window of slot {w['s']} T0 {w['T0']} has no timing entry")
        if first_check:
            if tm[(w["s"], t)][1] != w["A"] or not (60.0 * t <= w["T0"] < 60.0 * t + 60.0):
                raise RuntimeError(f"I-TIMING: timing table != window (slot {w['s']}, minute {t})")
            n += 1
        w["page"] = {u: H3(pol, w["s"], t, u, slots[w["s"]], gb, "bw") for u in w["A"]}
    if first_check:
        st["cw_tchk"] = max(n, 1)


# ------------------------------------------------------------------------------------------------ run
def run_cw(setting, cell, seeds, tag, root, cache, out, maps_file=None, maps_sha=None):
    M = NB._m2897()
    cfg, legacy = setting_cfg(setting)
    maps = RH.load_maps(maps_file, maps_sha) if cfg.get("map", "").startswith("MAP-") else None
    if cell != "D1":
        raise ValueError("the neighbour tier is defined for D1 only (prereg §2)")
    M.configure_v6(cfg["host"])
    V6 = install_cw()
    R = V6.V5.install()
    wd, row, rec, snap, r = V6.static_row_over(cell, seeds, tag, root, cache, None)
    row["cell"] = cell
    row["epar"] = None
    host = M.SETTINGS[cfg["host"]]
    HOOK3.reset(cfg, wd.bh, host, maps)
    row["s2936"] = dict(setting=setting, cfg=cfg, prereg=RH.PREREG, parent_prereg=NB.PREREG, h2d_tail=V6.H2D_TAIL, p_grid=list(V6.P_GRID),
                        r_nb=NB.R_NB, bh=wd.bh, world_h2d=float(wd.tph["h2d"]), seeds=list(seeds))
    row["s2938"] = dict(prereg=PREREG, setting=setting, legacy=legacy)
    if snap is not None:
        if abs(wd.tph["h2d"] - M.WORLD_H2D) > 1e-9:
            raise RuntimeError(f"I-WORLD: the static world's tail h2d is {wd.tph['h2d']}, not {M.WORLD_H2D}")
        row["v6"] = V6.overlay6(wd, row, rec, snap, r, seeds, R)
        rep = HOOK3.report()
        row["s2936"]["nb"] = rep
        if not rep:
            raise RuntimeError("I-NB-HOOK: the neighbour tier built no state")
        if not all(x["hook_sites_agree"] for x in rep):
            raise RuntimeError("I-NB-HOOK: build_windows and run_variant asked for different wakes")
        if not all(x["x2936"]["nested"] and x["x2936"]["cap_ok"] for x in rep):
            raise RuntimeError("I-NEST / I-CAP: a state was not nested or went over CAP")
        if not all(st.get("cw_tchk", 0) > 0 for st in HOOK3.states):
            raise RuntimeError("I-TIMING: a state's timing table was never checked against v6's windows")
        row["s2936"]["node_map"] = {str(k_): v for k_, v in sorted(HOOK3.states[0]["node_map"].items())}
        if maps is not None:
            i = cfg["map"].split("-")[1]
            got = {str(j): HOOK3.states[0]["node_map"][j] for j in RH.tail_slots_from(HOOK3.states[0])}
            if got != maps["draws"][i]:
                raise RuntimeError(f"I-MAP: seed-0 node map != manifest draw {i}")
            row["s2936"]["maps_sha"] = maps_sha
        row["s2938"]["timing_checked"] = [st.get("cw_tchk", 0) for st in HOOK3.states]
        if not legacy:
            rr = HOOK3.race_report()
            if any(not x["I"]["d60"] for x in rr):
                raise RuntimeError("I-D60: a drain ran 60 s or longer")
            row["s2938"]["race"] = rr
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------ checks over rows
def ifix_2936(new_file, ref_file, ignore_tag=False):
    """I-FIX-RHQ / -A1 / -PLC (prereg §8): every field but timing keys and s2938 (and the top-level tag with ignore_tag)."""
    M = NB._m2897()
    n, o = RH._one(new_file), RH._one(ref_file)
    a = {k_: v for k_, v in M.strip(n).items() if k_ != "s2938"}
    b = {k_: v for k_, v in M.strip(o).items() if k_ != "s2938"}
    if ignore_tag:
        a.pop("tag", None)
        b.pop("tag", None)
    top = sorted(k_ for k_ in set(a) | set(b) if k_ not in ("v6", "s2936") and a.get(k_) != b.get(k_))
    nv, ov = a.get("v6") or {}, b.get("v6") or {}
    vd = sorted(v for v in set(nv.get("variants", {})) | set(ov.get("variants", {})) if nv.get("variants", {}).get(v) != ov.get("variants", {}).get(v))
    vparts = sorted(p for p in set(nv) | set(ov) if p != "variants" and nv.get(p) != ov.get(p))
    ns, os_ = a.get("s2936") or {}, b.get("s2936") or {}
    sd = sorted(k_ for k_ in set(ns) | set(os_) if k_ != "nb" and ns.get(k_) != os_.get(k_))
    nbn, nbo = ns.get("nb") or [], os_.get("nb") or []
    nbd = [] if len(nbn) == len(nbo) else [f"nb entries {len(nbn)} != {len(nbo)}"]
    for i, (x, y) in enumerate(zip(nbn, nbo)):
        nbd += [f"nb[{i}].{k_}" for k_ in sorted(set(x) | set(y)) if x.get(k_) != y.get(k_)]
    ok = not top and not vd and not vparts and not sd and not nbd and len(ov.get("variants", {})) > 0 and len(nbo) > 0
    return dict(new=new_file, ref=ref_file, ignore_tag=ignore_tag, static_diffs=top[:10], variant_diffs=vd[:10], v6_part_diffs=vparts[:10],
                s2936_diffs=sd[:10], nb_diffs=nbd[:10], variants_compared=len(ov.get("variants", {})), nb_compared=len(nbo), pass_=ok)


# ------------------------------------------------------------------------------------------------ summary + DOOR-CW (prereg §5 / §6)
def paired_read(x_cw, x_rh):
    """§6 secondary: per map X(PLC-CW-i) vs X(PLC-RH-i), None lowest. PAIRED-ATTRIBUTABLE iff lower 0 and higher >= 7."""
    lo = hi = eq = 0
    for a, b in zip(x_cw, x_rh):
        va = -1.0 if a is None else a
        vb = -1.0 if b is None else b
        if va > vb:
            hi += 1
        elif va < vb:
            lo += 1
        else:
            eq += 1
    return dict(higher=hi, equal=eq, lower=lo, attributable=lo == 0 and hi >= 7)


def cowake_class(real, mx_shift, mx_day, mx_jit):
    """§4b frozen ladder on the seed-pooled real CWR vs each null family's max."""
    if real <= mx_shift:
        return "NOT-SYNCHRONIZED"
    if real <= mx_day:
        return "DIURNAL"
    if real <= mx_jit:
        return "LOCAL"
    return "EXACT-MINUTE"


def _race_of(d):
    out = {}
    for r in NB._m2897().load_rows(d):
        k_ = f"{r['tag']}|{r['cell']}"
        if k_ in out:
            raise ValueError(f"duplicate race row key {k_}")
        out[k_] = (r.get("s2938") or {}).get("race")
    return out


def _hbm(d, tag="main"):
    for r in NB._m2897().load_rows(d):
        if r["tag"] == tag:
            return {(e["pol"], e["ctx"]): e["x2936"]["replicates"] for e in r["s2936"]["nb"]}
    return None


def summarize_cw(out_dir):
    res = dict(settings={}, race={}, replay={}, seedshift={}, skipped=[], checks={})
    for st in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, st)
        if not os.path.isdir(d):
            continue
        try:
            setting_cfg(st)
        except ValueError:
            res["skipped"].append(st)
            continue
        if not any(f.endswith(".jsonl") for f in os.listdir(d)):
            res["skipped"].append(st)
            continue
        res["settings"][st], par = RH.setting_summary(d, st)
        res["replay"][st], res["seedshift"][st] = par["replay"], par["seedshift"]
        res["race"][st] = _race_of(d)
    S = res["settings"]

    def X(st, tag="main"):
        return S[st][tag]["X"] if st in S and tag in S[st] else None

    def door(a, b):
        if a not in S or b not in S:
            return None
        return RH.door_rh(X(a), X(b))
    doors = {nm: door(a, b) for nm, a, b in (("CW", "CW-Q", "CW-Q-SHUF"), ("NW", "CW-Q-NW", "CW-Q-NW-SHUF"), ("H", "CW-H", "CW-H-SHUF"),
                                             ("WORST", "CW-Q-WORST", "CW-Q-WORST-SHUF"),
                                             ("NW-WORST", "CW-Q-NW-WORST", "CW-Q-NW-WORST-SHUF"), ("SUB", "SUB-Q", "SUB-Q-SHUF"),
                                             ("LW", "LW-Q", "LW-Q-SHUF"), ("RH-Q", "RH-Q", "RH-Q-SHUF"),
                                             ("A1", "A1-RHS-Q", "A1-RHS-Q-SHUF"))}
    res["doors"] = doors
    dq = doors.get("CW")
    # I-HBM-EQ: every new setting's replicates = RH-Q's of the same map (main)
    hb = {}
    ref = {"ORDER": _hbm(os.path.join(out_dir, "RH-Q")) if "RH-Q" in S else None,
           "SHUF": _hbm(os.path.join(out_dir, "RH-Q-SHUF")) if "RH-Q-SHUF" in S else None}
    for st in S:
        cfg, legacy = setting_cfg(st)
        if legacy or cfg["map"] not in ref or ref[cfg["map"]] is None:
            continue
        hb[st] = _hbm(os.path.join(out_dir, st)) == ref[cfg["map"]]
    res["checks"]["I_HBM_EQ"] = hb
    if dq is not None:
        def rmax(st, key, sub=None):
            vals = []
            for kk, rr in res["race"].get(st, {}).items():
                if not kk.startswith("main|") or not rr:
                    continue
                for e in rr:
                    v = e["I"][key] if sub is None else e[key].get(sub)
                    if v is not None:
                        vals.append(v)
            return max(vals) if vals else None
        flags = dict(map_sensitive=dq["map_sensitive"],
                     guard_bound=S["CW-Q"]["main"]["guard_bound"] or S["CW-Q-SHUF"]["main"]["guard_bound"])
        for nm, fl in (("NW", "wait_sensitive"), ("H", "rate_sensitive"), ("WORST", "align_sensitive")):
            if doors.get(nm) is not None:
                flags[fl] = doors[nm]["door"] != dq["door"]
        if doors.get("NW-WORST") is not None and doors.get("NW") is not None:
            flags["align_sensitive_nw"] = doors["NW-WORST"]["door"] != doors["NW"]["door"]
        dm = [rmax(st, "drain_max") for st in ("CW-Q", "CW-Q-SHUF")]
        flags["drain_exposed"] = any(v is not None and v > P_STAR for v in dm)
        hm = [rmax(st, "hold", "0.842") for st in ("CW-Q", "CW-Q-SHUF")]
        flags["hold_exposed"] = any(v is not None and v > 0 for v in hm)
        res["drain_max_deciding"], res["hold_max_deciding_0842"] = dm, hm
        ss = res["seedshift"].get("CW-Q")
        if ss is not None:
            flags["seed_shift"] = bool(ss.get("UNATTRIBUTABLE"))
            res["seedshift_cwq"] = ss
        plc_rh = [f"PLC-RH-{i}" for i in range(K_MAPS) if f"PLC-RH-{i}" in S and "main" in S[f"PLC-RH-{i}"]]
        if plc_rh:
            spread, att = {}, {}
            for a in DEC_ARMS:
                pm = [S[st]["main"]["p_meet6"][a] for st in plc_rh]
                m8 = [S[st]["main"]["meet_0_842"][a] for st in plc_rh]
                m79 = [S[st]["main"]["meet_0_79"][a] for st in plc_rh]
                spread[a] = dict(p_meet6=pm, p_meet6_max=max((v for v in pm if v is not None), default=None),
                                 meet_0_842_true=sum(bool(v) for v in m8), meet_0_79_true=sum(bool(v) for v in m79))
                for arm_st in ("CW-Q", "CW-Q-SHUF"):
                    r = S[arm_st]["main"]
                    att[f"{arm_st}|{a}"] = dict(p_meet6=RH.attributable(r["p_meet6"][a], pm),
                                                meet_0_842=RH.attributable_bool(r["meet_0_842"][a], m8),
                                                meet_0_79=RH.attributable_bool(r["meet_0_79"][a], m79))
            aware = {}
            for arm_st, ctrl in (("CW-Q", "RH-Q"), ("CW-Q-SHUF", "RH-Q-SHUF")):
                r, cc = S[arm_st]["main"], S.get(ctrl, {}).get("main")
                pm = {a: (r["p_meet6"][a] if att[f"{arm_st}|{a}"]["p_meet6"] or cc is None else cc["p_meet6"][a]) for a in DEC_ARMS}
                aware[arm_st] = RH.x_of(pm, r["eligible"])[0]
            plc = dict(feasible=len(plc_rh), void=len(plc_rh) < 10, spread=spread, attributable=att,
                       any_attributable=any(v["p_meet6"] or v["meet_0_842"] or v["meet_0_79"] for v in att.values()),
                       aware_X=aware, door_aware=RH.door_rh(aware.get("CW-Q"), aware.get("CW-Q-SHUF")))
            plc_cw = [f"PLC-CW-{i}" for i in range(K_MAPS)]
            if all(st in S and "main" in S[st] for st in plc_cw) and len(plc_rh) == K_MAPS:
                xc, xr = [X(st) for st in plc_cw], [X(f"PLC-RH-{i}") for i in range(K_MAPS)]
                plc["paired"] = dict(paired_read(xc, xr), X_cw=xc, X_rh=xr)
                thr = {"OPEN-AT-STOCK-p": 0.842, "OPEN-WITH-p<=0.5": 0.5}.get(dq["door"])
                plc["robustness"] = dict(thr=thr, n_meet=None if thr is None else sum(1 for v in xc if v is not None and v >= thr - 1e-12),
                                         classes=dict(collections.Counter(NB.door_class(v) for v in xc)))
                flags["map_fragile"] = thr is not None and plc["robustness"]["n_meet"] < K_MAPS
                flags["paired_attributable"] = plc["paired"]["attributable"]
            res["placebo"] = plc
            flags["placebo_attributable_any"] = plc["any_attributable"]
        res["door_cw"] = dict(dq, flags=flags)
    return res


# ------------------------------------------------------------------------------------------------ co-wake synchronization (prereg §4b)
FAMILIES = ("N-SHIFT", "N-DAY", "N-JIT")
_JIT = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


def null_offset(fam, i, slot, week=10080):
    h = int(hashlib.sha256(f"2938|{fam}|{i}|{slot}".encode()).hexdigest(), 16)
    if fam == "N-SHIFT":
        return 1 + h % (week - 1)
    if fam == "N-DAY":
        return 1440 * (1 + h % 6)
    if fam == "N-JIT":
        return _JIT[h % 10]
    raise ValueError(f"unknown null family {fam}")


def cowake_events(eng, wake_of, n_min):
    """NEXT-mode co-wakes with their (owner, lender) pair: re-homes whose owner wakes in the reclaim minute."""
    ev = []
    for rc in eng["reclaims"]:
        for key, w, L2, ci in rc["items"]:
            if ci is not None and wake_of[key[0]][rc["t"]]:
                ev.append((key[0], rc["L"], rc["t"], w))
    return ev


def cowake_null(root, cache, out, ref_order, ref_shuf, K=20):
    import numpy as np
    M = NB._m2897()
    V6 = M.configure_v6("H12")
    install_cw()
    V3, V4 = V6.V3, V6.V4
    V6.V5.install()
    seeds = (0, 1, 2)
    wd, row, rec, snap, r = V6.static_row_over("D1", seeds, "cowake2938", root, cache, None)
    if snap is None:
        raise RuntimeError("the D1 static row did not complete")
    tr = wd.cx["tr"]
    train = ~(tr.minute >= V6.HELD)
    refs = {}
    for mp, f in (("ORDER", ref_order), ("SHUF", ref_shuf)):
        rr = RH._one(f)
        refs[mp] = {(e["pol"], e["ctx"]): (e["x2936"]["inflight_fallback"], e["stats"]["rehomed"]) for e in rr["s2936"]["nb"]}
    cap = 0.5 * wd.bh
    res = dict(prereg=PREREG, K=K, seeds=list(seeds), real={}, null={}, i_null0={}, pairs={}, hours={}, burst={})
    for si in range(len(seeds)):
        fi = rec["fi"][si]
        slots = V3.build_slots(wd, fi)
        bs = [V3.slot_busy(sl, fi) for sl in slots]
        khs = [V3.k_hist(tr.minute[train & np.isin(tr.mid, sl["models"])]) for sl in slots]
        tail = RH.tail_slots(slots)
        units_of = {i: dict(slots[i]["units"]) for i in tail}
        nms = {mp: NB.node_map(slots, mp) for mp in ("ORDER", "SHUF")}
        for pol in V6.POL6:
            def sched(shift):
                ho, wo = {}, {}
                for i in tail:
                    b = np.roll(np.asarray(bs[i], bool), shift.get(i, 0))
                    h, w = V3.held_static(b) if not slots[i]["host_ok"] else V4.slot_policy(pol, b, khs[i], False)
                    ho[i], wo[i] = h, w
                return ho, wo
            ho, wo = sched({})
            n_min = len(next(iter(ho.values())))
            allw = np.zeros(n_min, np.int64)
            for i in tail:
                allw += np.asarray(wo[i], np.int64)
            for mp in ("ORDER", "SHUF"):
                no = {i: nms[mp][i] for i in tail}
                eng = cw_engine(ho, wo, units_of, no, cap, "REHOME", n_min, "NEXT", None, True)
                a, b_ = eng["x"]["inflight_fallback"], eng["stats"]["rehomed"]
                want = refs[mp].get((pol, si))
                res["i_null0"][f"{pol}|{mp}|{si}"] = [a, b_, want]
                if want is None or [a, b_] != list(want):
                    raise RuntimeError(f"I-NULL0: zero shift {pol} {mp} seed {si}: {a}/{b_} != committed {want}")
                k_ = f"{pol}|{mp}"
                res["real"].setdefault(k_, []).append([a, b_])
                evs = cowake_events(eng, wo, n_min)
                pc = res["pairs"].setdefault(k_, collections.Counter())
                for O, L, t, w in evs:
                    pc[f"{O}->{L}"] += 1
                hc = res["hours"].setdefault(k_, [0] * 24)
                bc = res["burst"].setdefault(k_, dict(cowake_minute_wakes=[], mh=[0] * 60))
                for O, L, t, w in evs:
                    hc[(t % 1440) // 60] += 1
                    bc["cowake_minute_wakes"].append(int(allw[t]))
                    bc["mh"][t % 60] += 1
                bc.setdefault("all_wake_minute_wakes_mean", []).append(float(allw[allw > 0].mean()) if (allw > 0).any() else 0.0)
            for fam in FAMILIES:
                for i in range(K):
                    sh = {j: null_offset(fam, i, j) for j in tail}
                    hn, wn = sched(sh)
                    for mp in ("ORDER", "SHUF"):
                        no = {j: nms[mp][j] for j in tail}
                        e2 = RH.rh_simulate(hn, wn, units_of, no, cap, "REHOME", n_min, "NEXT", None, True)
                        res["null"].setdefault(f"{pol}|{mp}|{fam}", []).append([i, si, e2["x"]["inflight_fallback"], e2["stats"]["rehomed"]])
    cls = {}
    for k_, v in res["real"].items():
        real = sum(a for a, b in v) / max(1, sum(b for a, b in v))
        mx = {fam: max((a / b if b else 0.0) for i, si, a, b in res["null"][f"{k_}|{fam}"]) for fam in FAMILIES}
        med = {}
        for fam in FAMILIES:
            vals = sorted((a / b if b else 0.0) for i, si, a, b in res["null"][f"{k_}|{fam}"])
            med[fam] = [round(vals[0], 4), round(vals[len(vals) // 2], 4), round(vals[-1], 4)]
        cls[k_] = dict(real=round(real, 4), null_min_med_max=med, cls=cowake_class(real, mx["N-SHIFT"], mx["N-DAY"], mx["N-JIT"]))
    res["classes"] = cls
    res["pairs"] = {k_: dict(n=sum(v.values()), distinct=len(v), top5=v.most_common(5),
                             top5_share=round(sum(c for _, c in v.most_common(5)) / max(1, sum(v.values())), 4))
                    for k_, v in res["pairs"].items()}
    for k_, bc in res["burst"].items():
        cm = sorted(bc["cowake_minute_wakes"])
        bc["cowake_minute_wakes"] = dict(n=len(cm), median=cm[len(cm) // 2] if cm else None, p90=cm[int(0.9 * len(cm))] if cm else None)
    raw = (json.dumps(res, sort_keys=True) + "\n").encode()
    with open(out, "wb") as fh:
        fh.write(raw)
    print(json.dumps(dict(out=out, sha256=hashlib.sha256(raw).hexdigest(), classes=cls)))
    return 0


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    import numpy as np, tempfile
    NB._m2897().configure_v6("H12")
    V6 = install_cw()
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))
    r6 = lambda v: round(v, 6)  # noqa: E731
    # G1 drain
    dr = drain_items(100.0, [(10.0, 1, -INF), (4.0, 2, -INF)], 2, 0.25)
    chk("G1 drain serial, source side binds", [r6(dr[0][1] - dr[0][0]), r6(dr[0][1]), r6(dr[1][1] - dr[1][0]), r6(dr[1][1])],
        [0.338146, 100.338146, 0.147258, 100.485404])
    # G2 page options
    a = dict(w=15.8, p=0.842, tau=100.0, e_j=100.5, e_m=100.5, kl=2, kx=1, delta=0.25)

    def rp(race, wait, pi, worst=False, **o):
        z = dict(a, **o)
        return r6(race_page(race, wait, worst, z["w"], pi, z["p"], z["tau"], z["e_j"], z["e_m"], z["kl"], z["kx"], z["delta"])[0])
    chk("G2 host / nb k1 / nb k2", [r6(page_rate(15.8, 12.4)), r6(page_rate(15.8, 67.5)), r6(page_rate(15.8, 33.75))],
        [1.388155, 0.271335, 0.522670])
    chk("G2 in flight pi 100.2: SUB / LW / CW / CW-NW", [rp("SUB", None, 100.2), rp("LW", None, 100.2), rp("CW", True, 100.2), rp("CW", False, 100.2)],
        [1.388155, 0.571335, 0.571335, 0.522670])
    chk("G2 pre-reclaim 99.9 (all) / landed 100.6", [rp("SUB", None, 99.9), rp("LW", None, 99.9), rp("CW", True, 99.9), rp("CW", False, 99.9),
                                                      rp("CW", True, 100.6), rp("SUB", None, 100.6)],
        [0.522670, 0.522670, 0.522670, 0.522670, 0.271335, 0.271335])
    chk("G2b (added at code time) H unavailable after its copy landed even when it would be faster",
        rp("CW", True, 100.1, e_j=100.3, e_m=100.3, kl=1, kx=4), 1.225340)
    # G3 WAIT binds
    chk("G3 e 101.5 pi 100.1: CW / NW / SUB / LW", [rp("CW", True, 100.1, e_j=101.5, e_m=101.5), rp("CW", False, 100.1, e_j=101.5, e_m=101.5),
                                                    rp("SUB", None, 100.1, e_j=101.5, e_m=101.5), rp("LW", None, 100.1, e_j=101.5, e_m=101.5)],
        [1.264670, 0.522670, 1.388155, 1.388155])
    # G4 WORST
    chk("G4 WORST", [rp("CW", True, 0.0, True), rp("CW", False, 0.0, True), rp("CW", True, 0.0, True, e_j=101.5, e_m=101.5)],
        [0.771335, 0.522670, 1.364670])
    # G5 sequencing through wake_ready
    def fn5(u, pi, race="CW", wait=True):
        if u == "u1":
            return page_rate(2.0, 67.5)
        return race_page(race, wait, False, 15.8, pi, 0.5, 100.0, 100.5, 100.5, 2, 1, 0.25)[0]
    A5 = {"u1": 99.5, "u2": 99.6}
    pg, pis = wake_pages(A5, 99.5, 0.5, fn5)
    chk("G5 u1 page / pi2 / CW (H unavailable at e)", [r6(pg["u1"]), r6(pis["u2"]), r6(pg["u2"])], [0.051815, 100.051815, 0.719521])
    chk("G5 NW / SUB", [r6(wake_pages(A5, 99.5, 0.5, lambda u, pi: fn5(u, pi, "CW", False))[0]["u2"]),
                        r6(wake_pages(A5, 99.5, 0.5, lambda u, pi: fn5(u, pi, "SUB", None))[0]["u2"])], [0.522670, 1.388155])
    V3 = V6.V3
    rdy = V3.wake_ready(A5, pg, 100.0)
    chk("G5 pi = v3 wake_ready's start", [r6(rdy["u1"] - pg["u1"]), r6(rdy["u2"] - pg["u2"])], [r6(pis["u1"]), r6(pis["u2"])])
    # G6 chain
    d1 = drain_items(119.95, [(5.0, 1, -INF)], 1, 0.25)
    d2 = drain_items(120.01, [(5.0, 1, d1[0][1])], 1, 0.25)
    chk("G6 chain drains", [r6(d1[0][1]), r6(d2[0][0]), r6(d2[0][1])], [120.049536, 120.049536, 120.149073])
    g6 = dict(w=5.0, p=0.842, tau=119.95, e_j=d1[0][1], e_m=d2[0][1], kl=1, kx=1, delta=0.25)
    chk("G6 chain race pi 120.03: CW / NW / SUB",
        [r6(race_page("CW", True, False, g6["w"], 120.03, g6["p"], g6["tau"], g6["e_j"], g6["e_m"], 1, 1, 0.25)[0]),
         r6(race_page("CW", False, False, g6["w"], 120.03, g6["p"], g6["tau"], g6["e_j"], g6["e_m"], 1, 1, 0.25)[0]),
         r6(race_page("SUB", None, False, g6["w"], 120.03, g6["p"], g6["tau"], g6["e_j"], g6["e_m"], 1, 1, 0.25)[0])],
        [0.218609, 0.099536, 0.452960])
    # G7 nesting (300 random toys, SAME+charge and NEXT+charge and plain) + chain records on bounce toy
    rng = np.random.default_rng(2938)

    def toy(held, units, nodes, cap, eng, **kw):
        H = {s: np.array(h, bool) for s, h in held.items()}
        W = {}
        for s, h in H.items():
            W[s] = h & ~np.concatenate([[True], h[:-1]])
        n = len(next(iter(H.values())))
        return (cw_engine if eng == "cw" else RH.rh_simulate)(H, W, units, nodes, cap, "REHOME", n, **kw), W
    ok, ncmp = True, 0
    for _ in range(300):
        ns, nmin = int(rng.integers(3, 10)), int(rng.integers(30, 121))
        held, units, nodes = {}, {}, {}
        for s in range(ns):
            h, cur = np.zeros(nmin, bool), bool(rng.random() < 0.5)
            for t in range(nmin):
                if rng.random() < 0.15:
                    cur = not cur
                h[t] = cur
            held[s] = h.tolist()
            units[s] = {f"u{j}": float(rng.uniform(0.5, 30.0)) for j in range(int(rng.integers(0, 4)))}
            nodes[s] = int(rng.integers(0, 3))
        cap = float(rng.uniform(10, 40))
        for kw in (dict(avail="SAME", charge=True), dict(avail="NEXT", charge=True), dict()):
            a_, _ = toy(held, units, nodes, cap, "cw", **kw)
            b_, _ = toy(held, units, nodes, cap, "rh", **kw)
            ok = ok and nest_ok(a_, b_)
            ncmp += 1
    chk("G7 nesting (300 toys x 3 modes)", (ok, ncmp), (True, 900))
    g5 = dict(held={0: [0, 0, 1, 1, 1, 1], 1: [0, 0, 0, 1, 1, 1], 2: [0] * 6, 3: [1, 0, 0, 0, 0, 1]},
              units={0: {}, 1: {}, 2: {}, 3: {"g": 10.0}}, nodes={0: 0, 1: 0, 2: 0, 3: 0})
    e5, W5 = toy(g5["held"], g5["units"], g5["nodes"], 30.0, "cw", avail="SAME", charge=True)
    chn = [(e5["cps"][c]["kind"], e5["cps"][c]["src"], e5["cps"][c]["dst"], e5["cps"][c]["t"]) for c in e5["wake_chain"][(3, 5, "g")]]
    chk("G7 bounce chain recorded (park 3->0 t1, 0->1 t2, 1->2 t3)", chn, [("P", 3, 0, 1), ("R", 0, 1, 2), ("R", 1, 2, 3)])
    # G7b drains_of on the bounce toy: reclaims at 60 t (no timing for wakes not needed here: lenders 0 / 1 wake at t2 / t3)
    tm5 = {(0, 2): (120.5, {}), (1, 3): (180.25, {}), (3, 5): (300.0, {"g": 300.0})}
    se5, tau5, kl5, I5 = drains_of(e5, tm5, W5, 0.25, 6)
    chk("G7b reclaim instants from the lender's own T0, h counts owners waking at t or t+1",
        [tau5[(0, 2)], tau5[(1, 3)], kl5[(0, 2)], kl5[(1, 3)], I5["reclaim_nowake"]], [120.5, 180.25, 1, 1, 0])
    # G7c (added at code time) kL counts the handoff candidates: the owner wakes at the second reclaim minute (t+1 of the first)
    g7 = dict(held={0: [0, 0, 1, 1, 1, 1], 1: [0, 0, 0, 1, 1, 1], 2: [0] * 6, 3: [1, 0, 0, 1, 1, 1]},
              units={0: {}, 1: {}, 2: {}, 3: {"g": 10.0}}, nodes={0: 0, 1: 0, 2: 0, 3: 0})
    e7, W7 = toy(g7["held"], g7["units"], g7["nodes"], 30.0, "cw", avail="SAME", charge=True)
    se7, tau7, kl7, I7 = drains_of(e7, {(0, 2): (120.5, {}), (1, 3): (180.25, {}), (3, 3): (181.0, {"g": 181.0})}, W7, 0.25, 6)
    chk("G7c kL = 1 + candidates (owner wakes at t+1 / t)", [kl7[(0, 2)], kl7[(1, 3)]], [2, 2])
    # G8 old rules through the new plumbing: legacy hook = nb_page at every p; window refresh leaves pages unchanged
    HOOK3.reset(dict(LEGACY["A1-RHS-Q"]), 71.24, 12.4)
    st8, _ = toy(g5["held"], g5["units"], g5["nodes"], 30.0, "rh", avail="SAME", charge=True)
    st8["cw_timing"] = {(3, 5): (300.0, {"g": 300.0})}
    st8["cw_tchk"] = 0
    HOOK3.cur["KA-1"] = st8
    sl8 = dict(kind="tail", units={"g": 10.0})
    vals = []
    for p in (0.0, 0.5, 0.842, 30.0):
        HOOK3.cur_p, HOOK3.cur_pol = p, "KA-1"
        vals.append(HOOK3("KA-1", 3, 5, "g", sl8, 12.4, "rv") == NB.nb_page(st8, 3, 5, "g", 10.0, 12.4, 0.25))
    chk("G8 legacy hook = nb_page at every p", vals, [True] * 4)
    HOOK3.ctx = dict(slots={3: sl8})
    win = [dict(kind="tail", s=3, T0=300.0, A={"g": 300.0}, page={"g": -1.0}), dict(kind="anchor", s=9, T0=1.0, A={}, page={"x": 7.0})]
    HOOK3.cur_p = 0.842
    refresh_windows(win, 0.842)
    chk("G8 refresh: tail page = hook value, anchor untouched, I-TIMING counted",
        [win[0]["page"]["g"] == NB.nb_page(st8, 3, 5, "g", 10.0, 12.4, 0.25), win[1]["page"], st8["cw_tchk"]], [True, {"x": 7.0}, 1])
    try:
        refresh_windows(win, 0.5)
        chk("G8 refresh refuses a p it was not handed", "accepted", "refused")
    except RuntimeError:
        chk("G8 refresh refuses a p it was not handed", "refused", "refused")
    HOOK3.reset(dict(mode="OFF"), None, None)
    # G9 timing table (build_windows' rule) on a toy trace
    class _T:
        pass
    tr9 = _T()
    tr9.mid = np.array([0, 1, 0, 1, 0])
    mins9 = np.array([10, 10, 11, 12, 20])
    first9 = np.array([30.0, 5.0, 1.0, 2.0, 4.0])
    held9 = np.zeros(30, bool)
    held9[10:16] = True
    held9[20:22] = True
    wake9 = np.zeros(30, bool)
    wake9[10] = wake9[20] = True
    sl9 = [dict(kind="tail", unit_of={0: "a", 1: "b"}, units={"a": 1.0, "b": 2.0})]
    tt9 = timing_table("KA-5", [0], sl9, [(mins9, [0, 1, 2, 3, 4])], tr9, first9, {("KA-5", 0, False): (held9, wake9)})
    chk("G9 timing table", tt9, {(0, 10): (605.0, {"a": 630.0, "b": 605.0}), (0, 20): (1204.0, {"a": 1204.0})})
    # G10 shifts
    b10 = np.zeros(10080, bool)
    b10[[5, 1500, 10079]] = True
    offs = [null_offset(f, i, j) for f in FAMILIES for i in range(3) for j in range(4)]
    sh = [null_offset("N-SHIFT", i, j) for i in range(20) for j in range(80)]
    dy = [null_offset("N-DAY", i, j) for i in range(20) for j in range(80)]
    ji = [null_offset("N-JIT", i, j) for i in range(20) for j in range(80)]
    d0 = null_offset("N-DAY", 0, 0)
    rb = np.roll(b10, d0)
    chk("G10 ranges + deterministic + N-DAY keeps time of day + permutation",
        [min(sh) >= 1 and max(sh) <= 10079, set(dy) <= {1440 * k for k in range(1, 7)}, set(ji) <= set(_JIT),
         offs == [null_offset(f, i, j) for f in FAMILIES for i in range(3) for j in range(4)],
         sorted((np.flatnonzero(rb) % 1440).tolist()) == sorted((np.flatnonzero(b10) % 1440).tolist()), int(rb.sum()) == 3,
         bool(np.array_equal(np.roll(b10, 0), b10))], [True] * 7)
    # G11 summarize duplicates + paired read + template
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "a.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(tag="main", cell="D1")) + "\n" + json.dumps(dict(tag="main", cell="D1")) + "\n")
        try:
            _race_of(td)
            chk("G11 duplicates refused", "accepted", "refused")
        except ValueError:
            chk("G11 duplicates refused", "refused", "refused")
    chk("G11 paired read", [paired_read([0.842] * 7 + [0.5] * 13, [0.5] * 7 + [0.5] * 13)["attributable"],
                            paired_read([0.842] * 19 + [0.271], [0.5] * 19 + [0.5])["attributable"],
                            paired_read([0.842] * 6 + [0.5] * 14, [0.5] * 20)["attributable"],
                            paired_read([None, 0.5], [None, None])["higher"]], [True, False, False, 1])
    chk("G11 template: 0.842 vs spread max 0.842 is INSIDE", RH.attributable(0.842, [0.5, 0.842, None]), False)
    # G12 co-wake ladder
    chk("G12 ladder", [cowake_class(0.5, 0.6, 0.7, 0.8), cowake_class(0.5, 0.1, 0.6, 0.7), cowake_class(0.5, 0.1, 0.2, 0.6),
                       cowake_class(0.5, 0.1, 0.2, 0.3)], ["NOT-SYNCHRONIZED", "DIURNAL", "LOCAL", "EXACT-MINUTE"])
    chk("settings: names + legacy cfg = 's", [setting_cfg("PLC-CW-19")[0]["map"], setting_cfg("PLC-RH-3")[1], setting_cfg("CW-Q")[1],
                                                    setting_cfg("RH-Q")[0] == RH.SETTINGS["RH-Q"], _refuses(lambda: setting_cfg("PLC-CW-20")),
                                                    _refuses(lambda: setting_cfg("NB-P"))], ["MAP-19", True, False, True, True, True])
    chk("hook wired on every site", [NB.HOOK is HOOK3, V6.V5._NB_PAGE is HOOK3, V6._NB_PAGE is HOOK3,
                                     V6.V5.run_variant.__name__, V6.queue_pass.__name__], [True, True, True, "rv_wrap2938", "qp_wrap2938"])
    npass = sum(1 for r in res if r[1])
    for name, ok_, got, exp in res:
        if not ok_:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    print(f"SELF-TEST {'PASS' if npass == len(res) else 'FAIL'} {npass}/{len(res)}")
    return 0 if npass == len(res) else 1


def _refuses(f):
    try:
        f()
        return False
    except ValueError:
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--ifix-2936", nargs=2, metavar=("NEW_FILE", "REF_FILE"))
    m.add_argument("--ifix-nb", nargs=4, metavar=("NEW_DIR", "OFF_FILE", "SAME_FILE", "H12_FILE"))
    m.add_argument("--summarize-cw", metavar="OUT_DIR")
    m.add_argument("--cowake-null", action="store_true")
    ap.add_argument("--ignore-tag", action="store_true")
    ap.add_argument("--setting")
    ap.add_argument("--cell", default="D1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--maps")
    ap.add_argument("--maps-sha")
    ap.add_argument("--ref-order")
    ap.add_argument("--ref-shuf")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    try:
        if a.self_test:
            return self_test()
        if a.ifix_2936:
            r = ifix_2936(*a.ifix_2936, ignore_tag=a.ignore_tag)
            print(json.dumps(r))
            return 0 if r["pass_"] else 1
        if a.ifix_nb:
            install_cw()
            r = NB.ifix_nb(*a.ifix_nb)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize_cw:
            install_cw()
            print(json.dumps(summarize_cw(a.summarize_cw)))
            return 0
        if a.cowake_null:
            for k in ("root", "cache", "out", "ref_order", "ref_shuf"):
                if getattr(a, k) is None:
                    raise ValueError(f"--{k.replace('_', '-')} is required")
            return cowake_null(a.root, a.cache, a.out, a.ref_order, a.ref_shuf, a.k)
        for k in ("setting", "root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        V6 = NB._m2897()._v6()
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if len(seeds) * V6.NDAYS != V6.NREPS:
            raise ValueError(f"{len(seeds)} seeds x {V6.NDAYS} days != {V6.NREPS} replicates")
        return run_cw(a.setting, a.cell, seeds, a.tag, a.root, a.cache, a.out, a.maps, a.maps_sha)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
