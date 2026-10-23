#!/usr/bin/env python3
"""rehoming.py — RE-HOMING as the DECIDING design for the 97-model 2 s door (cloud sim v6, cell D1, NVLink neighbour
tier), with its HBM cost as a GUARD and the busy-lender copy rate as a SENSITIVITY. FORK BY IMPORT: no landed sim is edited.
wrapper (`neighbour_wake`) supplies the page-in hook machinery (both v6 page-in sites forked onto one `_NB_PAGE`
hook), `nb_page`, the node maps and its `ifix_nb`; this file adds a neighbour-cache engine (`rh_simulate`) that NESTS
`nb_simulate` (G2 on 300 random toys, I-NEST in every job, I-FIX-RH on the committed challenger row) and extends it with the frozen
re-homing definitions: in-flight availability, a bounce limit, the in-flight HBM charge, per-node / per-day HBM accounting, drains,
hold-home yardstick and the MAP placebo family.

Prereg (frozen before this file existed): (private report). On any divergence the prereg wins.

Output contract:
  --self-test                                  exit 0 + "SELF-TEST PASS n/n" (goldens G1-G14); exit 1 + FAIL lines.
  --freeze-maps OUT --root R --cache C         WRITES OUT (the MAP-0..19 manifest) ONLY if its checks pass, then prints ONE JSON line
                                               {out, sha256 = the sha256 of OUT's raw bytes, checks}; exit 0. A failed check writes
                                               nothing and exits 1 (QA r1: the digest is a field of that JSON, not a bare line).
  --run --setting S [--cell D1] [--seeds 0,1,2] [--tag T] --root R --cache C --out F [--maps M --maps-sha H]
                                               WRITES (appends) ONE JSON line to F: v6's row + "s2936" (setting, node map, the
                                               neighbour-tier stats per seed x policy incl. "x2936" HBM / drain / bounce numbers);
                                               exit 0. A failed in-job check exits 1. A MAP-i setting REFUSES (exit 2) without a manifest
                                               whose ok is true and whose raw sha256 equals --maps-sha.
  --ifix-row NEW_FILE REF_FILE                 read-only: I-FIX-RH / I-FIX-NBP / I-FIX-OFF (NEW row vs a committed row, every field
                                               but timing and provenance, plus every nb key); one JSON on stdout; exit 0 PASS / 1 FAIL.
  --ifix-nb NEW_DIR OFF_FILE SAME_FILE H12_FILE  read-only: I-NB-STATIC + I-NB-BRACKET + I-REPLAY over NEW_DIR; exit 0 / 1.
  --summarize-rh OUT_DIR                       read-only: per setting sub-directory summarize_v6 + the HBM guard + DOOR-RH,
                                               DOOR-RH@1/2, placebo attribution, robustness; REFUSES duplicate rows (exit 2); one JSON; exit 0.
  Bad input = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. ** forks look the hook up by NAME at call time** (`HOOK` in its module for `rv_wrap`, `_NB_PAGE` in v5's and v6's module
     globals); replacing the hook means re-pointing all three, or the old hook silently serves the run (install_rh does it and asserts).
  2. ** `_static` strips only its own provenance keys**; a row carrying "s2936" reads as a static diff unless STRIP_ROW is extended
     in-process (done at import).
  3. **`ssh host '... &'` holds the session open**: launch with `ssh -n` + `setsid -f`.
"""
import argparse, collections, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neighbour_wake as NB  # noqa: E402

PREREG = "98966be54"
GIB = NB.GIB
DAY = 1440
P_STAR = 0.842
S2 = 2.0
H_FB = 12.4
BREAK_W = (S2 - P_STAR - NB.PAGE_FIXED) * H_FB / GIB      # 13.142 GiB: host page + p* > S2 (prereg §3.4)
G_FLEET, G_NODE = 0.20, 0.25                               # prereg §5 HBM guard ceilings (<= passes)
N_FLEET = 112
MAP_SEED = "2936"
K_MAPS = 20
DEC_ARMS = NB.DEC_ARMS
POL_OF = {"KA-1|QINH": "KA-1", "KA-5|QINH": "KA-5"}
NB.STRIP_ROW = tuple(NB.STRIP_ROW) + ("s2936",)            # pitfall 2

_B = dict(mode="NB", host="H12", cap=0.5)
SETTINGS = {
    "RH-Q": dict(_B, map="ORDER", evict="REHOME", delta=0.25, avail="NEXT", bounce=None, charge=True),
    "RH-Q-SHUF": dict(_B, map="SHUF", evict="REHOME", delta=0.25, avail="NEXT", bounce=None, charge=True),
    "RH-H": dict(_B, map="ORDER", evict="REHOME", delta=0.5, avail="NEXT", bounce=None, charge=True),
    "RH-H-SHUF": dict(_B, map="SHUF", evict="REHOME", delta=0.5, avail="NEXT", bounce=None, charge=True),
    "RH-Q-NB1": dict(_B, map="ORDER", evict="REHOME", delta=0.25, avail="NEXT", bounce=1, charge=True),
    "DROP-Q": dict(_B, map="ORDER", evict="DROP", delta=0.25, avail="NEXT", bounce=None, charge=True),
    "DROP-Q-SHUF": dict(_B, map="SHUF", evict="DROP", delta=0.25, avail="NEXT", bounce=None, charge=True),
    "RH-2933": dict(_B, map="ORDER", evict="REHOME", delta=0.5, avail="SAME", bounce=None, charge=False),
    "DROP-2933": dict(_B, map="ORDER", evict="DROP", delta=0.5, avail="SAME", bounce=None, charge=False),
    "OFF-H12": dict(mode="OFF", host="H12"),
}
REF_OF = {"RH-2933": "NB-REHOME", "DROP-2933": "NB-P", "OFF-H12": "OFF-H12"}   # the committed row each one must equal


def setting_cfg(name):
    if name in SETTINGS:
        return dict(SETTINGS[name])
    m = re.fullmatch(r"PLC-(DROP|RH)-(\d+)", name or "")
    if not m or int(m.group(2)) >= K_MAPS:
        raise ValueError(f"unknown setting {name}")
    cfg = dict(SETTINGS["DROP-Q" if m.group(1) == "DROP" else "RH-Q"])
    cfg["map"] = f"MAP-{int(m.group(2))}"
    return cfg


# ------------------------------------------------------------------------------------------------ the engine (prereg §3.2)
def rh_simulate(held_of, wake_of, units_of, node_of, cap, evict, n_min, avail="SAME", bounce=None, charge=False, day=DAY):
    """ nb_simulate (prereg §3.3-3.7 of), extended per §3.2. With avail SAME, bounce None and charge False it
    returns exactly what nb_simulate returns in rec / k / copies / stats (G2, I-NEST). Extras in "x": per (node, day) guest GiB-min,
    the in-flight charge and lender GPU-min; re-home items for the drain; bounce histogram; in-flight fallbacks; drops at eviction."""
    import numpy as np
    if avail not in ("SAME", "NEXT") or evict not in ("DROP", "REHOME"):
        raise ValueError(f"bad engine mode {avail} / {evict}")
    slots = sorted(held_of)
    ev = collections.defaultdict(lambda: ([], [], []))           # t -> (becomes held, wakes, parks)
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
        k[(L, t)] += 1  # push
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

    def evictions(becomes, t):                                  # 1. start of the minute (§3.2a)
        for L in sorted(becomes):
            for key, w in sorted(guests[L].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                unplace(L, key, w)
                loc[key] = None
                st["evicted"] += 1
                if evict == "REHOME":
                    if bounce is not None and nre.get(key, 0) >= bounce:
                        x["nb1_dropped"] += 1
                        x["dropped_at_evict"] += 1
                        x["bounce_hist"][nre.get(key, 0)] += 1
                        continue
                    L2 = lender_for(L, w, t, {L, key[0]})            # §3.2b: same node, not L, not the owner, most free
                    if L2 is not None:
                        place(L2, key, w, t)                          # §3.2c: one copy touching L2 at t
                        st["rehomed"] += 1
                        nre[key] = nre.get(key, 0) + 1
                        x["rehome_items"].append((L, t, w, L2))
                        if avail == "NEXT":
                            inflight.add(key)                         # §3.2e: servable from t + 1
                        if charge:
                            c_day[node_of[L]][t // day] += w          # §3.2g: the in-flight minute, charged to L's node
                    else:
                        x["dropped_at_evict"] += 1
                        x["bounce_hist"][nre.get(key, 0)] += 1
                else:
                    x["dropped_at_evict"] += 1
                    x["bounce_hist"][nre.get(key, 0)] += 1
            if guests[L]:
                raise RuntimeError("I-NB-EVICT: guests left on a held lender")

    def wakes_(wakes, t):                                       # 2. pull, then leave the cache
        for s in sorted(wakes):
            for u, w in units_of[s].items():
                key = (s, u)
                L = loc.get(key)
                if L is not None and key in inflight:           # §3.2e: in flight at t -> host, the copy on L2 released
                    rec[(s, t, u)] = None
                    unplace(L, key, w)
                    x["inflight_fallback"] += 1
                    x["bounce_hist"][nre.get(key, 0)] += 1
                    loc[key] = None
                    continue
                rec[(s, t, u)] = L
                if L is not None:
                    k[(L, t)] += 1  # pull
                    copies.append((L, t, w))
                    unplace(L, key, w)
                    x["bounce_hist"][nre.get(key, 0)] += 1
                loc[key] = None

    def parks_(parks, t):                                       # 3. largest engine first
        for s in sorted(parks):
            for u, w in sorted(units_of[s].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                nre[(s, u)] = 0
                L = lender_for(s, w, t, {s})
                if L is None:
                    loc[(s, u)] = None
                    st["park_uncached"] += 1
                    continue
                place(L, (s, u), w, t)

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
    return dict(rec=rec, k=k, copies=copies, stats=st, x=x)


def same_core(a, b):
    """the four fields nb_simulate returns, compared exactly (I-NEST / G2)."""
    return a["rec"] == b["rec"] and dict(a["k"]) == dict(b["k"]) and a["copies"] == b["copies"] and a["stats"] == b["stats"]


def drains(x, k, delta):
    """§3.2h: per (L, t) serial drain of re-homed guests at each receiver's contended rate -> {(L, t): s}."""
    out = collections.defaultdict(float)
    for L, t, w, L2 in x["rehome_items"]:
        out[(L, t)] += w * GIB / (NB.R_NB * delta / k[(L2, t)])
    return out


def lodo_hbm(x, bh, hold_day=None, n_fleet=N_FLEET):
    """§3.4 per replicate (one left-out day): fleet tax, worst-node tax, fleet share, GPU-equivalents, hold-home yardstick."""
    nd, day = x["ndays"], x["day"]
    out = []
    for d_out in range(nd):
        days = [d for d in range(nd) if d != d_out]
        mins = day * len(days)
        g_n = {n: sum(x["g_day"][n][d] + (x["c_day"][n][d] if x["charge"] else 0.0) for d in days) for n in x["g_day"]}
        l_n = {n: sum(x["l_day"][n][d] for d in days) for n in x["l_day"]}
        G, Lm = sum(g_n.values()), sum(l_n.values())
        node_tax = {n: g_n[n] / (l_n[n] * bh) for n in g_n if l_n[n] > 0}
        out.append(dict(fleet_tax=None if Lm == 0 else G / (Lm * bh), worst_node_tax=max(node_tax.values()) if node_tax else 0.0,
                        worst_node=max(node_tax, key=node_tax.get) if node_tax else None, fleet_share=G / (n_fleet * mins * bh),
                        gpu_eq=G / (mins * bh), hold_eq=None if hold_day is None else sum(hold_day[d] for d in days) / mins,
                        guest_gib_min=G, lender_gpu_min=Lm))
    return out


def guard_ok(reps):
    """§5: both ceilings on every replicate; = passes."""
    return bool(reps) and all(r["fleet_tax"] is not None and r["fleet_tax"] <= G_FLEET and r["worst_node_tax"] <= G_NODE for r in reps)


# ------------------------------------------------------------------------------------------------ node maps (ORDER / SHUF + MAP-i)
def tail_slots(slots):
    return [i for i, sl in enumerate(slots) if sl["kind"] == "tail"]


def map_draw(slots, i):
    """MAP-i (prereg §6): anchors as ORDER; tail slots to nodes 4.., 8 per node, ordered by sha256('2936|i|slot')."""
    base = NB.node_map(slots, "ORDER")                          # refuses a cell it cannot map; anchors' nodes
    tail = tail_slots(slots)
    first = min(base[j] for j in tail)
    out = {j: base[j] for j in base if j not in tail}
    for r_, j in enumerate(sorted(tail, key=lambda j: hashlib.sha256(f"{MAP_SEED}|{i}|{j}".encode()).hexdigest())):
        out[j] = first + r_ // NB.GPN
    return out


def node_map2(slots, kind, maps=None):
    if kind in ("ORDER", "SHUF"):
        return NB.node_map(slots, kind)
    m = re.fullmatch(r"MAP-(\d+)", kind)
    if not m:
        raise ValueError(f"unknown map {kind}")
    return map_draw(slots, int(m.group(1)))


def maps_manifest(tail_by_seed, slots0):
    """the MAP-0..19 manifest + its checks (written only when ok)."""
    o, s = NB.node_map(slots0, "ORDER"), NB.node_map(slots0, "SHUF")
    tail = tail_slots(slots0)
    draws, checks = {}, {}
    for i in range(K_MAPS):
        m = map_draw(slots0, i)
        draws[str(i)] = {str(j): m[j] for j in tail}
    cnt_ok = all(sorted(collections.Counter(d.values()).values()) == [NB.GPN] * (len(tail) // NB.GPN) for d in draws.values())
    checks["n_tail"] = len(tail)
    checks["eight_per_node"] = cnt_ok
    checks["same_tail_slots_every_seed"] = all(v == tail for v in tail_by_seed.values())
    tl = [tuple(sorted(d.items())) for d in draws.values()]
    checks["draws_distinct"] = len(set(tl)) == len(tl)
    checks["no_draw_equals_order_or_shuf"] = all(d != {str(j): o[j] for j in tail} and d != {str(j): s[j] for j in tail}
                                                 for d in draws.values())
    ok = len(tail) == 80 and all(v is True for k_, v in checks.items() if k_ != "n_tail")
    return dict(seed=MAP_SEED, k=K_MAPS, rule="sha256(f'2936|{i}|{slot_index}') order, 8 per node from the first tail node",
                tail_slots=tail, seeds=sorted(tail_by_seed), draws=draws, checks=checks, prereg=PREREG, ok=ok)


def load_maps(path, sha):
    if not path or not sha:
        raise ValueError("a MAP-i setting needs --maps and --maps-sha")
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise ValueError(f"I-MAP: manifest sha256 {got[:12]}... != --maps-sha {sha[:12]}...")
    m = json.loads(raw)
    if m.get("ok") is not True:
        raise ValueError("I-MAP: the manifest's ok is not true")
    return m


# ------------------------------------------------------------------------------------------------ the hook (extends)
class RHook(NB.Hook):
    def __init__(self):
        super().__init__()
        self.maps = None

    def reset(self, cfg, bh, host, maps=None):
        self.__init__()
        self.cfg, self.bh, self.host, self.maps = dict(cfg), bh, host, maps

    def prepare(self, pol, slots, bs, khs, checks, pol_cache):
        """ prepare (pre-fill pol_cache with run_variant's own rule), then the engine + I-NEST on the real schedules."""
        if self.cfg["mode"] != "NB" or pol == "STATIC":
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
        nm = node_map2(slots, self.cfg["map"], self.maps)
        tail = tail_slots(slots)
        held_of = {i: pol_cache[(pol, i, False)][0] for i in tail}
        wake_of = {i: pol_cache[(pol, i, False)][1] for i in tail}
        units_of = {i: dict(slots[i]["units"]) for i in tail}
        node_of = {i: nm[i] for i in tail}
        n_min = len(next(iter(held_of.values())))
        cap = self.cfg["cap"] * self.bh
        c = self.cfg
        st = rh_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min, c["avail"], c["bounce"], c["charge"])
        nest_cfg = c["avail"] == "SAME" and c["bounce"] is None and not c["charge"]
        mine = st if nest_cfg else rh_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min)
        ref = NB.nb_simulate(held_of, wake_of, units_of, node_of, cap, c["evict"], n_min)
        if not same_core(mine, ref):
            raise RuntimeError(f"I-NEST: the engine != 's nb_simulate ({pol}, {c['evict']})")
        x = st["x"]
        if sum(sum(v) for v in x["g_day"].values()) - st["stats"]["guest_gib_min"] > 1e-9 * max(1.0, st["stats"]["guest_gib_min"]) or \
                st["stats"]["guest_gib_min"] - sum(sum(v) for v in x["g_day"].values()) > 1e-9 * max(1.0, st["stats"]["guest_gib_min"]):
            raise RuntimeError("I-HBM-SUM: per-day guest GiB-min != the whole-week total")
        brk = [i for i in tail if any(w > BREAK_W for w in units_of[i].values())]
        hold_day = [0] * x["ndays"]
        for i in brk:
            pk = ~held_of[i].astype(bool)
            for d in range(x["ndays"]):
                hold_day[d] += int(pk[d * x["day"]:(d + 1) * x["day"]].sum())
        st.update(tag=tag, pol=pol, ctx=sum(1 for y in self.states if y["pol"] == pol), pol_cache=pol_cache, units_of=units_of,
                  node_map=nm, nested=True, breakers=brk, hold_day=hold_day)
        self.states.append(st)
        self.cur[pol] = st

    def report(self):
        out = super().report()                                  # keys, unchanged
        for o, st in zip(out, self.states):
            x = st["x"]
            dr = drains(x, st["k"], self.cfg["delta"])
            reps = lodo_hbm(x, self.bh, st["hold_day"])
            nd, day = x["ndays"], x["day"]
            per_day = [collections.Counter() for _ in range(nd)]
            wmax = max(w for us in st["units_of"].values() for w in us.values())
            for (s, t, u), L in st["rec"].items():
                w = st["units_of"][s][u]
                dd = per_day[t // day]
                dd["wakes"] += 1
                dd["host"] += L is None
                if w > BREAK_W:
                    dd["break_wakes"] += 1
                    dd["break_host"] += L is None
                if abs(w - wmax) < 1e-9:
                    dd["wmax_host"] += L is None
            lodo_w = []
            for d_out in range(nd):
                agg = collections.Counter()
                for d in range(nd):
                    if d != d_out:
                        agg.update(per_day[d])
                lodo_w.append(dict(agg))
            o["x2936"] = dict(
                replicates=reps,                                # unrounded: the guard reads these (§5, = passes)
                guard_ok=guard_ok(reps), max_fleet_tax=round(max(r["fleet_tax"] or 0.0 for r in reps), 6),
                max_worst_node_tax=round(max(r["worst_node_tax"] for r in reps), 6),
                drain_max=round(max(dr.values()), 4) if dr else 0.0, drains_over_pstar=sum(1 for v in dr.values() if v > P_STAR),
                n_drains=len(dr), inflight_fallback=x["inflight_fallback"], dropped_at_evict=x["dropped_at_evict"],
                nb1_dropped=x["nb1_dropped"], bounce_hist={str(a): b for a, b in sorted(x["bounce_hist"].items())},
                breakers=st["breakers"], wakes_lodo=lodo_w, cap_ok=x["cap_ok"], nested=st.get("nested", False))
        return out


HOOK2 = RHook()


def install_rh():
    """ forks, re-pointed at the hook (pitfall 1)."""
    M = NB._m2897()
    V6 = M._v6()
    NB.install()
    NB.HOOK = HOOK2
    V6.V5._NB_PAGE = HOOK2
    V6._NB_PAGE = HOOK2
    if not (NB.HOOK is HOOK2 and V6.V5._NB_PAGE is HOOK2 and V6._NB_PAGE is HOOK2 and V6.V5.run_variant.__name__ == "rv_wrap"):
        raise RuntimeError("install_rh: the hook is not wired on every site")
    return V6


# ------------------------------------------------------------------------------------------------ run
def run_rh(setting, cell, seeds, tag, root, cache, out, maps_file=None, maps_sha=None):
    M = NB._m2897()
    cfg = setting_cfg(setting)
    maps = load_maps(maps_file, maps_sha) if cfg.get("map", "").startswith("MAP-") else None
    if cfg["mode"] != "OFF" and cell != "D1":
        raise ValueError("the neighbour tier is defined for D1 only (prereg §2)")
    M.configure_v6(cfg["host"])
    V6 = install_rh()
    R = V6.V5.install()
    wd, row, rec, snap, r = V6.static_row_over(cell, seeds, tag, root, cache, None)
    row["cell"] = cell
    row["epar"] = None
    host = M.SETTINGS[cfg["host"]]
    HOOK2.reset(cfg, wd.bh, host, maps)
    row["s2936"] = dict(setting=setting, cfg=cfg, prereg=PREREG, parent_prereg=NB.PREREG, h2d_tail=V6.H2D_TAIL, p_grid=list(V6.P_GRID),
                        r_nb=NB.R_NB, bh=wd.bh, world_h2d=float(wd.tph["h2d"]), seeds=list(seeds))
    if snap is not None:
        if abs(wd.tph["h2d"] - M.WORLD_H2D) > 1e-9:
            raise RuntimeError(f"I-WORLD: the static world's tail h2d is {wd.tph['h2d']}, not {M.WORLD_H2D}")
        row["v6"] = V6.overlay6(wd, row, rec, snap, r, seeds, R)
        rep = HOOK2.report()
        row["s2936"]["nb"] = rep
        if cfg["mode"] == "NB":
            if not rep:
                raise RuntimeError("I-NB-HOOK: the neighbour tier built no state")
            if not all(x["hook_sites_agree"] for x in rep):
                raise RuntimeError("I-NB-HOOK: build_windows and run_variant asked for different wakes")
            if not all(x["x2936"]["nested"] and x["x2936"]["cap_ok"] for x in rep):
                raise RuntimeError("I-NEST / I-CAP: a state was not nested or went over CAP")
            row["s2936"]["node_map"] = {str(k_): v for k_, v in sorted(HOOK2.states[0]["node_map"].items())}
            if maps is not None:
                i = cfg["map"].split("-")[1]
                want = maps["draws"][i]
                got = {str(j): HOOK2.states[0]["node_map"][j] for j in tail_slots_from(HOOK2.states[0])}
                if got != want:
                    raise RuntimeError(f"I-MAP: seed-0 node map != manifest draw {i}")
                row["s2936"]["maps_sha"] = maps_sha
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


def tail_slots_from(st):
    return sorted(st["units_of"])


def freeze_maps(out, root, cache):
    M = NB._m2897()
    V6 = M.configure_v6("H12")
    V3 = V6.V3
    V6.V5.install()
    wd, row, rec, snap, r = V3.static_row("D1", (0, 1, 2), "freeze2936", root, cache)
    tail_by_seed, slots0 = {}, None
    for si in range(3):
        slots = V3.build_slots(wd, rec["fi"][si])
        tail_by_seed[si] = tail_slots(slots)
        if si == 0:
            slots0 = slots
    man = maps_manifest(tail_by_seed, slots0)
    if not man["ok"]:
        print(json.dumps(man["checks"]), file=sys.stderr)
        return 1
    raw = (json.dumps(man, sort_keys=True) + "\n").encode()
    with open(out, "wb") as fh:
        fh.write(raw)
    print(json.dumps(dict(out=out, sha256=hashlib.sha256(raw).hexdigest(), checks=man["checks"])))
    return 0


# ------------------------------------------------------------------------------------------------ checks over rows
def _one(f):
    with open(f) as fh:
        x = [json.loads(l) for l in fh if l.strip()]
    if len(x) != 1:
        raise ValueError(f"{f}: {len(x)} rows, need 1")
    return x[0]


def ifix_row(new_file, ref_file):
    """I-FIX-RH / -NBP / -OFF (prereg §8): every field but timing and provenance, plus every nb key and the node map."""
    M = NB._m2897()
    n, o = _one(new_file), _one(ref_file)
    a = {k_: v for k_, v in M.strip(n).items() if k_ not in ("s2897", "s2933", "s2936")}
    b = {k_: v for k_, v in M.strip(o).items() if k_ not in ("s2897", "s2933", "s2936")}
    top = sorted(k_ for k_ in set(a) | set(b) if k_ != "v6" and a.get(k_) != b.get(k_))
    vd, vparts = [], []
    if "v6" in a or "v6" in b:
        nv, ov = a.get("v6") or {}, b.get("v6") or {}
        vd = sorted(v for v in set(nv.get("variants", {})) | set(ov.get("variants", {}))
                    if nv.get("variants", {}).get(v) != ov.get("variants", {}).get(v))
        vparts = sorted(p for p in set(nv) | set(ov) if p != "variants" and nv.get(p) != ov.get(p))
    nb_new, nb_ref = (n.get("s2936") or {}).get("nb") or [], (o.get("s2933") or {}).get("nb") or []
    nbd = []
    if len(nb_new) != len(nb_ref):
        nbd.append(f"nb entries {len(nb_new)} != {len(nb_ref)}")
    for i, (x, y) in enumerate(zip(nb_new, nb_ref)):
        for k_ in y:
            if x.get(k_) != y[k_]:
                nbd.append(f"nb[{i}].{k_}")
    nm = (n.get("s2936") or {}).get("node_map") != (o.get("s2933") or {}).get("node_map")
    ok = not top and not vd and not vparts and not nbd and not nm and len((b.get("v6") or {}).get("variants", {})) > 0
    return dict(new=new_file, ref=ref_file, static_diffs=top[:10], variant_diffs=vd[:10], v6_part_diffs=vparts[:10], nb_diffs=nbd[:10],
                node_map_diff=nm, variants_compared=len((b.get("v6") or {}).get("variants", {})), nb_compared=len(nb_ref), pass_=ok)


# ------------------------------------------------------------------------------------------------ summary + DOOR-RH (prereg §5 / §6)
def x_of(pm, elig):
    """X = max over ELIGIBLE deciding arms of P_MEET6 (None lowest); GUARD-BOUND iff an ineligible arm is above X."""
    vals = [pm[a] for a in DEC_ARMS if elig[a] and pm[a] is not None]
    X = max(vals) if vals else None
    gb = any((not elig[a]) and pm[a] is not None and (X is None or pm[a] > X) for a in DEC_ARMS)
    return X, gb


def _lo(a, b):
    if a is None or b is None:
        return None
    return min(a, b)


def attributable(v, draws):
    """§6: P_MEET6 strictly above the placebo max (None lowest)."""
    if v is None:
        return False
    vals = [d for d in draws if d is not None]
    return not vals or v > max(vals)


def attributable_bool(v, draws):
    return bool(v) and not any(draws)


def _rows(d, what):
    """(tag, cell) -> row, REFUSING a duplicate key (never last-wins)."""
    out = {}
    for r in NB._m2897().load_rows(d):
        k_ = (r["tag"], r["cell"])
        if k_ in out:
            raise ValueError(f"duplicate {what} row key {k_}")
        out[k_] = r
    return out


def setting_summary(d, st):
    M = NB._m2897()
    rows = _rows(d, st)
    M.configure_v6("H12")
    s = M.summarize_v6(d, "H12")
    out = {}
    for tag in sorted({k_[0] for k_ in rows}):
        r0 = rows.get((tag, "D1"))
        if r0 is None:
            continue
        ck = f"{tag}|D1"
        if f"{ck}|S2|{DEC_ARMS[0]}" not in s["readings"]:
            continue
        nb = (r0.get("s2936") or {}).get("nb") or []
        pols = collections.defaultdict(list)
        for e in nb:
            pols[e["pol"]].append(e)
        elig, guard = {}, {}
        for a in DEC_ARMS:
            es = pols.get(POL_OF[a], [])
            reps = [rr for e in es for rr in e["x2936"]["replicates"]]
            ok = len(reps) == 21 and guard_ok(reps)
            elig[a] = ok
            guard[a] = dict(n=len(reps), ok=ok, max_fleet_tax=max((rr["fleet_tax"] or 0.0) for rr in reps) if reps else None,
                            max_worst_node_tax=max(rr["worst_node_tax"] for rr in reps) if reps else None,
                            gpu_eq=[min(rr["gpu_eq"] for rr in reps), max(rr["gpu_eq"] for rr in reps)] if reps else None,
                            fleet_share=[min(rr["fleet_share"] for rr in reps), max(rr["fleet_share"] for rr in reps)] if reps else None,
                            hold_eq=[min(rr["hold_eq"] for rr in reps), max(rr["hold_eq"] for rr in reps)] if reps else None,
                            drain_max=max((e["x2936"]["drain_max"] for e in es), default=None),
                            drains_over_pstar=sum(e["x2936"]["drains_over_pstar"] for e in es),
                            inflight_fallback=[e["x2936"]["inflight_fallback"] for e in es],
                            evicted=[e["stats"]["evicted"] for e in es], rehomed=[e["stats"]["rehomed"] for e in es],
                            dropped_at_evict=[e["x2936"]["dropped_at_evict"] for e in es],
                            bounce_hist=[e["x2936"]["bounce_hist"] for e in es], k_max=[e["k_max"] for e in es],
                            wakes=[e["wakes"] for e in es], hbm_tax_week=[e["hbm_tax"] for e in es],
                            lender_batch_s_lost=[e["lender_batch_s_lost"] for e in es],
                            break_host_lodo_max=max((w_.get("break_host", 0) for e in es for w_ in e["x2936"]["wakes_lodo"]), default=None),
                            wmax_host_lodo_max=max((w_.get("wmax_host", 0) for e in es for w_ in e["x2936"]["wakes_lodo"]), default=None))
        pm = {a: s["readings"][f"{ck}|S2|{a}"]["p_meet6"] for a in DEC_ARMS}
        X, gb = x_of(pm, elig)
        rd = {a: s["readings"][f"{ck}|S2|{a}"] for a in DEC_ARMS}
        out[tag] = dict(p_meet6=pm, eligible=elig, X=X, guard_bound=gb, guard=guard,
                        meet_0_842={a: rd[a]["meet"]["0.842"] for a in DEC_ARMS}, meet_0_79={a: rd[a]["meet"]["0.79"] for a in DEC_ARMS},
                        first_fail={a: rd[a].get("first_fail") for a in DEC_ARMS},
                        mww_0={a: [min(rd[a]["mww"]["0"]), max(rd[a]["mww"]["0"])] for a in DEC_ARMS},
                        mww_0_842={a: [min(rd[a]["mww"]["0.842"]), max(rd[a]["mww"]["0.842"])] for a in DEC_ARMS},
                        idle={a: rd[a]["idle"] for a in DEC_ARMS},
                        add_s2={a2: s["readings"][f"{ck}|S2|{a2}"]["p_meet6"] for a2 in ("KA-1|ADD", "KA-5|ADD")},
                        s10={a: s["readings"][f"{ck}|S10|{a}"]["p_meet6"] for a in DEC_ARMS})
    return out, dict(replay=s["parent"].get("replay", {}).get("D1"), seedshift=s["parent"].get("seedshift", {}).get("D1"))


def door_rh(x_order, x_shuf):
    order = ["CLOSED", "OPEN-ONLY-BELOW-0.5", "OPEN-WITH-p<=0.5", "OPEN-AT-STOCK-p"]
    ca, cb = NB.door_class(x_order), NB.door_class(x_shuf)
    return dict(door=min((ca, cb), key=order.index), order=ca, shuf=cb, map_sensitive=ca != cb, y=_lo(x_order, x_shuf))


def summarize_rh(out_dir):
    res = dict(settings={}, replay={}, seedshift={}, skipped=[])
    for st in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, st)
        if not os.path.isdir(d):
            continue
        try:
            cfg = setting_cfg(st)
        except ValueError:
            res["skipped"].append(st)
            continue
        if cfg["mode"] != "NB" or not any(f.endswith(".jsonl") for f in os.listdir(d)):
            res["skipped"].append(st)
            continue
        res["settings"][st], par = setting_summary(d, st)
        res["replay"][st], res["seedshift"][st] = par["replay"], par["seedshift"]
    S = res["settings"]

    def X(st, tag="main"):
        return S[st][tag]["X"] if st in S and tag in S[st] else None
    if "RH-Q" in S and "RH-Q-SHUF" in S:
        dq = door_rh(X("RH-Q"), X("RH-Q-SHUF"))
        flags = dict(map_sensitive=dq["map_sensitive"],
                     guard_bound=S["RH-Q"]["main"]["guard_bound"] or S["RH-Q-SHUF"]["main"]["guard_bound"],
                     drain_exposed=any((S[x]["main"]["guard"][a]["drain_max"] or 0.0) > P_STAR for x in ("RH-Q", "RH-Q-SHUF")
                                       for a in DEC_ARMS))
        if "RH-H" in S and "RH-H-SHUF" in S:
            dh = door_rh(X("RH-H"), X("RH-H-SHUF"))
            res["door_half"] = dh
            flags["rate_sensitive"] = dh["door"] != dq["door"]
        ss = res["seedshift"].get("RH-Q")                  # QA r1: v6 keeps seed-shift in its parent section, not as a cell
        if ss is not None:
            flags["seed_shift"] = bool(ss.get("UNATTRIBUTABLE"))
            res["seedshift_rhq"] = ss
        plc = sorted(st for st in S if st.startswith("PLC-DROP-"))
        if plc:
            res["placebo"] = placebo_read(S, plc, dq)
            flags["placebo_attributable_any"] = res["placebo"]["any_attributable"]
        rob = sorted(st for st in S if st.startswith("PLC-RH-"))
        if rob:
            thr = {"OPEN-AT-STOCK-p": 0.842, "OPEN-WITH-p<=0.5": 0.5}.get(dq["door"])
            xs = {st: X(st) for st in rob}
            res["robustness"] = dict(n=len(rob), X=xs, thr=thr, n_meet=None if thr is None else
                                     sum(1 for v in xs.values() if v is not None and v >= thr - 1e-12),
                                     classes=dict(collections.Counter(NB.door_class(v) for v in xs.values())))
            flags["map_fragile"] = thr is not None and res["robustness"]["n_meet"] < K_MAPS
        res["door_rh"] = dict(dq, flags=flags)
    if "DROP-Q" in S and "DROP-Q-SHUF" in S:
        res["door_drop_q"] = door_rh(X("DROP-Q"), X("DROP-Q-SHUF"))
    return res


def placebo_read(S, plc, dq):
    """§6 MAP family: spread of PLC-DROP per axis; RH-Q / RH-Q-SHUF attribution; the placebo-aware door."""
    feas = [st for st in plc if "main" in S[st]]
    spread, att, aware = {}, {}, {}
    for a in DEC_ARMS:
        pm = [S[st]["main"]["p_meet6"][a] for st in feas]
        m8 = [S[st]["main"]["meet_0_842"][a] for st in feas]
        m79 = [S[st]["main"]["meet_0_79"][a] for st in feas]
        spread[a] = dict(p_meet6=pm, p_meet6_max=max((v for v in pm if v is not None), default=None), n_none=sum(v is None for v in pm),
                         meet_0_842_true=sum(bool(v) for v in m8), meet_0_79_true=sum(bool(v) for v in m79))
        for arm_st, ctrl in (("RH-Q", "DROP-Q"), ("RH-Q-SHUF", "DROP-Q-SHUF")):
            r = S[arm_st]["main"]
            att[f"{arm_st}|{a}"] = dict(p_meet6=attributable(r["p_meet6"][a], pm), meet_0_842=attributable_bool(r["meet_0_842"][a], m8),
                                        meet_0_79=attributable_bool(r["meet_0_79"][a], m79))
    for arm_st, ctrl in (("RH-Q", "DROP-Q"), ("RH-Q-SHUF", "DROP-Q-SHUF")):
        r = S[arm_st]["main"]
        c = S.get(ctrl, {}).get("main")
        pm = {a: (r["p_meet6"][a] if att[f"{arm_st}|{a}"]["p_meet6"] or c is None else c["p_meet6"][a]) for a in DEC_ARMS}
        aware[arm_st] = x_of(pm, r["eligible"])[0]
    da = door_rh(aware.get("RH-Q"), aware.get("RH-Q-SHUF"))
    return dict(feasible=len(feas), n=len(plc), void=len(feas) < 10, spread=spread, attributable=att,
                any_attributable=any(v["p_meet6"] or v["meet_0_842"] or v["meet_0_79"] for v in att.values()),
                aware_X=aware, door_aware=da)


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    import numpy as np, tempfile
    M = NB._m2897()
    V6 = install_rh()
    V3 = V6.V3
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))
    r4 = lambda v: round(v, 4)  # noqa: E731
    pr = NB.page_rate
    R = NB.R_NB
    # G1
    chk("G1 25.22 d.25 k1-4", [r4(pr(25.22, R * .25 / k_)) for k_ in (1, 2, 3, 4)], [0.4212, 0.8224, 1.2235, 1.6247])
    chk("G1 15.8 d.25 k1-3", [r4(pr(15.8, R * .25 / k_)) for k_ in (1, 2, 3)], [0.2713, 0.5227, 0.7740])
    chk("G1 d.5 k1 / host / min rule k6", [r4(pr(25.22, R * .5)), r4(pr(25.22, H_FB)),
                                          r4(NB.nb_page(dict(rec={(0, 0, "u"): 1}, k=collections.Counter({(1, 0): 6})), 0, 0, "u", 25.22,
                                                        H_FB, 0.25))], [0.2206, 2.2039, 2.2039])

    def toy(held, units, nodes, cap, evict, eng="rh", **kw):
        H = {s: np.array(h, bool) for s, h in held.items()}
        W = {}
        for s, h in H.items():
            prev = np.concatenate([[True], h[:-1]])
            W[s] = h & ~prev
        n = len(next(iter(H.values())))
        if eng == "nb":
            return NB.nb_simulate(H, W, units, nodes, cap, evict, n)
        return rh_simulate(H, W, units, nodes, cap, evict, n, **kw)
    # G2 nesting: 300 random toys, REHOME and DROP, + GN2 / GN10 toys
    rng = np.random.default_rng(2936)
    nest_ok, n_cmp = True, 0
    for _ in range(300):
        ns = int(rng.integers(3, 10))
        nmin = int(rng.integers(30, 121))
        held, units, nodes = {}, {}, {}
        for s in range(ns):
            h = np.zeros(nmin, bool)
            cur = bool(rng.random() < 0.5)
            for t in range(nmin):
                if rng.random() < 0.15:
                    cur = not cur
                h[t] = cur
            held[s] = h.tolist()
            units[s] = {f"u{j}": float(rng.uniform(0.5, 30.0)) for j in range(int(rng.integers(0, 4)))}
            nodes[s] = int(rng.integers(0, 3))
        cap = float(rng.uniform(10, 40))
        for ev_ in ("REHOME", "DROP"):
            a = toy(held, units, nodes, cap, ev_, "nb")
            b = toy(held, units, nodes, cap, ev_)
            nest_ok = nest_ok and same_core(a, b)
            n_cmp += 1
    g2 = dict(held={0: [1, 0, 0, 0, 1, 1], 1: [0] * 6, 2: [1, 1, 0, 0, 0, 1]}, units={0: {"u0": 25.0}, 1: {}, 2: {"u2": 5.0}},
              nodes={0: 0, 1: 0, 2: 0})
    for ev_ in ("DROP", "REHOME"):
        nest_ok = nest_ok and same_core(toy(g2["held"], g2["units"], g2["nodes"], 30.0, ev_, "nb"),
                                        toy(g2["held"], g2["units"], g2["nodes"], 30.0, ev_))
    chk("G2 nesting (300 random toys x REHOME/DROP + GN2/GN10)", (nest_ok, n_cmp), (True, 600))
    # G3 in-flight
    g3 = dict(held={0: [1, 0, 0, 1, 1, 1], 1: [0] * 6, 2: [1, 1, 0, 1, 1, 1]}, units={0: {"u0": 4.0}, 1: {}, 2: {"u2": 5.0}},
              nodes={0: 0, 1: 0, 2: 0})
    q = toy(g3["held"], g3["units"], g3["nodes"], 30.0, "REHOME", avail="NEXT", charge=True)
    chk("G3 RH-Q: u2 in flight -> host, u0 at k 2", [q["rec"][(2, 3, "u2")], r4(NB.nb_page(q, 2, 3, "u2", 5.0, H_FB, 0.25)),
                                                    q["k"][(1, 3)], r4(NB.nb_page(q, 0, 3, "u0", 4.0, H_FB, 0.25))], [None, 0.4530, 2, 0.1473])
    c = toy(g3["held"], g3["units"], g3["nodes"], 30.0, "REHOME")
    chk("G3 RH-2933: u2 pulled at k 3", [r4(NB.nb_page(c, 2, 3, "u2", 5.0, H_FB, 0.5)), r4(NB.nb_page(c, 0, 3, "u0", 4.0, H_FB, 0.5))],
        [0.1393, 0.1154])
    # G4 next-minute availability
    q4 = toy({0: [1, 0, 0, 1, 1, 1], 1: [0] * 6, 2: [1, 1, 0, 0, 1, 1]}, g3["units"], g3["nodes"], 30.0, "REHOME", avail="NEXT", charge=True)
    chk("G4 RH-Q: u2 from S1 at t4, k 1", [q4["rec"][(2, 4, "u2")], r4(NB.nb_page(q4, 2, 4, "u2", 5.0, H_FB, 0.25))], [1, 0.0995])
    # G5 bounce: unlimited vs NB1; owner never receives its own guest
    g5 = dict(held={0: [0, 0, 1, 1, 1, 1], 1: [0, 0, 0, 1, 1, 1], 2: [0] * 6, 3: [1, 0, 0, 0, 0, 1]},
              units={0: {}, 1: {}, 2: {}, 3: {"g": 10.0}}, nodes={0: 0, 1: 0, 2: 0, 3: 0})
    un = toy(g5["held"], g5["units"], g5["nodes"], 30.0, "REHOME", avail="NEXT", charge=True)
    b1 = toy(g5["held"], g5["units"], g5["nodes"], 30.0, "REHOME", avail="NEXT", bounce=1, charge=True)
    chk("G5 unlimited: 2 re-homes, nb at t5", [un["stats"]["rehomed"], un["rec"][(3, 5, "g")], r4(NB.nb_page(un, 3, 5, "g", 10.0, H_FB, .25))],
        [2, 2, 0.1791])
    chk("G5 NB1: 1 re-home, dropped -> host", [b1["stats"]["rehomed"], b1["x"]["nb1_dropped"], b1["rec"][(3, 5, "g")],
                                              r4(NB.nb_page(b1, 3, 5, "g", 10.0, H_FB, .25))], [1, 1, None, 0.8859])
    o5 = toy({0: [1, 0, 0, 0, 1], 1: [0, 0, 1, 1, 1], 2: [0] * 5}, {0: {"u": 8.0}, 1: {}, 2: {}}, {0: 0, 1: 0, 2: 0}, 30.0, "REHOME",
             avail="NEXT", charge=True)
    chk("G5b owner never receives its own guest", o5["rec"][(0, 4, "u")], 2)
    # G6 HBM accounting: per-day sums, the in-flight charge, LODO excludes the left-out day
    h6 = toy({0: [1, 0, 0, 0, 1, 1], 1: [0] * 6}, {0: {"u": 6.0}, 1: {}}, {0: 0, 1: 0}, 30.0, "DROP", day=3)
    chk("G6 per-day = total (18 GiB-min: 12 + 6)", [h6["x"]["g_day"][0], h6["stats"]["guest_gib_min"]], [[12.0, 6.0], 18.0])
    chk("G6 lender GPU-min per day", h6["x"]["l_day"][0], [5, 4])
    h6b = toy({0: [1, 0, 0, 1, 1, 1], 1: [0] * 6}, {0: {"u": 6.0}, 1: {}}, {0: 0, 1: 0}, 30.0, "DROP", day=3)
    reps = lodo_hbm(h6b["x"], 10.0, [0, 0], n_fleet=2)
    chk("G6 LODO: day-0-only guests read tax 0 at d 0", [reps[0]["fleet_tax"], r4(reps[1]["fleet_tax"])], [0.0, r4(12.0 / (5 * 10.0))])
    q6 = toy(g3["held"], g3["units"], g3["nodes"], 30.0, "REHOME", avail="NEXT", charge=True, day=3)
    chk("G6 in-flight charge: +w on the source's node", q6["x"]["c_day"][0], [0.0, 5.0])
    # G7 drain
    chk("G7 drain 5.0 GiB at k 2, d.25", {f"{a}|{b}": r4(v) for (a, b), v in drains(q["x"], q["k"], 0.25).items()}, {"0|3": 0.1591})
    # G8 guard
    mk = lambda f, n_: [dict(fleet_tax=f, worst_node_tax=n_)] * 21  # noqa: E731
    chk("G8 guard <= ceilings", [guard_ok(mk(0.20, 0.25)), guard_ok(mk(0.2000001, 0.1)), guard_ok(mk(0.1, 0.2500001)), guard_ok([])],
        [True, False, False, False])
    # G9 duplicates refused
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "a.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(tag="main", cell="D1")) + "\n" + json.dumps(dict(tag="main", cell="D1")) + "\n")
        try:
            _rows(td, "rh")
            chk("G9 duplicates refused", "accepted", "refused")
        except ValueError:
            chk("G9 duplicates refused", "refused", "refused")
    # G10 door with eligibility
    chk("G10 eligibility + GUARD-BOUND", [x_of({"KA-1|QINH": 0.5, "KA-5|QINH": 1.0}, {"KA-1|QINH": True, "KA-5|QINH": False}),
                                         x_of({"KA-1|QINH": None, "KA-5|QINH": None}, {"KA-1|QINH": True, "KA-5|QINH": True}),
                                         door_rh(None, 0.842)["door"], door_rh(0.842, 1.0)["door"], door_rh(0.5, 0.842)["map_sensitive"]],
        [(0.5, True), (None, False), "CLOSED", "OPEN-AT-STOCK-p", True])
    # G11 placebo read
    chk("G11 attribution", [attributable(0.75, [0.5, None, 0.25]), attributable(0.5, [0.5, 0.25]), attributable(None, [None]),
                            attributable(0.25, [None, None]), attributable_bool(True, [False] * 3), attributable_bool(True, [False, True])],
        [True, False, False, True, True, False])
    # G12 maps
    sl = [dict(kind="anchor", g=16)] * 2 + [dict(kind="tail", g=1)] * 80
    man = maps_manifest({0: tail_slots(sl), 1: tail_slots(sl), 2: tail_slots(sl)}, sl)
    nodes12 = sorted({v for d in man["draws"].values() for v in d.values()})
    chk("G12 manifest ok, 20 draws, nodes 4-13, deterministic", [man["ok"], len(man["draws"]), nodes12[0], nodes12[-1],
                                                                map_draw(sl, 7) == map_draw(sl, 7)], [True, 20, 4, 13, True])
    bad = maps_manifest({0: tail_slots(sl), 1: tail_slots(sl)[:-1]}, sl)
    chk("G12 adversarial: seeds with different slot lists -> not ok", bad["ok"], False)
    # G13 breaker cut
    chk("G13 breaker w > 13.142", [round(BREAK_W, 3), 25.22 > BREAK_W, 15.8 > BREAK_W, 13.04 > BREAK_W], [13.142, True, True, False])
    # G14 OFF identity through the hook
    HOOK2.reset(dict(mode="OFF"), 71.24, 12.4)
    ident = True
    for _ in range(200):
        w, gb = float(rng.uniform(0.5, 60)), float(rng.choice([12.4, 22.6, 53.8, 215.2]))
        sl_ = dict(kind=str(rng.choice(["tail", "anchor"])), units={"u": w})
        ident = ident and HOOK2("KA-1", 0, 0, "u", sl_, gb, "bw") == V3.page_in(w, gb)
    chk("G14 OFF == V3.page_in (200 pairs)", ident, True)
    chk("hook wired on every site", [NB.HOOK is HOOK2, V6.V5._NB_PAGE is HOOK2, V6._NB_PAGE is HOOK2], [True, True, True])
    chk("settings: placebo names + refuse unknown", [setting_cfg("PLC-RH-19")["map"], setting_cfg("PLC-DROP-0")["evict"],
                                                     _refuses(lambda: setting_cfg("PLC-RH-20")), _refuses(lambda: setting_cfg("NB-P"))],
        ["MAP-19", "DROP", True, True])
    HOOK2.reset(dict(mode="OFF"), None, None)
    npass = sum(1 for r in res if r[1])
    for name, ok, got, exp in res:
        if not ok:
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
    m.add_argument("--freeze-maps", metavar="OUT")
    m.add_argument("--ifix-row", nargs=2, metavar=("NEW_FILE", "REF_FILE"))
    m.add_argument("--ifix-nb", nargs=4, metavar=("NEW_DIR", "OFF_FILE", "SAME_FILE", "H12_FILE"))
    m.add_argument("--summarize-rh", metavar="OUT_DIR")
    ap.add_argument("--setting")
    ap.add_argument("--cell", default="D1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--maps")
    ap.add_argument("--maps-sha")
    a = ap.parse_args()
    try:
        if a.self_test:
            return self_test()
        if a.ifix_row:
            r = ifix_row(*a.ifix_row)
            print(json.dumps(r))
            return 0 if r["pass_"] else 1
        if a.ifix_nb:
            install_rh()
            r = NB.ifix_nb(*a.ifix_nb)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize_rh:
            install_rh()
            print(json.dumps(summarize_rh(a.summarize_rh)))
            return 0
        if a.freeze_maps:
            for k in ("root", "cache"):
                if getattr(a, k) is None:
                    raise ValueError(f"--{k} is required")
            return freeze_maps(a.freeze_maps, a.root, a.cache)
        for k in ("setting", "root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        V6 = NB._m2897()._v6()
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if len(seeds) * V6.NDAYS != V6.NREPS:
            raise ValueError(f"{len(seeds)} seeds x {V6.NDAYS} days != {V6.NREPS} replicates")
        return run_rh(a.setting, a.cell, seeds, a.tag, a.root, a.cache, a.out, a.maps, a.maps_sha)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
