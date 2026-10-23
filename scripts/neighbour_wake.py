#!/usr/bin/env python3
"""neighbour_wake.py — Part 2: the 97-model 2 s case (cloud sim v6, cell D1) with an NVLink NEIGHBOUR-GPU wake tier
in place of host page-in for tail engines. FORK BY IMPORT: no landed sim is edited. configurer
(`measured_p.configure_v6`) sets the grid and the host page-in rate; v6's `build_windows` and v5's `run_variant` are
re-compiled with v5's own `fork()` (exact-anchor edits, refused unless each anchor occurs once) so that both page-in sites call ONE hook,
`_NB_PAGE(pol, slot, minute, unit, sl, gb, site)`. With the tier OFF the hook returns `V3.page_in(sl["units"][u], gb)`: the golden is
I-FIX-OFF (OFF reproduces and committed rows bit for bit).

Prereg (frozen before this file existed): (private report). On any
divergence the prereg wins. Definitions (prereg §3): 8-GPU HGX nodes, every GPU of a node a neighbour of every other; lenders = parked
tail GPUs of the same node; CAP GiB per lender; park places engines largest-first on the most-free lender; a lender that becomes held
DROPs (or REHOMEs) its guests before any wake that minute; a cached wake pulls at R_NB x delta / k (k = copies touching the lender that
minute), never slower than the host rate; anchors neither lend nor borrow.

Output contract:
  --self-test                                  exit 0 + "SELF-TEST PASS n/n" (goldens GN1-GN12); exit 1 + FAIL lines.
  --run --setting S --cell D1 [--seeds 0,1,2] [--tag T] --root R --cache C --out F
                                               WRITES (appends) ONE JSON line to F: v6's row + "s2933" (setting, the node map, the
                                               neighbour-tier stats per seed x policy); exit 0. A failed in-job check exits 1.
  --ifix-off NEW_FILE REF_DIR                  read-only: I-FIX-OFF (NEW main row vs the committed main row of the same cell in REF_DIR);
                                               one JSON on stdout; exit 0 PASS / 1 FAIL.
  --ifix-nb NEW_DIR OFF_FILE SAME_FILE H12_FILE  read-only: I-NB-STATIC + I-NB-BRACKET + I-REPLAY over NEW_DIR; one JSON; exit 0 / 1.
  --summarize-nb OUT_DIR                       read-only: summarize_v6 per setting sub-directory + DOOR-NB (prereg §5) +
                                               the tier stats; REFUSES duplicate (tag, cell) rows (exit 2); one JSON on stdout; exit 0.
  Bad input = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **v6 computes every tail page-in TWICE** (`build_windows` and v5's `run_variant`) and checks them against each other (I-RES); a
     hook on one site only makes I-RES fail on the first wake (prereg §0).
  2. **v5's run_variant fills pol_cache slot by slot**, so at slot s's wakes the later slots' parking schedules do not exist yet; the
     neighbour state needs ALL slots, so the wrapper pre-fills pol_cache with run_variant's own rule (incl. its host-blocked counter).
  3. **`ssh host '... &'` holds the session open**: launch with `ssh -n` + `setsid -f`.
"""
import argparse, collections, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREREG = "d49e5e28f"
GIB = 1.073741824
PAGE_FIXED = 0.02
R_NB = 270.0                       # : 16.39 GB in 0.0599-0.0605 s over NV12, IDLE lender (n=5)
LENDER_PEN = 0.212                 #: PUSH lender max +21.2 % (PULL measured ~0); charged per copy-second, batch side only
BIG_W = 13.0                       # prereg §3.12: the 5 engines >= 13 GiB
GPN = 8                            # GPUs per HGX node
SETTINGS = {
    "OFF-H22": dict(mode="OFF", host="H22"),
    "OFF-H12": dict(mode="OFF", host="H12"),
    "NB-P": dict(mode="NB", host="H12", map="ORDER", cap=0.5, evict="DROP", delta=0.5),
    "NB-P-SHUF": dict(mode="NB", host="H12", map="SHUF", cap=0.5, evict="DROP", delta=0.5),
    "NB-CAP25": dict(mode="NB", host="H12", map="ORDER", cap=0.25, evict="DROP", delta=0.5),
    "NB-CAP100": dict(mode="NB", host="H12", map="ORDER", cap=1.0, evict="DROP", delta=0.5),
    "NB-REHOME": dict(mode="NB", host="H12", map="ORDER", cap=0.5, evict="REHOME", delta=0.5),
    "NB-IDLE": dict(mode="NB", host="H12", map="ORDER", cap=0.5, evict="DROP", delta=1.0),
    "NB-FREE": dict(mode="FREE", host="H12"),
}
DEC_ARMS = ("KA-1|QINH", "KA-5|QINH")
STRIP_ROW = ("v6", "s2897", "s2933")


def _m2897():
    import measured_p as M
    return M


# ------------------------------------------------------------------------------------------------ the neighbour tier (prereg §3)
def page_rate(w, gbps):
    return PAGE_FIXED + w * GIB / gbps


def node_map(slots, kind="ORDER"):
    """prereg §3.2: anchors (16-GPU groups) take whole nodes first; the 1-GPU tail slots fill the next nodes, 8 per node, in slot order
    (ORDER) or sorted by sha256('2933|<slot index>') (SHUF). -> {slot index: node}. Refuses a cell it cannot map."""
    anc = [i for i, sl in enumerate(slots) if sl["kind"] == "anchor"]
    tail = [i for i, sl in enumerate(slots) if sl["kind"] == "tail"]
    other = [i for i, sl in enumerate(slots) if sl["kind"] not in ("anchor", "tail")]
    if other:
        raise ValueError(f"node map: {len(other)} giant slots (D1 has none; this map is defined for D1 only)")
    if any(slots[i]["g"] != 1 for i in tail):
        raise ValueError("node map: a tail slot with g != 1")
    if any(slots[i]["g"] % GPN for i in anc) or len(tail) % GPN:
        raise ValueError(f"node map: GPU counts do not fill whole nodes ({len(tail)} tail slots)")
    out, node = {}, 0
    for i in anc:
        out[i] = node
        node += slots[i]["g"] // GPN
    if kind == "SHUF":
        tail = sorted(tail, key=lambda i: hashlib.sha256(f"2933|{i}".encode()).hexdigest())
    elif kind != "ORDER":
        raise ValueError(f"unknown map {kind}")
    for j, i in enumerate(tail):
        out[i] = node + j // GPN
    return out


def nb_simulate(held_of, wake_of, units_of, node_of, cap, evict, n_min):
    """prereg §3.3-3.7 on tail slots only. held_of / wake_of: {slot: bool array [n_min]}; units_of: {slot: {unit: GiB}};
    node_of: {slot: node}. -> dict(rec {(s, t, u): lender or None}, k {(L, t): copies}, copies [(L, t, w)], stats)."""
    import numpy as np
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
    guests = {s: {} for s in slots}                             # lender -> {(owner, unit): w}
    free = {s: float(cap) for s in slots}
    loc, rec, k, copies = {}, {}, collections.Counter(), []
    st = dict(evicted=0, rehomed=0, pushes=0, park_uncached=0, guest_gib_min=0.0,
              lender_gpu_min=int(sum(int((~np.asarray(held_of[s], bool)).sum()) for s in slots)))
    tot = dict(guest=0.0)

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

    def evictions(becomes, t):                                  # 1. start of the minute
        for L in sorted(becomes):
            for key, w in sorted(guests[L].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                del guests[L][key]
                free[L] += w
                loc[key] = None
                tot["guest"] -= w
                st["evicted"] += 1
                if evict == "REHOME":
                    L2 = lender_for(L, w, t, {L, key[0]})
                    if L2 is not None:
                        place(L2, key, w, t)
                        st["rehomed"] += 1
            if guests[L]:
                raise RuntimeError("I-NB-EVICT: guests left on a held lender")

    def wakes_(wakes, t):                                       # 2. pull, then leave the cache
        for s in sorted(wakes):
            for u, w in units_of[s].items():
                L = loc.get((s, u))
                rec[(s, t, u)] = L
                if L is not None:
                    k[(L, t)] += 1  # pull
                    copies.append((L, t, w))
                    del guests[L][(s, u)]
                    free[L] += w
                    tot["guest"] -= w
                loc[(s, u)] = None

    def parks_(parks, t):                                       # 3. largest engine first
        for s in sorted(parks):
            for u, w in sorted(units_of[s].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                L = lender_for(s, w, t, {s})
                if L is None:
                    loc[(s, u)] = None
                    st["park_uncached"] += 1
                    continue
                place(L, (s, u), w, t)

    t_last = 0
    for t in sorted(ev):
        st["guest_gib_min"] += tot["guest"] * (t - t_last)
        t_last = t
        becomes, wakes, parks = ev[t]
        evictions(becomes, t)
        wakes_(wakes, t)
        parks_(parks, t)
    st["guest_gib_min"] += tot["guest"] * (n_min - t_last)
    return dict(rec=rec, k=k, copies=copies, stats=st)


def nb_page(state, s, t, u, w, host_gbps, delta):
    """prereg §3.8: cached -> min(0.02 + w GiB / (R_NB x delta / k), host page); uncached -> host page."""
    key = (s, t, u)
    if key not in state["rec"]:
        raise RuntimeError(f"I-NB-HOOK: no wake record for slot {s} minute {t} unit {u}")
    L = state["rec"][key]
    host = page_rate(w, host_gbps)
    if L is None:
        return host
    kk = state["k"][(L, t)]
    if kk < 1:
        raise RuntimeError("I-NB-HOOK: a cached wake with k = 0")
    return min(page_rate(w, R_NB * delta / kk), host)


class Hook:
    """the one page-in hook both v6 sites call (prereg §0); state per (seed context, policy)."""

    def __init__(self):
        self.cfg = dict(mode="OFF")
        self.bh = None
        self.host = None
        self.cur = {}
        self.states = []
        self.calls = collections.defaultdict(lambda: {"bw": set(), "rv": set()})
        self.bad = []

    def reset(self, cfg, bh, host):
        self.__init__()
        self.cfg, self.bh, self.host = dict(cfg), bh, host

    def __call__(self, pol, s, t, u, sl, gb, site):
        V3 = _m2897()._v6().V3
        w = sl["units"][u]
        mode = self.cfg["mode"]
        if mode == "OFF" or sl["kind"] != "tail":
            return V3.page_in(w, gb)
        if abs(gb - self.host) > 1e-12:
            raise RuntimeError(f"I-NB-HOOK: tail gb {gb} != the setting's host rate {self.host}")
        host = V3.page_in(w, gb)
        if mode == "FREE":
            v = page_rate(w, R_NB)
        else:
            st = self.cur.get(pol)
            if st is None:
                raise RuntimeError(f"I-NB-HOOK: no neighbour state for {pol}")
            v = nb_page(st, s, int(t), u, w, gb, self.cfg["delta"])
            self.calls[id(st)][site].add((s, int(t), u))
        if not (PAGE_FIXED - 1e-12 <= v <= host + 1e-12):
            self.bad.append((pol, s, int(t), str(u), v, host))
            raise RuntimeError(f"I-NB-HOOK: page {v} outside [0.02, host {host}]")
        return v

    def prepare(self, pol, slots, bs, khs, checks, pol_cache):
        """pre-fill pol_cache with v5 run_variant's own rule (pitfall 2), then build the neighbour state for this (seed, pol)."""
        if self.cfg["mode"] != "NB" or pol == "STATIC":
            return
        M = _m2897()
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
        nm = node_map(slots, self.cfg["map"])
        tail = [i for i, sl in enumerate(slots) if sl["kind"] == "tail"]
        held_of = {i: pol_cache[(pol, i, False)][0] for i in tail}
        wake_of = {i: pol_cache[(pol, i, False)][1] for i in tail}
        units_of = {i: dict(slots[i]["units"]) for i in tail}
        n_min = len(next(iter(held_of.values())))
        st = nb_simulate(held_of, wake_of, units_of, {i: nm[i] for i in tail}, self.cfg["cap"] * self.bh, self.cfg["evict"], n_min)
        st.update(tag=tag, pol=pol, ctx=sum(1 for x in self.states if x["pol"] == pol), pol_cache=pol_cache, units_of=units_of,
                  node_map=nm)
        self.states.append(st)
        self.cur[pol] = st

    def report(self):
        out = []
        for st in self.states:
            rec, k, units_of = st["rec"], st["k"], st["units_of"]
            wmax = max(w for us in units_of.values() for w in us.values())
            n = dict(unit_wakes=0, cached=0, host=0, big_wakes=0, big_cached=0, big_host=0, wmax=round(wmax, 2), wmax_host=0, wmax_cached=0)
            kh = collections.Counter()
            for (s, t, u), L in rec.items():
                w = units_of[s][u]
                n["unit_wakes"] += 1
                c = L is not None
                n["cached" if c else "host"] += 1
                if c:
                    kh[k[(L, t)]] += 1
                if w >= BIG_W:
                    n["big_wakes"] += 1
                    n["big_cached" if c else "big_host"] += 1
                if abs(w - wmax) < 1e-9:
                    n["wmax_cached" if c else "wmax_host"] += 1
            lost = sum(w * GIB / (R_NB * self.cfg["delta"] / k[(L, t)]) for L, t, w in st["copies"]) * LENDER_PEN
            s2 = dict(st["stats"])
            tax = s2["guest_gib_min"] / (s2["lender_gpu_min"] * self.bh) if s2["lender_gpu_min"] else None
            calls = self.calls.get(id(st), {"bw": set(), "rv": set()})
            out.append(dict(pol=st["pol"], ctx=st["ctx"], wakes=n, k_hist={str(a): b for a, b in sorted(kh.items())},
                            k_max=max(k.values()) if k else 0, stats=s2, hbm_tax=None if tax is None else round(tax, 6),
                            lender_batch_s_lost=round(lost, 3), hook_bw=len(calls["bw"]), hook_rv=len(calls["rv"]),
                            hook_sites_agree=calls["bw"] == calls["rv"]))
        return out


HOOK = Hook()
_INSTALLED = {}


def install():
    """fork both page-in sites onto the hook (idempotent). Refuses if either anchor is not found exactly once."""
    M = _m2897()
    V6 = M._v6()
    V5 = V6.V5
    if _INSTALLED:
        return V6
    bw = V5.fork(V6.build_windows, [(
        '            page = {u: V3.page_in(sl["units"][u], gb) for u in A}\n',
        '            page = {u: _NB_PAGE(pol, s_i, int(t), u, sl, gb, "bw") for u in A}\n')])
    rv = V5.fork(V5.run_variant, [(
        '(lambda u, _sl=sl, _gb=gb: V3.page_in(_sl["units"][u], _gb))',
        '(lambda u, _sl=sl, _gb=gb, _si=s_i, _tt=int(t): _NB_PAGE(pol, _si, _tt, u, _sl, _gb, "rv"))')])

    def rv_wrap(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache):
        if not any(hset):
            HOOK.prepare(pol, slots, bs, khs, checks, pol_cache)
        elif HOOK.cfg["mode"] == "NB":
            raise RuntimeError("the neighbour tier is defined without hold sets (v6 runs none)")
        return rv(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache)
    V5._NB_PAGE = HOOK
    V6._NB_PAGE = HOOK
    V6.build_windows = bw
    V5.run_variant = rv_wrap
    _INSTALLED.update(bw=bw, rv=rv)
    return V6


def configure(setting):
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting}")
    cfg = SETTINGS[setting]
    M = _m2897()
    V6 = M.configure_v6(cfg["host"])
    install()
    return V6, cfg


# ------------------------------------------------------------------------------------------------ run
def run_nb(setting, cell, seeds, tag, root, cache, out):
    M = _m2897()
    V6, cfg = configure(setting)
    if cfg["mode"] != "OFF" and cell != "D1":
        raise ValueError("the neighbour tier is defined for D1 only (prereg §2)")
    R = V6.V5.install()
    wd, row, rec, snap, r = V6.static_row_over(cell, seeds, tag, root, cache, None)
    row["cell"] = cell
    row["epar"] = None
    host = M.SETTINGS[cfg["host"]]
    HOOK.reset(cfg, wd.bh, host)
    row["s2933"] = dict(setting=setting, cfg=cfg, prereg=PREREG, h2d_tail=V6.H2D_TAIL, p_grid=list(V6.P_GRID), r_nb=R_NB,
                        bh=wd.bh, world_h2d=float(wd.tph["h2d"]))
    if snap is not None:
        if abs(wd.tph["h2d"] - M.WORLD_H2D) > 1e-9:
            raise RuntimeError(f"I-WORLD: the static world's tail h2d is {wd.tph['h2d']}, not {M.WORLD_H2D}")
        row["v6"] = V6.overlay6(wd, row, rec, snap, r, seeds, R)
        rep = HOOK.report()
        row["s2933"]["nb"] = rep
        if cfg["mode"] == "NB":
            if not rep:
                raise RuntimeError("I-NB-HOOK: the neighbour tier built no state")
            if not all(x["hook_sites_agree"] for x in rep):
                raise RuntimeError("I-NB-HOOK: build_windows and run_variant asked for different wakes")
            row["s2933"]["node_map"] = {str(k_): v for k_, v in sorted(HOOK.states[0]["node_map"].items())}
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks over rows
def _key(r):
    return (r["tag"], r["cell"])


def _unique(rows, what):
    """Refuse a duplicate (tag, cell), never last-wins."""
    out = {}
    for r in rows:
        k_ = _key(r)
        if k_ in out:
            raise ValueError(f"duplicate {what} row key {k_}")
        out[k_] = r
    return out


def _static(r):
    M = _m2897()
    return {k: v for k, v in M.strip(r).items() if k not in STRIP_ROW}


def ifix_off(new_file, ref_dir):
    """I-FIX-OFF (prereg §8): the NEW main row equals the committed main row of the same cell: static fields, every committed variant,
    rep_req / platform_burn / platform_hours, consts (minus P_GRID / n_variants) and the I_* checks."""
    M = _m2897()
    with open(new_file) as fh:
        new = [json.loads(l) for l in fh if l.strip()]
    if len(new) != 1:
        raise ValueError(f"{new_file}: {len(new)} rows, need 1")
    n = new[0]
    ref = _unique(M.load_rows(ref_dir), "ref v6")
    o = ref.get(_key(n))
    if o is None:
        raise ValueError(f"no committed row for {_key(n)} in {ref_dir}")
    sa, sb = _static(n), _static(o)
    diffs = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    nv, ov = n["v6"], o["v6"]
    vd = [v for v in ov["variants"] if M.strip(nv["variants"].get(v)) != M.strip(ov["variants"][v])]
    for part in ("rep_req", "platform_burn", "platform_hours"):
        if nv.get(part) != ov.get(part):
            diffs.append(f"v6.{part}")
    if {c: v for c, v in nv["consts"].items() if c not in ("P_GRID", "n_variants")} != \
            {c: v for c, v in ov["consts"].items() if c not in ("P_GRID", "n_variants")}:
        diffs.append("v6.consts")
    if {c: v for c, v in nv["checks"].items() if c.startswith("I_")} != {c: v for c, v in ov["checks"].items() if c.startswith("I_")}:
        diffs.append("v6.checks(I_*)")
    return dict(key=list(_key(n)), setting=n.get("s2933", {}).get("setting"), static_diffs=diffs[:10], variant_diffs=vd[:10],
                variants_compared=len(ov["variants"]), pass_=not diffs and not vd and len(ov["variants"]) > 0)


def ifix_nb(new_dir, off_file, same_file, h12_file):
    """I-NB-STATIC (static fields = OFF-H12's), I-NB-BRACKET (worst-wake per replicate SAME <= NB <= H12, every variant), I-REPLAY
    (main = rep_b = rep_c when present), in-job checks."""
    M = _m2897()
    rows = _unique(M.load_rows(new_dir), "nb")

    def one(f):
        with open(f) as fh:
            x = [json.loads(l) for l in fh if l.strip()]
        if len(x) != 1:
            raise ValueError(f"{f}: {len(x)} rows")
        return x[0]
    off, same, h12 = one(off_file), one(same_file), one(h12_file)
    res, ok = {}, True
    so = _static(off)
    for k_, r in sorted(rows.items()):
        if k_[0] not in ("main", "rep_b", "rep_c"):
            continue
        d = sorted(k for k in set(so) | set(_static(r)) if so.get(k) != _static(r).get(k) and k != "tag")
        res[f"I_NB_STATIC|{k_[0]}"] = d[:10] or "0 diffs"
        ok = ok and not d
        bad = [c for c, v in (r["v6"].get("checks") or {}).items() if c.startswith("I_") and v is False]
        if bad:
            res[f"checks|{k_[0]}"] = bad
            ok = False
        nviol, ncmp = 0, 0
        for v, x in r["v6"]["variants"].items():
            if v == "STATIC":
                continue
            a, b, c = same["v6"]["variants"][v]["mww"], x["mww"], h12["v6"]["variants"][v]["mww"]
            if not (len(a) == len(b) == len(c) == 21):
                raise ValueError(f"I-NB-BRACKET: {v} has {len(a)}/{len(b)}/{len(c)} replicates")
            for lo, mid, hi in zip(a, b, c):
                ncmp += 1
                if mid < lo - 1e-9 or mid > hi + 1e-9:
                    nviol += 1
        res[f"I_NB_BRACKET|{k_[0]}"] = dict(compared=ncmp, violations=nviol)
        ok = ok and nviol == 0 and ncmp > 0
    base = rows.get(("main", "D1"))
    for t in ("rep_b", "rep_c"):
        o = rows.get((t, "D1"))
        if base is not None and o is not None:
            a, b = M.strip(base), M.strip(o)
            for x in (a, b):
                x.pop("tag", None)
            res[f"I_REPLAY|{t}"] = "0 diffs" if a == b else "DIFFERS"
            ok = ok and a == b
    res["rows"] = len(rows)
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summary + DOOR-NB (prereg §5)
def door_class(y):
    if y is None:
        return "CLOSED"
    if y >= 0.842 - 1e-12:
        return "OPEN-AT-STOCK-p"
    if y >= 0.5 - 1e-12:
        return "OPEN-WITH-p<=0.5"
    return "OPEN-ONLY-BELOW-0.5"


def door_nb(x_order, x_shuf):
    """X = max over the deciding arms of P_MEET6 per map (None = NO at any grid p); the lower class decides."""
    order = ["CLOSED", "OPEN-ONLY-BELOW-0.5", "OPEN-WITH-p<=0.5", "OPEN-AT-STOCK-p"]
    ca, cb = door_class(x_order), door_class(x_shuf)
    lo = min((ca, cb), key=order.index)
    return dict(door=lo, order=ca, shuf=cb, map_sensitive=ca != cb)


def _best(readings, cell="main|D1", S="S2"):
    vals = [readings[f"{cell}|{S}|{a}"]["p_meet6"] for a in DEC_ARMS]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def summarize_nb(out_dir):
    M = _m2897()
    res = dict(settings={}, door=None, skipped=[])
    for st in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, st)
        if not os.path.isdir(d):
            continue
        if st not in SETTINGS or not any(f.endswith(".jsonl") for f in os.listdir(d)):
            res["skipped"].append(st)                           # QA r1: a misnamed or empty directory is named, never silent
            continue
        cfg = SETTINGS[st]
        rows = _unique(M.load_rows(d), st)
        configure(st)
        s = M.summarize_v6(d, cfg["host"])
        r0 = rows.get(("main", "D1"))
        one = dict(readings={k: v for k, v in s["readings"].items() if k.startswith("main|D1|")},
                   qmain={S: q for S, q in s["parent"]["cells"].get("main|D1", {}).get("qmain", {}).items()},
                   replay=s["parent"].get("replay", {}).get("D1"), seedshift=s["parent"].get("seedshift", {}).get("D1"),
                   best_p_meet6_S2=_best(s["readings"]) if r0 else None,
                   point_0_79={a: s["readings"][f"main|D1|S2|{a}"]["meet"]["0.79"] for a in DEC_ARMS} if r0 else None,
                   nb=(r0 or {}).get("s2933", {}).get("nb"))
        res["settings"][st] = one
    a, b = res["settings"].get("NB-P"), res["settings"].get("NB-P-SHUF")
    if a and b:
        res["door"] = door_nb(a["best_p_meet6_S2"], b["best_p_meet6_S2"])
    return res


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    import numpy as np, tempfile
    M = _m2897()
    V6 = install()
    V3 = V6.V3
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))

    r4 = lambda x: round(x, 4)  # noqa: E731
    # GN1
    chk("GN1 host 25.22", r4(page_rate(25.22, 12.4)), 2.2039)
    chk("GN1 nb 25.22 d.5 k1/k2/k3", [r4(page_rate(25.22, R_NB * .5 / k_)) for k_ in (1, 2, 3)], [0.2206, 0.4212, 0.6218])
    chk("GN1 nb 25.22 d1 k1", r4(page_rate(25.22, R_NB)), 0.1203)
    chk("GN1 15.8 host / nb", [r4(page_rate(15.8, 12.4)), r4(page_rate(15.8, R_NB * .5))], [1.3882, 0.1457])

    def toy(held, units, nodes, cap, evict):
        H = {s: np.array(h, bool) for s, h in held.items()}
        W = {}
        for s, h in H.items():
            prev = np.concatenate([[True], h[:-1]])
            W[s] = h & ~prev
        return nb_simulate(H, W, units, nodes, cap, evict, len(next(iter(H.values()))))
    # GN2: placement, eviction, fallback
    g2 = dict(held={0: [1, 0, 0, 0, 1, 1], 1: [0] * 6, 2: [1, 1, 0, 0, 0, 1]}, units={0: {"u0": 25.0}, 1: {}, 2: {"u2": 5.0}},
              nodes={0: 0, 1: 0, 2: 0})
    st2 = toy(g2["held"], g2["units"], g2["nodes"], 30.0, "DROP")
    chk("GN2 u0 from S1 at t4", r4(nb_page(st2, 0, 4, "u0", 25.0, 12.4, 0.5)), 0.2188)
    chk("GN2 u2 evicted -> host at t5", r4(nb_page(st2, 2, 5, "u2", 5.0, 12.4, 0.5)), 0.4530)
    chk("GN2 lenders recorded", [st2["rec"][(0, 4, "u0")], st2["rec"][(2, 5, "u2")]], [1, None])
    # GN2b (adversarial, prereg §3.7): the lender becomes held in the SAME minute its guest's owner wakes -> evicted first -> host
    st2b = toy({0: [1, 1, 0, 1], 1: [1, 0, 0, 1]}, {0: {"u0": 5.0}, 1: {}}, {0: 0, 1: 0}, 30.0, "DROP")
    chk("GN2b same-minute eviction -> host", [st2b["rec"][(0, 3, "u0")], r4(nb_page(st2b, 0, 3, "u0", 5.0, 12.4, 0.5))], [None, 0.4530])
    # GN3: contention (wake pull + park push on one lender in one minute)
    st3 = toy({0: [1, 0, 0, 1], 1: [0] * 4, 2: [1, 1, 1, 0]}, {0: {"u0": 10.0}, 1: {}, 2: {"u2": 5.0}}, {0: 0, 1: 0, 2: 0}, 40.0, "DROP")
    chk("GN3 k(S1,3) = 2", st3["k"][(1, 3)], 2)
    chk("GN3 u0 page", r4(nb_page(st3, 0, 3, "u0", 10.0, 12.4, 0.5)), 0.1791)
    # GN4: a unit above CAP is never cached
    st4 = toy({0: [1, 0, 1], 1: [0] * 3}, {0: {"u": 40.0}, 1: {}}, {0: 0, 1: 0}, 35.62, "DROP")
    chk("GN4 above cap -> host", st4["rec"][(0, 2, "u")], None)
    # GN5: no lender across nodes
    st5 = toy({0: [1, 0, 1], 1: [0] * 3}, {0: {"u": 5.0}, 1: {}}, {0: 0, 1: 1}, 35.62, "DROP")
    chk("GN5 cross-node -> host", st5["rec"][(0, 2, "u")], None)
    # GN6: OFF is the identity
    HOOK.reset(dict(mode="OFF"), 71.24, 12.4)
    rng = np.random.default_rng(2933)
    ident = True
    for _ in range(200):
        w, gb = float(rng.uniform(0.5, 60)), float(rng.choice([12.4, 22.6, 53.8, 215.2]))
        sl = dict(kind=str(rng.choice(["tail", "anchor"])), units={"u": w})
        ident = ident and HOOK("KA-1", 0, 0, "u", sl, gb, "bw") == V3.page_in(w, gb)
    chk("GN6 OFF == V3.page_in (200 pairs)", ident, True)
    # GN7: anchors pass through in NB mode; anchors are never mapped as lenders
    HOOK.reset(dict(SETTINGS["NB-P"]), 71.24, 12.4)
    chk("GN7 anchor page", HOOK("KA-1", 0, 0, "a", dict(kind="anchor", units={"a": 553.4}), 16 * 53.8, "bw"),
        V3.page_in(553.4, 16 * 53.8))
    # GN8: min(nb, host)
    st8 = dict(rec={(0, 0, "u"): 1}, k=collections.Counter({(1, 0): 11}))
    chk("GN8 min rule at k 11", r4(nb_page(st8, 0, 0, "u", 25.22, 12.4, 0.5)), 2.2039)
    # GN9: maps
    sl9 = [dict(kind="anchor", g=16)] * 2 + [dict(kind="tail", g=1)] * 80
    mo, ms = node_map(sl9, "ORDER"), node_map(sl9, "SHUF")
    cnt = lambda m: sorted(collections.Counter(v for k_, v in m.items() if k_ >= 2).values())  # noqa: E731
    chk("GN9 ORDER 8 per node, nodes 4-13", (cnt(mo), min(v for k_, v in mo.items() if k_ >= 2), mo[0], mo[1]), ([8] * 10, 4, 0, 2))
    chk("GN9 SHUF permutation + deterministic", (cnt(ms), ms == node_map(sl9, "SHUF"), ms != mo), ([8] * 10, True, True))
    # GN10: REHOME
    st10 = toy(g2["held"], g2["units"], g2["nodes"], 30.0, "REHOME")
    chk("GN10 rehome: u0 k=2 at t4, u2 nb at t5", [r4(nb_page(st10, 0, 4, "u0", 25.0, 12.4, 0.5)),
                                                    r4(nb_page(st10, 2, 5, "u2", 5.0, 12.4, 0.5))], [0.4177, 0.0598])
    # GN11: duplicate rows refused
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "a.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(tag="main", cell="D1")) + "\n" + json.dumps(dict(tag="main", cell="D1")) + "\n")
        try:
            _unique(M.load_rows(td), "nb")
            chk("GN11 duplicates refused", "accepted", "refused")
        except ValueError:
            chk("GN11 duplicates refused", "refused", "refused")
    # GN12: beta arithmetic (Part 1 constants, prereg §7)
    chk("GN12 beta", [round(6.5 * (1 + 5.85), 3), round(6.5 * (1 + 1.65), 3), round((1 + 5.85) * 1.212 - 1, 3)], [44.525, 17.225, 7.302])
    # door mapping (prereg §5)
    chk("door classes", [door_nb(0.842, 1.0)["door"], door_nb(0.5, 0.842)["door"], door_nb(0.25, 0.75)["door"],
                         door_nb(None, 0.75)["door"], door_nb(0.5, 0.75)["map_sensitive"]],
        ["OPEN-AT-STOCK-p", "OPEN-WITH-p<=0.5", "OPEN-ONLY-BELOW-0.5", "CLOSED", False])
    # the forks are installed on both sites (names searched through nested code objects: comprehensions are separate code on 3.11)
    def names(co):
        out = set(co.co_names)
        for c in co.co_consts:
            if hasattr(c, "co_names"):
                out |= names(c)
        return out
    chk("forks installed", [V6.build_windows is _INSTALLED["bw"], V6.V5.run_variant.__name__ == "rv_wrap",
                            "_NB_PAGE" in names(V6.build_windows.__code__), "_NB_PAGE" in names(_INSTALLED["rv"].__code__),
                            "page_in" in names(V6.build_windows.__code__) or "page_in" in names(_INSTALLED["rv"].__code__)],
        [True, True, True, True, False])
    HOOK.reset(dict(mode="OFF"), None, None)
    npass = sum(1 for r in res if r[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    print(f"SELF-TEST {'PASS' if npass == len(res) else 'FAIL'} {npass}/{len(res)}")
    return 0 if npass == len(res) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--ifix-off", nargs=2, metavar=("NEW_FILE", "REF_DIR"))
    m.add_argument("--ifix-nb", nargs=4, metavar=("NEW_DIR", "OFF_FILE", "SAME_FILE", "H12_FILE"))
    m.add_argument("--summarize-nb", metavar="OUT_DIR")
    ap.add_argument("--setting", choices=sorted(SETTINGS))
    ap.add_argument("--cell", default="D1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        if a.self_test:
            return self_test()
        if a.ifix_off:
            r = ifix_off(*a.ifix_off)
            print(json.dumps(r))
            return 0 if r["pass_"] else 1
        if a.ifix_nb:
            r = ifix_nb(*a.ifix_nb)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize_nb:
            print(json.dumps(summarize_nb(a.summarize_nb)))
            return 0
        for k in ("setting", "root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        V6 = _m2897()._v6()
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if len(seeds) * V6.NDAYS != V6.NREPS:
            raise ValueError(f"{len(seeds)} seeds x {V6.NDAYS} days != {V6.NREPS} replicates")
        return run_nb(a.setting, a.cell, seeds, a.tag, a.root, a.cache, a.out)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
