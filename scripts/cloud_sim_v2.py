#!/usr/bin/env python3
"""cloud_sim_v2.py — cloud fleet with (a) co-activity giant homing, (b) a swept KV shed on giant groups,
(c) a demand-sized fleet (the fewest GPUs that keep the SLO for a fixed catalogue, and the idle % it implies), plus (d) hour/day
elastic idle floors (descriptive).

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. FORK BY IMPORT: Fleet / walk / evaluate / World (cloud_sim) and sleep core (quiet_giants) run
UNCHANGED. v2 adds: FleetCA (a Fleet subclass, co-activity homing), run_giants (a fork of sleep_groups_real carrying the shed
tau, patched in for run_dgroups during a job; equal at tau None, I-FIX2753), a capture of the per-minute busy GPUs at
fleet_idle call (elastic floors), and the scan / N* / summary code.

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n"; exit 1 + FAIL lines. Pure goldens + a toy sleep path (no trace).
  --run --arm A --fmt F --N n [--tau T] [--window W] [--days D] [--seeds 0,1,2] [--set k=v] [--tag T] --root R --cache C --out F
                                    WRITES (appends) one JSON line per n-grid point to F (row + "v2" + "v2x"); exit 0.
                                    Anchor flags as (exit 3 on ANCHOR-NOT-PLACEABLE after one flagged row).
  --scan --arm A --cat CAT128P|ALL --fmt F [--tau T] [--set k=v] [--tag T] --root R --cache C --out F
                                    WRITES (appends) one JSON line per scanned N (the catalogue's full walk on N GPUs); exit 0.
  --ifix NEW_DIR OLD_DIR            read-only: I-FIX2753 + I-STALL-FLEET + I-REPLAY over the rows; one JSON object; exit 0 PASS / 1 FAIL.
  --summarize DIR                   read-only: prereg §4 verdicts over DIR/*.jsonl; one JSON object on stdout; exit 0.
  Bad input (unknown arm/cat, missing file) or a held-out manifest mismatch = exit 2 + one "ERROR:" line on stderr.
Arms: P, BT, BS BSS (BS + --tau, swept), CA (co-activity homing), CAS (CA + tau 10), BC (descriptive, scan only).

Top pitfalls (observed earlier, inherited here):
  1. **`ssh host '... &'` holds the session open** — launch grids with `ssh -n` + `setsid -f`.
  2. **The seed is blind for all-awake arms** — P / BT / the tail and the anchor are seed-blind; only idle offsets and the
     giant sleep path move with the seed.
  3. **A unit index of -1 wraps numpy** — arrays are indexed by model, never by unit.
"""
import argparse, collections, contextlib, hashlib, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import node_sim as NS        # noqa: E402  (not edited)
import quiet_giants as QG    # noqa: E402  (not edited)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
TRAIN0 = HELD - WEEK_MIN
SLOS = CS.SLOS
NREPS = 21
CELL_DIGEST = "a475b6f0a63faab162da4d2641220511acc9c637cb75aa8e3a4fe20fc6c8b2ec"   # prereg §2
HELD_TOTAL = 3687246
CAT128_SHA = "aed68e28a2dc6b9402f8c76f321e222be221532d143195d9463807269710272e"     # prereg §0.5
TAU_POINT = 10.0                     # prereg §3: point, fixed before any number
TAUS = (600.0, 60.0, 10.0, 2.0)      # prereg §3: swept
WIN_POINT = 60                       # prereg §3: 60-minute windows over the 7 TRAIN days
SCAN = {"CAT128P": tuple(range(40, 129, 8)), "ALL": tuple(range(128, 545, 8))}
ARMS = {"P": ("P", None), "BT": ("BT", None), "BS": ("BS", "rho"), "BSS": ("BS", "rho"), "CA": ("BS", "coact"),
        "CAS": ("BS", "coact"), "BC": ("BC", "rho")}
HOUR = 60


class ManifestError(ValueError):
    """The loaded world's held-out cells are not the prereg's (§2): never run on them."""


# ------------------------------------------------------------------------------------------------ (a) co-activity homing
def train_windows(minutes, window, days=7):
    """TRAIN windows (index of `window`-minute buckets from TRAIN0) holding >= 1 request; days < 7 keeps the trailing `days` only."""
    m = np.asarray(minutes, np.int64)
    m = m[(m >= TRAIN0) & (m < HELD)]
    w = set(((m - TRAIN0) // window).tolist())
    if days < 7:
        lo = -(-((7 - days) * DAY_MIN) // window)
        w = {x for x in w if x >= lo}
    return w


class FleetCA(CS.Fleet):
    """prereg §3 CA: a giant c joins group g iff c and every member are feasible alone, c shares no TRAIN window with any member, and
    §4.3 (b)-(c) hold (host RAM, KV floor). No rho cap. An infeasible-alone giant opens its own group, which nobody joins."""

    def __init__(self, *a, hours=None, feas=None, **k):
        super().__init__(*a, **k)
        self.hours, self.feas, self._cur = hours, feas, None

    def join_ok(self, g, G, W_, rho):
        c = self._cur
        if g.get("solo") or not self.feas[c]:
            return False
        if any(self.hours[c] & self.hours[x] for x in g["members"]):
            return False
        return self.fits_group(G, g["W"] + [W_])

    def try_giant(self, cid, G, W_, rho):
        self._cur = cid
        n0 = len(self.groups)
        ok = super().try_giant(cid, G, W_, rho)
        if ok and len(self.groups) > n0 and not self.feas[cid]:
            self.groups[-1]["solo"] = True
        return ok


def ca_violations(groups, hours, feas):
    """I-CA: list of violations (shared window or infeasible giant with company) in snapshot groups."""
    bad = []
    for g in groups:
        mem = g["members"]
        if len(mem) < 2:
            continue
        for i, a in enumerate(mem):
            if not feas[a]:
                bad.append(("infeasible-with-company", a[:8]))
            for b in mem[i + 1:]:
                if hours[a] & hours[b]:
                    bad.append(("shared-window", a[:8], b[:8]))
    return bad


def feasible_alone(wd, cid):
    """prereg §3: the giant's TRAIN cells alone, dedicated on its G with KV = L0, p95 <= 10 s (or < 20 TRAIN requests)."""
    cx = wd.cx
    d = cx["tab"][cid]
    m = cx["cid_m"][cid]
    st = NS.subtrace(cx["tr"], [m])
    k = st.minute < HELD
    st2 = T.Trace.__new__(T.Trace)
    st2.minute, st2.mid, st2.n, st2.it, st2.ot = st.minute[k], st.mid[k], st.n[k], st.it[k], st.ot[k]
    st2.models, st2.explicit = st.models, None
    kt = [-1.0] * cx["M"]
    kt[m] = NS.kv_tokens(d["G"], d["L0"], cx["S"]["pop"]["bpt"][m], d["dup"])
    Cs = [1.0] * cx["M"]
    Cs[m] = wd.cg(d["G"])
    wk, _ = CS.dedicated_rep(st2, kt, NS.giant_phys(wd.np, wd.tph), Cs, 1, score_from_min=TRAIN0)
    n = int(sum(wk[m].values()))
    if n < QG.MIN_REQ:
        return True, n, None
    p = T._pct(wk[m], 0.95)
    return bool(p <= 10.0), n, p


def coact_inputs(wd, window, days):
    tr = wd.cx["tr"]
    hours, feas, info = {}, {}, {}
    for c in wd.cands:
        if c["kind"] != "giant":
            continue
        cid = c["cid"]
        hours[cid] = train_windows(tr.minute[tr.mid == c["m"]], window, days)
        f, n, p = feasible_alone(wd, cid)
        feas[cid] = f
        info[cid[:8]] = dict(feasible=f, train_req=n, train_p95=None if p is None else (str(p) if p == math.inf else round(p, 3)),
                             windows=len(hours[cid]))
    return dict(hours=hours, feas=feas, info=info)


# ------------------------------------------------------------------------------------------------ (b) the shed giant runner
def shed_core(trace, members, Gc, np_, tph, seed, tau):
    """ sleep_core with the KV shed tau (None = never-shed sleep path)."""
    return QG.sleep_core(trace, members, Gc, np_, tph, seed, shed_tau=tau)


def run_giants(cx, np_, groups, seed, tau):
    """FORK of sleep_groups_real (= run_dgroups at tau None, I-SLEEP) routed through shed_core."""
    tab, tr = cx["tab"], cx["tr"]
    byG = collections.defaultdict(list)
    for G, mem in groups:
        byG[G].append(list(mem))
    ow, od, stats = {}, {}, {}
    for G, gs in sorted(byG.items()):
        members = []
        for gi, g in enumerate(gs):
            for cid in g:
                m = cx["cid_m"][cid]
                members.append((m, gi, tab[cid]["W"], cx["S"]["pop"]["bpt"][m] * tab[cid]["dup"], tr.models[m][1], cid))
        w, dy, stt = shed_core(tr, members, G, np_, cx["tph"], seed, tau)
        ow.update(w)
        od.update(dy)
        stats[str(G)] = stt
    return ow, od, stats


@contextlib.contextmanager
def giant_tau(tau):
    """route evaluate's giant runner through run_giants(tau) for the duration of a job."""
    orig = CS.run_dgroups
    CS.run_dgroups = lambda cx, np_, groups, seed: run_giants(cx, np_, groups, seed, tau)
    try:
        yield
    finally:
        CS.run_dgroups = orig


# ------------------------------------------------------------------------------------------------ (d) elastic floors (TOP-PRO)
def busy_gpus_per_min(B, slots, anc_cnt, anc_r, anc_g, n_min):
    """busy GPUs per minute with §5's definition: slot of g GPUs busy iff any homed model in flight; anchor g x min(r, in flight)."""
    b = np.zeros(n_min, np.int64)
    for g, ms in slots:
        if ms:
            b += g * np.any(B[list(ms)], axis=0)[:n_min].astype(np.int64)
    if anc_cnt is not None and anc_r:
        b += anc_g * np.minimum(np.asarray(anc_cnt, np.int64)[:n_min], anc_r)
    return b


def elastic_reps(b, day_len, hour_len):
    """LODO replicates of hour-elastic and day-elastic idle: 1 - busy / capacity, capacity = each hour's (day's) peak x its length."""
    b = np.asarray(b, np.int64)
    nd = len(b) // day_len
    H = b[:nd * day_len].reshape(-1, hour_len)
    hb = H.sum(axis=1).reshape(nd, -1).sum(axis=1)
    hc = (H.max(axis=1) * hour_len).reshape(nd, -1).sum(axis=1)
    D = b[:nd * day_len].reshape(nd, day_len)
    db, dc = D.sum(axis=1), D.max(axis=1) * day_len

    def reps(bb, cc):
        out = []
        for d in range(nd):
            c = int(cc.sum() - cc[d])
            out.append(None if c <= 0 else 1.0 - (int(bb.sum() - bb[d])) / float(c))
        return out
    return reps(hb, hc), reps(db, dc)


@contextlib.contextmanager
def capture_elastic(store):
    orig = CS.fleet_idle

    def wrapped(B, slots, anc_cnt, anc_r, anc_g, N, n_min, day_len):
        b = busy_gpus_per_min(B, slots, anc_cnt, anc_r, anc_g, n_min)
        ref = CS.slot_busy_by_day(B, slots, anc_cnt, anc_r, anc_g, n_min, day_len)
        if not np.array_equal(b.reshape(-1, day_len).sum(axis=1), ref):
            raise RuntimeError("I-BUSY: per-minute busy GPUs disagree with 's per-day busy GPU-minutes")
        h, d = elastic_reps(b, day_len, HOUR)
        store.append(dict(hour=h, day=d, peak=int(b.max()), mean=float(b.mean())))
        return orig(B, slots, anc_cnt, anc_r, anc_g, N, n_min, day_len)
    CS.fleet_idle = wrapped
    try:
        yield
    finally:
        CS.fleet_idle = orig


# ------------------------------------------------------------------------------------------------ (c) demand sizing
def n_star(rows, nreps=NREPS):
    """prereg §4c: rows = [(N, complete, passes | None)] -> first / stable / non_monotone / sized. Incomplete rows never count."""
    ok = {}
    for N, complete, passes in rows:
        if passes is not None and len(passes) != nreps:
            raise ValueError(f"N={N}: {len(passes)} replicates, need {nreps}")
        ok[N] = bool(complete) and passes is not None and all(passes)
    Ns = sorted(ok)
    first = next((N for N in Ns if ok[N]), None)
    stable = None
    for N in reversed(Ns):
        if not ok[N]:
            break
        stable = N
    return dict(first=first, stable=stable, non_monotone=first is not None and stable != first, sized=stable is not None)


def gpus_verdict(x, ref):
    """prereg §4c C2: FEWER-GPUS iff x <= ref - 8; MORE-GPUS iff x >= ref + 8; else SAME (incl. any NOT-SIZED)."""
    if x is None or ref is None:
        return "SAME" if x is None and ref is None else ("NOT-SIZED" if x is None else "SIZED-VS-NOT-SIZED")
    if x <= ref - 8:
        return "FEWER-GPUS"
    if x >= ref + 8:
        return "MORE-GPUS"
    return "SAME"


# ------------------------------------------------------------------------------------------------ world, plan, jobs
def load_world(fmt, over, extra, root, cache):
    wd = CS.make_world(fmt, over, extra, root, cache)
    tr = wd.cx["tr"]
    h = tr.minute >= HELD
    cells = sorted(zip([tr.models[m][0] for m in tr.mid[h]], tr.minute[h].tolist(), tr.n[h].tolist()))
    dig = hashlib.sha256("\n".join(f"{c}\t{mi}\t{n}" for c, mi, n in cells).encode()).hexdigest()
    tot = int(sum(n for _, _, n in cells))
    if dig != CELL_DIGEST or tot != HELD_TOTAL:
        raise ManifestError(f"I-MANIFEST: held-out digest {dig[:12]} / {tot} != prereg {CELL_DIGEST[:12]} / {HELD_TOTAL}")
    return wd


def catalogue(wd, cat):
    if cat == "ALL":
        return {c["cid"] for c in wd.cands}
    if cat == "CAT128P":
        _, sn = CS.plan(wd, "P", 128)
        ids = sn[CS.N_GRID[-1]]["admitted"]
        sha = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        if sha != CAT128_SHA:
            raise ManifestError(f"I-CAT: CAT-128P sha {sha[:12]} != prereg {CAT128_SHA[:12]}")
        return set(ids)
    raise ValueError(f"unknown catalogue {cat}")


def plan2(wd, base, homing, N, ca, grid, cands):
    """ plan() with the v2 Fleet and an explicit candidate list."""
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
    tm, gm = CS.ARM_MODES[base]
    units = {u: dict(cid=x["cid"], f=x["f"], w=x["w"], host=x["host"]) for u, x in enumerate(wd.uinfo)}
    if homing == "coact":
        fl = FleetCA(N, tm, gm, G, rmin, units, wd.gp, hours=ca["hours"], feas=ca["feas"])
    else:
        fl = CS.Fleet(N, tm, gm, G, rmin, units, wd.gp)
    return dict(flag=None, r_min=rmin, r=rmin), CS.walk(fl, cands, grid)


def evaluate2(wd, base, snap, N, r, beta, seeds, ca):
    store = []
    with capture_elastic(store):
        row = CS.evaluate(wd, base, snap, N, r, beta, seeds)
    v2x = dict(el_hour=[x for s in store for x in s["hour"]], el_day=[x for s in store for x in s["day"]],
               busy_peak=[s["peak"] for s in store], busy_mean=[round(s["mean"], 3) for s in store],
               solo=[g["members"][0][:8] for g in snap["groups"] if g.get("solo")],
               shed_held={s_: {G: st.get("shed_req_held", 0) for G, st in v.items()} for s_, v in row["d_stats"].items()},
               abandoned={s_: {G: st.get("abandoned_wakes", 0) for G, st in v.items()} for s_, v in row["d_stats"].items()})
    if ca is not None:
        bad = ca_violations(snap["groups"], ca["hours"], ca["feas"])
        if bad:
            raise RuntimeError(f"I-CA: {bad[:3]}")
    row["v2x"] = v2x
    return row


def label_of(arm, tau):
    return f"BSS-{int(tau)}" if arm == "BSS" else arm


def job_setup(arm, fmt, over, extra, root, cache, tau, window, days):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if arm == "BSS" and tau is None:
        raise ValueError("BSS needs --tau")
    if arm == "CAS" and tau is None:
        tau = TAU_POINT
    if arm in ("P", "BT", "BS", "CA", "BC") and tau is not None:
        raise ValueError(f"arm {arm} takes no --tau (prereg §3)")
    base, homing = ARMS[arm]
    wd = load_world(fmt, over, extra, root, cache)
    ca = coact_inputs(wd, window, days) if homing == "coact" else None
    v2 = dict(label=label_of(arm, tau), homing=homing, tau=tau, window=window if homing == "coact" else None,
              days=days if homing == "coact" else None, feas=ca["info"] if ca else None)
    return wd, base, homing, ca, tau, v2


def run_walk(arm, fmt, N, over, extra, tag, root, cache, out, tau, window, days, seeds):
    wd, base_arm, homing, ca, tau, v2 = job_setup(arm, fmt, over, extra, root, cache, tau, window, days)
    v2["seeds"] = list(seeds)
    beta = wd.np["beta"]
    base = dict(arm=base_arm, fmt=fmt, N=N, tag=tag, nphys=wd.np, extra=extra)
    with open(out, "a") as fh, giant_tau(tau):
        try:
            info, snaps = plan2(wd, base_arm, homing, N, ca, CS.N_GRID, wd.cands)
        except CS.AnchorError as e:
            fh.write(json.dumps(dict(base, v2=v2, flag="ANCHOR-NOT-PLACEABLE", error=str(e))) + "\n")
            raise
        if info["flag"]:
            fh.write(json.dumps(dict(base, v2=v2, flag=info["flag"], r_min=info["r_min"])) + "\n")
            return 0
        prev, prev_len = None, None
        for n in CS.N_GRID:
            sn = snaps[n]
            t0 = time.time()
            if prev is not None and len(sn["admitted"]) == prev_len:
                row = dict(prev, n=n, reused_from=prev["n"], elapsed_s=0.0)
            else:
                row = dict(base, n=n, r_min=info["r_min"])
                row.update(evaluate2(wd, base_arm, sn, N, info["r"], beta, seeds, ca))
                row.update(base)
                row["v2"] = v2
                row["elapsed_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            prev, prev_len = row, len(sn["admitted"])
    return 0


def run_scan(arm, cat, fmt, over, extra, tag, root, cache, out, tau, window, days, seeds, Ns=None):
    wd, base_arm, homing, ca, tau, v2 = job_setup(arm, fmt, over, extra, root, cache, tau, window, days)
    v2["seeds"] = list(seeds)
    beta = wd.np["beta"]
    ids = catalogue(wd, cat)
    cands = [c for c in wd.cands if c["cid"] in ids]
    base = dict(arm=base_arm, fmt=fmt, tag=tag, cat=cat, nphys=wd.np, extra=extra, n_cat=len(ids), v2=v2)
    with open(out, "a") as fh, giant_tau(tau):
        for N in (Ns or SCAN[cat]):
            t0 = time.time()
            try:
                info, snaps = plan2(wd, base_arm, homing, N, ca, [len(cands)], cands)
            except CS.AnchorError as e:
                fh.write(json.dumps(dict(base, N=N, flag="ANCHOR-NOT-PLACEABLE", error=str(e), complete=False)) + "\n")
                continue
            if info["flag"]:
                fh.write(json.dumps(dict(base, N=N, flag=info["flag"], r_min=info["r_min"], complete=False)) + "\n")
                continue
            sn = snaps[len(cands)]
            complete = set(sn["admitted"]) == ids
            row = dict(base, N=N, r_min=info["r_min"], complete=complete, n_admitted=len(sn["admitted"]))
            if complete:
                row.update(evaluate2(wd, base_arm, sn, N, info["r"], beta, seeds, ca))
                row.update(base, N=N, complete=True)
                row["pass"] = {s: bool(all(row["slo"][s]["pass"])) for s in row["slo"]}
            row["elapsed_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return 0


# ------------------------------------------------------------------------------------------------ checks over rows
TIMING = {"elapsed_s", "sim_s"}


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _rows(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            out += _load(os.path.join(d, f))
    return out


def same_row(new, old):
    keys = set(old) - TIMING
    diff = [k for k in sorted(keys) if new.get(k) != old.get(k)]
    return diff


def ifix(new_dir, old_dir):
    """I-FIX2753: v2 main P/BT/BS rows == grid rows (timing + v2 keys excluded); I-STALL-FLEET; I-REPLAY (main/rep_b/rep_c)."""
    res, ok = {}, True
    for fmt in ("int4", "fp8"):
        for N in (256, 384, 512):
            for arm in ("P", "BT", "BS"):
                fn = f"main_{fmt}_{N}_{arm}.jsonl"
                po, pn = os.path.join(old_dir, fn), os.path.join(new_dir, fn)
                if not (os.path.exists(po) and os.path.exists(pn)):
                    res[fn] = "MISSING"
                    ok = False
                    continue
                o, n = _load(po), _load(pn)
                bad = [(a.get("n"), same_row(b, a)) for a, b in zip(o, n) if same_row(b, a)]
                res[fn] = dict(rows=[len(o), len(n)], diffs=bad[:3])
                ok = ok and len(o) == len(n) and not bad
    st = [r for r in _load(os.path.join(new_dir, "main_int4_256_BS.jsonl")) if r.get("n") == 200]
    ab = st[0]["d_stats"]["0"].get("4", {}).get("abandoned_wakes") if st else None
    res["I_STALL_FLEET"] = dict(abandoned_seed0_G4=ab, pass_=ab == 1771)
    ok = ok and ab == 1771
    for arm in ("P", "BT", "BS"):
        a = _load(os.path.join(new_dir, f"main_int4_256_{arm}.jsonl"))
        for t in ("rep_b", "rep_c"):
            p = os.path.join(new_dir, f"{t}_int4_256_{arm}.jsonl")
            if not os.path.exists(p):
                res[f"I_REPLAY_{arm}_{t}"] = "MISSING"
                ok = False
                continue
            b = _load(p)
            bad = [x.get("n") for x, y in zip(a, b) if same_row(dict(y, tag=x["tag"]), x)]
            res[f"I_REPLAY_{arm}_{t}"] = dict(rows=[len(a), len(b)], diffs=bad[:5])
            ok = ok and len(a) == len(b) and not bad
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summarize (prereg §4)
def _rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def stable_cat(rr, s):
    """ §4.4 at SLO s over {n: row}: -> summary dict at n_match (or None)."""
    pm = {n: all(r_["slo"][s]["pass"]) for n, r_ in rr.items()}
    nm = CS.stable_n(pm)
    if nm is None:
        return None
    r0 = rr[nm]
    sl = r0["slo"][s]
    x = r0.get("v2x", {})
    return dict(n_match=nm, hosted=r0["hosted"], n_mg=r0["n_mg"], at=sl["at"], at_rng=_rng(sl["at"]), RS=sl["RS"],
                RS_rng=_rng(sl["RS"]), idle=r0["idle"], idle_rng=_rng(r0["idle"]), idle_w_rng=_rng(r0["idle_w"]),
                allocated=r0["allocated"], groups=r0["groups"], solo=x.get("solo"),
                abandoned_s0=x.get("abandoned", {}).get("0"), shed_held_s0=x.get("shed_held", {}).get("0"),
                el_hour_rng=_rng(x.get("el_hour", [])), el_day_rng=_rng(x.get("el_day", [])))


def cat_v(a, b):
    return CS.verdict_hi(a["at"], b["at"]) if a and b else "NO-CATALOGUE"


def summarize(d):
    rows = _rows(d)
    walk = collections.defaultdict(dict)
    scan = collections.defaultdict(list)
    for r in rows:
        if r.get("flag") and "cat" not in r:
            continue
        lab = r.get("v2", {}).get("label", r["arm"])
        if "cat" in r:
            scan[(r["tag"], lab, r["cat"], r["fmt"])].append(r)
            continue
        walk[(r["tag"], r["fmt"], r["N"], lab)][r["n"]] = r
    out = dict(walk={}, verdicts={}, scan={}, c={})
    for (tag, fmt, N, lab), rr in sorted(walk.items()):
        out["walk"][f"{tag}|{fmt}|{N}|{lab}"] = {str(x): stable_cat(rr, str(x)) for x in SLOS}
    W = out["walk"]

    def g(tag, fmt, N, lab, s="10.0"):
        return (W.get(f"{tag}|{fmt}|{N}|{lab}") or {}).get(s)
    for fmt in ("int4", "fp8"):
        for N in (256, 384, 512):
            v = {}
            for s in (str(x) for x in SLOS):
                bs, bt, p, ca, cas = (g("main", fmt, N, x, s) for x in ("BS", "BT", "P", "CA", "CAS"))
                vv = dict(A1_CA_vs_BS=cat_v(ca, bs), A2_CA_vs_BT=cat_v(ca, bt), CAS_vs_CA=cat_v(cas, ca), CAS_vs_BS=cat_v(cas, bs),
                          CAS_vs_BT=cat_v(cas, bt), CA_idle_vs_BT=CS.verdict_idle(ca["idle"], bt["idle"]) if ca and bt else None)
                for tau in TAUS:
                    bss = g("main", fmt, N, f"BSS-{int(tau)}", s)
                    vv[f"BSS-{int(tau)}_vs_BS"] = cat_v(bss, bs)
                    vv[f"BSS-{int(tau)}_vs_BT"] = cat_v(bss, bt)
                for lab, x in (("CA", ca), ("CAS", cas)):
                    if x and p:
                        c_, i_, r_ = CS.verdict_hi(x["at"], p["at"]), CS.verdict_idle(x["idle"], p["idle"]), CS.verdict_rs(x["RS"], p["RS"])
                        vv[f"{lab}_vs_P"] = ("GPU-BUNK LIFTS" if c_ == "MORE" and i_ == "LOWER-IDLE" and r_ != "LOWERS" else
                                             "NO-LIFT" if c_ != "MORE" and i_ != "LOWER-IDLE" else f"MIXED ({c_}, {i_}, {r_})")
                v[s] = vv
            out["verdicts"][f"{fmt}|{N}"] = v
    cells = [f"{f}|{N}" for f in ("int4", "fp8") for N in (256, 384, 512)]
    S = "10.0"
    a1 = sum(1 for c in cells if out["verdicts"][c].get(S, {}).get("A1_CA_vs_BS") == "MORE")
    out["headline_A1"] = ("CO-ACTIVITY HOMING UN-STALLS" if a1 == 6 else "PARTIAL" if a1 else "DOES NOT UN-STALL") + f" ({a1}/6)"
    out["A2"] = {c: out["verdicts"][c].get(S, {}).get("A2_CA_vs_BT") for c in cells}
    un = {}
    for tau in TAUS:
        un[str(int(tau))] = all(out["verdicts"][c].get(S, {}).get(f"BSS-{int(tau)}_vs_BS") == "MORE" for c in cells)
    out["tau_unstalls"] = un
    out["tau_max"] = next((int(t) for t in TAUS if un[str(int(t))]), None)
    # seed-shift replay (prereg §5)
    ss = {}
    for s in (str(x) for x in SLOS):
        bs0, cas0, bt0 = g("main", "int4", 256, "BS", s), g("main", "int4", 256, "CAS", s), g("main", "int4", 256, "BT", s)
        bs1, cas1 = g("seedshift", "int4", 256, "BS", s), g("seedshift", "int4", 256, "CAS", s)
        ss[s] = dict(CAS_vs_BS=[cat_v(cas0, bs0), cat_v(cas1, bs1)], CAS_vs_BT=[cat_v(cas0, bt0), cat_v(cas1, bt0)],
                     BS_at=[bs0 and bs0["at_rng"], bs1 and bs1["at_rng"]], CAS_at=[cas0 and cas0["at_rng"], cas1 and cas1["at_rng"]],
                     BS_idle=[bs0 and bs0["idle_rng"], bs1 and bs1["idle_rng"]])
        ss[s]["UNATTRIBUTABLE_256_int4"] = any(len(set(v_)) > 1 for k_, v_ in ss[s].items() if k_ in ("CAS_vs_BS", "CAS_vs_BT"))
    out["seedshift"] = ss
    # scans (prereg §4c)
    for (tag, lab, cat, fmt), rr in sorted(scan.items()):
        rr = sorted(rr, key=lambda x: x["N"])
        for s in (str(x) for x in SLOS):
            ns = n_star([(r["N"], r.get("complete"), r["slo"][s]["pass"] if r.get("complete") else None) for r in rr])
            ns95 = n_star([(r["N"], r.get("complete"), [m is None or m >= 0.95 - 1e-12 for m in r["slo"][s]["M"]]
                            if r.get("complete") else None) for r in rr])
            at = {r["N"]: r for r in rr}
            r0 = at.get(ns["stable"])
            e = dict(N_stable=ns["stable"], N_first=ns["first"], non_monotone=ns["non_monotone"], N95_stable=ns95["stable"],
                     first_complete=next((r["N"] for r in rr if r.get("complete")), None),
                     scanned=[rr[0]["N"], rr[-1]["N"]])
            if r0:
                x = r0.get("v2x", {})
                e.update(idle=r0["idle"], idle_rng=_rng(r0["idle"]), idle_w_rng=_rng(r0["idle_w"]), RS_rng=_rng(r0["slo"][s]["RS"]),
                         at_rng=_rng(r0["slo"][s]["at"]), hosted=r0["hosted"], allocated=r0["allocated"],
                         el_hour_rng=_rng(x.get("el_hour", [])), el_day_rng=_rng(x.get("el_day", [])), busy_peak=x.get("busy_peak"),
                         gap_to_10pct=[round(v_ - 0.10, 4) for v_ in _rng(r0["idle"])])
            out["scan"][f"{tag}|{lab}|{cat}|{fmt}|{s}"] = e
    sc = out["scan"]
    for fmt in ("int4", "fp8"):
        for s in (str(x) for x in SLOS):
            p128, bt128 = sc.get(f"main|P|CAT128P|{fmt}|{s}"), sc.get(f"main|BT|CAT128P|{fmt}|{s}")
            c1 = None
            if p128 and bt128:
                gv = ("FEWER-GPUS" if bt128["N_stable"] is not None and bt128["N_stable"] <= 120 else
                      "SAME" if bt128["N_stable"] == 128 else "NOT-SIZED" if bt128["N_stable"] is None else "OTHER")
                iv = CS.verdict_idle(bt128["idle"], p128["idle"]) if bt128.get("idle") and p128.get("idle") else None
                head = ("DEMAND-SIZING CUTS IDLE" if gv == "FEWER-GPUS" and iv == "LOWER-IDLE" else
                        "SIZING ONLY" if gv == "FEWER-GPUS" else "NO CUT")
                c1 = dict(N_P=p128["N_stable"], N_BT=bt128["N_stable"], gpus=gv, idle=iv, headline=head,
                          surplus_on_128=None if bt128["N_stable"] is None else 128 - bt128["N_stable"])
            pa, bta, csa = (sc.get(f"main|{x}|ALL|{fmt}|{s}") for x in ("P", "BT", "CAS"))
            c2 = None
            if pa and bta and csa:
                c2 = dict(N_P=pa["N_stable"], N_BT=bta["N_stable"], N_CAS=csa["N_stable"],
                          CAS_vs_P=gpus_verdict(csa["N_stable"], pa["N_stable"]), CAS_vs_BT=gpus_verdict(csa["N_stable"], bta["N_stable"]),
                          idle_CAS_vs_P=CS.verdict_idle(csa["idle"], pa["idle"]) if csa.get("idle") and pa.get("idle") else None,
                          idle_CAS_vs_BT=CS.verdict_idle(csa["idle"], bta["idle"]) if csa.get("idle") and bta.get("idle") else None)
            out["c"][f"{fmt}|{s}"] = dict(C1=c1, C2=c2)
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §6)
def self_test():
    res = []

    def chk(name, got, exp, tol=None):
        if tol is None:
            ok = got == exp
        else:
            ok = all(abs(a - b) <= tol for a, b in zip(got, exp)) and len(got) == len(exp)
        res.append((name, ok, got, exp))

    # GV1 homing through the real Fleet code
    gp = dict(bh=80.0, h_tot=1e9, node_ram=1e9, r_pin=1.0, o_g=0.0, s_g=0.0, rho_cap=0.5)
    order = [("D", 0.05), ("A", 0.20), ("B", 0.20), ("C", 0.20), ("E", 0.20)]
    hours = {"D": {9}, "A": {1, 2, 3}, "B": {3, 4}, "C": {7}, "E": {20}}
    feas = {"D": False, "A": True, "B": True, "C": True, "E": True}
    cands = [dict(cid=c, kind="giant", uid=None, G=4, W=100.0, rho=r) for c, r in order]

    def groups_of(fl):
        return [sorted(g["members"]) for g in fl.groups]
    for N, exp_rho, exp_ca, exp_ca_adm in ((48, [["A", "B", "D"], ["C", "E"]], [["D"], ["A", "C", "E"], ["B"]], ["D", "A", "B", "C", "E"]),
                                           (24, [["A", "B", "D"], ["C", "E"]], [["D"], ["A", "C", "E"]], ["D", "A", "C", "E"])):
        f1 = CS.Fleet(N, "SPREAD", "SHARE", 16, 1, {}, gp)
        CS.walk(f1, cands, [len(cands)])
        f2 = FleetCA(N, "SPREAD", "SHARE", 16, 1, {}, gp, hours=hours, feas=feas)
        CS.walk(f2, cands, [len(cands)])
        tag = "GV1" if N == 48 else "GV1b"
        chk(f"{tag} rho groups N={N}", groups_of(f1), exp_rho)
        chk(f"{tag} rho admits all N={N}", sorted(f1.admitted), sorted(["A", "B", "C", "D", "E"]))
        chk(f"{tag} co-activity groups N={N}", groups_of(f2), exp_ca)
        chk(f"{tag} co-activity admitted N={N}", f2.admitted, exp_ca_adm)
        chk(f"{tag} D solo N={N}", [g.get("solo", False) for g in f2.groups][0], True)
        chk(f"{tag} I-CA clean N={N}", ca_violations([dict(g) for g in f2.groups], hours, feas), [])
    f1 = CS.Fleet(48, "SPREAD", "SHARE", 16, 1, {}, gp)
    CS.walk(f1, cands, [len(cands)])
    f2 = FleetCA(48, "SPREAD", "SHARE", 16, 1, {}, gp, hours=hours, feas=feas)
    CS.walk(f2, cands, [len(cands)])
    chk("GV1 GPUs rho 24 / co-activity 28", (f1.group_gpus(), f2.group_gpus()), (24, 28))
    chk("GV1 I-CA flags the rho groups", len(ca_violations([dict(g) for g in f1.groups], hours, feas)) > 0, True)
    # GV2 shed through the sleep path (toy, one G = 1 group)
    np_ = NS.nphys_with({"fmt": "int4"})
    tph = NS.tail_phys(np_, "STATIC")
    ta = np.arange(TRAIN0, HELD, dtype=np.int64)
    tb = np.array([HELD + 60 + 100 * k for k in range(10)], np.int64)
    models = [("A", 100.0, 10.0, "int4", 30.0, 30.0), ("B", 100.0, 10.0, "int4", 30.0, 30.0)]
    tr = QG.synth_trace([(ta, np.ones(len(ta), np.int64), np.full(len(ta), 150000.0), np.full(len(ta), 10000.0)),
                         (tb, np.ones(10, np.int64), np.full(10, 1000.0), np.full(10, 100.0))], models)
    mem = [(0, 0, 30.0, 2e5, 100.0, "A"), (1, 0, 30.0, 2e5, 100.0, "B")]
    w0, _, st0 = shed_core(tr, mem, 1, np_, tph, 0, None)
    w1, _, st1 = shed_core(tr, mem, 1, np_, tph, 0, 10.0)
    chk("GV2 no shed: B 10/10 at +inf", w0[1].get(T.INF_KEY, 0), 10)
    chk("GV2 no shed: abandoned wakes >= 10", st0.get("abandoned_wakes", 0) >= 10, True)
    chk("GV2 tau 10: B 0/10 at +inf", w1[1].get(T.INF_KEY, 0), 0)
    chk("GV2 tau 10: B all <= 10 s", sum(v for k, v in w1[1].items() if k <= T._key(10.0)), 10)
    chk("GV2 tau 10: A shed > 0", st1.get("shed_req", 0) > 0, True)
    # GV2 through the patched entry point (M5 path): run_giants on a toy cx
    cx = dict(tab={"A": dict(W=30.0, dup=1.0), "B": dict(W=30.0, dup=1.0)}, tr=tr, cid_m={"A": 0, "B": 1},
              S=dict(pop=dict(bpt=[2e5, 2e5])), tph=tph)
    with giant_tau(10.0):
        wg, _, _ = CS.run_dgroups(cx, np_, [(1, ["A", "B"])], 0)
    chk("GV2 patched runner carries tau", wg[1].get(T.INF_KEY, 0), 0)
    with giant_tau(None):
        wn, _, _ = CS.run_dgroups(cx, np_, [(1, ["A", "B"])], 0)
    chk("GV2 patched runner at tau None stalls", wn[1].get(T.INF_KEY, 0), 10)
    chk("GV2 patch restored", CS.run_dgroups.__name__, "run_dgroups")
    # GV3 demand sizing (adversarial)
    A = [True] * NREPS

    def fail_at(i):
        x = list(A)
        x[i] = False
        return x
    rows = [(40, False, A), (48, True, fail_at(5)), (56, True, A), (64, True, fail_at(0)), (72, True, A), (80, True, A)]
    chk("GV3 N* first/stable/non-monotone", n_star(rows), dict(first=56, stable=72, non_monotone=True, sized=True))
    rows2 = rows[:-1] + [(80, True, fail_at(20))]
    chk("GV3 top fails -> NOT-SIZED", n_star(rows2), dict(first=56, stable=None, non_monotone=True, sized=False))
    try:
        n_star([(56, True, [True] * 20)])
        chk("GV3 20 replicates raise", False, True)
    except ValueError:
        chk("GV3 20 replicates raise", True, True)
    chk("GV3 incomplete never counts", n_star([(40, False, A)])["sized"], False)
    chk("GV3 gpus verdicts", [gpus_verdict(120, 128), gpus_verdict(128, 128), gpus_verdict(136, 128), gpus_verdict(None, 128)],
        ["FEWER-GPUS", "SAME", "MORE-GPUS", "NOT-SIZED"])
    # GV4 idle over N* (idle_reps, unchanged)
    pdb = np.array([600, 1200, 0, 0, 0, 0, 0])
    chk("GV4 idle N=1", [round(x, 6) for x in CS.idle_reps(pdb, 1, 1440)], [0.861111, 0.930556] + [0.791667] * 5)
    chk("GV4 idle N=2 rep 0", round(CS.idle_reps(pdb, 2, 1440)[0], 6), 0.930556)
    # GV5 busy GPUs + elastic floors
    B = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]], bool)
    chk("GV5 busy GPUs per minute", busy_gpus_per_min(B, [(4, [0]), (1, [1, 2])], np.array([0, 3, 1, 0]), 2, 16, 4).tolist(),
        [4, 33, 17, 4])
    h, dd = elastic_reps([2, 0, 1, 1, 0, 0, 3, 2], 4, 2)
    chk("GV5 hour-elastic", [round(x, 6) for x in h], [0.166667, 0.333333])
    chk("GV5 day-elastic", [round(x, 6) for x in dd], [0.583333, 0.5])
    # GV6 TRAIN windows
    chk("GV6 window index", train_windows([TRAIN0 + 125], 60), {2})
    chk("GV6 trailing 3 days keeps 96, drops 95", train_windows([TRAIN0 + 5765, TRAIN0 + 5759], 60, 3), {96})
    chk("GV6 held-out minutes ignored", train_windows([HELD + 5, TRAIN0 - 1], 60), set())
    n_ok = sum(1 for r in res if r[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    print(f"SELF-TEST {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")
    return 0 if n_ok == len(res) else 1


# ------------------------------------------------------------------------------------------------ CLI
def _sets(lst):
    over, extra = {}, dict(CS.EXTRA0)
    for s in lst or []:
        k, v = s.split("=", 1)
        if k in NS.NPHYS0:
            over[k] = v
        elif k in extra:
            extra[k] = float(v)
        else:
            raise ValueError(f"unknown param {k}")
    return over, extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--scan", action="store_true")
    m.add_argument("--ifix", nargs=2, metavar=("NEW_DIR", "OLD_DIR"))
    m.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--arm")
    ap.add_argument("--fmt", default="int4")
    ap.add_argument("--N", type=int)
    ap.add_argument("--cat")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--window", type=int, default=WIN_POINT)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", action="append")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.ifix:
            r = ifix(*a.ifix)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"no such dir {a.summarize}")
            print(json.dumps(summarize(a.summarize)))
            return 0
        for k in ("arm", "root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        if a.fmt not in ("int4", "fp8") or a.days not in (3, 7) or a.window not in (10, 60):
            raise ValueError("fmt int4|fp8, days 3|7, window 10|60")
        for p in (a.root, a.cache):
            if not os.path.exists(p):
                raise ValueError(f"missing {p}")
        over, extra = _sets(a.set)
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if a.run:
            if a.N is None:
                raise ValueError("--N is required")
            return run_walk(a.arm, a.fmt, a.N, over, extra, a.tag, a.root, a.cache, a.out, a.tau, a.window, a.days, seeds)
        if a.cat not in SCAN:
            raise ValueError(f"--cat must be one of {sorted(SCAN)}")
        return run_scan(a.arm, a.cat, a.fmt, over, extra, a.tag, a.root, a.cache, a.out, a.tau, a.window, a.days, seeds)
    except CS.AnchorError as e:
        print(f"ANCHOR-NOT-PLACEABLE: {e}", file=sys.stderr)
        return 3
    except (ValueError, ManifestError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
