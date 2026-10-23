#!/usr/bin/env python3
"""measured_p.py — re-read (cloud sim v6), (idle backfill) and (quiet giants OFF) at the
delays MEASURED. FORK BY IMPORT: no landed sim is edited. This sets the parents' module globals in-process (v6's P_GRID,
VARIANTS6 and the elastic layer's TAIL page-in rate; idle_backfill's P_GRID) and calls the parents' own functions. The static world, the
catalogue, the giants' G x 53.8 page-in and every other knob stay frozen.

Prereg (frozen before this file existed): (private report). On any divergence the
prereg wins.

Output contract:
  --self-test                              exit 0 + "SELF-TEST PASS n/n" (goldens G1-G8); exit 1 + FAIL lines.
  --run-v6 --cell D1|D2 --setting H22|H12|SAME [--seeds 0,1,2] [--tag T] [--epar X] --root R --cache C --out F
                                           WRITES (appends) ONE JSON line to F: v6's row + "s2897" provenance; exit 0. A failed
                                           in-job check (the parent's, or I-WORLD) exits 1.
  --run-ib --arm A --fmt F --N n [--tag T] [--set k=v ...] --root R --cache C --out F
                                           WRITES (appends) idle_backfill's rows at GRID_B to F; exit = the parent's run_job (3 = anchor).
  --ifix-v6 NEW_DIR [--ref REF_DIR]        read-only: I-WORLD + I-REPLAY-A (+ A0 vs REF when given); one JSON on stdout; exit 0/1.
  --ifix-ib NEW_DIR REF_DIR [--rep REP_DIR] read-only: B0 + I-PIND (+ I-REPLAY-B); one JSON on stdout; exit 0/1.
  --ifix-qg NEW_DIR REF_DIR                read-only: C0 (every NEW row == the committed row, timing stripped); exit 0/1.
  --summarize-v6 DIR --setting S           read-only: v6's frozen verdicts over GRID_A + the measured-p readings; REFUSES duplicate
                                           (tag, cell) rows (exit 2); one JSON on stdout; exit 0.
  --summarize-ib DIR                       read-only: B1 verdicts at GRID_B + Q-DOOR-S2 (the parent's summarize refuses duplicates);
                                           one JSON on stdout; exit 0.
  Bad input = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **v6's VARIANTS6 is built at import from P_GRID**, so patching P_GRID alone runs the 11 frozen points only (M1 / G3).
  2. **v6's run_cell guards the world's h2d against its OWN H2D_TAIL**, so a patched H2D_TAIL must not reach that guard; this file keeps
     the guard against the frozen world value (I-WORLD) and never touches v3 / v4's H2D_TAIL (M2 / G5).
  3. **v6's summarize keeps the last row per (tag, cell)**; this file refuses duplicates before calling it (G8).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREREG = "6b65763ab"
WORLD_H2D = 22.6                                   # the frozen world's tph h2d never changed here
SETTINGS = {"H22": 22.6, "H12": 12.4, "SAME": 1e9}  # frozen control measured / same-model bound (page-in = 0.02 s fixed only)
GRID_A = (0.0, 0.25, 0.271, 0.5, 0.75, 0.79, 0.842, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
FROZEN_A = (0.0, 0.25, 0.5, 0.79, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
Y_MAX, W_MAX = 0.8422, 1.3301                      # a2_split.json: yield max n=17, wake max n=22
P_OTHER = round(Y_MAX + W_MAX, 3)                  # 2.172: other-model extra TTFT = yield + wake
P_SAME_AMPLE, P_SAME_FULL = 0.271, 0.75
P_READ_A = 0.842                                   # the measured other-model yield (the v6 sims' p)
GRID_B = (0.05, P_SAME_AMPLE, P_SAME_FULL, 0.79, 1.6, P_OTHER, 3.04)
FROZEN_B = (0.05, 0.79, 1.6)
TIMING = {"elapsed_s", "sim_s", "overlay_s", "overlay5_s", "overlay6_s"}
DEC_ARMS = ("KA-1|QINH", "KA-5|QINH")


def _v6():
    import cloud_sim_v6 as V6
    return V6


def _ib():
    import idle_backfill as IB
    return IB


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def load_rows(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            with open(os.path.join(d, f)) as fh:
                out += [json.loads(l) for l in fh if l.strip()]
    return out


def by_unique(rows, keyf, what):
    """Refuse duplicate keys, never last-wins."""
    out = {}
    for r in rows:
        k = keyf(r)
        if k in out:
            raise ValueError(f"duplicate {what} row key {k}")
        out[k] = r
    return out


# ------------------------------------------------------------------------------------------------ Part A: v6
def configure_v6(setting, grid=GRID_A):
    V6 = _v6()
    V6.P_GRID = tuple(grid)
    var = {"STATIC": ("STATIC", None, "ADD")}
    for pol in V6.POL6:
        for c in ("ADD",) + V6.COMPS:
            for p in V6.P_GRID:
                var[V6.vname(pol, c, p)] = (pol, p, c)
    V6.VARIANTS6 = var
    V6.H2D_TAIL = SETTINGS[setting]
    return V6


def run_v6(cell, seeds, tag, root, cache, out, setting, epar=None):
    """v6.run_cell, with the world guard held at the frozen world value (prereg §5 I-WORLD)."""
    V6 = configure_v6(setting)
    if cell not in V6.CELLS6:
        raise ValueError(f"unknown cell {cell}")
    R = V6.V5.install()
    wd, row, rec, snap, r = V6.static_row_over(cell, seeds, tag, root, cache, epar)
    row["cell"] = cell
    row["epar"] = epar
    row["s2897"] = dict(setting=setting, h2d_tail=V6.H2D_TAIL, p_grid=list(V6.P_GRID), prereg=PREREG,
                        world_h2d=float(wd.tph["h2d"]))
    if snap is not None:
        if abs(wd.tph["h2d"] - WORLD_H2D) > 1e-9:
            raise RuntimeError(f"I-WORLD: the static world's tail h2d is {wd.tph['h2d']}, not {WORLD_H2D}")
        if epar is not None and abs(wd.np["e_par"] - float(epar)) > 1e-12:
            raise RuntimeError(f"the calibrated world reads e_par {wd.np['e_par']}, not {epar}")
        row["v6"] = V6.overlay6(wd, row, rec, snap, r, seeds, R)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


def _v6_key(r):
    return (r["tag"], r["cell"])


def ifix_v6(new_dir, ref_dir=None):
    """I-WORLD, I-REPLAY-A (main = rep_b = rep_c, timing stripped) and, with ref_dir, A0: every frozen-grid variant and the static row
    equal the committed rows."""
    new = by_unique(load_rows(new_dir), _v6_key, "v6")
    res, ok = {}, True
    for k, r in new.items():
        w = (r.get("s2897") or {}).get("world_h2d")
        if w is None or abs(w - WORLD_H2D) > 1e-9:
            res[f"I_WORLD|{k[0]}|{k[1]}"] = w
            ok = False
        bad = [c for c, v in (r.get("v6", {}).get("checks") or {}).items() if c.startswith("I_") and v is False]
        if bad:
            res[f"checks|{k[0]}|{k[1]}"] = bad
            ok = False
    for cell in ("D1", "D2"):
        base = new.get(("main", cell))
        for t in ("rep_b", "rep_c"):
            o = new.get((t, cell))
            if base is None or o is None:
                res[f"I_REPLAY|{cell}|{t}"] = "MISSING"
                ok = False
                continue
            a, b = strip(base), strip(o)
            for x in (a, b):
                x.pop("tag", None)
            same = a == b
            res[f"I_REPLAY|{cell}|{t}"] = "0 diffs" if same else "DIFFERS"
            ok = ok and same
    if ref_dir:
        ref = by_unique(load_rows(ref_dir), _v6_key, "ref v6")
        for k, o in sorted(ref.items()):
            n = new.get(k)
            if n is None:
                res[f"A0|{k[0]}|{k[1]}"] = "MISSING"
                ok = False
                continue
            sa = {kk: v for kk, v in strip(n).items() if kk not in ("v6", "s2897")}
            sb = {kk: v for kk, v in strip(o).items() if kk not in ("v6", "s2897")}
            diffs = [kk for kk in set(sa) | set(sb) if sa.get(kk) != sb.get(kk)]
            if "v6" in o:
                nv, ov = n["v6"], o["v6"]
                vd = [v for v in ov["variants"] if strip(nv["variants"].get(v)) != strip(ov["variants"][v])]
                for part in ("rep_req", "platform_burn", "platform_hours"):
                    if nv.get(part) != ov.get(part):
                        diffs.append(f"v6.{part}")
                cn = {c: v for c, v in nv["consts"].items() if c not in ("P_GRID", "n_variants")}
                co = {c: v for c, v in ov["consts"].items() if c not in ("P_GRID", "n_variants")}
                if cn != co:
                    diffs.append("v6.consts")
                bn = {c: v for c, v in nv["checks"].items() if c.startswith("I_")}
                bo = {c: v for c, v in ov["checks"].items() if c.startswith("I_")}
                if bn != bo:
                    diffs.append("v6.checks(I_*)")
                n_cmp = len(ov["variants"])
            else:
                vd, n_cmp = [], 0
            res[f"A0|{k[0]}|{k[1]}"] = dict(static_diffs=sorted(diffs)[:10], variant_diffs=vd[:10], variants_compared=n_cmp)
            ok = ok and not diffs and not vd
    res["rows"] = len(new)
    res["pass"] = bool(ok)
    return res


def reading(arm_row, p_star=P_READ_A):
    """prereg §3: MEETS iff P_MEET6 >= p* (p* on the grid); else FAILS with the first failing grid point."""
    pm = arm_row["p_meet6"]
    if pm is not None and pm >= p_star - 1e-12:
        return dict(reading="MEETS", p_meet6=pm)
    brk = arm_row["p_break6"]
    return dict(reading="FAILS", p_meet6=pm, first_fail=brk)


def summarize_v6(d, setting):
    V6 = configure_v6(setting)
    rows = load_rows(d)
    by_unique(rows, _v6_key, "v6")                 # refuse duplicates before the parent's last-wins dict
    s = V6.summarize(d)
    out = dict(setting=setting, h2d_tail=SETTINGS[setting], grid=list(GRID_A), parent=s, readings={})
    for ck, c in s["cells"].items():
        for S in ("S2", "S10"):
            for arm in DEC_ARMS + ("KA-1|ADD", "KA-5|ADD"):
                a = c["arms"][f"{S}|{arm}"]
                bp = a["by_p"]
                out["readings"][f"{ck}|{S}|{arm}"] = dict(
                    reading(a), p_break6=a["p_break6"],
                    meet={p: bp[p]["meet"] for p in ("0", "0.25", "0.271", "0.5", "0.75", "0.79", "0.842", "1", "5", "10")},
                    fail_at={p: bp[p]["fail"] for p in ("0", "0.271", "0.5", "0.75", "0.79", "0.842")},
                    mww={p: bp[p]["mww"] for p in ("0", "0.271", "0.75", "0.79", "0.842")},
                    idle=bp["0.842"]["idle"])
    return out


# ------------------------------------------------------------------------------------------------ Part B: idle backfill
def configure_ib():
    IB = _ib()
    IB.P_GRID = tuple(GRID_B)
    return IB


def run_ib(arm, fmt, N, tag, root, cache, out, sets):
    IB = configure_ib()
    over, extra = IB._parse_sets(sets)
    return IB.run_job(arm, fmt, N, over, extra, tag, root, cache, out)


def _ib_key(r):
    return (r["arm"], r["fmt"], r["N"], r.get("tag", ""), json.dumps(r["nphys"], sort_keys=True),
            json.dumps(r["extra"], sort_keys=True), r.get("n"))


def _ib_frozen(r):
    x = strip(r)
    if "bf_slo" in x:
        x["bf_slo"] = {k: v for k, v in x["bf_slo"].items() if float(k.split("|")[2]) in FROZEN_B}
    return x


def ifix_ib(new_dir, ref_dir, rep_dir=None):
    new = by_unique(load_rows(new_dir), _ib_key, "ib")
    ref = by_unique(load_rows(ref_dir), _ib_key, "ref ib")
    res, ok = {}, True
    n_cmp, diffs, pind = 0, [], 0
    for k, r in sorted(new.items()):
        o = ref.get(k)
        if o is None:
            diffs.append(f"MISSING {k[0]} {k[2]} n={k[6]}")
            continue
        n_cmp += 1
        a, b = _ib_frozen(r), _ib_frozen(o)
        if a != b:
            diffs.append(f"{k[0]} N={k[2]} n={k[6]}: " + ",".join(sorted(kk for kk in set(a) | set(b) if a.get(kk) != b.get(kk)))[:200])
        if r.get("bf") != o.get("bf"):
            pind += 1
    res["B0"] = dict(rows_compared=n_cmp, diffs=diffs[:10], n_diffs=len(diffs))
    res["I_PIND"] = dict(bf_rows_differing=pind)
    ok = not diffs and pind == 0 and n_cmp > 0
    if rep_dir:
        rep = by_unique(load_rows(rep_dir), _ib_key, "rep ib")
        rd = []
        for k, r in sorted(rep.items()):
            km = (k[0], k[1], k[2], "main", k[4], k[5], k[6])
            m = new.get(km)
            a, b = strip(r), strip(m) if m else None
            if b is None:
                rd.append(f"MISSING main for {k}")
                continue
            a.pop("tag", None)
            b.pop("tag", None)
            if a != b:
                rd.append(f"{k[0]} N={k[2]} {k[3]} n={k[6]}")
        res["I_REPLAY_B"] = dict(rows=len(rep), diffs=rd[:10], n_diffs=len(rd))
        ok = ok and not rd and len(rep) > 0
    res["pass"] = bool(ok)
    return res


def door(safe_ample, safe_full):
    """prereg §5 Q-DOOR-S2."""
    if not safe_ample:
        return "DOES-NOT-RESCUE"
    return "RESCUES" if safe_full else "RESCUES-WITH-AMPLE-KV-ONLY"


def summarize_ib(d):
    IB = configure_ib()
    s = IB.summarize(d)                            # raises ValueError on a duplicate row (the parent's own check)
    rows = by_unique(load_rows(d), _ib_key, "ib")
    out = dict(grid=list(GRID_B), cells={}, door={})
    pk = f"CONS|1|{int(IB.POINT['lam'])}"
    for one in s:                                  # the parent returns [dict(cell, key=[fmt, N, tag]), ...]
        fmt, N, tag = one["key"]
        for arm, a in one["arms"].items():
            for S, e in a.items():
                nm = e.get("n_match")
                if nm is None:
                    continue
                r0 = [r for kk, r in rows.items() if kk[0] == arm and kk[1] == fmt and kk[2] == N and kk[3] == tag and kk[6] == nm]
                if len(r0) != 1:
                    raise ValueError(f"n_match row for {arm} {fmt} {N} {tag} n={nm}: {len(r0)} rows")
                r0 = r0[0]
                if "bf" not in r0:
                    continue
                fr = r0["bf"][pk]["frac"]
                ent = dict(n_match=nm, frac_rng=[min(x or 0.0 for x in fr), max(x or 0.0 for x in fr)], by_p={})
                for p in GRID_B:
                    sl = r0["bf_slo"][f"CONS|1|{p}|{float(S)}"]
                    ent["by_p"][f"{p:g}"] = dict(verdict=IB.bf_verdict(fr, sl["safe"]), safe_all=all(sl["safe"]),
                                                 safe_reps=sum(1 for x in sl["safe"] if x), at_risk_max=max(sl["at_risk"]),
                                                 at_risk_rng=[min(sl["at_risk"]), max(sl["at_risk"])],
                                                 hit_share_rng=[min(sl["hit_share"]), max(sl["hit_share"])])
                out["cells"][f"{fmt}|{N}|{tag}|{arm}|S{float(S):g}"] = ent
    for arm in ("P", "BS"):
        e = out["cells"].get(f"int4|128|main|{arm}|S2")
        if e:
            out["door"][arm] = dict(safe_0_271=e["by_p"]["0.271"]["safe_all"], safe_0_75=e["by_p"]["0.75"]["safe_all"])
    if set(out["door"]) == {"P", "BS"}:
        out["door"]["Q_DOOR_S2"] = door(all(out["door"][a]["safe_0_271"] for a in ("P", "BS")),
                                         all(out["door"][a]["safe_0_75"] for a in ("P", "BS")))
    return out


# ------------------------------------------------------------------------------------------------ Part C: quiet giants
def ifix_qg(new_dir, ref_dir):
    def rows_of(d):
        out = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl") and f.startswith("F_"):
                with open(os.path.join(d, f)) as fh:
                    out[f] = [strip(json.loads(l)) for l in fh if l.strip()]
        return out
    new, ref = rows_of(new_dir), rows_of(ref_dir)
    diffs = [f for f in new if new[f] != ref.get(f)]
    missing = sorted(set(ref) - set(new))
    return dict(files_compared=len(new), rows_compared=sum(len(v) for v in new.values()), diffs=diffs[:10], n_diffs=len(diffs),
                missing=missing[:10], pass_=not diffs and not missing and len(new) > 0)


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    import tempfile
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))

    V6 = configure_v6("H12")
    V3, V4, T = V6.V3, V6.V4, V6.T
    chk("G1 page_in(25.2, 12.4)", round(V3.page_in(25.2, 12.4), 4), 2.2021)
    chk("G1 page_in(25.2, 22.6) = v4 GV2b", round(V3.page_in(25.2, 22.6), 4), 1.2173)
    chk("G1 page_in(25.2, SAME)", round(V3.page_in(25.2, SETTINGS["SAME"]), 4), 0.02)
    chk("G2 P_OTHER", P_OTHER, 2.172)
    chk("G2 P_BIG", round(P_READ_A + round(V3.page_in(25.2, 12.4), 4), 2), 3.04)
    chk("G3 GRID_A sorted, 14, frozen subset", [list(GRID_A) == sorted(GRID_A), len(GRID_A), set(FROZEN_A) <= set(GRID_A)],
        [True, 14, True])
    chk("G3 frozen grid = v4's", list(FROZEN_A), list(V4.P_GRID))
    chk("G3 113 variants", len(V6.VARIANTS6), 113)
    chk("G3 vname 0.842", "KA-1|QINH|p0.842" in V6.VARIANTS6 and V6.VARIANTS6["KA-1|QINH|p0.842"] == ("KA-1", 0.842, "QINH"), True)
    chk("G3 GRID_B", GRID_B, (0.05, 0.271, 0.75, 0.79, 1.6, 2.172, 3.04))
    chk("G4 keys", [T._key(10 - 2.172), T._key(2 - 0.271), T._key(2 - 0.75), T._key(2 - 2.172)], [783, 173, 125, -17])
    chk("G5 world untouched (v6 12.4, v3 / v4 22.6)", [V6.H2D_TAIL, V3.H2D_TAIL, V4.H2D_TAIL], [12.4, 22.6, 22.6])
    V6b = configure_v6("H22")
    chk("G5 H22 restores 22.6", V6b.H2D_TAIL, 22.6)
    g = {"p_meet6": 0.271, "p_break6": [0.5, "PM6-p99"]}
    chk("G6 reading FAILS", reading(g), dict(reading="FAILS", p_meet6=0.271, first_fail=[0.5, "PM6-p99"]))
    chk("G6 reading MEETS", reading({"p_meet6": 30.0, "p_break6": None})["reading"], "MEETS")
    chk("G6 reading NONE", reading({"p_meet6": None, "p_break6": [0.0, "PM6-MWW"]})["reading"], "FAILS")
    chk("G7 door", [door(True, True), door(True, False), door(False, True), door(False, False)],
        ["RESCUES", "RESCUES-WITH-AMPLE-KV-ONLY", "DOES-NOT-RESCUE", "DOES-NOT-RESCUE"])
    with tempfile.TemporaryDirectory() as td:
        row = dict(tag="main", cell="D1")
        with open(os.path.join(td, "a.jsonl"), "w") as fh:
            fh.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        try:
            by_unique(load_rows(td), _v6_key, "v6")
            chk("G8 duplicate v6 rows refused", "accepted", "refused")
        except ValueError:
            chk("G8 duplicate v6 rows refused", "refused", "refused")
        ib_row = dict(arm="P", fmt="int4", N=128, tag="main", nphys={}, extra={}, n=8)
        with open(os.path.join(td, "a.jsonl"), "w") as fh:
            fh.write(json.dumps(ib_row) + "\n" + json.dumps(ib_row) + "\n")
        try:
            by_unique(load_rows(td), _ib_key, "ib")
            chk("G8 duplicate ib rows refused", "accepted", "refused")
        except ValueError:
            chk("G8 duplicate ib rows refused", "refused", "refused")
    IB = configure_ib()
    chk("G3 idle_backfill P_GRID", IB.P_GRID, GRID_B)
    chk("G7b bf_verdict frozen semantics", [IB.bf_verdict([0.96] * 21, [True] * 21), IB.bf_verdict([0.96] * 21, [True] * 20 + [False])],
        ["RECOVERS-MOST", "SLO-COST"])
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
    m.add_argument("--run-v6", action="store_true")
    m.add_argument("--run-ib", action="store_true")
    m.add_argument("--ifix-v6", metavar="NEW_DIR")
    m.add_argument("--ifix-ib", nargs=2, metavar=("NEW_DIR", "REF_DIR"))
    m.add_argument("--ifix-qg", nargs=2, metavar=("NEW_DIR", "REF_DIR"))
    m.add_argument("--summarize-v6", metavar="DIR")
    m.add_argument("--summarize-ib", metavar="DIR")
    ap.add_argument("--ref")
    ap.add_argument("--rep")
    ap.add_argument("--cell")
    ap.add_argument("--setting", choices=sorted(SETTINGS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epar", type=float)
    ap.add_argument("--arm")
    ap.add_argument("--fmt")
    ap.add_argument("--N", type=int)
    ap.add_argument("--set", action="append")
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        if a.self_test:
            return self_test()
        if a.ifix_v6:
            r = ifix_v6(a.ifix_v6, a.ref)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.ifix_ib:
            r = ifix_ib(a.ifix_ib[0], a.ifix_ib[1], a.rep)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.ifix_qg:
            r = ifix_qg(*a.ifix_qg)
            print(json.dumps(r))
            return 0 if r["pass_"] else 1
        if a.summarize_v6:
            if not a.setting:
                raise ValueError("--summarize-v6 needs --setting")
            print(json.dumps(summarize_v6(a.summarize_v6, a.setting)))
            return 0
        if a.summarize_ib:
            print(json.dumps(summarize_ib(a.summarize_ib)))
            return 0
        for k in ("root", "cache", "out"):
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        if a.run_v6:
            if not (a.cell and a.setting):
                raise ValueError("--run-v6 needs --cell and --setting")
            V6 = _v6()
            seeds = tuple(int(x) for x in a.seeds.split(","))
            if len(seeds) * V6.NDAYS != V6.NREPS:
                raise ValueError(f"{len(seeds)} seeds x {V6.NDAYS} days != {V6.NREPS} replicates")
            if a.epar is not None and a.epar not in V6.EPAR_CAL.values():
                raise ValueError(f"--epar must be one of {sorted(V6.EPAR_CAL.values())}")
            return run_v6(a.cell, seeds, a.tag, a.root, a.cache, a.out, a.setting, a.epar)
        if a.run_ib:
            if not (a.arm and a.fmt and a.N is not None):
                raise ValueError("--run-ib needs --arm, --fmt and --N")
            return run_ib(a.arm, a.fmt, a.N, a.tag, a.root, a.cache, a.out, a.set)
        raise ValueError("no mode")
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
