#!/usr/bin/env python3
"""cloud_sim_v6.py — does 's thin 2 s margin survive QUEUEING on a woken engine? v5 composed each wake's delay
additively per request (a woken engine's backlog never delayed the requests behind it). v6 puts every request of a wake window on an
FCFS queue at the slot's INHERITED service rate (the static sims' own fluid C), splits the error budget into PLATFORM burn (static
over S) and ELASTIC burn (requests the wakes push over S), gates on the elastic burn with a multi-window fast-burn alert, and re-runs
both cells under calibrated giant compute.

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. FORK BY IMPORT: v5's recorder, run_variant (the ADD control, v5 exactly), pairing and gates run UNCHANGED; no inherited file is
edited (I-FIXV5 proves the ADD arm reproduces committed rows).

Output contract:
  --self-test                        exit 0 + "SELF-TEST PASS n/n" (v5's own self-test + GQ1-GQ6, GQ3b); exit 1 + FAIL lines.
  --premise --cell D1|D2 --root R --cache C
                                     TRAIN-ONLY read (seed 0): per-request work seconds, TRAIN p99 headroom, at-risk counts; one JSON
                                     object on stdout; exit 0. Prints no held-out number.
  --run --cell D1|D2 [--seeds 0,1,2] [--tag T] [--epar X] --root R --cache C --out F
                                     WRITES (appends) ONE JSON line to F: static row + "v6" (89 variants, in-job I-checks);
                                     exit 0. A failed in-job I-check exits 1. --epar runs the calibrated world (catalogue = point's).
  --ifix V6_DIR V5_DIR               read-only: I-FIXV5, I-REPLAY, in-job checks; one JSON on stdout; exit 0 PASS / 1 FAIL.
  --summarize DIR                    read-only: prereg §5 verdicts over DIR/*.jsonl; one JSON object on stdout; exit 0.
  Bad input (unknown cell, missing file) or a held-out manifest mismatch = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **A calibrated world re-plans the catalogue** (`V.catalogue` runs plan P@128 inside the world it is given), so CAT-128P must be
     read from the POINT world, or I-CAT refuses the run (premise read of `cloud_sim_v2.catalogue`).
  2. **v5's run_variant writes the held / wake arrays into pol_cache**; the queue windows must read the SAME arrays (host-blocked
     slots are held static there), never recompute them.
  3. **`ssh host '... &'` holds the session open**: launch with `ssh -n` + `setsid -f`.
"""
import argparse, collections, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import cloud_sim_v2 as V     # noqa: E402  (not edited)
import cloud_sim_v3 as V3    # noqa: E402  (not edited)
import cloud_sim_v4 as V4    # noqa: E402  (not edited)
import cloud_sim_v5 as V5    # noqa: E402  (not edited)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
SLOS = CS.SLOS
NREPS = V3.NREPS
INF = V3.INF
P_GRID = V4.P_GRID
P_POINT = V4.P_POINT
H2D_TAIL = V4.H2D_TAIL
WARM = V4.WARM
C_REF = CS.C_REF
ACTIVE_REF_B = CS.ACTIVE_REF_B
B_E = 0.001                          # prereg §5: the elastic layer may spend 10 % of a 1 % budget (EBR <= 0.001, inclusive)
FAST_BURN = 14.4                     # prereg §4: SRE-workbook page pair (2 % of a budget in 1 h)
WIN_LONG, WIN_SHORT = 60, 5
TENANT_N = 100                       # descriptive tenant fast-burn hours: model-hours with n >= 100
POL6 = ("KA-1", "KA-5")
COMPS = ("QINH", "QGROSS", "QPRE")
DEC_COMP = "QINH"
EPAR_CAL = {"CALP": 5.0, "CALD": 1.07}
CELLS6 = ("D1", "D2")
TIMING6 = set(V5.TIMING5) | {"overlay6_s"}
V5_FIELDS = ("idle_el", "held_gpuh", "peak_held", "busy_day", "held_day", "wakes", "wake_hit", "mww", "n_split", "hold_slots",
             "hold_gpus")
V5_XFIELDS = ("p95_pass", "p99_pass", "p99_scope", "eb", "eb_elig", "mct_scoped", "mct_all", "over_by_wake")


def vname(pol, comp, p):
    return f"{pol}|{'BASE' if comp == 'ADD' else comp}|p{p:g}"


VARIANTS6 = {"STATIC": ("STATIC", None, "ADD")}
for _pol in POL6:
    for _c in ("ADD",) + COMPS:
        for _p in P_GRID:
            VARIANTS6[vname(_pol, _c, _p)] = (_pol, _p, _c)


# ------------------------------------------------------------------------------------------------ the queue (prereg §3)
def cell_work(it, ot, n, ab):
    """per-request work (work-tokens) of a cell: QINH / QGROSS (it + ot), QPRE prefill only; the static sims' own unit."""
    wq = (it + ot) / n * ab / ACTIVE_REF_B
    wp = it / n * ab / ACTIVE_REF_B
    return wq, wp


def giant_c(g, m_h, e_par):
    """a giant group's / anchor replica's service rate (work tok/s): the static runners' C (`World.cg`)."""
    return g * C_REF * m_h * e_par


def lindley(rel, svc):
    """FCFS single server, empty at the start, in the given order: -> start times. start_j = max(rel_j, finish_{j-1})."""
    rel = np.asarray(rel, float)
    svc = np.asarray(svc, float)
    if not len(rel):
        return rel
    prev = np.concatenate([[0.0], np.cumsum(svc)[:-1]])
    return prev + np.maximum.accumulate(rel - prev)


def _queue_waits(rel, svc, keys):
    """waits (start - release, >= 0) in the original order; FCFS by lexsort(keys) (last key primary)."""
    o = np.lexsort(keys)
    q = np.empty(len(rel))
    q[o] = lindley(rel[o], svc[o]) - rel[o]
    return np.maximum(q, 0.0)


def queue_window(kind, T0, p, A, page, trig, a, uinv, uu, svc_q, svc_p, cid, rank):
    """one wake window -> (res, d_inh, d_gross, d_pre, burst, nclip), arrays aligned with the requests a (abs s).
    kind 'tail': engines page in at their first request, one at a time (V3.wake_ready); 'giant' / 'anchor': only `trig` pages in.
    tail and giant queue every request of the window; an anchor replica queues only its pre-ready requests (prereg §3)."""
    gpu = T0 + p
    if kind == "tail":
        ready = V3.wake_ready(A, page, gpu)
        r_arr = np.array([ready[u] for u in uu], float)[uinv]
    else:
        r_arr = np.full(len(a), gpu + page[trig])
    lt = a < r_arr
    res = np.where(lt, r_arr - a, 0.0)
    rho = np.where(lt, r_arr, a)
    inq = lt if kind == "anchor" else np.ones(len(a), bool)
    d_inh, d_gross, d_pre = res.copy(), res.copy(), res.copy()
    nclip = 0
    idx = np.flatnonzero(inq)
    if len(idx):
        aq, rq, cq, kq = a[idx], rho[idx], cid[idx], rank[idx]
        for svc, mode in ((svc_q, "q"), (svc_p, "p")):
            s = np.asarray(svc, float)[idx]
            Qw = _queue_waits(rq, s, (kq, cq, aq, rq))
            Q0 = _queue_waits(aq, s, (kq, cq, aq))
            ext_inh = np.maximum(0.0, Qw - Q0)
            if mode == "q":
                d_inh[idx] += ext_inh
                d_gross[idx] += Qw
                nclip = int(np.sum(Qw < Q0 - 1e-12))
            else:
                d_pre[idx] += ext_inh
    return res, d_inh, d_gross, d_pre, int(lt.sum()), nclip


# ------------------------------------------------------------------------------------------------ burn (prereg §4 / §5)
def elastic_cross(s, e, sk):
    """requests the elastic layer pushes over S: static key <= S and elastic key > S (a rank already over S is platform burn)."""
    s = np.asarray(s, np.int64)
    e = np.asarray(e, np.int64)
    return int(((s <= sk) & (e > sk)).sum())


def ebe_gate(over, req, alerts):
    """EBE: on every replicate EBR = over / req <= B_E (inclusive; integer form 1000 * over <= req) and 0 alert minutes."""
    if not (len(over) == len(req) == len(alerts) == NREPS):
        raise ValueError(f"I-REPS: {len(over)} / {len(req)} / {len(alerts)} EBE replicates, need {NREPS}")
    share_ok = all(1000 * int(o) <= int(q) for o, q in zip(over, req))
    return "EBE-HOLDS" if share_ok and all(int(x) == 0 for x in alerts) else "EBE-FAILS"


def alert_minutes(E, N):
    """bool per minute: burn rate > FAST_BURN over BOTH the trailing 60- and 5-minute windows (inclusive; shorter at the start)."""
    E = np.asarray(E, float)
    N = np.asarray(N, float)
    cE = np.concatenate([[0.0], np.cumsum(E)])
    cN = np.concatenate([[0.0], np.cumsum(N)])
    t = np.arange(len(E))

    def burn(L):
        lo = np.maximum(t + 1 - L, 0)
        se, sn = cE[t + 1] - cE[lo], cN[t + 1] - cN[lo]
        return np.where(sn > 0, se / np.where(sn > 0, sn, 1.0), 0.0) / B_E
    bl, bs_ = burn(WIN_LONG), burn(WIN_SHORT)
    al = (bl > FAST_BURN) & (bs_ > FAST_BURN)
    return al


def alert_reps(E, N):
    """per LODO replicate: alert minutes inside the six days != d."""
    al = alert_minutes(E, N)
    per = al[:NDAYS * DAY_MIN].reshape(NDAYS, DAY_MIN).sum(axis=1)
    return V5.lodo_sum([int(x) for x in per])


def hour_bins(over, nmh):
    """over: {model-hour: count}, nmh: {model-hour: n} -> model-hours (n >= 1) by over count {0, 1, 2-4, 5-9, >= 10} + tenant fast."""
    b = [0, 0, 0, 0, 0]
    fast = 0
    mx = 0
    for k, n in nmh.items():
        c = int(over.get(k, 0))
        b[0 if c == 0 else 1 if c == 1 else 2 if c <= 4 else 3 if c <= 9 else 4] += 1
        mx = max(mx, c)
        if n >= TENANT_N and c / float(n) > FAST_BURN * B_E:
            fast += 1
    return dict(bins=b, tenant_fast=fast, max_mh=mx)


# ------------------------------------------------------------------------------------------------ windows (prereg §3)
def build_windows(pol, slots, slot_cells, tr, first, offs, pol_cache, wq_cell, wp_cell, Cs):
    """per slot, per wake: the window's request arrays (p-independent). Reads v5's held / wake arrays from pol_cache."""
    out = []
    for s_i, sl in enumerate(slots):
        held, wake = pol_cache[(pol, s_i, False)]
        wk_t = np.flatnonzero(wake)
        if not len(wk_t):
            continue
        mins, ids = slot_cells[s_i]
        kind = sl["kind"]
        gb = H2D_TAIL if kind == "tail" else sl["gbps"]
        for t in wk_t:
            te = V3.window_end(pol, held, int(t))
            a_, z_ = np.searchsorted(mins, t, "left"), np.searchsorted(mins, te, "right")
            cl = [(int(mins[q]), float(first[ids[q]]), int(tr.mid[ids[q]]), sl["unit_of"][int(tr.mid[ids[q]])], ids[q])
                  for q in range(a_, z_)]
            at = [c for c in cl if c[0] == t]
            if not at:
                raise RuntimeError(f"I-WAKE: no cell of slot {s_i} starts at wake minute {t}")
            T0 = min(60.0 * c[0] + c[1] for c in at)
            if kind == "tail":
                A = {}
                for c in cl:
                    tt = 60.0 * c[0] + c[1]
                    if c[3] not in A or tt < A[c[3]]:
                        A[c[3]] = tt
                use = cl
                trig = None
            else:
                trc = min(at, key=lambda c: (60.0 * c[0] + c[1], c[4]))
                trig = trc[3]
                A = {trig: 60.0 * trc[0] + trc[1]}
                use = [c for c in cl if c[3] == trig]
            page = {u: V3.page_in(sl["units"][u], gb) for u in A}
            aa, un, ci, rk, wq, wp, blocks = [], [], [], [], [], [], []
            lo = 0
            for c in use:
                ts = 60.0 * c[0] + offs(c[4])
                n = len(ts)
                aa.append(ts)
                un.append(np.full(n, c[3]))
                ci.append(np.full(n, c[4], np.int64))
                rk.append(np.arange(n, dtype=np.int64))
                wq.append(np.full(n, wq_cell[c[4]]))
                wp.append(np.full(n, wp_cell[c[4]]))
                blocks.append((int(c[4]), lo, lo + n))
                lo += n
            unit = np.concatenate(un)
            uu, uinv = np.unique(unit, return_inverse=True)
            C = Cs[s_i]
            out.append(dict(kind=kind, s=s_i, T0=T0, A=A, page=page, trig=trig, a=np.concatenate(aa), uu=uu.tolist(), uinv=uinv,
                            cid=np.concatenate(ci), rank=np.concatenate(rk), sq=np.concatenate(wq) / C, sp=np.concatenate(wp) / C,
                            blocks=blocks))
    return out


def queue_pass(windows, p):
    """one (pol, p): -> dict comp -> {cell id: delays s} (max over windows), the residual dict (v5's semantics), stats."""
    cds = {c: {} for c in COMPS}
    resd = {}
    bursts = []
    kmax = {c: collections.defaultdict(float) for c in COMPS}
    nclip = 0
    for w in windows:
        res, di, dg, dp, burst, nc = queue_window(w["kind"], w["T0"], p, w["A"], w["page"], w["trig"], w["a"], w["uinv"], w["uu"],
                                                  w["sq"], w["sp"], w["cid"], w["rank"])
        bursts.append(burst)
        nclip += nc
        for comp, arr in (("QINH", di), ("QGROSS", dg), ("QPRE", dp)):
            ex = arr - res
            if len(ex):
                kmax[comp][w["kind"]] = max(kmax[comp][w["kind"]], float(ex.max()))
        for cid, lo, hi in w["blocks"]:
            r_ = res[lo:hi]
            if r_[0] > 0.0:                                   # v3 returns a cell iff its first rank arrives before ready
                prev = resd.get(cid)
                resd[cid] = r_ if prev is None else np.maximum(prev, r_)
            for comp, arr in (("QINH", di), ("QGROSS", dg), ("QPRE", dp)):
                x = arr[lo:hi]
                if (x > 0.0).any():
                    prev = cds[comp].get(cid)
                    cds[comp][cid] = x if prev is None else np.maximum(prev, x)
    return cds, resd, dict(bursts=bursts, nclip=nclip, kmax={c: dict(v) for c, v in kmax.items()})


def _cs(x):
    return np.round(np.asarray(x, float) * 100.0).astype(np.int64)


# ------------------------------------------------------------------------------------------------ the v6 overlay
def _new_res():
    return dict(idle_el=[], held_gpuh=[], peak_held=[], busy_day=[], held_day=[], wakes=[], wake_hit=[], mww=[], n_split=[],
                hold_slots=[], hold_gpus=[], burst_max=[], burst_p99=[], n_windows=[], nclip=[], kmax=[],
                x={str(S): dict(p95_pass=[], p99_pass=[], p99_scope=[], eb=[], eb_elig=[], mct_scoped=[], mct_all=[], over_by_wake=[],
                                alert=[], mww_s=[], hours=[], p99_fail={}) for S in SLOS})


def score_variant(o, cd, busy, ctx, seed_i):
    """v5's per-variant block (identity pairing, exact stats, PM inputs, EB) + v6's burn, alerts, hourly bins, scoped MWW."""
    busy_d, held_d, wake_d, held_tot = busy
    tr, R, sc, hosted, st_stats, st_p99, rst, allm, sk = (ctx[k] for k in ("tr", "R", "sc", "hosted", "st_stats", "st_p99", "rst",
                                                                          "allm", "sk"))
    o["idle_el"].extend(V3.lodo(busy_d, held_d))
    H_ = held_d.sum()
    o["held_gpuh"].extend([(H_ - held_d[d]) / 60.0 for d in range(NDAYS)])
    o["wakes"].extend([int(wake_d.sum() - wake_d[d]) for d in range(NDAYS)])
    V4.peak_check(held_tot, ctx["tot"])
    o["peak_held"].extend(V4.peak_reps(held_tot, WARM))
    o["busy_day"].append([int(x) for x in busy_d])
    o["held_day"].append([int(x) for x in held_d])
    o["hold_slots"].append(0)
    o["hold_gpus"].append(0)
    moves = collections.defaultdict(lambda: [[] for _ in range(NDAYS)])
    hit_d = np.zeros(NDAYS, np.int64)
    dmax = np.zeros(NDAYS, np.int64)
    nspl = np.zeros(NDAYS, np.int64)
    per_s = {s: dict(mct_s=np.zeros(NDAYS, np.int64), over=np.zeros(NDAYS, np.int64), dmax_s=np.zeros(NDAYS, np.int64),
                     E=np.zeros(WEEK_MIN, np.int64), mh=collections.Counter()) for s in sk}
    mct_a = np.zeros(NDAYS, np.int64)
    for cid, dd in cd.items():
        dc = _cs(dd)
        if not (dc > 0).any():
            continue
        m, mi = int(tr.mid[cid]), int(tr.minute[cid])
        d = (mi - HELD) // DAY_MIN
        if m not in sc:
            raise RuntimeError(f"a delay on a model that is not hosted: {m}")
        s_, split = R.keys_of(sc[m][1][d], mi)
        if s_ is None:
            raise RuntimeError(f"I-PERREQ: delayed cell {cid} has no static record")
        moves[m][d].extend(V5.pair_cell(s_, dc))
        pos = dc[dc > 0]
        hit_d[d] += len(pos)
        dmax[d] = max(dmax[d], int(pos.max()))
        if split:
            nspl[d] += len(pos)
        k = dc > 0
        sv = np.asarray(s_, np.int64)[k]
        ev = np.where(sv >= INF, INF, sv + dc[k])
        mct_a[d] = max(mct_a[d], int(np.minimum(ev, 10 ** 9).max()))
        for s, kk_ in sk.items():
            P = per_s[s]
            sc_ = sv <= kk_
            if sc_.any():
                P["mct_s"][d] = max(P["mct_s"][d], int(np.minimum(ev[sc_], 10 ** 9).max()))
                P["dmax_s"][d] = max(P["dmax_s"][d], int(dc[k][sc_].max()))
            cnt = elastic_cross(sv, ev, kk_)
            if cnt:
                P["over"][d] += cnt
                P["E"][mi - HELD] += cnt
                P["mh"][m * 1000 + (mi - HELD) // 60] += cnt
    o["wake_hit"].extend(V5.lodo_sum(hit_d))
    o["mww"].extend(V4.mww_reps(dmax))
    o["n_split"].extend(V5.lodo_sum(nspl))
    ost = dict(st_stats)
    ex99 = {}
    for m in moves:
        ost[m], ex99[m] = V5.compose_exact(sc[m][1], moves[m])
    for x in SLOS:
        s = str(x)
        xs = o["x"][s]
        P = per_s[s]
        sr = CS.score_reps(hosted, ost, rst, allm, x)
        xs["p95_pass"].extend(sr["pass"])
        for d in range(NDAYS):
            mods = [(ost[m]["n"][d], st_p99[m][d], ex99[m][d] if m in ex99 else st_p99[m][d],
                     ex99[m][d] if m in ex99 else st_p99[m][d]) for m in hosted]
            pw, _, ns, _ = V4.p99_rep_pass(mods, x)
            xs["p99_pass"].append(pw)
            xs["p99_scope"].append(ns)
            for m in ex99:
                if ost[m]["n"][d] >= V4.P99_FLOOR and st_p99[m][d] is not None and st_p99[m][d] <= x and ex99[m][d] > x:
                    k8 = tr.models[m][0][:8]
                    f = xs["p99_fail"].setdefault(k8, [0, 0.0])
                    f[0] += 1
                    f[1] = max(f[1], ex99[m][d] if ex99[m][d] != math.inf else 1e9)
            xs["eb"].append(V5.eb_share([ost[m]["ok"][s][d] for m in hosted], [ost[m]["n"][d] for m in hosted]))
            fe = [m for m in hosted if rst[m]["n"][d] >= 20 and rst[m]["p95"][d] <= x]
            xs["eb_elig"].append(V5.eb_share([ost[m]["ok"][s][d] for m in fe], [ost[m]["n"][d] for m in fe]))
        xs["mct_scoped"].extend([y / 100.0 for y in V5.lodo_max(P["mct_s"])])
        xs["mct_all"].extend([y / 100.0 for y in V5.lodo_max(mct_a)])
        xs["over_by_wake"].extend(V5.lodo_sum(P["over"]))
        xs["alert"].extend(alert_reps(P["E"], ctx["Nmin"]))
        xs["mww_s"].extend(V4.mww_reps(P["dmax_s"]))
        xs["hours"].append(dict(seed_i=seed_i, el=hour_bins(P["mh"], ctx["nmh"])))
    return moves


def overlay6(wd, row, rec, snap, r, seeds, R, variants=None):
    variants = variants or list(VARIANTS6)
    cx = wd.cx
    tr = cx["tr"]
    M = cx["M"]
    adm = [cx["cid_m"][c] for c in snap["admitted"]]
    hosted = [wd.am] + adm
    allm = list(range(M))
    rst = {m: wd.ref_stats(m, r) for m in allm}
    rd = wd.req_day()
    held_mask = tr.minute >= HELD
    kk = np.flatnonzero(held_mask)
    train = ~held_mask
    t_start = time.time()
    sk = {str(S): T._key(S) for S in SLOS}
    res = {v: _new_res() for v in variants}
    checks = dict(I_PERREQ=True, I_CELLUNIQ=True, I_EXACT0=True, I_HELD=True, I_WAKE=True, I_RES=True, I_QGE=True, I_ZW=True,
                  I_EBE0=True, I_DEN=True, I_HOST_parked_blocked=0, res_compared=0, qge_compared=0, qge_violations=0, peri_cells=0,
                  split_cells=0, n_train_unready=0)
    keyc = tr.mid[kk].astype(np.int64) * (WEEK_MIN + HELD + 10 ** 6) + tr.minute[kk].astype(np.int64)
    if len(np.unique(keyc)) != len(kk):
        checks["I_CELLUNIQ"] = False
        raise RuntimeError("I-CELLUNIQ: (model, minute) is not unique over held-out cells")
    hk = kk[np.isin(tr.mid[kk], hosted)]
    Nmin = np.bincount(tr.minute[hk] - HELD, weights=tr.n[hk].astype(float), minlength=WEEK_MIN)[:WEEK_MIN].astype(np.int64)
    mhk = tr.mid[hk].astype(np.int64) * 1000 + (tr.minute[hk] - HELD) // 60
    u_, inv_ = np.unique(mhk, return_inverse=True)
    nmh = dict(zip(u_.tolist(), np.bincount(inv_, weights=tr.n[hk].astype(float)).astype(np.int64).tolist()))
    ab = np.array([mm[2] for mm in tr.models], float)
    wq_cell, wp_cell = cell_work(tr.it.astype(float), tr.ot.astype(float), tr.n.astype(float), ab[tr.mid])
    rep_req_all, pb_all, pl_hours = [], {str(S): [] for S in SLOS}, []
    for si, seed in enumerate(seeds):
        fi = rec["fi"][si]
        slots = V3.build_slots(wd, fi)
        bs = [V3.slot_busy(sl, fi) for sl in slots]
        tot = np.zeros(WEEK_MIN, np.int64)
        for sl, b in zip(slots, bs):
            tot += sl["g"] * b.astype(np.int64)
        Cs = [wd.Ch if sl["kind"] == "tail" else wd.cg(sl["g"]) for sl in slots]
        first, last, _ = T.arrivals(tr, seed)
        cells_of = collections.defaultdict(list)
        for i in kk:
            cells_of[int(tr.mid[i])].append(int(i))
        slot_cells = []
        for sl in slots:
            ids = sorted((c for m in sl["models"] for c in cells_of.get(m, [])), key=lambda c: (int(tr.minute[c]), float(first[c]), c))
            mins = np.array([int(tr.minute[c]) - HELD for c in ids], np.int64)
            slot_cells.append((mins, ids))
        offs_cache = {}

        def offs(c):
            if c not in offs_cache:
                offs_cache[c] = T.cell_times(tr, c, first, last, seed)
            return offs_cache[c]
        khs = []
        for sl in slots:
            k = train & np.isin(tr.mid, sl["models"])
            khs.append(V3.k_hist(tr.minute[k]))
        sc = V3.static_counters(wd, rec, snap, r, seed)
        st_stats = {m: CS.cstats(*sc[m]) for m in hosted}
        st_p99 = {m: V4.static_pq(sc[m][0], sc[m][1], 0.99) for m in hosted}
        for m in hosted:                                        # v5's I-PERREQ + I-EXACT0, unchanged
            per_day = collections.defaultdict(dict)
            for c in cells_of.get(m, []):
                per_day[(int(tr.minute[c]) - HELD) // DAY_MIN][int(tr.minute[c])] = int(tr.n[c])
            for d in range(NDAYS):
                obj = sc[m][1][d]
                if R.counter_of(obj) != +collections.Counter(obj) or R.cells_of(obj) != per_day.get(d, {}):
                    checks["I_PERREQ"] = False
                    raise RuntimeError(f"I-PERREQ: model {tr.models[m][0][:8]} day {d}: the record != the scored Counter / cells")
                checks["peri_cells"] += len(per_day.get(d, {}))
                checks["split_cells"] += len(R.split.get(id(obj), {}))
            ex0, p0 = V5.compose_exact(sc[m][1], [[] for _ in range(NDAYS)])
            s0 = st_stats[m]
            if ex0["n"] != s0["n"] or ex0["p95"] != s0["p95"] or ex0["ok"] != s0["ok"] or p0 != st_p99[m]:
                checks["I_EXACT0"] = False
                raise RuntimeError("I-EXACT0: zero-move exact composition != the static stats")
        checks["n_train_unready"] = R.n_train_unready
        rep_req = [int(sum(int(rd[:, m].sum() - rd[x, m]) for m in hosted)) for x in range(NDAYS)]
        for d in range(NDAYS):                                  # I-DEN: the burn denominator = the scored n
            if sum(st_stats[m]["n"][d] for m in hosted) != rep_req[d]:
                checks["I_DEN"] = False
                raise RuntimeError(f"I-DEN: replicate {d}: scored n != hosted requests")
        rep_req_all.extend(rep_req)
        for x in SLOS:
            s = str(x)
            pb_all[s].extend([(rep_req[d] - sum(st_stats[m]["ok"][s][d] for m in hosted)) / float(rep_req[d]) for d in range(NDAYS)])
        plmh = {s: collections.Counter() for s in sk}           # platform over per model-hour (descriptive)
        for m in hosted:
            for d in range(NDAYS):
                obj = sc[m][1][d]
                for mi, (n, key) in R.const.get(id(obj), {}).items():
                    for s, kv in sk.items():
                        if key > kv:
                            plmh[s][m * 1000 + (mi - HELD) // 60] += n
                for mi, (n, segs) in R.split.get(id(obj), {}).items():
                    arr = V5.expand(n, segs)
                    for s, kv in sk.items():
                        c_ = int((arr > kv).sum())
                        if c_:
                            plmh[s][m * 1000 + (mi - HELD) // 60] += c_
        pl_hours.append({s: hour_bins(plmh[s], nmh) for s in sk})
        ctx = dict(tr=tr, R=R, sc=sc, hosted=hosted, st_stats=st_stats, st_p99=st_p99, rst=rst, allm=allm, sk=sk, tot=tot,
                   Nmin=Nmin, nmh=nmh)
        pol_cache = {}
        nof = [False] * len(slots)
        if "STATIC" in res:
            b4 = V5.run_variant("STATIC", None, None, nof, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache)
            if b4[4]:
                checks["I_ZW"] = False
                raise RuntimeError("I-ZW: STATIC carries a delay")
            score_variant(res["STATIC"], {}, b4[:4], ctx, si)
        for pol in POL6:
            wins = None
            for p in P_GRID:
                busy_d, held_d, wake_d, held_tot, add_cd = V5.run_variant(pol, p, H2D_TAIL, nof, slots, bs, slot_cells, khs, tr,
                                                                          first, offs, checks, pol_cache)
                if wins is None:
                    wins = build_windows(pol, slots, slot_cells, tr, first, offs, pol_cache, wq_cell, wp_cell, Cs)
                    wslots = {w["s"] for w in wins}
                cds, resd, st = queue_pass(wins, p)
                # I-RES: the queue's residuals are v5's, bit for bit
                checks["res_compared"] += 1
                if set(resd) != set(add_cd) or any(not np.array_equal(resd[c], add_cd[c]) for c in add_cd):
                    checks["I_RES"] = False
                    raise RuntimeError(f"I-RES: {pol} p{p:g}: queue residuals != v5's window_delays")
                # I-ZW: delays only on cells of slots that woke
                woke = {c for s_i in wslots for c in slot_cells[s_i][1]}
                if any(c not in woke for comp in COMPS for c in cds[comp]):
                    checks["I_ZW"] = False
                    raise RuntimeError(f"I-ZW: {pol} p{p:g}: a delay on a slot with no wake")
                # I-QGE: request by request, QINH >= ADD, QGROSS >= QINH, QPRE >= ADD (centiseconds)
                for c in set().union(*[set(cds[x]) for x in COMPS], add_cd):
                    n_ = int(tr.n[c])
                    z = np.zeros(n_)
                    ad, qi, qg, qp = (_cs(add_cd.get(c, z)), _cs(cds["QINH"].get(c, z)), _cs(cds["QGROSS"].get(c, z)),
                                      _cs(cds["QPRE"].get(c, z)))
                    checks["qge_compared"] += n_
                    bad = int(np.sum(qi < ad) + np.sum(qg < qi) + np.sum(qp < ad))
                    if bad:
                        checks["qge_violations"] += bad
                        checks["I_QGE"] = False
                        raise RuntimeError(f"I-QGE: {pol} p{p:g} cell {c}: queued < additive")
                bur = st["bursts"]
                for comp in ("ADD",) + COMPS:
                    v = vname(pol, comp, p)
                    if v not in res:
                        continue
                    o = res[v]
                    score_variant(o, add_cd if comp == "ADD" else cds[comp], (busy_d, held_d, wake_d, held_tot), ctx, si)
                    o["burst_max"].append(int(max(bur)) if bur else 0)
                    o["burst_p99"].append(float(np.percentile(bur, 99)) if bur else 0.0)
                    o["n_windows"].append(len(bur))
                    o["nclip"].append(st["nclip"] if comp != "ADD" else 0)
                    o["kmax"].append(st["kmax"].get(comp, {}) if comp != "ADD" else {})
    for v in variants:
        res[v]["idle_el"] = [None if x is None else round(x, 6) for x in res[v]["idle_el"]]
    if "STATIC" in res:
        for x in SLOS:
            xs = res["STATIC"]["x"][str(x)]
            if any(xs["over_by_wake"]) or any(xs["alert"]):
                checks["I_EBE0"] = False
                raise RuntimeError("I-EBE0: STATIC has elastic burn")
    return dict(variants=res, checks=checks, rep_req=rep_req_all, platform_burn=pb_all, platform_hours=pl_hours,
                overlay6_s=round(time.time() - t_start, 1),
                consts=dict(B_E=B_E, FAST_BURN=FAST_BURN, WIN=[WIN_LONG, WIN_SHORT], P_GRID=list(P_GRID), POL6=list(POL6),
                            COMPS=list(COMPS), H2D_TAIL=H2D_TAIL, Ch=wd.Ch, e_par=wd.np["e_par"], m_h=wd.np["m_h"],
                            n_variants=len(variants)))


# ------------------------------------------------------------------------------------------------ the static row (point or calibrated)
def static_row_over(cell, seeds, tag, root, cache, epar=None):
    """V3.static_row at the point physics; with epar, the same code path in a world with e_par := epar and the POINT catalogue."""
    if epar is None:
        return V3.static_row(cell, seeds, tag, root, cache)
    arm, cat, N = V3.CELLS[cell]
    fmt = "int4"
    extra = dict(CS.EXTRA0)
    if cat == "CAT128P":
        wd0 = V.load_world(fmt, {}, dict(CS.EXTRA0), root, cache)
        ids = V.catalogue(wd0, cat)
        del wd0
    over = {"e_par": float(epar)}
    wd, base_arm, homing, ca, tau, v2 = V.job_setup(arm, fmt, over, extra, root, cache, None, V.WIN_POINT, 7)
    if cat != "CAT128P":
        ids = V.catalogue(wd, cat)
    v2["seeds"] = list(seeds)
    beta = wd.np["beta"]
    cands = [c for c in wd.cands if c["cid"] in ids]
    base = dict(arm=base_arm, fmt=fmt, tag=tag, cat=cat, nphys=wd.np, extra=extra, n_cat=len(ids), v2=v2)
    rec = dict(tail=[], giant={}, fi=[])
    t0 = time.time()
    with V.giant_tau(tau):
        info, snaps = V.plan2(wd, base_arm, homing, N, ca, [len(cands)], cands)
        if info["flag"]:
            return wd, dict(base, N=N, flag=f"ANCHOR:{info['flag']}"), rec, None, None
        sn = snaps[len(cands)]
        if set(sn["admitted"]) != ids:
            return wd, dict(base, N=N, flag="ADMISSION-INCOMPLETE", n_admitted=len(sn["admitted"])), rec, None, None
        row = dict(base, N=N, r_min=info["r_min"], complete=True, n_admitted=len(sn["admitted"]))
        with V3.record(rec):
            row.update(V.evaluate2(wd, base_arm, sn, N, info["r"], beta, seeds, ca))
        row.update(base, N=N, complete=True)
        row["pass"] = {s: bool(all(row["slo"][s]["pass"])) for s in row["slo"]}
    row["elapsed_s"] = round(time.time() - t0, 1)
    if len(rec["fi"]) != len(seeds):
        raise RuntimeError(f"recorded {len(rec['fi'])} fleet_idle calls for {len(seeds)} seeds")
    return wd, row, rec, sn, info["r"]


def run_cell(cell, seeds, tag, root, cache, out, epar=None):
    if cell not in CELLS6:
        raise ValueError(f"unknown cell {cell} (v6 runs {CELLS6})")
    R = V5.install()
    wd, row, rec, snap, r = static_row_over(cell, seeds, tag, root, cache, epar)
    row["cell"] = cell
    row["epar"] = epar
    if snap is None:
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return 0
    if abs(wd.tph["h2d"] - H2D_TAIL) > 1e-9:
        raise RuntimeError(f"prereg §2: the cloud sim's tail h2d is {wd.tph['h2d']}, not {H2D_TAIL}")
    if epar is not None and abs(wd.np["e_par"] - float(epar)) > 1e-12:
        raise RuntimeError(f"the calibrated world reads e_par {wd.np['e_par']}, not {epar}")
    row["v6"] = overlay6(wd, row, rec, snap, r, seeds, R)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------ the TRAIN prior (prereg §1)
def _wq(vals, wts, q):
    if not len(vals):
        return None
    o = np.argsort(vals)
    cw = np.cumsum(np.asarray(wts, float)[o])
    j = int(np.searchsorted(cw, q * cw[-1], "left"))
    return float(np.asarray(vals)[o][min(j, len(o) - 1)])


def premise(cell, root, cache):
    if cell not in CELLS6:
        raise ValueError("--premise takes D1 or D2")
    R = V5.install()
    wd, row, rec, snap, r = V3.static_row(cell, (0,), "premise", root, cache)
    cx = wd.cx
    tr = cx["tr"]
    hosted = [wd.am] + [cx["cid_m"][c] for c in snap["admitted"]]
    sc = V3.static_counters(wd, rec, snap, r, 0)
    ptr = V5.train_p99(R, sc, hosted)
    slots = V3.build_slots(wd, rec["fi"][0])
    slot_of = {}
    for sl in slots:
        for m in sl["models"]:
            slot_of.setdefault(m, sl)
    ab = np.array([mm[2] for mm in tr.models], float)
    trn = tr.minute < HELD
    out = dict(cell=cell, seed=0, hosted=len(hosted), Ch=wd.Ch, cg8=wd.cg(8), e_par=wd.np["e_par"], at_risk={}, work={})
    allq, allp, allw = [], [], []
    per = {}
    for m in hosted:
        sl = slot_of.get(m)
        if sl is None:
            continue
        C = wd.Ch if sl["kind"] == "tail" else wd.cg(sl["g"])
        k = trn & (tr.mid == m)
        if not k.any():
            per[m] = None
            continue
        wq, wp = cell_work(tr.it[k].astype(float), tr.ot[k].astype(float), tr.n[k].astype(float), ab[m])
        n = tr.n[k]
        per[m] = dict(q50=_wq(wq / C, n, 0.5), q99=_wq(wq / C, n, 0.99), p50=_wq(wp / C, n, 0.5), own=V5.wake_costs(sl, m, H2D_TAIL)[0],
                      kind=sl["kind"], engines=len(sl["units"]))
        allq.append(wq / C)
        allp.append(wp / C)
        allw.append(n)
    if allq:
        q, pp, w = np.concatenate(allq), np.concatenate(allp), np.concatenate(allw)
        out["work"] = dict(qinh_s=[_wq(q, w, x) for x in (0.5, 0.9, 0.99)], qpre_s=[_wq(pp, w, x) for x in (0.5, 0.9, 0.99)])
    for S in SLOS:
        for p in (0.0, 0.5):
            v5r, v6r, v6r99, new = 0, 0, 0, []
            for m in hosted:
                nt, p99 = ptr.get(m, (0, None))
                if nt < 1 or p99 is None or per.get(m) is None:
                    continue
                h = S - p99
                if h < 0:
                    continue
                pm = per[m]
                a5 = h < pm["own"] + p
                a6 = h < pm["own"] + p + pm["q50"]
                v5r += a5
                v6r += a6
                v6r99 += h < pm["own"] + p + pm["q99"]
                if a6 and not a5:
                    new.append(tr.models[m][0][:8])
            out["at_risk"][f"S{S:g}|p{p:g}"] = dict(v5=v5r, v6_q50=v6r, v6_q99=v6r99, new_under_v6=sorted(new)[:12], n_new=len(new))
    out["engines_per_tail_slot"] = dict(collections.Counter(len(sl["units"]) for sl in slots if sl["kind"] == "tail"))
    print(json.dumps(out))
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks over rows
def _strip6(x):
    if isinstance(x, dict):
        return {k: _strip6(v) for k, v in x.items() if k not in TIMING6}
    if isinstance(x, list):
        return [_strip6(v) for v in x]
    return x


def ifix(v6_dir, v5_dir):
    rows = V3._rows(v6_dir)
    by = {(r["tag"], r["cell"]): r for r in rows if r.get("epar") is None}
    olds = [o for o in V3._rows(v5_dir) if o.get("cell") in CELLS6]
    res, ok = {}, True
    for o in olds:
        key = (o["tag"], o["cell"])
        n = by.get(key)
        if n is None:
            res[f"I_FIXV5_{key[0]}_{key[1]}"] = "MISSING"
            ok = False
            continue
        dfs = [k for k in sorted(set(o) - TIMING6 - {"v4", "v5"}) if _strip6(n.get(k)) != _strip6(o.get(k))]
        vd = []
        nv, ov = n["v6"]["variants"], o["v5"]["variants"]
        shared = [v for v in nv if VARIANTS6[v][2] == "ADD"]
        for v in shared:
            if v not in ov:
                vd.append(f"{v}: not in v5")
                continue
            for k in V5_FIELDS:
                if _strip6(nv[v][k]) != _strip6(ov[v][k]):
                    vd.append(f"{v}.{k}")
            for s in nv[v]["x"]:
                for k in V5_XFIELDS:
                    if nv[v]["x"][s][k] != ov[v]["x"][s][k]:
                        vd.append(f"{v}.x.{s}.{k}")
        res[f"I_FIXV5_{key[0]}_{key[1]}"] = dict(static_diffs=dfs[:10], variant_diffs=vd[:10], n_shared=len(shared))
        ok = ok and not dfs and not vd and len(shared) == 1 + 2 * len(P_GRID)
    res["I_FIXV5_rows"] = len(olds)
    ok = ok and len(olds) == 8
    for (tag, cell), r in sorted(by.items()):
        ch = r["v6"]["checks"]
        res[f"checks_{tag}_{cell}"] = ch
        ok = ok and all(ch[k] for k in ("I_PERREQ", "I_CELLUNIQ", "I_EXACT0", "I_HELD", "I_WAKE", "I_RES", "I_QGE", "I_ZW", "I_EBE0",
                                        "I_DEN"))
        ok = ok and ch["qge_violations"] == 0 and ch["res_compared"] == len(POL6) * len(P_GRID) * 3
    for cell in CELLS6:
        a = by.get(("main", cell))
        for t in ("rep_b", "rep_c"):
            b = by.get((t, cell))
            if a is None or b is None:
                res[f"I_REPLAY_{cell}_{t}"] = "MISSING"
                ok = False
                continue
            sa, sb = _strip6(dict(a, tag=None)), _strip6(dict(b, tag=None))
            diff = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
            res[f"I_REPLAY_{cell}_{t}"] = dict(diffs=diff[:5])
            ok = ok and not diff
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summarize (prereg §5)
def _rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None


def static_reading(X, x):
    st = X["STATIC"]["x"][str(x)]
    if V5.eb_gate(st["eb"]) == "EB-HOLDS":
        return "EB"
    if V5.eb_gate(st["eb_elig"]) == "EB-HOLDS":
        return "EB-ELIG"
    return "NONE"


def gates6(o, x, reading, rep_req):
    s = str(x)
    xs = o["x"][s]
    p95 = all(xs["p95_pass"])
    mw = all(m <= x for m in o["mww"])
    p99 = all(xs["p99_pass"])
    pm = V5.pm_gate(xs["p95_pass"], o["mww"], xs["p99_pass"], x)
    ebe = ebe_gate(xs["over_by_wake"], rep_req, xs["alert"])
    share_ok = all(1000 * int(a) <= int(b) for a, b in zip(xs["over_by_wake"], rep_req))
    alert_ok = all(int(a) == 0 for a in xs["alert"])
    ebx = True if reading == "NONE" else V5.eb_gate(xs["eb" if reading == "EB" else "eb_elig"]) == "EB-HOLDS"
    bar = V3.bar_v(o["idle_el"])
    fail = [nm for nm, okk in (("BAR", bar == "UNDER-BAR"), ("PM6-p95", p95), ("PM6-MWW", mw), ("PM6-p99", p99),
                                ("EBE-share", share_ok), ("EBE-alert", alert_ok), ("EBX", ebx)) if not okk]
    ebr = [a / float(b) for a, b in zip(xs["over_by_wake"], rep_req)]
    return dict(meet=not fail, fail=fail, pm=pm, ebe=ebe, ebx=ebx, bar=bar, idle=_rng(o["idle_el"]), mww=_rng(o["mww"]),
                mww_s=_rng(xs["mww_s"]), ebr=_rng(ebr), alert=_rng(xs["alert"]), over=_rng(xs["over_by_wake"]),
                eb=_rng(xs["eb"]), ebe_elig=_rng(xs["eb_elig"]), burst_max=max(o["burst_max"]) if o["burst_max"] else None,
                kmax=_kmax(o["kmax"]), p99_fail={k: v for k, v in sorted(xs["p99_fail"].items())[:8]})


def _kmax(lst):
    out = {}
    for d in lst:
        for k, v in d.items():
            out[k] = round(max(out.get(k, 0.0), v), 3)
    return out


def p_meet6(rows):
    k = 0
    while k < len(P_GRID) and rows[P_GRID[k]]["meet"]:
        k += 1
    return (P_GRID[k - 1] if k else None), ([P_GRID[k], rows[P_GRID[k]]["fail"]] if k < len(P_GRID) else None)


def survive(pm):
    if pm is None:
        return "FALLS"
    return "SURVIVES" if pm >= 0.5 else f"SHRINKS TO p <= {pm:g}"


def cell_v6(row):
    X = row["v6"]["variants"]
    rr = row["v6"]["rep_req"]
    out = dict(reading={}, static={}, arms={}, qmain={}, attrib={}, platform={})
    for x in SLOS:
        s = f"S{x:g}"
        rd_ = static_reading(X, x)
        out["reading"][s] = rd_
        out["static"][s] = gates6(X["STATIC"], x, rd_, rr)
        out["platform"][s] = _rng(row["v6"]["platform_burn"][str(x)])
        best, best_add = None, None
        for pol in POL6:
            for comp in ("ADD",) + COMPS:
                rows = {p: gates6(X[vname(pol, comp, p)], x, rd_, rr) for p in P_GRID}
                pmv, brk = p_meet6(rows)
                pm_only = 0
                while pm_only < len(P_GRID) and rows[P_GRID[pm_only]]["pm"] == "PM-HOLDS" and rows[P_GRID[pm_only]]["bar"] == "UNDER-BAR":
                    pm_only += 1
                out["arms"][f"{s}|{pol}|{comp}"] = dict(p_meet6=pmv, p_break6=brk,
                                                       p_meet_pm_bar=P_GRID[pm_only - 1] if pm_only else None,
                                                       by_p={f"{p:g}": rows[p] for p in P_GRID})
                if comp == DEC_COMP and pmv is not None and (best is None or pmv > best[0]):
                    best = (pmv, f"{pol}|{comp}")
                if comp == "ADD" and pmv is not None and (best_add is None or pmv > best_add[0]):
                    best_add = (pmv, f"{pol}|ADD")
        st = out["static"][s]
        held_ok = st["pm"] == "PM-HOLDS" and st["ebe"] == "EBE-HOLDS" and st["ebx"]
        ans = (f"YES for p <= {best[0]:g} ({best[1]})" if best else
               "NO at any grid p; " + ("HELD-HOME-ONLY" if held_ok else f"NOT MET EVEN HELD-HOME ({st['fail']})"))
        out["qmain"][s] = dict(answer=ans, largest_p=best[0] if best else None, survive=survive(best[0] if best else None))
        out["attrib"][s] = dict(add_v6_gates=best_add[0] if best_add else None,
                                add_pm_only=max([out["arms"][f"{s}|{pol}|ADD"]["p_meet_pm_bar"] or -1 for pol in POL6]),
                                qinh=best[0] if best else None)
    out["platform_hours_seed0"] = row["v6"]["platform_hours"][0] if row["v6"]["platform_hours"] else None
    out["consts"] = row["v6"]["consts"]
    return out


def verdict_key6(c):
    k = {}
    for a, x in c["arms"].items():
        k[f"arm|{a}"] = x["p_meet6"]
    for s, x in c["qmain"].items():
        k[f"qmain|{s}"] = x["answer"]
    return k


def summarize(d):
    rows = V3._rows(d)
    by = {(r["tag"], r["cell"]): r for r in rows}
    out = dict(cells={}, seedshift={}, replay={}, flags={})
    for (tag, cell), r in sorted(by.items()):
        if "v6" not in r:
            out["flags"][f"{tag}|{cell}"] = r.get("flag")
            continue
        if tag in ("main", "calp", "cald"):
            out["cells"][f"{tag}|{cell}"] = cell_v6(r)
            out["cells"][f"{tag}|{cell}"]["N"] = r["N"]
    for cell in CELLS6:
        a = out["cells"].get(f"main|{cell}")
        ss = by.get(("seedshift", cell))
        if a and ss and "v6" in ss:
            ka, kb = verdict_key6(a), verdict_key6(cell_v6(ss))
            out["seedshift"][cell] = dict(UNATTRIBUTABLE={k: [ka[k], kb.get(k)] for k in ka if ka[k] != kb.get(k)}, n_compared=len(ka))
        vals = {}
        for t in ("main", "rep_b", "rep_c"):
            r = by.get((t, cell))
            if r and "v6" in r:
                c = cell_v6(r)
                for s, q in c["qmain"].items():
                    vals.setdefault(s, []).append(q["largest_p"])
        out["replay"][cell] = {s: dict(runs=len(v), spread=[min(x for x in v if x is not None), max(x for x in v if x is not None)]
                                       if any(x is not None for x in v) else None) for s, v in vals.items()}
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §8)
def self_test():
    rc5 = V5.self_test()
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))

    def qw(kind, offs_, svc, p=0.0, page=1.0):
        a = np.asarray(offs_, float)
        n = len(a)
        unit = np.zeros(n, np.int64)
        uu, uinv = np.unique(unit, return_inverse=True)
        sv = np.full(n, svc)
        return queue_window(kind, float(a[0]), p, {0: float(a[0])}, {0: page}, 0, a, uinv, uu.tolist(), sv, sv,
                            np.zeros(n, np.int64), np.arange(n, dtype=np.int64))
    r1 = qw("tail", [0.0, 0.5, 1.1], 0.3)
    chk("GQ1 ADD cs [100, 50, 0]", _cs(r1[0]).tolist(), [100, 50, 0])
    chk("GQ1 QINH cs [100, 80, 50] (rank 1 behind rank 0; rank 2, never delayed by v5, waits the backlog)", _cs(r1[1]).tolist(),
        [100, 80, 50])
    chk("GQ1 QGROSS cs [100, 80, 50]; burst 2", [_cs(r1[2]).tolist(), r1[4]], [[100, 80, 50], 2])
    r1b = qw("tail", [0.0, 0.1], 0.3)
    chk("GQ1b ADD [100, 90] QINH [100, 100] QGROSS [100, 120]", [_cs(r1b[0]).tolist(), _cs(r1b[1]).tolist(), _cs(r1b[2]).tolist()],
        [[100, 90], [100, 100], [100, 120]])
    r1c = qw("tail", [0.0, 0.1], 0.3, p=0.5)
    chk("GQ1b p 0.5: ADD [150, 140] QINH [150, 150]", [_cs(r1c[0]).tolist(), _cs(r1c[1]).tolist()], [[150, 140], [150, 150]])
    r2 = qw("tail", [0.0, 5.0], 0.3)
    chk("GQ2 a wake with no other request in it: QINH = ADD = [100, 0]", [_cs(r2[0]).tolist(), _cs(r2[1]).tolist(),
                                                                       _cs(r2[3]).tolist()], [[100, 0], [100, 0], [100, 0]])
    held, wake = V4.slot_policy("STATIC", np.array([0, 1, 1, 0, 1], bool), None, False)
    chk("GQ2 STATIC: no wake, so no window", int(wake.sum()), 0)
    r6 = qw("anchor", [0.0, 0.5, 1.1], 0.3)
    chk("GQ6 anchor: only pre-ready ranks queue: QINH [100, 80, 0]", _cs(r6[1]).tolist(), [100, 80, 0])
    rg = qw("giant", [0.0, 0.5, 1.1], 0.3)
    chk("GQ6 giant trigger queues its whole window: QINH [100, 80, 50]", _cs(rg[1]).tolist(), [100, 80, 50])
    # QPRE uses its own service column
    a = np.array([0.0, 0.1])
    unit = np.zeros(2, np.int64)
    rp = queue_window("tail", 0.0, 0.0, {0: 0.0}, {0: 1.0}, None, a, np.zeros(2, np.int64), [0], np.full(2, 0.3), np.full(2, 0.15),
                      unit, np.arange(2, dtype=np.int64))
    chk("GQ1b QPRE (service 0.15): Qw 0.15, Q0 0.05 -> [100, 100]", _cs(rp[3]).tolist(), [100, 100])
    # GQ3: the elastic-burn gate is per replicate, inclusive, and independent of the platform budget
    R = NREPS
    chk("GQ3 11 / 10,000 on every replicate -> EBE-FAILS", ebe_gate([11] * R, [10000] * R, [0] * R), "EBE-FAILS")
    chk("GQ3 10 / 10,000 -> EBE-HOLDS (inclusive)", ebe_gate([10] * R, [10000] * R, [0] * R), "EBE-HOLDS")
    chk("GQ3 one replicate at 11 -> EBE-FAILS", ebe_gate([5] * (R - 1) + [11], [10000] * R, [0] * R), "EBE-FAILS")
    chk("GQ3 EBX vacuous + PM holds still FAILS on EBE", gates_meet_stub("PM-HOLDS", ebe_gate([11] * R, [10000] * R, [0] * R), True,
                                                                         "UNDER-BAR"), False)
    chk("GQ3b no double count: only the 1.50 -> 2.10 rank is elastic burn at S2",
        elastic_cross([150, 190, 250, INF], [210, 195, 300, INF], T._key(2.0)), 1)
    # GQ4: the fast burn needs BOTH windows
    N = np.full(1200, 300)
    E = np.zeros(1200, np.int64)
    E[100:160] = 5
    al = alert_minutes(E, N)
    chk("GQ4 alert minutes t 151..159 = 9", [int(al.sum()), int(np.flatnonzero(al)[0]), int(np.flatnonzero(al)[-1])], [9, 151, 159])
    chk("GQ4 the share alone passes (300 / 360,000 <= 0.001) but alerts fail EBE",
        [1000 * int(E.sum()) <= int(N.sum()), ebe_gate([int(E.sum())] * R, [int(N.sum())] * R, [9] * R)], [True, "EBE-FAILS"])
    E2 = np.zeros(NDAYS * DAY_MIN, np.int64)
    E2[DAY_MIN + 100:DAY_MIN + 160] = 5
    chk("GQ4 per replicate: the day holding the alerts drops them in its own replicate only",
        alert_reps(E2, np.full(NDAYS * DAY_MIN, 300)), [9, 0, 9, 9, 9, 9, 9])
    # GQ5: rates (inherited)
    wq, wp = cell_work(3000.0, 600.0, 2.0, 6.2)
    chk("GQ5 work per request QINH 3,600 / QPRE 3,000", [round(wq, 9), round(wp, 9)], [3600.0, 3000.0])
    chk("GQ5 at C 18,000: 0.200000 / 0.166667 s", [round(wq / 18000.0, 6), round(wp / 18000.0, 6)], [0.2, 0.166667])
    chk("GQ5 giant C g 8: e_par 0.5 / 5.0 / 1.07", [round(giant_c(8, 1.1, e), 6) for e in (0.5, 5.0, 1.07)],
        [82618.864686, 826188.646858, 176804.370428])
    chk("lindley [1.0, 1.0, 1.1] x 0.3 -> starts [1.0, 1.3, 1.6]", [round(x, 9) for x in lindley([1.0, 1.0, 1.1], [0.3] * 3)],
        [1.0, 1.3, 1.6])
    chk("hour bins: {0: 0, 1: 1, 2: 3, 3: 12}, tenant fast at n >= 100 and share > 0.0144",
        hour_bins({1: 1, 2: 3, 3: 12}, {0: 50, 1: 50, 2: 100, 3: 500}), dict(bins=[1, 1, 1, 0, 1], tenant_fast=2, max_mh=12))
    chk("variants: 89", len(VARIANTS6), 89)
    chk("ADD names are v5's", all(v in V5.VARIANTS5 for v, s in VARIANTS6.items() if s[2] == "ADD"), True)
    n_ok = sum(1 for r_ in res if r_[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    allok = n_ok == len(res) and rc5 == 0
    print(f"SELF-TEST {'PASS' if allok else 'FAIL'} {n_ok}/{len(res)} (v6) + v5 rc {rc5}")
    return 0 if allok else 1


def gates_meet_stub(pm, ebe, ebx, bar):
    """MEET6 at one p from its parts (prereg §5): PM6 ∧ EBE ∧ EBX ∧ UNDER-BAR."""
    return pm == "PM-HOLDS" and ebe == "EBE-HOLDS" and bool(ebx) and bar == "UNDER-BAR"


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--premise", action="store_true")
    m.add_argument("--ifix", nargs=2, metavar=("V6_DIR", "V5_DIR"))
    m.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--cell")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epar", type=float)
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.ifix:
            for p in a.ifix:
                if not os.path.isdir(p):
                    raise ValueError(f"no such dir {p}")
            r = ifix(*a.ifix)
            print(json.dumps(r))
            return 0 if r["pass"] else 1
        if a.summarize:
            if not os.path.isdir(a.summarize):
                raise ValueError(f"no such dir {a.summarize}")
            print(json.dumps(summarize(a.summarize)))
            return 0
        need = ("cell", "root", "cache") + (() if a.premise else ("out",))
        for k in need:
            if getattr(a, k) is None:
                raise ValueError(f"--{k} is required")
        for p in (a.root, a.cache):
            if not os.path.exists(p):
                raise ValueError(f"missing {p}")
        if a.premise:
            return premise(a.cell, a.root, a.cache)
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if len(seeds) * NDAYS != NREPS:
            raise ValueError(f"{len(seeds)} seeds x {NDAYS} days != {NREPS} replicates")
        if a.epar is not None and a.epar not in EPAR_CAL.values():
            raise ValueError(f"--epar must be one of the prereg's calibrations {sorted(EPAR_CAL.values())}")
        return run_cell(a.cell, seeds, a.tag, a.root, a.cache, a.out, a.epar)
    except (ValueError, V.ManifestError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
