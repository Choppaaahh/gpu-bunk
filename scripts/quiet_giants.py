#!/usr/bin/env python3
"""quiet_giants.py — can a quiet big MoE run on fewer H100 GPUs at the SLO, or is it cooked?

Prereg (frozen before this file existed): (private report) + amendments. On any divergence
the prereg wins. Imports `node_sim`, `cloud_sim` and v1/v2 UNCHANGED. Forks, each equivalence-checked:
`dedicated_by_day` -> `gengine` (adds the expert-offload host resource and the decode floor; I-EQ) and `run_dgroups` -> `sleep_core`
(adds a KV-shed tau and an h2d per GPU; I-SLEEP). Levers, LOSSLESS FIRST (prereg §3):
  OFF   expert offloading: every expert in pinned host RAM, a resident fraction f on a G-GPU group (roofline, prereg §4)
  PAIR  giant sleep with CO-ACTIVITY homing (zero shared TRAIN active hours) + a KV-shed rule (v2 sleep path)
  LOSSLESS = OFF + PAIR composed (no int4), built in --summarize
  INT4  weights in int4, SEPARATE and quality-labelled (counts only for the two natively-4-bit giants)
  CTRL  dedicated fp8 group, r replicas (engine)

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n" (prereg §9 goldens GQ1-GQ9, I-EQ on synthetic traces, the CLI golden);
                                    exit 1 + FAIL lines.
  --table [--root R --cache C]      read-only: one JSON line per MG model (fp8/int4 G, W/E/NE, f_max per G, host fit); without --root,
                                    only the anchor from its checkpoint constants. exit 0.
  --check EQ|SLEEP|STALL|THIN|CONS|SEED|MONO --root R --cache C
                                    read-only instrument checks (prereg §8); one JSON object; exit 0 PASS / 1 FAIL (STALL and MONO are
                                    descriptive: exit 0 unless they crash).
  --ladder --giant CID8 --donor ANCHOR|QWEN --cm POINT|PHCAL [--sens NAME] [--arms CTRL,INT4,OFF] [--noprune] --root R --cache C --out F
                                    WRITES (appends) one JSON line per (config, seed, rung) to F; exit 0.
  --pair-ladder --giant CID8 --n 2|4 --donor D --cm M --root R --cache C --out F
                                    WRITES (appends) one JSON line per (seed, rung) to F; exit 0.
  --fleet --giant CID8 --cm M --root R --cache C --out F
                                    WRITES (appends) real-traffic CTRL / INT4 / OFF rows and one TRAIN-feasibility row to F; exit 0.
  --pair-fleet --cm M --tau X|none --window 60|10 --root R --cache C --out F
                                    WRITES (appends) the homing and one row per (multi-member group, seed) to F; exit 0.
  --summarize DIR                   read-only: prereg §5-§6 over DIR/*.jsonl; one JSON object on stdout.
  --mono DIR                        read-only (amendment 1, I-MONO): non-monotone OFF series among UNPRUNED ladder rows in DIR.
  --synth-table SID [--root R --cache C]
                                    read-only: one JSON line per (reading, format) of a synthetic giant's memory, G_min, f_max,
                                    host fit; with --root also its borrowed occupancy and bpt per borrowed profile. exit 0.
  Synthetic giants (addendum (private report)): --giant muse816b on --ladder /
  --pair-ladder / --fleet, optional --borrow CID8 (default the nearest quiet giant), extra donor BORROW (ladder / pair-ladder), and extra
  arms BF16CTRL,BF16OFF on --ladder ONLY (--fleet and --pair-ladder take no --arms; BF16 arms elsewhere = exit 2). Census giants never
  reach that code; their rows are byte-identical to 2056cb93.
  Bad input (unknown giant/arm/sens, missing file, sha mismatch) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed earlier, inherited here):
  1. **`ssh host '... &'` holds the session**: launch on a CPU server with `ssh -n` + `setsid -f`.
  2. **The seed is blind for all-awake arms**: this engine takes no seed; replicates are LODO days (x thinning seeds on the
     ladder). Only the sleep path (PAIR) reads the seed.
  3. **A unit index of -1 wraps numpy** (v2 pitfall 2): every array here is indexed by model, never by a unit index.
"""
import argparse, collections, json, math, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_sim as NS        # noqa: E402  (not edited)
import cloud_sim as CS       # noqa: E402  (not edited; run_dgroups reference for I-SLEEP)
import fabric_sim as F       # noqa: E402  (v1, not edited)
import fabric_sim_v2 as V2   # noqa: E402  (v2, not edited)
import tailpool_sim as T     # noqa: E402  (not edited)
import wakestorm_sim as W    # noqa: E402  (not edited)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

G30 = NS.G30
HELD = NS.HELD
NDAYS = NS.NDAYS
DAY_MIN = NS.DAY_MIN
WEEK_MIN = NDAYS * DAY_MIN
TRAIN0 = HELD - WEEK_MIN              # day 300
KV_MIN = NS.KV_MIN
SLOS = (2.0, 10.0)
SEEDS = (0, 1, 2)
MIN_REQ = 20
U_MAX = 0.95                          # prereg §4: prefill share of a minute is capped
RUNGS = (30, 100, 300, 1000, 3000, 10000, 30000, 100000)
DONORS = {"ANCHOR": "51c86bd2", "QWEN": "aa605b6d"}
DONOR_IDX = {"ANCHOR": 0, "QWEN": 1}
E_PAR = {"POINT": 0.5, "PHCAL": 5.0}
G_OFF = (1, 2, 4, 8)
F_GRID = (0.0, 1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8)
R_GRID = (1, 2, 4)
FLOOR_KEYS = (("ttft", None), ("f50", 0.05), ("f100", 0.10), ("f200", 0.20))
FK_POINT = "f100"
TAU_POINT = 10.0
NATIVE4 = {"51c86bd2", "9b227eb9"}    # prereg §0 item 4: the checkpoint itself ships in 4 bits
O_PROC = F.PHYS0["o_proc"]

# ---- prereg §1 sourced constants
H2D = NS.H2D_PCIE                                     # 53.8 GB/s per H100, ICC bandwidthTest pinned
DS_ROUTED = 8 * 3 * 7168 * 2048 * 58                  # DeepSeek-V3 routed params per token
Q4KM_BYTES = 4.85 / 8.0
BW_CPU_NODE = 12.208 * DS_ROUTED * Q4KM_BYTES / 1e9   # 151.24 GB/s (KTransformers dual-socket 6454S decode, batch 1)
F_CPU_NODE = 2.0 * DS_ROUTED * 255.26                 # 1.0432e13 FLOP/s (KTransformers V0.3 AMX prefill)
ETA_NVBW = 26.0 / 53.8                                # nvbandwidth sample / bandwidthTest

OPHYS0 = dict(eta_f=1.0, dma_cap=0.0, bw_cpu=BW_CPU_NODE, overlap="SUM", loc="UNIFORM", gpu_scale=0, chunk=8192,
              cpu_share="G8", mode="HYB")
SENS = {"base": {}, "eta048": {"eta_f": ETA_NVBW}, "dma200": {"dma_cap": 200.0}, "bwc100": {"bw_cpu": 100.0},
        "bwc300": {"bw_cpu": 300.0}, "ovmax": {"overlap": "MAX"}, "loc": {"loc": "LOC"}, "gscale": {"gpu_scale": 1},
        "chunk64k": {"chunk": 65536}, "cpunode": {"cpu_share": "NODE"}, "fetch": {"mode": "FETCH"}, "cpu": {"mode": "CPU"}}


# ------------------------------------------------------------------------------------------------ pure physics (prereg §3-§4)
def f_max(G, E, NE, bh, o_g):
    """Largest resident expert fraction with only the KV floor left (GiB in, fraction out; may be < 0 = nothing fits)."""
    if E <= 0:
        return None
    return (G * (bh - o_g - KV_MIN) - NE) / E


def kv_room(G, f, E, NE, bh, o_g):
    """KV GiB left on a G-GPU group holding NE + f E (prereg §3.1)."""
    return G * (bh - o_g) - NE - f * E


def host_fit(W, G, node_ram, r_pin):
    """prereg §3.1: every expert pinned, W r_pin + G o_proc <= max(G, 1)/8 of a node's RAM."""
    return W * r_pin + G * O_PROC <= max(G, 1) / 8.0 * node_ram + 1e-9


def distinct(B, n, k):
    """Expected distinct experts per layer touched by B tokens choosing k of n (uniform routing)."""
    return n * (1.0 - (1.0 - k / n) ** B)


def hit_rate(f, loc):
    return f if loc == "UNIFORM" else 1.0 - (1.0 - f) ** 2


def host_decode(B, s):
    """prereg §4: host seconds one decode step adds at batch B (FETCH / CPU / HYB = min)."""
    miss = (1.0 - s["h"]) * distinct(B, s["n"], s["k"])
    gb = miss * s["L"] * s["e_gb"]
    fetch = gb / s["bwf"]
    cpu = gb / s["bwc"] + 2.0 * B * (1.0 - s["h"]) * s["A_r"] / s["fc"]
    if s["mode"] == "FETCH":
        return fetch
    if s["mode"] == "CPU":
        return cpu
    return min(fetch, cpu)


def host_prefill(in_r, s):
    """prereg §4: host seconds of one request's prefill with in_r input tokens."""
    if in_r <= 0.0 or s["h"] >= 1.0:
        return 0.0
    chunks = max(1, math.ceil(in_r / s["chunk"] - 1e-12))
    fetch = chunks * (1.0 - s["h"]) * s["E_gb"] / s["bwf"]
    cpu = in_r * 2.0 * (1.0 - s["h"]) * s["A_r"] / s["fc"]
    if s["mode"] == "FETCH":
        return fetch
    if s["mode"] == "CPU":
        return cpu
    return min(fetch, cpu)


def off_spec(geo, G, f, op):
    """Offload spec for a giant's geometry at G GPUs and resident fraction f under offload physics op (prereg §4)."""
    bwf = G * H2D * op["eta_f"]
    if op["dma_cap"] > 0:
        bwf = min(bwf, max(1.0, G / 8.0) * op["dma_cap"])
    share = G / 8.0 if op["cpu_share"] == "G8" else max(1.0, G / 8.0)
    return dict(h=hit_rate(f, op["loc"]), f=f, n=geo["n"], k=geo["k"], L=geo["L"], e_gb=geo["e_gb"], E_gb=geo["E_gb"],
                A_r=geo["A_r"], bwf=bwf, bwc=share * op["bw_cpu"], fc=share * F_CPU_NODE, mode=op["mode"],
                overlap=op["overlap"], chunk=op["chunk"])


def wake_s(W_bytes, G, h2d=H2D, node_cap=0.0):
    """Wake time of W_bytes from pinned host RAM over G GPUs at h2d GB/s each (node_cap GB/s per node when set)."""
    bw = G * h2d
    if node_cap > 0:
        bw = min(bw, max(1.0, G / 8.0) * node_cap)
    return W_bytes / 1e9 / bw


def verdict(ctrl, x):
    """prereg §6 / GQ4: -> (verdict, GPUs the fleet books for this giant in arm X). A failing lever never saves GPUs."""
    if ctrl is None:
        return "DEGENERATE", None
    if x is None:
        return "X-FAILS", ctrl
    if x < ctrl:
        return "FEWER-GPUS", x
    if x > ctrl:
        return "MORE-GPUS", ctrl
    return "SAME", ctrl


def break_even(rungs, fewer):
    """prereg §5.1 / GQ5: the largest rung with FEWER at it and at every lower rung; None if the first rung fails."""
    best = None
    for r, ok in zip(rungs, fewer):
        if not ok:
            break
        best = r
    return best


def meets(reps, slo):
    """reps = [(n, p95|None|inf) ...]: MEETS iff >= 1 replicate has >= MIN_REQ requests and every such replicate has p95 <= slo.
    -> True / False / None (no scored replicate)."""
    sc = [p for n, p in reps if n >= MIN_REQ]
    if not sc:
        return None
    return all(p is not None and p <= slo for p in sc)


def floor_key(base, tpe, fl):
    """prereg §2: a request misses (+inf) if its decode TPOT exceeds the floor, whatever its TTFT."""
    return T.INF_KEY if (fl is not None and tpe > fl) else T._key(base)


# ------------------------------------------------------------------------------------------------ the engine (fork of dedicated_by_day)
def gengine(trace, specs, gphys, window=(HELD, None)):
    """FORK of node_sim.dedicated_by_day. specs {m: dict(kvtok, C, mult, off)}; off None = nothing offloaded. Scores cells with
    window[0] <= minute < window[1] (None = open). -> {m: {fk: (week Counter, [7 day Counters])}}. With off None, the 'ttft' Counters
    equal dedicated_by_day exactly (I-EQ)."""
    lo, hi = window
    rcap = F.rho_cap(gphys)
    st, res = {}, {}
    for m in specs:
        st[m] = dict(backlog=0.0, bk=0.0, fut={}, kv_last=None, last=None, hb=0.0, hlast=None, rfut={})
        res[m] = {fk: (collections.Counter(), [collections.Counter() for _ in range(NDAYS)]) for fk, _ in FLOOR_KEYS}
    for i in range(len(trace.n)):
        m = int(trace.mid[i])
        sp = specs.get(m)
        if sp is None:
            continue
        n = int(trace.n[i])
        mi = int(trace.minute[i])
        scored = mi >= lo and (hi is None or mi < hi)
        dd = (mi - HELD) // DAY_MIN if mi >= HELD else None
        s = st[m]
        rm = res[m]
        ab = trace.models[m][2]
        Cm = sp["C"]
        cap_min = Cm * 60.0
        kvt = sp["kvtok"]
        it = float(trace.it[i])
        ot = float(trace.ot[i])
        if (it + ot) / n > kvt:
            if scored:
                for fk, _ in FLOOR_KEYS:
                    rm[fk][0][T.INF_KEY] += n
                    if dd is not None:
                        rm[fk][1][dd][T.INF_KEY] += n
            continue
        Wm = (it + ot) * (ab / ACTIVE_REF_B)
        if s["last"] is not None:
            s["backlog"] = max(0.0, s["backlog"] - cap_min * (mi - s["last"] - 1))
        s["last"] = mi
        s["bk"] = F.kv_catch_up(s["bk"], s["fut"], kvt, s["kv_last"], mi)
        s["kv_last"] = mi
        rho = min(rcap, (s["backlog"] + Wm) / cap_min)
        pre = (it / n) * (ab / ACTIVE_REF_B) / Cm
        tp = F.tpot_of(rho, gphys) * sp["mult"]
        off = sp["off"]
        if off is None:
            pf = 0.0
            qh = 0.0
            tpe = tp
        else:
            pf = host_prefill(it / n, off)
            H = n * pf
            if s["hlast"] is not None:
                s["hb"] = max(0.0, s["hb"] - 60.0 * (mi - s["hlast"] - 1))
            s["hlast"] = mi
            u = min(U_MAX, (s["hb"] + H) / 60.0)
            qh = s["hb"] + 0.5 * max(0.0, H - 60.0)
            s["hb"] = max(0.0, s["hb"] + H - 60.0)
            rf = s["rfut"]
            for k in [k for k in rf if k < mi]:
                del rf[k]
            B = max(1.0, rf.pop(mi, 0.0) / 60.0 + n / 2.0)
            hd = host_decode(B, off)
            step = tp + hd if off["overlap"] == "SUM" else max(tp, hd)
            tpe = step / (1.0 - u)
        R = pre + pf + (ot / n) * tpe
        occ = it / n + gphys["kappa"] * ot / n
        D = s["fut"].pop(mi, 0.0) + F.spread(s["fut"], mi, n * occ, R)
        kw, s["bk"] = F.kv_step(s["bk"], D, kvt)
        if off is not None:
            F.spread(s["rfut"], mi, float(n), R)
        q = (s["backlog"] + 0.5 * max(0.0, Wm - cap_min)) / Cm
        s["backlog"] = max(0.0, s["backlog"] + Wm - cap_min)
        if scored:
            base = q + pre + kw + qh + pf
            for fk, fl in FLOOR_KEYS:
                k_ = floor_key(base, tpe, fl)
                rm[fk][0][k_] += n
                if dd is not None:
                    rm[fk][1][dd][k_] += n
    return res


def reps_of(week, days):
    """7 LODO replicates -> [(n, p95 s)]."""
    out = []
    for d in range(NDAYS):
        c = week - days[d]
        out.append((int(sum(c.values())), T._pct(c, 0.95)))
    return out


def jnum(v):
    if v is None:
        return None
    return "inf" if v == math.inf else round(float(v), 4)


def unj(v):
    if v is None:
        return None
    return math.inf if v == "inf" else float(v)


# ------------------------------------------------------------------------------------------------ giants, geometry, configs
_CX = {}


def cx_of(root, cache, fmt):
    key = (root, cache, fmt)
    if key not in _CX:
        np_ = NS.nphys_with({"fmt": fmt})
        cx = NS.ctx(root, cache, np_, "STATIC")
        cx["np"] = np_
        cx["gp"] = NS.giant_phys(np_, cx["tph"])
        cx["by8"] = {c[:8]: c for c in cx["tab"]}
        _CX[key] = cx
    return _CX[key]


def full_cid(cx, cid8):
    c = cx["by8"].get(cid8)
    if c is None:
        raise ValueError(f"unknown giant {cid8}")
    return c


def geometry(cx, cid):
    """Expert geometry (prereg §4): n, k, L, hidden (recovered as mg_weights), routed active params, expert GiB/GB."""
    d = cx["tab"][cid]
    if not d["moe"]:
        return None
    row = cx["rows"][cid]
    if cid == NS.ANCHOR:
        n, k, mi_, L, hidden = 384, 8, 2048, 60, 7168
    else:
        mo = row["moe"]
        n, k, mi_, L = mo["experts"], mo["topk"], mo["moe_inter"], mo["moe_layers"]
        hidden = (row["params_total_R2"] - row["params_active_R2"]) / ((n - k) * 3 * mi_ * L)
    E_gb = d["E"] * G30 / 1e9
    return dict(n=n, k=k, L=L, hidden=hidden, A_r=k * 3 * hidden * mi_ * L, E_gb=E_gb, e_gb=E_gb / (n * L), E=d["E"], NE=d["NE"],
                W=d["W"])


def bpt_eff(cx, cid):
    m = cx["cid_m"][cid]
    return cx["S"]["pop"]["bpt"][m] * cx["tab"][cid]["dup"]


def ctrl_specs(cx, cid, e_par):
    """CTRL (fp8 ctx) or INT4 (int4 ctx) at r = 1, 2, 4: one engine with r x KV and r x compute (prereg §3.0)."""
    d = cx["tab"][cid]
    if not d.get("placeable"):
        return []
    kt = NS.kv_tokens(d["G"], d["L0"], cx["S"]["pop"]["bpt"][cx["cid_m"][cid]], d["dup"])
    C = d["G"] * C_REF * cx["np"]["m_h"] * e_par
    return [dict(r=r, G=d["G"], gpus=r * d["G"], kvtok=r * kt, C=r * C, mult=1.0, off=None) for r in R_GRID]


def off_configs(cx, cid, e_par, op):
    """Feasible OFF configs (prereg §3.1): G < G_ctrl, f in F_GRID, KV room >= G KV_MIN, host RAM fits."""
    d = cx["tab"][cid]
    geo = geometry(cx, cid)
    if geo is None or not d.get("placeable"):
        return []
    np_ = cx["np"]
    bh = NS.b_h(np_)
    out = []
    for G in G_OFF:
        if G >= d["G"] or not host_fit(d["W"], G, np_["node_ram"], np_["r_pin"]):
            continue
        for f in F_GRID:
            kr = kv_room(G, f, geo["E"], geo["NE"], bh, np_["o_g"])
            if kr < G * KV_MIN - 1e-9:
                continue
            kt = NS.kv_tokens(G, kr / G, cx["S"]["pop"]["bpt"][cx["cid_m"][cid]], d["dup"])
            mult = (d["G"] / G) if op["gpu_scale"] else 1.0
            out.append(dict(G=G, f=f, gpus=G, kvtok=kt, C=G * C_REF * np_["m_h"] * e_par, mult=mult,
                            off=off_spec(geo, G, f, op), kv_gib=kr))
    return out


# ------------------------------------------------------------------------------------------------ traces
def donor_cells(cx, donor):
    m = cx["cid_m"][full_cid(cx, DONORS[donor])]
    tr = cx["tr"]
    k = (tr.mid == m) & (tr.minute >= HELD)
    return dict(minute=tr.minute[k].astype(np.int64), n=tr.n[k].astype(np.int64), it=tr.it[k].astype(float),
                ot=tr.ot[k].astype(float), N=int(tr.n[k].sum()))


def thin(dc, lam, seed, ri, di, j=0, rot=0):
    """prereg §5.1: per-cell n' ~ floor(p) n + Binomial(n, p - floor(p)), p = lam / N_donor; tokens scale n'/n; copy j uses seed
    1000 j + seed and a cyclic rotation of rot minutes inside the held-out week."""
    rng = np.random.default_rng([1000 * j + seed, 2761, ri, di])
    p = lam / dc["N"]
    fl = math.floor(p)
    n2 = fl * dc["n"] + rng.binomial(dc["n"], p - fl)
    keep = n2 > 0
    sc = n2[keep] / dc["n"][keep]
    mins = dc["minute"][keep]
    if rot:
        mins = HELD + ((mins - HELD + rot) % WEEK_MIN)
    return mins, n2[keep], dc["it"][keep] * sc, dc["ot"][keep] * sc


def synth_trace(parts, models):
    """parts = [(minute, n, it, ot)] one per model index -> a T.Trace (sorted by minute, one cell per (minute, model))."""
    mins = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    mid = np.concatenate([np.full(len(p[0]), j, np.int64) for j, p in enumerate(parts)]) if parts else np.zeros(0, np.int64)
    n = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
    it = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0)
    ot = np.concatenate([p[3] for p in parts]) if parts else np.zeros(0)
    return T.Trace(mins, mid, n, it, ot, models)


def giant_model(cx, cid, tag=""):
    m = cx["cid_m"][cid]
    return (cid + tag,) + tuple(cx["tr"].models[m][1:])


# ------------------------------------------------------------------------------------------------ the sleep path (fork of run_dgroups)
def sleep_core(trace, members, Gc, np_, tph, seed, shed_tau=None, h2d_gpu=H2D):
    """FORK of cloud_sim.run_dgroups for ONE G-class: members = [(m, home, W, bpt_eff, tb, cid)] on a trace with
    len(trace.models) = M. Adds shed_tau (v2's KV shed) and h2d per GPU. -> (week {m: Counter}, day {m: [7]}, stats) or raises."""
    M = len(trace.models)
    bh = NS.b_h(np_)
    card = Gc * bh
    s = np_["s_g"] * Gc
    homes = [mb[1] for mb in members]
    Kc = max(homes) + 1
    nh = collections.Counter(homes)
    units, kt, bpt = [], [], []
    host = collections.Counter()
    for k, (m, h, Wg, b, tb, nm) in enumerate(members):
        kv = card - Wg - Gc * np_["o_g"] - s * (nh[h] - 1)
        if kv < KV_MIN * Gc - 1e-9:
            raise RuntimeError(f"PAIR KV floor: member {k} {kv:.2f} GiB")
        host[h] += Wg * np_["r_pin"] + Gc * O_PROC
        units.append(dict(cid=nm, rep=m, members=[m], tb=tb, prec=np_["fmt"], w=Wg, f=Wg + Gc * np_["o_g"] + kv,
                          shape=(tb, np_["fmt"])))
        kt.append(kv * G30 / b)
        bpt.append(b)
    for g, hh in host.items():
        if hh > NS.d_host_cap(Gc, np_) + 1e-9:
            raise RuntimeError(f"PAIR host RAM: group {g} {hh:.1f} GiB")
    cfg, _ = V2.arm_cfg2("ALL")
    cfg = dict(cfg, copies=False, topo="direct", h2d=Gc * h2d_gpu, d2h=Gc * NS.D2H_PCIE)
    cfg["protect"] = tph["protect"]
    gp = NS.giant_phys(np_, tph)
    unit_of = [-1] * M
    for k, mb in enumerate(members):
        unit_of[mb[0]] = k
    keep = [mb[0] for mb in members]
    st = NS.relabel_days(NS.subtrace(trace, keep), M)
    out, met = V2.simulate(st, units, unit_of * (1 + NDAYS), ("card", homes), Kc, cfg, seed, None, card_gib=card, s=s, phys=gp,
                           kvtok=kt, copies=None, C=Gc * C_REF * np_["m_h"] * np_["e_par"], mech=None, shed_tau=shed_tau,
                           bpt=bpt, resize=None)
    w, dy = NS.split_days(out, M)
    return {m: w[m] for m in keep}, {m: [dy[d][m] for d in range(NDAYS)] for m in keep}, met["stats"]


def sleep_groups_real(cx, np_, groups, seed, shed_tau=None, h2d_gpu=H2D):
    """Real-trace wrapper with run_dgroups' signature: groups = [(G, [cid ...]) ...]."""
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
        w, dy, stt = sleep_core(tr, members, G, np_, cx["tph"], seed, shed_tau, h2d_gpu)
        ow.update(w)
        od.update(dy)
        stats[str(G)] = stt
    return ow, od, stats


# ------------------------------------------------------------------------------------------------ job runners
def np_cm(cx, cm):
    return dict(cx["np"], e_par=E_PAR[cm])


def _rep_rows(res_m):
    return {fk: [[n, jnum(p)] for n, p in reps_of(res_m[fk][0], res_m[fk][1])] for fk, _ in FLOOR_KEYS}


def run_ladder(a, root, cache):
    cx8 = _cx_for(a, root, cache, "fp8")
    cid = full_cid(cx8, a.giant)
    e_par = E_PAR[a.cm]
    op = dict(OPHYS0, **SENS[a.sens])
    arms = a.arms.split(",")
    dc = _donor_cells(cx8, a.donor)
    di = _donor_idx(a.donor)
    gmodel = [giant_model(cx8, cid)]
    configs = []
    if "CTRL" in arms:
        configs += [("CTRL", c) for c in ctrl_specs(cx8, cid, e_par)]
    if "INT4" in arms:
        cx4 = _cx_for(a, root, cache, "int4")
        configs += [("INT4", c) for c in ctrl_specs(cx4, cid, e_par)]
    if "OFF" in arms:
        configs += [("OFF", c) for c in off_configs(cx8, cid, e_par, op)]
    if "BF16CTRL" in arms:                                   # synthetic giants only (main() refuses it otherwise)
        configs += [("BF16CTRL", c) for c in bf16_ctrl_specs(cx8, cid, e_par)]
    if "BF16OFF" in arms:
        configs += [("BF16OFF", c) for c in bf16_off_configs(cx8, cid, e_par, op)]
    gp = cx8["gp"]
    rows = 0
    with open(a.out, "a") as fh:
        for seed in SEEDS:
            traces = {}
            alive = {i: True for i in range(len(configs))}
            for ri, lam in enumerate(RUNGS):
                if ri not in traces:
                    traces[ri] = synth_trace([thin(dc, lam, seed, ri, di)], gmodel)
                tr = traces[ri]
                for ci, (arm, c) in enumerate(configs):
                    if not alive[ci]:
                        continue
                    t0 = time.time()
                    res = gengine(tr, {0: dict(kvtok=c["kvtok"], C=c["C"], mult=c["mult"], off=c["off"])}, gp)
                    rr = _rep_rows(res[0])
                    row = dict(kind="ladder", giant=a.giant, donor=a.donor, cm=a.cm, sens=a.sens, arm=arm, G=c["G"],
                               r=c.get("r", 1), f=c.get("f"), gpus=c["gpus"], seed=seed, rung=lam, ri=ri,
                               n_week=int(sum(res[0]["ttft"][0].values())), reps=rr, elapsed_s=round(time.time() - t0, 3))
                    if "synth" in cx8:
                        row["borrow"] = cx8["synth"]["borrow"][:8]
                    fh.write(json.dumps(row) + "\n")
                    rows += 1
                    if arm in ("OFF", "BF16OFF") and not a.noprune:
                        lenient = [(n, unj(p)) for n, p in rr["ttft"]]
                        if meets(lenient, 10.0) is False:
                            alive[ci] = False
            fh.flush()
    return rows


def run_pair_ladder(a, root, cache):
    cx8 = _cx_for(a, root, cache, "fp8")
    cid = full_cid(cx8, a.giant)
    d = cx8["tab"][cid]
    np2 = np_cm(cx8, a.cm)
    dc = _donor_cells(cx8, a.donor)
    di = _donor_idx(a.donor)
    n = a.n
    tau = None if a.tau == "none" else float(a.tau)
    m0 = cx8["cid_m"][cid]
    b = cx8["S"]["pop"]["bpt"][m0] * d["dup"]
    tb = cx8["tr"].models[m0][1]
    models = [giant_model(cx8, cid, f"#{j}") for j in range(n)]
    members = [(j, 0, d["W"], b, tb, f"{cid}#{j}") for j in range(n)]
    rows = 0
    with open(a.out, "a") as fh:
        try:                                   # feasibility of n sleepers on one group (prereg §5.1)
            sleep_core(synth_trace([(np.array([HELD], np.int64), np.array([1]), np.array([1.0]), np.array([1.0]))] +
                                   [(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0))] * (n - 1), models),
                       members, d["G"], np2, cx8["tph"], 0, tau)
        except RuntimeError as e:
            fh.write(json.dumps(dict(kind="pair_ladder", giant=a.giant, n=n, donor=a.donor, cm=a.cm, tau=a.tau,
                                     infeasible=str(e))) + "\n")
            return 1
        for seed in SEEDS:
            for ri, lam in enumerate(RUNGS):
                parts = [thin(dc, lam, seed, ri, di, j=j, rot=j * (WEEK_MIN // n)) for j in range(n)]
                tr = synth_trace(parts, models)
                t0 = time.time()
                w, dy, stt = sleep_core(tr, members, d["G"], np2, cx8["tph"], seed, tau)
                rr = {str(j): [[nn, jnum(p)] for nn, p in reps_of(w[j], dy[j])] for j in range(n)}
                row = dict(kind="pair_ladder", giant=a.giant, n=n, donor=a.donor, cm=a.cm, tau=a.tau, G=d["G"], gpus=d["G"] / n,
                           seed=seed, rung=lam, ri=ri, reps=rr, stats={k: stt.get(k, 0) for k in
                                                                      ("wakes", "evictions", "abandoned_wakes", "shed_req_held")},
                           elapsed_s=round(time.time() - t0, 3))
                fh.write(json.dumps(row) + "\n")
                rows += 1
                fail = any(meets([(nn, unj(p)) for nn, p in rr[str(j)]], 10.0) is False for j in range(n))
                if fail:
                    break
            fh.flush()
    return rows


def run_fleet(a, root, cache):
    cx8 = _cx_for(a, root, cache, "fp8")
    cid = full_cid(cx8, a.giant)
    m = cx8["cid_m"][cid]
    e_par = E_PAR[a.cm]
    cx4 = _cx_for(a, root, cache, "int4")
    st = _fleet_trace(cx8, cid, m)
    configs = [("CTRL", c) for c in ctrl_specs(cx8, cid, e_par)] + [("INT4", c) for c in ctrl_specs(cx4, cid, e_par)]
    configs += [("OFF", c) for c in off_configs(cx8, cid, e_par, OPHYS0)]
    rows = 0
    with open(a.out, "a") as fh:
        for arm, c in configs:
            t0 = time.time()
            res = gengine(st, {m: dict(kvtok=c["kvtok"], C=c["C"], mult=c["mult"], off=c["off"])}, cx8["gp"])
            row = dict(kind="fleet", giant=a.giant, cm=a.cm, arm=arm, G=c["G"], r=c.get("r", 1), f=c.get("f"), gpus=c["gpus"],
                       n_week=int(sum(res[m]["ttft"][0].values())), reps=_rep_rows(res[m]), elapsed_s=round(time.time() - t0, 3))
            fh.write(json.dumps(row) + "\n")
            rows += 1
        c = ctrl_specs(cx8, cid, e_par)
        if c:
            res = gengine(st, {m: dict(kvtok=c[0]["kvtok"], C=c[0]["C"], mult=1.0, off=None)}, cx8["gp"], window=(TRAIN0, HELD))
            fh.write(json.dumps(dict(kind="train_feas", giant=a.giant, cm=a.cm,
                                     train={fk: [int(sum(res[m][fk][0].values())), jnum(T._pct(res[m][fk][0], 0.95))]
                                            for fk, _ in FLOOR_KEYS})) + "\n")
            rows += 1
    return rows


def train_hours(cx, cid, window_min):
    m = cx["cid_m"][cid]
    tr = cx["tr"]
    k = (tr.mid == m) & (tr.minute >= TRAIN0) & (tr.minute < HELD)
    return set(((tr.minute[k] - TRAIN0) // window_min).tolist())


def home_class(cands, hours, feas, W, Gc, np_):
    """prereg §3.2 homing for one G-class. cands sorted by (-train_req, cid). -> (groups dedicated [cid])."""
    bh = NS.b_h(np_)
    s = np_["s_g"] * Gc
    groups, ded = [], []
    for c in cands:
        if not feas[c]:
            ded.append(c)
            continue
        placed = False
        for g in groups:
            if any(hours[c] & hours[x] for x in g):
                continue
            mem = g + [c]
            if any(Gc * bh - W[x] - Gc * np_["o_g"] - s * (len(mem) - 1) < KV_MIN * Gc - 1e-9 for x in mem):
                continue
            if sum(W[x] * np_["r_pin"] + Gc * O_PROC for x in mem) > NS.d_host_cap(Gc, np_) + 1e-9:
                continue
            g.append(c)
            placed = True
            break
        if not placed:
            groups.append([c])
    return groups, ded


def run_pair_fleet(a, root, cache):
    cx8 = cx_of(root, cache, "fp8")
    np2 = np_cm(cx8, a.cm)
    e_par = E_PAR[a.cm]
    tau = None if a.tau == "none" else float(a.tau)
    tab = cx8["tab"]
    tr = cx8["tr"]
    feas, hours, W, treq = {}, {}, {}, {}
    for cid, d in tab.items():
        if not d.get("placeable"):
            continue
        m = cx8["cid_m"][cid]
        c = ctrl_specs(cx8, cid, e_par)[0]
        res = gengine(NS.subtrace(tr, [m]), {m: dict(kvtok=c["kvtok"], C=c["C"], mult=1.0, off=None)}, cx8["gp"],
                      window=(TRAIN0, HELD))
        cnt = res[m][FK_POINT][0]
        ntr = int(sum(cnt.values()))
        feas[cid] = True if ntr < MIN_REQ else (T._pct(cnt, 0.95) <= 10.0)
        hours[cid] = train_hours(cx8, cid, a.window)
        W[cid] = d["W"]
        treq[cid] = d.get("train_req") or 0
    byG = collections.defaultdict(list)
    for cid in feas:
        byG[tab[cid]["G"]].append(cid)
    plan = {}
    for G, cs in byG.items():
        cs = sorted(cs, key=lambda c: (-treq[c], c))
        plan[G] = home_class(cs, hours, feas, W, G, np2)
    rows = 0
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(kind="pair_plan", cm=a.cm, tau=a.tau, window=a.window,
                                 feas={c[:8]: feas[c] for c in feas},
                                 plan={str(G): dict(groups=[[c[:8] for c in g] for g in p[0]], dedicated=[c[:8] for c in p[1]])
                                       for G, p in plan.items()})) + "\n")
        rows += 1
        for G, (groups, _) in sorted(plan.items()):
            for g in groups:
                if len(g) < 2:
                    continue
                for seed in SEEDS:
                    t0 = time.time()
                    w, dy, stt = sleep_groups_real(cx8, np2, [(G, g)], seed, tau)
                    rr = {c[:8]: [[n, jnum(p)] for n, p in reps_of(w[cx8["cid_m"][c]], dy[cx8["cid_m"][c]])] for c in g}
                    fh.write(json.dumps(dict(kind="pair_group", cm=a.cm, tau=a.tau, window=a.window, G=G,
                                             members=[c[:8] for c in g], seed=seed, reps=rr, stats=stt.get(str(G), stt),
                                             elapsed_s=round(time.time() - t0, 2))) + "\n")
                    rows += 1
    return rows


# ------------------------------------------------------------------------------------------------ synthetic giants (ADDITIVE)
# A giant born after the census (no trace): a census-shaped row from a published config, KV occupancy and real traffic BORROWED from a
# census giant ((private report)). The census context is COPIED, never mutated, and only
# a synthetic --giant reaches this block, so every census giant's output is byte-identical (addendum §6).
SYNTH = {
    # Muse Spark Open (Meta, 2026-10-15). P: HF API safetensors total (BF16 821,906,490,368 + 130,039,296), meta-models repo sha
    # 0fb02478; P_text: HF API gguf.total of unsloth/Muse-Spark-Open-816B-A42B-GGUF (sha c0067056) = the hand sum of its 1,429 tensor
    # shapes. Geometry: Meta card (62 layers, 128 Q / 16 KV heads x 64, 256 experts top-8) + GGUF header (expert_feed_forward_length
    # 4096, moe_latent_size 4096, sliding_window 2048, pattern 46 sliding / 16 full, no shared-expert or dense-FFN tensor). Read
    # 2026-10-15T20:35Z-20:36Z.
    "muse816b": dict(name="meta-models/Muse-Spark-Open-816B-A42B", P=822036529664, P_text=819991575808, experts=256, topk=8,
                     moe_inter=4096, latent=4096, moe_layers=62, layers=62, full_layers=16, sliding_layers=46, window=2048, n_kv=16,
                     head_dim=64, n_heads=128, borrow="a4d621aa", borrow_sens="51c86bd2"),
}
SYNTH_DONOR_IDX = {"BORROW": 2}       # D-BORROW: the borrowed giant's TRAIN week shifted into the held-out week (addendum §3)
BF16_G = (1, 2, 4, 8, 16, 32)         # bf16 G_min search goes past mg_layout's 16 (Muse bf16 needs 32)
BF16_G_OFF = (8, 16)                  # BF16OFF groups (addendum §4; 16 is outside the frozen G_OFF)
SYNTH_ARMS = ("BF16CTRL", "BF16OFF")
B_BF16 = 2.0


def synth_full(sid):
    return f"{sid}-synth-2784"


def synth_memory(sid, fmt, P=None, ne_rate=None):
    """Pure arithmetic -> (W, E, NE GiB, routed expert params, A_R2). fp8 / int4 at the sim's rates (B_EXP, B_NE); bf16 at 2 B/param.
    ne_rate overrides the non-expert bytes per param (the fp8 sensitivity, addendum §1)."""
    s = SYNTH[sid]
    P = s["P"] if P is None else P
    Ep = s["experts"] * 3 * s["latent"] * s["moe_inter"] * s["moe_layers"]
    A = P - (s["experts"] - s["topk"]) * 3 * s["latent"] * s["moe_inter"] * s["moe_layers"]
    be = B_BF16 if fmt == "bf16" else NS.B_EXP[fmt]
    bn = ne_rate if ne_rate is not None else (B_BF16 if fmt == "bf16" else NS.B_NE[fmt])
    e = Ep * be / G30
    ne = (P - Ep) * bn / G30
    return e + ne, e, ne, Ep, A


def synth_kv(sid):
    """KV per token as the census writes it (bf16 KV, GQA): per = 2 n_kv head_dim 2 B per layer."""
    s = SYNTH[sid]
    per = 2.0 * s["n_kv"] * s["head_dim"] * V2.SC.KV_BYTES
    return dict(layers=s["layers"], full_layers=s["full_layers"], sliding_layers=s["sliding_layers"], linear_layers=0, other_layers=0,
                window=s["window"], layer_src="gguf_sliding_window_pattern", attn="GQA", n_kv=s["n_kv"], head_dim=s["head_dim"],
                n_heads=s["n_heads"], bpt_all=per * s["layers"], bpt_global=per * s["full_layers"], bpt_sw=per * s["sliding_layers"])


def bf16_layout(W, bh, o_g):
    for G in BF16_G:
        if W / G + o_g + KV_MIN <= bh + 1e-12:
            return G, W / G, bh - W / G - o_g
    return None


def occ_L(cx, m):
    """v2 build_pop2's KV occupancy for model m: (sum in + kappa sum out) / sum n over the bpt_occ window (bincount, same floats)."""
    tr, ph = cx["tr"], cx["tph"]
    k = (tr.minute >= HELD) if ph["bpt_occ"] == "heldout" else (tr.minute < HELD)
    M = len(tr.models)
    sn = np.bincount(tr.mid[k], weights=tr.n[k].astype(float), minlength=M)
    si = np.bincount(tr.mid[k], weights=tr.it[k], minlength=M)
    so = np.bincount(tr.mid[k], weights=tr.ot[k], minlength=M)
    L = np.where(sn > 0, (si + ph["kappa"] * so) / np.maximum(sn, 1), 0.0)
    return float(L[m])


def synth_cx(root, cache, fmt, sid, borrow8=None):
    """A COPY of the census context with one synthetic giant appended (model index M). Raises if the occupancy recomputation does
    not reproduce the census bpt of every sliding-window census MG giant (addendum §7)."""
    borrow8 = borrow8 or SYNTH[sid]["borrow"]
    key = ("synth", root, cache, fmt, sid, borrow8)
    if key in _CX:
        return _CX[key]
    base = cx_of(root, cache, fmt)
    for c, d in base["tab"].items():                     # the recomputation must reproduce v2's bpt where occupancy matters
        r = base["rows"][c]
        if (r.get("kv_sliding_layers") or 0) > 0:
            kd = {a[3:]: r[a] for a in r if a.startswith("kv_")}
            mc = base["cid_m"][c]
            if V2.SC.bpt_effective(kd, occ_L(base, mc), "SW_AWARE") != base["S"]["pop"]["bpt"][mc]:
                raise ValueError(f"occupancy recomputation does not reproduce the census bpt of {c[:8]}")
    s = SYNTH[sid]
    cid = synth_full(sid)
    bc = full_cid(base, borrow8)
    mb = base["cid_m"][bc]
    P = s["P"]
    _, _, _, Ep, A = synth_memory(sid, "fp8")
    kv = synth_kv(sid)
    row = dict(cid=cid, name=s["name"], status="SYNTH", params_total_R2=P, params_active_R2=A,
               moe=dict(experts=s["experts"], topk=s["topk"], moe_inter=s["moe_inter"], moe_layers=s["moe_layers"]),
               train_req=base["tab"][bc].get("train_req"), held_req=base["tab"][bc].get("held_req"),
               **{"kv_" + k: v for k, v in kv.items()})
    tab1 = NS.mg_table({cid: row}, [cid], base["np"])
    L = occ_L(base, mb)
    bpt = V2.SC.bpt_effective(kv, L, "SW_AWARE")
    M = len(base["tr"].models)
    w8 = V2.SC.weights_gib_fp8(P)
    model = (cid, P / 1e9, A / 1e9, "fp8", w8, w8 + V2.OVH + T.KV_GIB)
    tr0 = base["tr"]
    tr2 = T.Trace.__new__(T.Trace)
    tr2.minute, tr2.mid, tr2.n, tr2.it, tr2.ot, tr2.explicit = tr0.minute, tr0.mid, tr0.n, tr0.it, tr0.ot, tr0.explicit
    tr2.models = list(tr0.models) + [model]
    np_ = base["np"]
    bh = NS.b_h(np_)
    Wb, Eb, NEb, _, _ = synth_memory(sid, "bf16")
    lb = bf16_layout(Wb, bh, np_["o_g"])
    sx = dict(base)
    sx["tab"] = dict(base["tab"], **{cid: tab1[cid]})
    sx["rows"] = dict(base["rows"], **{cid: row})
    sx["cid_m"] = dict(base["cid_m"], **{cid: M})
    sx["by8"] = dict(base["by8"], **{sid: cid})
    sx["tr"] = tr2
    sx["M"] = M + 1
    sx["S"] = dict(base["S"], pop=dict(base["S"]["pop"], bpt=list(base["S"]["pop"]["bpt"]) + [bpt]))
    sx["synth"] = dict(sid=sid, cid=cid, borrow=bc, mb=mb, L=L, bpt=bpt, A=A, Ep=Ep)
    sx["bf16"] = dict(W=Wb, E=Eb, NE=NEb, placeable=lb is not None, G=lb[0] if lb else None, L0=lb[2] if lb else None, dup=1.0)
    _CX[key] = sx
    return sx


def _cx_for(a, root, cache, fmt):
    if a.giant in SYNTH:
        return synth_cx(root, cache, fmt, a.giant, getattr(a, "borrow", None))
    return cx_of(root, cache, fmt)


def _donor_cells(cx, donor):
    if donor not in SYNTH_DONOR_IDX:
        return donor_cells(cx, donor)
    mb = cx["synth"]["mb"]
    tr = cx["tr"]
    k = (tr.mid == mb) & (tr.minute >= TRAIN0) & (tr.minute < HELD)
    return dict(minute=tr.minute[k].astype(np.int64) + WEEK_MIN, n=tr.n[k].astype(np.int64), it=tr.it[k].astype(float),
                ot=tr.ot[k].astype(float), N=int(tr.n[k].sum()))


def _donor_idx(donor):
    return DONOR_IDX[donor] if donor in DONOR_IDX else SYNTH_DONOR_IDX[donor]


def _fleet_trace(cx, cid, m):
    """Census giant: its own subtrace (unchanged). Synthetic: the BORROWED giant's cells relabelled to the synthetic model index."""
    if "synth" not in cx or cid != cx["synth"]["cid"]:
        return NS.subtrace(cx["tr"], [m])
    st = NS.subtrace(cx["tr"], [cx["synth"]["mb"]])
    st.mid = np.full_like(st.mid, m)
    return st


def bf16_ctrl_specs(cx, cid, e_par):
    """BF16CTRL (addendum §4): dedicated bf16 at its own G_min (past 16), r = 1, 2, 4; same engine, KV and compute rules as CTRL."""
    b = cx["bf16"]
    if not b["placeable"]:
        return []
    kt = NS.kv_tokens(b["G"], b["L0"], cx["S"]["pop"]["bpt"][cx["cid_m"][cid]], b["dup"])
    C = b["G"] * C_REF * cx["np"]["m_h"] * e_par
    return [dict(r=r, G=b["G"], gpus=r * b["G"], kvtok=r * kt, C=r * C, mult=1.0, off=None) for r in R_GRID]


def bf16_off_configs(cx, cid, e_par, op):
    """BF16OFF (addendum §4): bf16 experts in pinned host RAM, G in BF16_G_OFF below the bf16 G_min, prereg §3.1's HBM + host rules."""
    b = cx["bf16"]
    geo = geometry(cx, cid)
    if geo is None or not b["placeable"]:
        return []
    np_ = cx["np"]
    bh = NS.b_h(np_)
    Egb = b["E"] * G30 / 1e9
    geo = dict(geo, E_gb=Egb, e_gb=Egb / (geo["n"] * geo["L"]), E=b["E"], NE=b["NE"], W=b["W"])
    out = []
    for G in BF16_G_OFF:
        if G >= b["G"] or not host_fit(b["W"], G, np_["node_ram"], np_["r_pin"]):
            continue
        for f in F_GRID:
            kr = kv_room(G, f, b["E"], b["NE"], bh, np_["o_g"])
            if kr < G * KV_MIN - 1e-9:
                continue
            kt = NS.kv_tokens(G, kr / G, cx["S"]["pop"]["bpt"][cx["cid_m"][cid]], b["dup"])
            mult = (b["G"] / G) if op["gpu_scale"] else 1.0
            out.append(dict(G=G, f=f, gpus=G, kvtok=kt, C=G * C_REF * np_["m_h"] * e_par, mult=mult, off=off_spec(geo, G, f, op),
                            kv_gib=kr))
    return out


def synth_table(sid, root=None, cache=None):
    """Read-only: memory per format (point, text-only P, fp8 with bf16 non-experts), G_min, f_max per G, host fit; with --root also the
    borrowed occupancy and bpt per borrowed profile."""
    s = SYNTH[sid]
    np8 = NS.nphys_with({"fmt": "fp8"})
    bh = NS.b_h(np8)
    out = []
    for tag, fmt, P, ner in (("point", "fp8", None, None), ("point", "int4", None, None), ("point", "bf16", None, None),
                             ("text_only", "fp8", s["P_text"], None), ("text_only", "bf16", s["P_text"], None),
                             ("ne_bf16", "fp8", None, B_BF16)):
        W, E, NE, Ep, A = synth_memory(sid, fmt, P, ner)
        if fmt == "bf16":
            lay = bf16_layout(W, bh, np8["o_g"])
            G, L0 = (lay[0], lay[2]) if lay else (None, None)
        else:
            lay = NS.mg_layout(W, E, NE, True, "DCP", bh, np8["o_g"])
            G, L0 = (lay[0], lay[2]) if lay else (None, None)
        gs = (1, 2, 4, 8, 16)
        out.append(dict(giant=sid, reading=tag, fmt=fmt, P=P or s["P"], A_R2=A, E_params=Ep, W=round(W, 4), E=round(E, 4),
                        NE=round(NE, 4), W_GB=round(W * G30 / 1e9, 2), G=G, L0=round(L0, 4) if L0 is not None else None,
                        f_max={g: round(f_max(g, E, NE, bh, np8["o_g"]), 4) for g in gs},
                        host_fit={g: host_fit(W, g, np8["node_ram"], np8["r_pin"]) for g in gs},
                        host_pin_gib={g: round(W * np8["r_pin"] + g * O_PROC, 1) for g in gs},
                        fits_16_unit=bool(G is not None and G <= 16)))
    if root:
        for b8 in (s["borrow"], s["borrow_sens"]):
            sx = synth_cx(root, cache, "fp8", sid, b8)
            out.append(dict(giant=sid, reading="borrow", borrow=b8, L_occ=round(sx["synth"]["L"], 4), bpt=sx["synth"]["bpt"],
                            kvtok_ctrl_r1=round(ctrl_specs(sx, sx["synth"]["cid"], E_PAR["POINT"])[0]["kvtok"], 1),
                            held_req=sx["tab"][sx["synth"]["cid"]]["held_req"], train_req=sx["tab"][sx["synth"]["cid"]]["train_req"]))
    return out


# ------------------------------------------------------------------------------------------------ instrument checks (prereg §8)
def check(which, root, cache):
    cx8 = cx_of(root, cache, "fp8")
    out = dict(check=which)
    if which == "EQ":
        ok = True
        for c8 in ("51c86bd2", "aa605b6d"):
            cid = full_cid(cx8, c8)
            m = cx8["cid_m"][cid]
            st = NS.subtrace(cx8["tr"], [m])
            c = ctrl_specs(cx8, cid, E_PAR["POINT"])[0]
            res = gengine(st, {m: dict(kvtok=c["kvtok"], C=c["C"], mult=1.0, off=None)}, cx8["gp"])
            M = cx8["M"]
            kt = [-1.0] * M
            kt[m] = c["kvtok"]
            Cs = [1.0] * M
            Cs[m] = c["C"]
            w, dy = NS.dedicated_by_day(st, kt, cx8["gp"], Cs)
            eq = res[m]["ttft"][0] == w[m] and all(res[m]["ttft"][1][d] == dy[d][m] for d in range(NDAYS))
            out[c8] = dict(equal=eq, n=int(sum(w[m].values())))
            ok = ok and eq
        out["pass"] = ok
        return out
    if which == "SLEEP":
        np_ = cx8["np"]
        ok = True
        for G, mem in ((4, ("aa605b6d", "29c2ca39")), (8, ("7b546c7d", "ae206d64"))):
            g = [full_cid(cx8, x) for x in mem]
            a1 = CS.run_dgroups(cx8, np_, [(G, g)], 0)
            a2 = sleep_groups_real(cx8, np_, [(G, g)], 0)
            eq = a1[0] == a2[0] and a1[1] == a2[1] and a1[2] == a2[2]
            out[f"G{G}"] = dict(equal=eq, members=list(mem), n={x: int(sum(a1[0][cx8["cid_m"][c]].values())) for x, c in zip(mem, g)})
            ok = ok and eq
        out["pass"] = ok
        return out
    if which == "STALL":
        cx4 = cx_of(root, cache, "int4")
        np4 = cx4["np"]
        g = [full_cid(cx4, "ae206d64"), full_cid(cx4, "aa605b6d")]
        m = cx4["cid_m"][g[1]]
        for tau in (None, TAU_POINT):
            for seed in SEEDS:
                w, dy, stt = sleep_groups_real(cx4, np4, [(4, g)], seed, tau)
                out[f"tau{tau}_s{seed}"] = dict(p95_aa605b6d=jnum(T._pct(w[m], 0.95)), n=int(sum(w[m].values())),
                                                 abandoned=stt["4"].get("abandoned_wakes", 0), shed=stt["4"].get("shed_req_held", 0))
        out["shape_reproduced"] = all(out[f"tauNone_s{s}"]["p95_aa605b6d"] == "inf" and out[f"tauNone_s{s}"]["abandoned"] > 0
                                      for s in SEEDS)
        out["pass"] = True
        return out
    if which == "THIN":
        ok = True
        for donor in DONORS:
            dc = donor_cells(cx8, donor)
            for ri, lam in enumerate(RUNGS):
                for seed in SEEDS:
                    a = thin(dc, lam, seed, ri, DONOR_IDX[donor])
                    b = thin(dc, lam, seed, ri, DONOR_IDX[donor])
                    det = all(np.array_equal(x, y) for x, y in zip(a, b))
                    tot = int(a[1].sum())
                    p = lam / dc["N"]
                    fr = p - math.floor(p)
                    sd = math.sqrt(dc["N"] * fr * (1 - fr)) if fr > 0 else 0.0
                    within = abs(tot - lam) <= 5 * sd + 1e-9
                    ok = ok and det and within
                    out[f"{donor}_{lam}_s{seed}"] = [tot, round(sd, 2), det, within]
        out["pass"] = ok
        return out
    if which == "CONS":
        ok = True
        for cid, d in cx8["tab"].items():
            m = cx8["cid_m"][cid]
            st = NS.subtrace(cx8["tr"], [m])
            held = int(cx8["tr"].n[(cx8["tr"].mid == m) & (cx8["tr"].minute >= HELD)].sum())
            c = ctrl_specs(cx8, cid, E_PAR["POINT"])
            if not c:
                continue
            res = gengine(st, {m: dict(kvtok=c[0]["kvtok"], C=c[0]["C"], mult=1.0, off=None)}, cx8["gp"])
            wk, dy = res[m]["ttft"]
            good = int(sum(wk.values())) == held == (d.get("held_req") or 0) and \
                sum(sum(x.values()) for x in (wk - dy[dd] for dd in range(NDAYS))) == 6 * held
            out[cid[:8]] = [held, good]
            ok = ok and good
        out["pass"] = ok
        return out
    if which == "SEED":
        np_ = cx8["np"]
        cid = full_cid(cx8, "aa605b6d")
        m = cx8["cid_m"][cid]
        st = NS.subtrace(cx8["tr"], [m])
        c = ctrl_specs(cx8, cid, E_PAR["POINT"])[0]
        r1 = gengine(st, {m: dict(kvtok=c["kvtok"], C=c["C"], mult=1.0, off=None)}, cx8["gp"])
        r2 = gengine(st, {m: dict(kvtok=c["kvtok"], C=c["C"], mult=1.0, off=None)}, cx8["gp"])
        out["engine_takes_no_seed_and_repeats"] = r1 == r2
        g = [full_cid(cx8, "aa605b6d"), full_cid(cx8, "29c2ca39")]
        a0 = sleep_groups_real(cx8, np_, [(4, g)], 0)
        a0b = sleep_groups_real(cx8, np_, [(4, g)], 0)
        out["pair_same_seed_identical"] = a0[0] == a0b[0] and a0[1] == a0b[1]
        out["pass"] = out["engine_takes_no_seed_and_repeats"] and out["pair_same_seed_identical"]
        return out
    raise ValueError(f"unknown check {which}")


# ------------------------------------------------------------------------------------------------ summarize (prereg §5-§6)
def _rows(d):
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(d, fn)) as fh:
                rows += [json.loads(x) for x in fh if x.strip()]
    return rows


def _reps(lst):
    return [(n, unj(p)) for n, p in lst]


def ladder_tables(rows):
    """-> {(giant, donor, cm, sens): {arm-config: {ri: [21 reps per fk]}}} aggregated over seeds."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
    seen = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
    for r in rows:
        if r.get("kind") != "ladder":
            continue
        k = (r["giant"], r["donor"], r["cm"], r["sens"])
        ck = (r["arm"], r["G"], r["r"], r["f"])
        if r["seed"] in seen[k][ck][r["ri"]]:
            raise ValueError(f"duplicate ladder row {k} {ck} ri={r['ri']} seed={r['seed']}")
        seen[k][ck][r["ri"]].add(r["seed"])
        for fk, lst in r["reps"].items():
            acc[k][ck][r["ri"]][fk] += _reps(lst)
    return acc, seen


def lever_curve(cfgs, seen, arm, fk, slo, stable):
    """Per rung: the smallest GPUs of `arm` configs meeting on every replicate of all 3 seeds (stable = met at every rung <= ri)."""
    out = []
    for ri in range(len(RUNGS)):
        best = None
        for ck, byri in cfgs.items():
            if ck[0] != arm:
                continue
            ok = True
            for rj in (range(ri + 1) if stable else (ri,)):
                if len(seen[ck].get(rj, ())) < len(SEEDS) or meets(byri[rj][fk], slo) is not True:
                    ok = False
                    break
            if ok:
                g = ck[1] * ck[2]
                best = g if best is None else min(best, g)
        out.append(best)
    return out


def ctrl_r1_meets(cfgs, seen, fk, slo, ri):
    """CTRL at r = 1 meets the reading at rung ri on all 3 seeds x 7 replicates (amendment 1's PAIR decode-floor proxy)."""
    for ck, byri in cfgs.items():
        if ck[0] == "CTRL" and ck[2] == 1:
            return len(seen[ck].get(ri, ())) == len(SEEDS) and meets(byri[ri][fk], slo) is True
    return False


def summarize(d):
    rows = _rows(d)
    out = dict(n_rows=len(rows))
    acc, seen = ladder_tables(rows)
    lad = {}
    for k, cfgs in sorted(acc.items(), key=lambda kv: kv[0][3] != "base"):
        ent = {}
        bk = (k[0], k[1], k[2], "base")
        for slo in SLOS:
            for fk, _ in FLOOR_KEYS:
                # sensitivity jobs run OFF only (prereg §7): their control and int4 curves are the base key's (a later fix)
                src = cfgs if k[3] == "base" else acc.get(bk, {})
                srs = seen[k] if k[3] == "base" else seen[bk]
                ctrl = lever_curve(src, srs, "CTRL", fk, slo, False)
                offc = lever_curve(cfgs, seen[k], "OFF", fk, slo, True)
                i4 = lever_curve(src, srs, "INT4", fk, slo, False)
                fe_off = [ctrl[i] is not None and offc[i] is not None and offc[i] < ctrl[i] for i in range(len(RUNGS))]
                fe_i4 = [ctrl[i] is not None and i4[i] is not None and i4[i] < ctrl[i] for i in range(len(RUNGS))]
                ent[f"S{int(slo)}_{fk}"] = dict(ctrl=ctrl, off=offc, int4=i4, be_off=break_even(RUNGS, fe_off),
                                                be_int4=break_even(RUNGS, fe_i4))
        lad["|".join(k)] = ent
    out["ladder"] = lad
    # PAIR ladder: per (giant, n, donor, cm, tau): a rung PASSES iff all n copies meet on every replicate of all 3 seeds (TTFT keys
    # from the v2 sleep path); a floor reading also needs CTRL r = 1 to meet it at that rung (amendment 1)
    pl = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    pls = collections.defaultdict(lambda: collections.defaultdict(set))
    pinf = {}
    for r in rows:
        if r.get("kind") != "pair_ladder":
            continue
        k = (r["giant"], str(r["n"]), r["donor"], r["cm"], str(r["tau"]))
        if "infeasible" in r:
            pinf["|".join(k)] = r["infeasible"]
            continue
        if r["seed"] in pls[k][r["ri"]]:
            raise ValueError(f"duplicate pair_ladder row {k} ri={r['ri']} seed={r['seed']}")
        pls[k][r["ri"]].add(r["seed"])
        for j, lst in r["reps"].items():
            pl[k][r["ri"]][j] += _reps(lst)
    pair = {}
    for k, byri in pl.items():
        g, n, donor, cm, tau = k
        bk = (g, donor, cm, "base")
        ent = {}
        for slo in SLOS:
            for fk, _ in FLOOR_KEYS:
                ctrl = lad.get("|".join(bk), {}).get(f"S{int(slo)}_{fk}", {}).get("ctrl", [None] * len(RUNGS))
                passes = []
                for ri in range(len(RUNGS)):
                    ok = len(pls[k].get(ri, ())) == len(SEEDS) and len(byri[ri]) == int(n) and \
                        all(meets(v, slo) is True for v in byri[ri].values())
                    if ok and fk != "ttft":
                        ok = bk in acc and ctrl_r1_meets(acc[bk], seen[bk], fk, slo, ri)
                    passes.append(bool(ok))
                fewer = [passes[i] and ctrl[i] is not None for i in range(len(RUNGS))]
                ent[f"S{int(slo)}_{fk}"] = dict(pass_=passes, be=break_even(RUNGS, fewer))
        pair["|".join(k)] = ent
    out["pair_ladder"] = pair
    out["pair_ladder_infeasible"] = pinf
    out["fleet"] = fleet_summary(rows, lad, pair)
    out["sens"] = sens_summary(lad)
    return out


def _fleet_gpus(frows, arm, fk, slo):
    best = None
    for r in frows:
        if r["arm"] != arm:
            continue
        if meets(_reps(r["reps"][fk]), slo) is True:
            best = r["gpus"] if best is None else min(best, r["gpus"])
    return best


def fleet_summary(rows, lad, pair):
    """prereg §5.2: per giant per arm GPUs at real traffic; a giant with no scored held-out replicate (zero or < 20 requests per
    replicate) is judged by AVAILABILITY at the ladder's lowest rung, D-ANCHOR (amendment 1). PAIR groups, LOSSLESS composition,
    totals and the 128 / 256 lines, per compute model and deciding."""
    fr = collections.defaultdict(list)
    scored = collections.defaultdict(bool)
    gctrl = {}
    for r in rows:
        if r.get("kind") == "fleet":
            fr[(r["giant"], r["cm"])].append(r)
            if r["arm"] == "CTRL" and r["r"] == 1:
                gctrl[r["giant"]] = r["G"]
                if any(n >= MIN_REQ for n, _ in r["reps"]["ttft"]):
                    scored[r["giant"]] = True
    giants = sorted({g for g, _ in fr})
    res = {}
    for slo in SLOS:
        for fk in ("f100", "ttft", "f50", "f200"):
            key = f"S{int(slo)}_{fk}"
            per = {}
            for cm in E_PAR:
                pg = {}
                for g in giants:
                    if scored[g]:
                        frows = fr.get((g, cm), [])
                        ctrl = _fleet_gpus(frows, "CTRL", fk, slo)
                        off = _fleet_gpus(frows, "OFF", fk, slo)
                        i4 = _fleet_gpus(frows, "INT4", fk, slo)
                        src = "real"
                    else:
                        ent = lad.get(f"{g}|ANCHOR|{cm}|base", {}).get(key, {})
                        ctrl = (ent.get("ctrl") or [None])[0]
                        off = (ent.get("off") or [None])[0]
                        i4 = (ent.get("int4") or [None])[0]
                        src = "avail"
                    pg[g] = dict(ctrl=ctrl, off=off, int4=i4, src=src, gctrl=gctrl.get(g))
                per[cm] = pg
            res[key] = dict(per_giant=per, pair=pair_fleet_summary(rows, per, pair, slo, fk))
            res[key]["totals"] = fleet_totals(res[key], giants)
    return res


def pair_fleet_summary(rows, per, pair, slo, fk):
    """PAIR groups at real traffic: a group meets iff every scored member meets on all scored replicates of all 3 seeds (floor
    readings also need that member's real-traffic CTRL to meet them, amendment 1) and every unscored member passes PAIR(2)
    availability at the lowest rung (D-ANCHOR, tau point)."""
    plans = [r for r in rows if r.get("kind") == "pair_plan"]
    grp = collections.defaultdict(lambda: collections.defaultdict(list))
    gseeds = collections.defaultdict(set)
    for r in rows:
        if r.get("kind") == "pair_group":
            k = (r["cm"], str(r["tau"]), str(r["window"]), tuple(r["members"]))
            if r["seed"] in gseeds[k]:
                raise ValueError(f"duplicate pair_group row {k} seed={r['seed']}")
            gseeds[k].add(r["seed"])
            for c, lst in r["reps"].items():
                grp[k][c] += _reps(lst)
    out = {}
    for p in plans:
        cm, tau, win = p["cm"], str(p["tau"]), str(p["window"])
        pg = per.get(cm, {})
        gl = []
        for G, pl in p["plan"].items():
            for g in pl["groups"]:
                if len(g) < 2:
                    continue
                k = (cm, tau, win, tuple(g))
                ok = len(gseeds.get(k, ())) == len(SEEDS)
                detail = {}
                for c in g:
                    mr = meets(grp[k].get(c, []), slo)
                    if mr is None:
                        av = pair.get(f"{c}|2|ANCHOR|{cm}|{TAU_POINT}", {}).get(f"S{int(slo)}_{fk}", {}).get("pass_", [False])
                        mr = bool(av[0])
                        detail[c] = ["avail", mr]
                    else:
                        if fk != "ttft":
                            mr = mr and pg.get(c, {}).get("ctrl") is not None
                        detail[c] = ["real", mr]
                    ok = ok and bool(mr)
                gl.append(dict(G=int(G), members=g, meets=ok, detail=detail))
        out[f"{cm}|tau{tau}|w{win}"] = gl
    return out


ARMS_TOT = ("CTRL", "OFF", "PAIR", "LOSSLESS", "INT4_Q", "INT4_U")


def fleet_totals(entry, giants):
    """Totals per compute model, and DECIDING (prereg §2 + amendment 1): per giant, the model(s) under which CTRL meets; if both, the one
    with the smaller LOSSLESS saving (ties POINT) supplies every arm for that giant; neither -> 4 x G_ctrl in every arm."""
    per = entry["per_giant"]
    tot = {}
    alloc = {}
    for cm in E_PAR:
        groups = entry["pair"].get(f"{cm}|tau{TAU_POINT}|w60", [])
        alloc[cm] = _alloc(per[cm], groups, giants)
        t = collections.Counter()
        for g in giants:
            for a in ARMS_TOT:
                t[a] += alloc[cm][g][a]
        tot[cm] = dict(t)
        tot[cm]["ctrl_fails"] = [g for g in giants if per[cm][g]["ctrl"] is None]
    dec = collections.Counter()
    which = {}
    for g in giants:
        ok = [cm for cm in E_PAR if per[cm][g]["ctrl"] is not None]
        if not ok:
            which[g] = None
            for a in ARMS_TOT:
                dec[a] += 4 * (per["POINT"][g]["gctrl"] or 0)
            continue
        cm = min(ok, key=lambda c: (alloc[c][g]["CTRL"] - alloc[c][g]["LOSSLESS"], c != "POINT"))
        which[g] = cm
        for a in ARMS_TOT:
            dec[a] += alloc[cm][g][a]
    tot["deciding"] = dict(dec)
    tot["deciding_model"] = which
    for k in list(tot):
        if isinstance(tot[k], dict) and "CTRL" in tot[k]:
            tot[k]["left_128"] = {a: 128 - tot[k][a] for a in ARMS_TOT}
            tot[k]["left_256"] = {a: 256 - tot[k][a] for a in ARMS_TOT}
    return tot


def _alloc(pg, groups, giants):
    """Per-giant GPU allocation per arm (groups split equally). CTRL-FAILS -> 4 x G_ctrl in every arm (prereg §5.2)."""
    out = {}
    for g in giants:
        e = pg[g]
        gc = e["gctrl"] or 0
        if e["ctrl"] is None:
            out[g] = {a: 4 * gc for a in ("CTRL", "OFF", "PAIR", "LOSSLESS", "INT4_Q", "INT4_U")}
            continue
        c = e["ctrl"]
        off = verdict(c, e["off"])[1]
        i4 = verdict(c, e["int4"])[1]
        out[g] = dict(CTRL=c, OFF=off, PAIR=c, LOSSLESS=min(c, off), INT4_Q=(i4 if g in NATIVE4 else c), INT4_U=i4)
    for grp in groups:
        mem = [c for c in grp["members"] if c in out]
        if len(mem) < 2 or not grp["meets"]:
            continue
        if any(pg[c]["ctrl"] is None for c in mem):
            continue
        G = grp["G"]
        for c in mem:
            out[c]["PAIR"] = G / len(mem)
        best_alone = sum(min(pg[c]["ctrl"], verdict(pg[c]["ctrl"], pg[c]["off"])[1]) for c in mem)
        if G < best_alone:
            for c in mem:
                out[c]["LOSSLESS"] = G / len(mem)
    return out


def mono_count(rows):
    """Amendment 1 / I-MONO (descriptive): over UNPRUNED ladder rows, count (giant, donor, cm, config, seed) series that fail the
    lenient reading (S10, TTFT-only) at a rung and pass it at a higher rung — the cases where pruning could change a break-even."""
    ser = collections.defaultdict(dict)
    for r in rows:
        if r.get("kind") != "ladder" or r["arm"] != "OFF":
            continue
        ser[(r["giant"], r["donor"], r["cm"], r["G"], r["f"], r["seed"])][r["ri"]] = meets(_reps(r["reps"]["ttft"]), 10.0)
    bad = []
    for k, d in ser.items():
        seq = [d.get(i) for i in range(len(RUNGS)) if i in d]
        for i, v in enumerate(seq):
            if v is False and any(x is True for x in seq[i + 1:]):
                bad.append(list(k))
                break
    return dict(series=len(ser), non_monotone=len(bad), examples=bad[:10])


def sens_summary(lad):
    out = {}
    for k, ent in lad.items():
        g, donor, cm, sens = k.split("|")
        if donor != "ANCHOR":
            continue
        e = ent.get(f"S10_{FK_POINT}", {})
        out.setdefault(f"{g}|{cm}", {})[sens] = e.get("be_off")
    return out


# ------------------------------------------------------------------------------------------------ table
def anchor_table(fmt, np_):
    W, E, NE, moe = NS.mg_weights({"cid": NS.ANCHOR}, fmt)
    bh = NS.b_h(np_)
    lay = NS.mg_layout(W, E, NE, moe, "DCP", bh, np_["o_g"])
    return dict(cid="51c86bd2", fmt=fmt, W=round(W, 4), E=round(E, 4), NE=round(NE, 4), G=lay[0], L0=round(lay[2], 4),
                need8=round(W / 8 + np_["o_g"] + KV_MIN, 4),
                f_max={G: round(f_max(G, E, NE, bh, np_["o_g"]), 4) for G in G_OFF},
                host_fit={G: host_fit(W, G, np_["node_ram"], np_["r_pin"]) for G in G_OFF})


def table_rows(root, cache):
    if not root:
        return [anchor_table(fmt, NS.nphys_with({"fmt": fmt})) for fmt in ("fp8", "int4")]
    out = []
    for fmt in ("fp8", "int4"):
        cx = cx_of(root, cache, fmt)
        np_ = cx["np"]
        bh = NS.b_h(np_)
        for cid, d in sorted(cx["tab"].items(), key=lambda kv: -(kv[1].get("held_req") or 0)):
            geo = geometry(cx, cid)
            out.append(dict(cid=cid[:8], fmt=fmt, G=d.get("G"), L0=round(d.get("L0", 0), 4) if d.get("placeable") else None,
                            W=round(d["W"], 4), E=round(d["E"], 4), NE=round(d["NE"], 4), held=d.get("held_req"),
                            train=d.get("train_req"), native4=cid[:8] in NATIVE4,
                            f_max={G: (round(f_max(G, d["E"], d["NE"], bh, np_["o_g"]), 4) if geo else None) for G in G_OFF},
                            host_fit={G: host_fit(d["W"], G, np_["node_ram"], np_["r_pin"]) for G in G_OFF},
                            A_r=geo["A_r"] if geo else None, e_gb=round(geo["e_gb"], 6) if geo else None))
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §9)
def self_test():
    res = []

    def chk(name, got, want, tol=None):
        if tol is None:
            ok = got == want
        else:
            ok = got is not None and abs(got - want) <= tol
        res.append(ok)
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": got {got!r} want {want!r}"))

    np8 = NS.nphys_with({"fmt": "fp8"})
    np4 = NS.nphys_with({"fmt": "int4"})
    bh = NS.b_h(np8)
    # GQ1
    a8, a4 = anchor_table("fp8", np8), anchor_table("int4", np4)
    chk("GQ1 kimi fp8 W", a8["W"], 958.5088)
    chk("GQ1 kimi fp8 E", a8["E"], 945.2307)
    chk("GQ1 kimi fp8 NE", a8["NE"], 13.2781)
    chk("GQ1 kimi fp8 G, L0", (a8["G"], a8["L0"]), (16, 5.3365))
    chk("GQ1 kimi int4 W/E/NE", (a4["W"], a4["E"], a4["NE"]), (553.4222, 531.5625, 21.8597))
    chk("GQ1 kimi int4 G, L0 (int4 does NOT fit 8 GPUs)", (a4["G"], a4["L0"]), (16, 30.6544))
    chk("GQ1 kimi int4 per-GPU need on 8", a4["need8"], 75.4278)
    l1 = NS.mg_layout(338.45, 0.0, 338.45, False, "DCP", bh, np8["o_g"])
    l2 = NS.mg_layout(217.95, 0.0, 217.95, False, "DCP", bh, np8["o_g"])
    chk("GQ1 GLM-358B fp8 G, L0", (l1[0], round(l1[2], 4)), (8, 22.937))
    chk("GQ1 GLM-358B int4 G, L0", (l2[0], round(l2[2], 4)), (4, 10.7558))
    # GQ2
    W8, E8, NE8, _ = NS.mg_weights({"cid": NS.ANCHOR}, "fp8")
    for G, want in ((1, 0.0547), (2, 0.1235), (4, 0.2610), (8, 0.5360)):
        chk(f"GQ2 kimi fp8 f_max G={G}", round(f_max(G, E8, NE8, bh, np8["o_g"]), 4), want)
    chk("GQ2 KV room G8 f0.5", round(kv_room(8, 0.5, E8, NE8, bh, np8["o_g"]), 4), 36.0527)
    # GQ3 toy
    toy = dict(h=0.5, n=8, k=2, L=1, e_gb=1.0, E_gb=8.0, A_r=0.0, bwf=10.0, bwc=1e9, fc=1e30, mode="FETCH", overlap="SUM",
               chunk=8192)
    chk("GQ3 toy distinct B=4", round(distinct(4, 8, 2), 4), 5.4688)
    chk("GQ3 toy B=1 host", round(host_decode(1, toy), 4), 0.1)
    chk("GQ3 toy B=4 host", round(host_decode(4, toy), 4), 0.2734)
    chk("GQ3 toy B=4 tok/s", round(4 / (0.02 + host_decode(4, toy)), 3), 13.632)
    toyc = dict(toy, bwc=5.0, mode="CPU")
    chk("GQ3 toy B=1 cpu at 5 GB/s", round(host_decode(1, toyc), 4), 0.2)
    chk("GQ3 toy HYB picks fetch", round(host_decode(1, dict(toyc, mode="HYB")), 4), 0.1)
    chk("GQ3 toy prefill one chunk", round(host_prefill(5000, toy), 4), 0.4)
    chk("GQ3 toy prefill in=10000 two chunks", round(host_prefill(10000, toy), 4), 0.8)
    # GQ3b kimi real
    geo = dict(n=384, k=8, L=60, hidden=7168, A_r=8 * 3 * 7168 * 2048 * 60, E_gb=E8 * G30 / 1e9, e_gb=E8 * G30 / 1e9 / (384 * 60))
    f8 = f_max(8, E8, NE8, bh, np8["o_g"])
    s8 = off_spec(geo, 8, f8, dict(OPHYS0, mode="FETCH"))
    chk("GQ3b e_gb", round(geo["e_gb"], 6), 0.044051)
    chk("GQ3b B=1 host", round(host_decode(1, s8), 5), 0.02279)
    gt = NS.GIANT_TPOT
    for B, want in ((1, 0.04911), (2, 0.07143), (3, 0.09328), (4, 0.11468)):
        chk(f"GQ3b TPOT B={B}", round(gt + host_decode(B, s8), 5), want)
    chk("GQ3b prefill pass", round(host_prefill(6000, s8), 4), 1.0941)
    s4 = off_spec(geo, 4, f_max(4, E8, NE8, bh, np8["o_g"]), dict(OPHYS0, mode="FETCH"))
    chk("GQ3b G4 B=1 TPOT", round(gt + host_decode(1, s4), 5), 0.09893)
    chk("GQ3b G4 pass", round(host_prefill(6000, s4), 4), 3.4853)
    sh = off_spec(geo, 8, f8, OPHYS0)
    chk("GQ3b HYB <= FETCH on Kimi G8", host_decode(1, sh) <= host_decode(1, s8) + 1e-12, True)
    # GQ4
    for c, x, want in ((16, None, ("X-FAILS", 16)), (None, 8, ("DEGENERATE", None)), (16, 8, ("FEWER-GPUS", 8)), (8, 8, ("SAME", 8)),
                       (8, 16, ("MORE-GPUS", 8))):
        chk(f"GQ4 verdict({c},{x})", verdict(c, x), want)
    pg = {"a": dict(ctrl=16, off=None, int4=None, gctrl=16), "b": dict(ctrl=None, off=8, int4=8, gctrl=8)}
    al = _alloc(pg, [], ["a", "b"])
    chk("GQ4 fleet books a failing lever at CTRL", al["a"]["OFF"], 16)
    chk("GQ4 fleet books a CTRL-FAILS giant 4 x G_ctrl in OFF", al["b"]["OFF"], 32)
    pg2 = {"a": dict(ctrl=8, off=None, int4=None, gctrl=8), "b": dict(ctrl=8, off=4, int4=None, gctrl=8)}
    al2 = _alloc(pg2, [dict(G=8, members=["a", "b"], meets=True)], ["a", "b"])
    chk("GQ4 LOSSLESS group 8 vs best-alone 12 -> 4 each", (al2["a"]["LOSSLESS"], al2["b"]["LOSSLESS"], al2["a"]["PAIR"]), (4.0, 4.0, 4.0))
    al3 = _alloc(pg2, [dict(G=8, members=["a", "b"], meets=False)], ["a", "b"])
    chk("GQ4 a failing PAIR group saves nothing", (al3["a"]["PAIR"], al3["b"]["LOSSLESS"]), (8, 4))
    # GQ5
    chk("GQ5 break_even [T,T,F,T]", break_even([30, 100, 300, 1000], [True, True, False, True]), 100)
    chk("GQ5 break_even [F,T]", break_even([30, 100], [False, True]), None)
    # GQ6
    chk("GQ6 kimi int4 wake 16", round(wake_s(594232561304, 16), 4), 0.6903)
    chk("GQ6 kimi fp8 wake 16", round(wake_s(1029190981272, 16), 4), 1.1956)
    chk("GQ6 kimi int4 wake capped", round(wake_s(594232561304, 16, node_cap=200.0), 4), 1.4856)
    # GQ7
    chk("GQ7 BW_cpu", round(BW_CPU_NODE, 2), 151.24)
    chk("GQ7 F_cpu", f"{F_CPU_NODE:.4e}", "1.0432e+13")
    chk("GQ7 kimi cross-check", round(BW_CPU_NODE / (8 * 3 * 7168 * 2048 * 60 * Q4KM_BYTES / 1e9), 2), 11.80)
    # GQ8
    chk("GQ8 kimi fp8 host fit G8", host_fit(W8, 8, np8["node_ram"], np8["r_pin"]), True)
    chk("GQ8 kimi fp8 host fit G4 (infeasible)", host_fit(W8, 4, np8["node_ram"], np8["r_pin"]), False)
    chk("GQ8 two kimi fp8 one node", host_fit(2 * W8 + 8 * O_PROC / np8["r_pin"], 8, np8["node_ram"], np8["r_pin"]), False)
    # GQ9
    chk("GQ9 floor meets", floor_key(1.0, 0.0989, 0.10), T._key(1.0))
    chk("GQ9 floor misses", floor_key(1.0, 0.1013, 0.10), T.INF_KEY)
    chk("GQ9 no floor", floor_key(1.0, 0.9, None), T._key(1.0))
    # meets rule
    chk("meets: a < 20 replicate is NOT-SCORED", meets([(5, 99.0), (30, 1.0)], 10.0), True)
    chk("meets: no scored replicate -> None", meets([(5, 1.0)], 10.0), None)
    chk("meets: inf fails", meets([(30, math.inf)], 10.0), False)
    # I-EQ on synthetic traces: gengine (off None) == dedicated_by_day
    rng = np.random.default_rng(7)
    for trial in range(3):
        K = 300
        mins = np.sort(rng.choice(np.arange(HELD - 200, HELD + WEEK_MIN), K, replace=False))
        nn = rng.integers(1, 6, K)
        it = nn * rng.uniform(100, 9000, K)
        ot = nn * rng.uniform(10, 4000, K)
        models = [("g", 300.0, 30.0 + 10 * trial)]
        tr = T.Trace(mins, np.zeros(K, np.int64), nn, it, ot, models)
        gp = dict(F.PHYS0, tpot_q=NS.GIANT_TPOT, tpot_sat=NS.GIANT_TPOT * NS.TPOT_SAT_RATIO)
        kt = [2e5 / (trial + 1)]
        Cs = [C_REF * (2 + trial)]
        w, dy = NS.dedicated_by_day(tr, kt, gp, Cs)
        r = gengine(tr, {0: dict(kvtok=kt[0], C=Cs[0], mult=1.0, off=None)}, gp)
        chk(f"I-EQ synthetic trial {trial} week", r[0]["ttft"][0] == w[0], True)
        chk(f"I-EQ synthetic trial {trial} days", all(r[0]["ttft"][1][d] == dy[d][0] for d in range(NDAYS)), True)
        # offload adds host terms: every key >= the no-offload key in distribution (stochastically larger p95)
        so = off_spec(dict(n=8, k=2, L=4, e_gb=0.5, E_gb=16.0, A_r=1e9, hidden=1), 1, 0.25, OPHYS0)
        r2 = gengine(tr, {0: dict(kvtok=kt[0], C=Cs[0], mult=1.0, off=so)}, gp)
        chk(f"offload p95 >= no-offload p95 trial {trial}", T._pct(r2[0]["ttft"][0], 0.95) >= T._pct(r[0]["ttft"][0], 0.95), True)
        chk(f"floor keys only add +inf trial {trial}",
            sum(r2[0]["f100"][0].values()) == sum(r2[0]["ttft"][0].values()) and
            r2[0]["f100"][0].get(T.INF_KEY, 0) >= r2[0]["ttft"][0].get(T.INF_KEY, 0), True)
    # thinning
    dc = dict(minute=np.arange(HELD, HELD + 1000, dtype=np.int64), n=np.full(1000, 10, np.int64), it=np.full(1000, 1000.0),
              ot=np.full(1000, 100.0), N=10000)
    t1, t2 = thin(dc, 3000, 1, 4, 0), thin(dc, 3000, 1, 4, 0)
    chk("thin deterministic", all(np.array_equal(x, y) for x, y in zip(t1, t2)), True)
    chk("thin count within 5 sigma", abs(int(t1[1].sum()) - 3000) <= 5 * math.sqrt(10000 * 0.3 * 0.7), True)
    chk("thin keeps tokens per request", bool(np.allclose(t1[2] / t1[1], 100.0)), True)
    t3 = thin(dc, 25000, 0, 5, 0)
    chk("thin p > 1 replicates", int(t3[1].min()) >= 20, True)
    t4 = thin(dc, 3000, 1, 4, 0, j=1, rot=5040)
    chk("thin rotation stays in the week", bool(((t4[0] >= HELD) & (t4[0] < HELD + WEEK_MIN)).all()), True)
    # homing golden: co-activity refusal, KV floor, host RAM
    npx = dict(np8)
    hours = {"a": {1, 2}, "b": {2, 3}, "c": {5}, "d": {9}}
    feas = {"a": True, "b": True, "c": True, "d": False}
    Wd = {x: 100.0 for x in hours}
    groups, ded = home_class(["a", "b", "c", "d"], hours, feas, Wd, 4, npx)
    chk("homing: overlap refused, disjoint joined, infeasible dedicated", (groups, ded), ([["a", "c"], ["b"]], ["d"]))
    Wbig = {x: 260.0 for x in hours}
    groups2, _ = home_class(["a", "c"], hours, feas, Wbig, 4, npx)
    chk("homing: KV floor with a sleeper refuses (4 x 71.24 - 260 - 24 - 4 < 1)", groups2, [["a"], ["c"]])
    # break-even through lever_curve on synthetic rows (degenerate rung stops it)
    rows = []
    for seed in SEEDS:
        for ri in range(3):
            for arm, G, ok in (("CTRL", 16, ri != 1), ("OFF", 8, True)):
                rows.append(dict(kind="ladder", giant="x", donor="ANCHOR", cm="POINT", sens="base", arm=arm, G=G, r=1,
                                 f=0.5 if arm == "OFF" else None, gpus=G, seed=seed, rung=RUNGS[ri], ri=ri,
                                 reps={fk: [[30, 1.0 if ok else "inf"]] * 7 for fk, _ in FLOOR_KEYS}))
    acc, seen = ladder_tables(rows)
    k = ("x", "ANCHOR", "POINT", "base")
    ctrl = lever_curve(acc[k], seen[k], "CTRL", "f100", 10.0, False)
    offc = lever_curve(acc[k], seen[k], "OFF", "f100", 10.0, True)
    chk("ladder: CTRL degenerate at rung 1", ctrl[:3], [16, None, 16])
    fe = [ctrl[i] is not None and offc[i] is not None and offc[i] < ctrl[i] for i in range(3)]
    chk("ladder: break-even stops at the degenerate rung", break_even(RUNGS[:3], fe), 30)
    # GX1-GX3 (added after the freeze, hand-computed): one request, in 10,000 / out 100 tokens, ab 31 B (10 x the reference), C =
    # 10 C_REF -> GPU prefill 10000 / C_REF = 0.53257 s, no queue, no KV wait; toy FETCH host: 2 chunks x 0.4 s = 0.8 s -> TTFT 1.33 s.
    # Decode: B = 1, host 0.1 s, GPU 0.02 / (1 - 101000 / (60 * 10 C_REF)) = 0.020181 s, u = 0.8 / 60 -> TPOT 0.121805 s:
    # f100 / f50 miss (+inf), f200 meets at 1.33 s.
    trx = T.Trace(np.array([HELD + 10]), np.array([0]), np.array([1]), np.array([10000.0]), np.array([100.0]),
                  [("g", 300.0, 31.0)])
    gpx = dict(F.PHYS0, tpot_q=0.02, tpot_sat=0.2)
    rx = gengine(trx, {0: dict(kvtok=1e9, C=10 * C_REF, mult=1.0, off=toy)}, gpx)
    chk("GX1 TTFT carries the host prefill (1.33 s)", T._pct(rx[0]["ttft"][0], 0.95), 1.33)
    chk("GX2 host decode pushes TPOT past the 100 ms floor", T._pct(rx[0]["f100"][0], 0.95), math.inf)
    chk("GX2 the 200 ms floor still meets", T._pct(rx[0]["f200"][0], 0.95), 1.33)
    pg4 = {"a": dict(ctrl=8, off=2, int4=None, gctrl=8), "b": dict(ctrl=8, off=2, int4=None, gctrl=8)}
    al4 = _alloc(pg4, [dict(G=8, members=["a", "b"], meets=True)], ["a", "b"])
    chk("GX3 LOSSLESS keeps best-alone when the group is not cheaper", (al4["a"]["LOSSLESS"], al4["a"]["PAIR"]), (2, 4.0))
    # CLI golden through the real entry point
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--table"], capture_output=True, text=True, timeout=120)
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    a = [x for x in lines if x["fmt"] == "fp8"]
    chk("CLI --table rc", p.returncode, 0)
    chk("CLI --table anchor fp8 G/L0/f_max8", (a[0]["G"], a[0]["L0"], a[0]["f_max"]["8"]) if a else None, (16, 5.3365, 0.536))
    b = [x for x in lines if x["fmt"] == "int4"]
    chk("CLI --table anchor int4 G/need8", (b[0]["G"], b[0]["need8"]) if b else None, (16, 75.4278))
    # GM1-GM8 (hand-computed in the addendum before any Muse row): E = 4096^2 x 3 x 256 x 62 = 2^30 x 744 exactly
    W8, E8m, NE8m, Ep, Am = synth_memory("muse816b", "fp8")
    chk("GM1 Muse routed expert params", Ep, 798863917056)
    chk("GM2 Muse A_R2 = P - 248 x 3 x 4096^2 x 62", Am, 48137110016)
    chk("GM3 brief golden: 816e9 x 2 B = 1.632 TB bf16", synth_memory("muse816b", "bf16", P=816e9)[0] * G30, 1.632e12, 1e-3)
    chk("GM3 brief golden: 816e9 at ~1 B (fp8 experts + 1/4096 scales, 1 B non-experts) = 816.195 GB",
        synth_memory("muse816b", "fp8", P=816e9, ne_rate=1.0)[0] * G30 / 1e9, 816.195035136, 1e-6)
    chk("GM4 Muse bf16 W GiB (1,644,073,059,328 B)", round(synth_memory("muse816b", "bf16")[0], 4), 1531.1624)
    chk("GM5 Muse fp8 expert GiB = 744 x (1 + 1/4096)", round(E8m, 6), 744.181641)
    lay8 = NS.mg_layout(W8, E8m, NE8m, True, "DCP", bh, np8["o_g"])
    W4m, E4m, NE4m, _, _ = synth_memory("muse816b", "int4")
    lay4 = NS.mg_layout(W4m, E4m, NE4m, True, "DCP", bh, np8["o_g"])
    Wbm = synth_memory("muse816b", "bf16")[0]
    chk("GM6 Muse G_min fp8 / int4 / bf16", (lay8[0], lay4[0], bf16_layout(Wbm, bh, np8["o_g"])[0]), (16, 8, 32))
    chk("GM6 bf16 on 16 needs 95.70 GiB/GPU > B_h", round(Wbm / 16, 2), 95.70)
    chk("GM7 Muse bpt at occupancy 4,096 = 65,536 + 188,416 / 2", V2.SC.bpt_effective(synth_kv("muse816b"), 4096.0), 159744.0)
    chk("GM8 bf16 host fit G8 False / G16 True", (host_fit(Wbm, 8, np8["node_ram"], np8["r_pin"]),
                                                    host_fit(Wbm, 16, np8["node_ram"], np8["r_pin"])), (False, True))
    p2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--synth-table", "muse816b"], capture_output=True, text=True,
                        timeout=120)
    sl = [json.loads(x) for x in p2.stdout.splitlines() if x.strip()]
    pt = {x["fmt"]: x["G"] for x in sl if x.get("reading") == "point"}
    chk("CLI --synth-table rc + point G per format", (p2.returncode, pt), (0, {"fp8": 16, "int4": 8, "bf16": 32}))
    n_ok = sum(res)
    print(f"SELF-TEST {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")
    return n_ok == len(res)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--check")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--pair-ladder", action="store_true")
    ap.add_argument("--fleet", action="store_true")
    ap.add_argument("--pair-fleet", action="store_true")
    ap.add_argument("--summarize")
    ap.add_argument("--mono")
    ap.add_argument("--giant")
    ap.add_argument("--donor", default="ANCHOR")
    ap.add_argument("--cm", default="POINT")
    ap.add_argument("--sens", default="base")
    ap.add_argument("--arms", default="CTRL,INT4,OFF")
    ap.add_argument("--noprune", action="store_true")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--tau", default=str(TAU_POINT))
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--root")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--borrow")                              #: census CID8 a synthetic giant borrows occupancy/traffic from
    ap.add_argument("--synth-table")                         #: read-only memory/borrow table of a synthetic giant
    a = ap.parse_args()

    def die(msg):
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)

    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.synth_table:
        if a.synth_table not in SYNTH:
            die(f"unknown synthetic giant {a.synth_table}")
        if a.root and (not os.path.isdir(a.root) or not a.cache or not os.path.isfile(a.cache)):
            die("--root DIR needs --cache FILE")
        try:
            for r in synth_table(a.synth_table, a.root, a.cache):
                print(json.dumps(r))
        except ValueError as e:
            die(str(e))
        return
    synth = a.giant in SYNTH
    if a.borrow and not synth:
        die("--borrow needs a synthetic --giant")
    if synth and a.donor in SYNTH_DONOR_IDX:                 # the synthetic donor bypasses the census-donor check below only
        a_donor_ok = True
    else:
        a_donor_ok = a.donor in DONORS
    if synth and a.arms != "CTRL,INT4,OFF" and any(x in SYNTH_ARMS for x in a.arms.split(",")):
        if any(x not in ("CTRL", "INT4", "OFF") + SYNTH_ARMS for x in a.arms.split(",")):
            die("bad --arms")
        if not a.ladder:                                     # QA r1: --fleet / --pair-ladder take no --arms; refuse, never ignore
            die("BF16CTRL / BF16OFF run on --ladder only (--fleet and --pair-ladder take no --arms)")
        a_arms_ok = True
    else:
        a_arms_ok = False
    if a.table:
        for r in table_rows(a.root, a.cache):
            print(json.dumps(r))
        return
    if a.mono:
        if not os.path.isdir(a.mono):
            die(f"no dir {a.mono}")
        print(json.dumps(mono_count(_rows(a.mono))))
        return
    if a.summarize:
        if not os.path.isdir(a.summarize):
            die(f"no dir {a.summarize}")
        print(json.dumps(summarize(a.summarize), sort_keys=True))
        return
    if a.cm not in E_PAR or not a_donor_ok or a.sens not in SENS or a.n not in (2, 4) or a.window not in (60, 10):
        die("bad --cm / --donor / --sens / --n / --window")
    if not a_arms_ok and any(x not in ("CTRL", "INT4", "OFF") for x in a.arms.split(",")):
        die("bad --arms")
    if not a.root or not a.cache or not os.path.isdir(a.root) or not os.path.isfile(a.cache):
        die("--root DIR and --cache FILE are required")
    try:
        if a.check:
            r = check(a.check, a.root, a.cache)
            print(json.dumps(r))
            sys.exit(0 if r.get("pass") else 1)
        if not a.out:
            die("--out is required")
        if a.ladder:
            n = run_ladder(a, a.root, a.cache)
        elif a.pair_ladder:
            n = run_pair_ladder(a, a.root, a.cache)
        elif a.fleet:
            n = run_fleet(a, a.root, a.cache)
        elif a.pair_fleet:
            n = run_pair_fleet(a, a.root, a.cache)
        else:
            die("no action")
        print(json.dumps(dict(rows=n, out=a.out)))
    except ValueError as e:
        die(str(e))


if __name__ == "__main__":
    main()
