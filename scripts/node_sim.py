#!/usr/bin/env python3
"""node_sim.py — the gpu-bunk fabric at the provider's unit (8-GPU NVLink nodes of H100 SXM 80 GB).

Prereg (frozen before this file existed): (private report) + its amendments. On any
divergence the prereg wins. Imports v1 (`fabric_sim`), v2 (`fabric_sim_v2`) and their dependencies UNCHANGED.
The node layer COMPOSES unchanged v2 `simulate()` calls, one per uniform card class (cards never interact inside it). Only v1's
`dedicated()` is forked, to add per-day output (golden GN1: identical week Counters). Arms (prereg §4):
  A   multi-GPU (MG) models on dedicated G-GPU groups (KV = all leftover); the tail on K full H100 cards
  B_r the tail packed into the MG GPUs' leftover HBM, each MG GPU keeping a KV reserve r (SZ | Q25 | Q50 | Q75)
  C_r B_r with node-scope cross-GPU KV borrowing (one v2 card per 8-GPU node; remote-KV TPOT penalty beta)
  D   MG models time-sharing G-GPU groups by sleep/wake (host RAM as a hard constraint)
  WAKE tail sleep stack (v2 BAL_LEN) with host-RAM (PCIe) vs neighbour-GPU (NVLink, upper bound) wake bandwidth
Replicates: leave-one-held-out-day-out (7) via a per-day model relabel (prereg §4.7); sleep arms add seeds 0-2.

Output contract:
  --self-test            exit 0 + "SELF-TEST PASS n/n" (prereg §8 goldens GH0-GH11, GN1-GN3, decision rules); exit 1 + FAIL lines.
  --giant-table [--set k=v ...] [--root R]
                         read-only: one JSON line per MG model (format, layout, G, weights, leftover, KV dup) at the given physics.
  --check REPRO|DAY --root R --cache F [--v2rows DIR]
                         read-only instrument checks (prereg §7); one JSON object; exit 0 if PASS, 1 otherwise.
  --run --arm SPEC [--ks 0,4 | --kgrid] [--seeds 0] [--set k=v ...] [--tag T] --root R --cache F --out F
                         simulates and APPENDS one JSON line per (K, seed) to F (WRITES F); exit 0.
                         SPEC: A:<STATIC|POOL_LEN> | B:<stack>:<SZ|Q25|Q50|Q75|HELD> | C:<stack>:<r> | D | WAKE:<PCIE|NVL>
  --summarize DIR        read-only: prereg §5 Q1-Q5 over DIR/*.jsonl; one JSON object on stdout.
  Bad input (sha mismatch, unknown arm/param, missing file) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed in v1/v2/the census, inherited here):
  1. **A unit index of -1 wraps numpy** — MG models are NO-CARD units in v2 (`unit_of = -1`); every array here is indexed by
     model, never by a unit index that can be -1 (v2 pitfall 2).
  2. **`ssh host 'write && ... &'` backgrounds the whole chain** — launch runners in separate calls on a CPU server.
  3. **The 3-seed replay is blind for all-awake arms** — replicates are held-out days, not seeds, for A/B/C.
"""
import argparse, collections, copy, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric_sim as F       # noqa: E402  (v1 not edited)
import fabric_sim_v2 as V2   # noqa: E402  (v2 not edited)
import tailpool_sim as T     # noqa: E402  (not edited)
import wakestorm_sim as W    # noqa: E402  (not edited)
from real_trace_analysis import C_REF, ACTIVE_REF_B  # noqa: E402

import numpy as np  # noqa: E402

G30 = float(2 ** 30)
OVH = F.OVH
HELD = F.HELD
NDAYS = 7
DAY_MIN = 1440
KV_MIN = F.KV_MIN_GIB
K_GRID = tuple(range(0, 257, 4))
K_TOP = 256
K_STEP = 4

# ---- prereg §1 sourced constants
NVSMI_MIB = 81559                  # H100 80GB HBM3 nvidia-smi total (ICC HGX 8xH100 benchmark, measured)
SMI_GAP_MIB = 500                  # nvidia-smi vs CUDA total
H2D_PCIE, D2H_PCIE = 53.8, 53.9    # ICC bandwidthTest pinned, GB/s per GPU
NVL_P2P = 350.0                    # ICC busGrind P2P unidirectional, GB/s
NODE_RAM_GCP = 1872e9 / G30        # GCP a3-highgpu-8g, 1,872 GB
NODE_RAM_AWS = 2048.0              # AWS p5.48xlarge, 2 TiB
PIN_FRAC = 78.2 / 135.0            # v1's pinnable share of host RAM (h_pin / h_tot)
GIANT_TPOT = 1.0 / 38.0            # median of 3 provider p50 throughputs (32, 38, 172.5 tok/s), OpenRouter 2026-10-14
ANCHOR = "51c86bd2-eed8-5db2-8a78-f87c2296aaf5"
KIMI_FP8_BYTES = 1029190981272     # HF API moonshotai/Kimi-K2-Instruct @fd1984e2
KIMI_INT4_BYTES = 594232561304     # HF API moonshotai/Kimi-K2-Thinking @a51ccc05
KIMI_PARAMS = {"fp8": 1026408235864, "int4": 1026408240256}
KIMI_E = 384 * 3 * 7168 * 2048 * 60
B_EXP = {"fp8": 1.0 + 4.0 / (128 * 128), "int4": 0.5 + 2.0 / 32}
B_NE = {"fp8": (KIMI_FP8_BYTES - KIMI_E * B_EXP["fp8"]) / (KIMI_PARAMS["fp8"] - KIMI_E),
        "int4": (KIMI_INT4_BYTES - KIMI_E * B_EXP["int4"]) / (KIMI_PARAMS["int4"] - KIMI_E)}
TPOT_SAT_RATIO = 0.06 / 0.0065     # v2: tpot_sat / tpot_q (rho_cap kept when TPOT is scaled)
V2_M_H = 1.1                       # v2 big_mult; amendment 1: tail TPOT x (1.1 / m_h)

# prereg §1 point values; every key overridable with --set k=v and recorded in each row
NPHYS0 = dict(fmt="int4", layout="DCP", util=0.9, o_g=6.0, sigma=0.3, phi=0.5, beta=1.0, e_par=0.5, m_h=1.1,
              g_tpot=GIANT_TPOT, kappa=0.5, node_ram=NODE_RAM_GCP, r_pin=1.27, s_g=1.0, bh=0.0, nomg=0.0)
NPHYS_STR = {"fmt", "layout"}
FMTS = ("fp8", "int4")
LAYOUTS = ("DCP", "TP", "DPA")
STACKS = {"STATIC": "BIG_RESIZE", "POOL_LEN": "BIG_POOL_LEN"}
RESERVES = ("SZ", "Q25", "Q50", "Q75", "HELD")


def nphys_with(over=None):
    p = dict(NPHYS0)
    for k, v in (over or {}).items():
        if k not in NPHYS0:
            raise ValueError(f"unknown param {k}")
        p[k] = v if k in NPHYS_STR else float(v)
    if p["fmt"] not in FMTS or p["layout"] not in LAYOUTS:
        raise ValueError(f"fmt must be in {FMTS} and layout in {LAYOUTS}")
    return p


def b_h(np_):
    """prereg §1: H100 card budget = util x (nvidia-smi total - gap); --set bh=X overrides (I-REPRO)."""
    return np_["bh"] if np_["bh"] > 0 else np_["util"] * (NVSMI_MIB - SMI_GAP_MIB) / 1024.0


# ------------------------------------------------------------------------------------------------ MG arithmetic (§2)
def kv_dup(layout, G, H):
    """prereg §2: DCP / DPA 1; TP G / min(G, H) (vLLM v0.19 doc: tp_size / H; MLA H = 1)."""
    if layout in ("DCP", "DPA"):
        return 1.0
    if layout == "TP":
        return G / min(G, H)
    raise ValueError(f"unknown layout {layout}")


def mg_weights(row, fmt):
    """-> (W GiB, expert GiB, non-expert GiB, moe?) for a census row; the anchor uses its checkpoint bytes (§1)."""
    if row["cid"] == ANCHOR:
        tot = KIMI_FP8_BYTES if fmt == "fp8" else KIMI_INT4_BYTES
        e = KIMI_E * B_EXP[fmt]
        return tot / G30, e / G30, (tot - e) / G30, True
    m = row.get("moe") or {}
    if not m.get("experts"):
        w = float(row["weights_gib_fp8"])
        return w, 0.0, w, False
    P, A = row["params_total_R2"], row["params_active_R2"]
    hidden = (P - A) / ((m["experts"] - m["topk"]) * 3 * m["moe_inter"] * m["moe_layers"])
    E = m["experts"] * 3 * hidden * m["moe_inter"] * m["moe_layers"]
    e = E * B_EXP[fmt]
    ne = (P - E) * B_NE[fmt]
    return (e + ne) / G30, e / G30, ne / G30, True


def mg_layout(W, Eg, NEg, moe, layout, bh, o_g):
    """-> (G, weights per GPU, L0 leftover per GPU, layout used) or None (NOT-PLACEABLE). Amendment 1(b): DPA is MoE only;
    a dense model uses DCP in the DPA scenario."""
    lay = layout if (layout != "DPA" or moe) else "DCP"
    for G in (1, 2, 4, 8, 16):
        wg = (Eg / G + NEg) if lay == "DPA" else W / G
        if wg + o_g + KV_MIN <= bh + 1e-12:
            return G, wg, bh - wg - o_g, lay
    return None


def kv_heads(row):
    return 1 if str(row.get("kv_attn", "")).startswith("MLA") else int(row.get("kv_n_kv") or 1)


def mg_table(rows_by_cid, mg_cids, np_):
    """prereg §2 table for the MG set at physics np_ -> {cid: dict}."""
    bh = b_h(np_)
    out = {}
    for cid in mg_cids:
        r = rows_by_cid[cid]
        W, Eg, NEg, moe = mg_weights(r, np_["fmt"] if moe_ok(r) else "fp8")
        lay = mg_layout(W, Eg, NEg, moe, np_["layout"], bh, np_["o_g"])
        d = dict(cid=cid, name=r["name"], moe=moe, W=W, E=Eg, NE=NEg, H=kv_heads(r), train_req=r.get("train_req"),
                 held_req=r.get("held_req"), placeable=lay is not None)
        if lay is not None:
            G, wg, L0, used = lay
            d.update(G=G, w_gpu=wg, L0=L0, layout=used, dup=kv_dup(used, G, d["H"]))
        out[cid] = d
    return out


def moe_ok(row):
    """INT4 applies to MoE MG models only (prereg §2); dense MG keep census fp8."""
    return row["cid"] == ANCHOR or bool((row.get("moe") or {}).get("experts"))


def kv_tokens(G, R, bpt, dup):
    """prereg §2: capacity tokens = G x R x 2^30 / (bpt x dup)."""
    return G * R * G30 / (bpt * dup)


# ------------------------------------------------------------------------------------------------ placement (§4.5, §4.6)
def hetero_spread(units, cards):
    """prereg §4.5 B: f descending (ties cid); the card with the most free GPU memory where f AND the host total fit
    (ties lowest index). cards: list of [free GiB, host GiB] (mutated). unit = dict(cid, f, host). None fits -> ValueError."""
    homes = [-1] * len(units)
    for u in sorted(range(len(units)), key=lambda k: (-units[k]["f"], units[k]["cid"])):
        f, h = units[u]["f"], units[u]["host"]
        feas = [c for c in range(len(cards)) if f <= cards[c][0] + 1e-9 and h <= cards[c][1] + 1e-9]
        if not feas:
            raise ValueError(f"hetero SPREAD infeasible: unit {units[u]['cid']} f={f:.3f} fits no card")
        c = max(feas, key=lambda j: (cards[j][0], -j))
        homes[u] = c
        cards[c][0] -= f; cards[c][1] -= h
    return homes


def b_cards(tab, groups, R_of, K, bh, h_tot):
    """prereg §4 B: each MG GPU is a card with tail room L0 - R (never below 0), then K full cards at B_h.
    -> (cards owner cid | None per card)."""
    cards, owner = [], []
    for cid, G in groups:
        room = tab[cid]["L0"] - R_of[cid]
        for _ in range(G):
            cards.append([max(0.0, room), h_tot]); owner.append(cid)
    for _ in range(K):
        cards.append([bh, h_tot]); owner.append(None)
    return cards, owner


def tpot_mults(tab, cid_m, hosted, sigma, M):
    """prereg §3: giant TPOT x (1 + sigma) only for an MG model hosting >= 1 tail unit."""
    mult = [1.0] * M
    for cid, d in tab.items():
        if d["placeable"] and hosted.get(cid):
            mult[cid_m[cid]] = 1.0 + sigma
    return mult


def pack_nodes(groups, k_full):
    """prereg §4.6: MG groups [(key, G)] by G desc (ties key) first-fit into 8-GPU nodes (a 16-group = 2 new nodes); full cards
    fill partial nodes' free slots first (lowest index), then new nodes. -> list of nodes = list of slots ('mg', key, i) |
    ('full', k)."""
    nodes = []
    for key, G in sorted(groups, key=lambda x: (-x[1], x[0])):
        if G > 8:
            if G % 8:
                raise ValueError(f"group {key} G={G} is not a whole number of nodes")
            for j in range(G // 8):
                nodes.append([("mg", key, 8 * j + i) for i in range(8)])
            continue
        for nd in nodes:
            if 8 - len(nd) >= G:
                nd.extend(("mg", key, i) for i in range(G))
                break
        else:
            nodes.append([("mg", key, i) for i in range(G)])
    k = 0
    for nd in nodes:
        while len(nd) < 8 and k < k_full:
            nd.append(("full", k)); k += 1
    while k < k_full:
        nd = []
        while len(nd) < 8 and k < k_full:
            nd.append(("full", k)); k += 1
        nodes.append(nd)
    return nodes


def node_place(units, nodes):
    """prereg §4.5 C: f descending (ties cid); a node is feasible iff node free >= f AND one GPU's weights-free >= w + 0.82
    AND that GPU's host >= the unit's host need; the node with the most free memory (ties lowest index), then its GPU with the
    most weights-free (ties lowest). nodes: list of dict(free, gpus=) (mutated). -> (node, gpu) per unit."""
    homes = [None] * len(units)
    for u in sorted(range(len(units)), key=lambda k: (-units[k]["f"], units[k]["cid"])):
        U = units[u]
        need_w = U["w"] + OVH

        def gpu_ok(g):
            return g[0] >= need_w - 1e-9 and g[1] >= U["host"] - 1e-9
        feas = [n for n in range(len(nodes)) if nodes[n]["free"] >= U["f"] - 1e-9 and any(gpu_ok(g) for g in nodes[n]["gpus"])]
        if not feas:
            raise ValueError(f"node placement infeasible: unit {U['cid']} f={U['f']:.3f} fits no node")
        n = max(feas, key=lambda j: (nodes[j]["free"], -j))
        gi = max((i for i, g in enumerate(nodes[n]["gpus"]) if gpu_ok(g)), key=lambda i: (nodes[n]["gpus"][i][0], -i))
        nodes[n]["free"] -= U["f"]
        nodes[n]["gpus"][gi][0] -= need_w
        nodes[n]["gpus"][gi][1] -= U["host"]
        homes[u] = (n, gi)
    return homes


# ------------------------------------------------------------------------------------------------ trace helpers (§4.7)
def subtrace(tr, keep_models):
    """Cells of the given models only; model ids and the models list unchanged."""
    mask = np.isin(tr.mid, np.asarray(sorted(keep_models), dtype=tr.mid.dtype))
    t2 = T.Trace.__new__(T.Trace)
    t2.minute, t2.mid, t2.n, t2.it, t2.ot = tr.minute[mask], tr.mid[mask], tr.n[mask], tr.it[mask], tr.ot[mask]
    t2.models = tr.models
    t2.explicit = None if tr.explicit is None else [e for e, k in zip(tr.explicit, mask) if k]   # QA r1: goldens carry offsets
    return t2


def relabel_days(tr, M):
    """prereg §4.7: held-out cells of model m on held-out day d -> m + M (1 + d); models extended 8x. TRAIN cells keep m."""
    t2 = T.Trace.__new__(T.Trace)
    t2.minute, t2.n, t2.it, t2.ot, t2.explicit = tr.minute, tr.n, tr.it, tr.ot, tr.explicit
    mid = tr.mid.astype(np.int64).copy()
    held = tr.minute >= HELD
    d = (tr.minute[held] - HELD) // DAY_MIN
    if len(d) and (d.min() < 0 or d.max() >= NDAYS):
        raise ValueError("held-out cells outside days 0..6")
    mid[held] += M * (1 + d)
    t2.mid = mid
    t2.models = list(tr.models) * (1 + NDAYS)
    return t2


def split_days(out, M):
    """-> (week Counters [M], day Counters [7][M]) from a relabelled run's out (len 8M). Day slot 0 must be empty."""
    if len(out) != M * (1 + NDAYS):
        raise ValueError("relabelled out has the wrong length")
    if any(sum(out[m].values()) for m in range(M)):
        raise ValueError("a TRAIN-slot Counter is non-empty: scored cells were not relabelled")
    day = [[out[m + M * (1 + d)] for m in range(M)] for d in range(NDAYS)]
    week = [sum((day[d][m] for d in range(NDAYS)), collections.Counter()) for m in range(M)]
    return week, day


def dedicated_by_day(trace, kvtok_ded, phys, C, tpot_mult=None, score_from_min=HELD):
    """FORK of fabric_sim.dedicated (v1): adds per-held-out-day Counters, a per-model compute C and an
    optional per-model TPOT multiplier (prereg §3: straggler). With scalar C and no multiplier the week equals v1 (GN1)."""
    M = len(trace.models)
    out = [collections.Counter() for _ in range(M)]
    day = [[collections.Counter() for _ in range(M)] for _ in range(NDAYS)]
    Cs = list(C) if isinstance(C, (list, tuple, np.ndarray)) else [C] * M
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
        cap_min = Cm * 60.0
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
            bk[m] = F.kv_catch_up(bk[m], fut[m], kvtok_ded[m], kv_last[m], mi)
            kv_last[m] = mi
            rho = min(rcap, (backlog[m] + Wm) / cap_min)
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm
            tp = F.tpot_of(rho, phys)
            if tpot_mult is not None:
                tp = tp * tpot_mult[m]
            R = pre + (trace.ot[i] / n) * tp
            occ = trace.it[i] / n + phys["kappa"] * trace.ot[i] / n
            D = fut[m].pop(mi, 0.0) + F.spread(fut[m], mi, n * occ, R)
            kw, bk[m] = F.kv_step(bk[m], D, kvtok_ded[m])
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / Cm
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        if mi >= score_from_min:
            k_ = T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm + kw)
            out[m][k_] += n
            day[(mi - HELD) // DAY_MIN][m][k_] += n
    return out, day


# ------------------------------------------------------------------------------------------------ scoring (§5)
def lodo(week, day):
    """7 leave-one-day-out replicates of per-model Counters."""
    return [[week[m] - day[d][m] for m in range(len(week))] for d in range(NDAYS)]


def score_all(out_w, out_d, ref_w, ref_d, slos=T.SLOS):
    """-> {slo: dict(M, R, meet, feasible, eligible, M_reps[7], R_reps[7])}."""
    reps_o, reps_r = lodo(out_w, out_d), lodo(ref_w, ref_d)
    res = {}
    for x in slos:
        a = T.score(out_w, ref_w, x)
        rr = [T.score(reps_o[d], reps_r[d], x) for d in range(NDAYS)]
        res[str(x)] = dict(M=a["M_frac"], R=a["R_ratio"], meet=a["meet"], feasible=a["slo_feasible_models"],
                           eligible=a["eligible"], M_reps=[r["M_frac"] for r in rr], R_reps=[r["R_ratio"] for r in rr],
                           meet_reps=[r["meet"] for r in rr])
    return res


def verdict_gpus(a_reps, x_reps):
    """prereg §5 Q1."""
    if max(x_reps) < min(a_reps):
        return "FEWER-GPUS"
    if min(x_reps) > max(a_reps):
        return "MORE-GPUS"
    return "UNRESOLVED"


def verdict_m(a_reps, x_reps):
    """prereg §5 Q2 (None = infeasible)."""
    if x_reps is None or any(v is None for v in x_reps):
        return "INFEASIBLE"
    if a_reps is None or any(v is None for v in a_reps):
        return "CONTROL-INFEASIBLE"
    if min(x_reps) > max(a_reps):
        return "LIFTS"
    if max(x_reps) < min(a_reps):
        return "LOWERS"
    return "UNATTRIBUTABLE"


def k_match_stable(passmap, top=K_TOP, step=K_STEP):
    """prereg §5: smallest grid K from which every K' up to top passes; never -> top + step. passmap {K: bool}."""
    best = None
    for k in sorted(passmap, reverse=True):
        if passmap[k]:
            best = k
        else:
            break
    return best if best is not None else top + step


# ------------------------------------------------------------------------------------------------ physics plumbing
def tail_phys(np_, stack=None, extra=None):
    """v2 physics for the tail on the H100 class: big_card_gib = B_h, big_mult = m_h, tail TPOT x (1.1 / m_h)
    (amendment 1), kappa, host RAM per GPU = node RAM / 8 with v1's pinned share, r_pin."""
    over = {"big_card_gib": b_h(np_), "big_mult": np_["m_h"], "kappa": np_["kappa"],
            "h_tot": np_["node_ram"] / 8.0, "h_pin": np_["node_ram"] / 8.0 * PIN_FRAC, "r_pin": np_["r_pin"]}
    if np_["bh"] > 0:                    # I-REPRO: v2's own host RAM and TPOT
        over.pop("h_tot"); over.pop("h_pin")
    f = V2_M_H / np_["m_h"]
    if f != 1.0:
        over["tpot_q"] = V2.PHYS0["tpot_q"] * f
        over["tpot_sat"] = V2.PHYS0["tpot_sat"] * f
    if stack == "STATIC":
        over["resize_h"] = 1e8
    over.update(extra or {})
    return V2.phys_with(over)


def giant_phys(np_, tphys):
    p = dict(tphys)
    p["tpot_q"] = np_["g_tpot"]
    p["tpot_sat"] = np_["g_tpot"] * TPOT_SAT_RATIO
    return p


_C = {}


def ctx(root, cache, np_, stack):
    """Population, v2 setup for the stack, census rows, MG table (cached per process)."""
    key = ("ctx", root, json.dumps(np_, sort_keys=True), stack)
    if key in _C:
        return _C[key]
    tph = tail_phys(np_, stack)
    S = V2.setup2(STACKS[stack] if stack else "BIG_POOL_LEN", tph, root, cache)
    tr = S["tr"]
    M = len(tr.models)
    rows, _ = V2.load_census(root)
    mg = [m for m in range(M) if S["unit_of"][m] < 0]
    if np_["nomg"] > 0:                 # I-REPRO: v2's NO-CARD convention (+inf), no MG engines
        tab = {}
    else:
        tab = mg_table(rows, [tr.models[m][0] for m in mg], np_)
    cid_m = {tr.models[m][0]: m for m in range(M)}
    _C[key] = dict(S=S, tr=tr, M=M, mg=mg, tab=tab, cid_m=cid_m, tph=tph, rows=rows)
    return _C[key]


def tail_ref(cx, np_):
    """Tail reference: one dedicated H100 per model (v2 ref_scores convention) with per-day output; MG -> -1 here."""
    key = ("tref", json.dumps(np_, sort_keys=True))
    if key not in _C:
        S, tr, tph = cx["S"], cx["tr"], cx["tph"]
        pop = S["pop"]
        cg = tph["big_card_gib"]
        C = C_REF * tph["cref_mult"] * tph["big_mult"]
        kt = [((cg - mm[4] - OVH) * G30 / pop["bpt"][k]) if cg - mm[4] - OVH > 0 else -1.0 for k, mm in enumerate(tr.models)]
        _C[key] = dedicated_by_day(tr, kt, tph, C)
    return _C[key]


def giant_run(cx, np_, R_of, hosted, sigma):
    """MG engines (prereg §3): per-model kvtok from R_of[cid] (GiB per GPU), C_g, TPOT x (1 + sigma) if hosted[cid]."""
    tr, M, tab, S = cx["tr"], cx["M"], cx["tab"], cx["S"]
    gp = giant_phys(np_, cx["tph"])
    kt = [-1.0] * M
    Cs = [1.0] * M
    mult = tpot_mults(tab, cx["cid_m"], hosted, sigma, M)
    keep = []
    for cid, d in tab.items():
        m = cx["cid_m"][cid]
        keep.append(m)
        if not d["placeable"]:
            continue
        kt[m] = kv_tokens(d["G"], R_of[cid], S["pop"]["bpt"][m], d["dup"])
        Cs[m] = d["G"] * C_REF * np_["m_h"] * np_["e_par"]
    if not keep:
        return {}, {}
    st = subtrace(tr, keep)
    w, dy = dedicated_by_day(st, kt, gp, Cs, tpot_mult=mult)
    return ({m: w[m] for m in keep}, {m: [dy[d][m] for d in range(NDAYS)] for m in keep})


def sz_reserve(cx, np_, held=False):
    """prereg §4 B SZ: v1 sizing on TRAIN only (max(TRAIN p99 length, TRAIN q95 in-flight at design TPOT / 0.8)) with the
    giant's TPOT and C_g, per GPU x dup / G, clamped [0.25, L0]. held=True -> the HELD-OUT oracle (named challenger)."""
    tr, tab, S = cx["tr"], cx["tab"], cx["S"]
    gp = giant_phys(np_, cx["tph"])
    out = {}
    for cid, d in tab.items():
        if not d["placeable"]:
            continue
        m = cx["cid_m"][cid]
        st = subtrace(tr, [m])
        if held:
            k = st.minute >= HELD
            st2 = T.Trace.__new__(T.Trace)
            st2.minute, st2.mid, st2.n, st2.it, st2.ot = st.minute[k], st.mid[k], st.n[k], st.it[k], st.ot[k]
            st2.models, st2.explicit = st.models, None
            st, hl = st2, 10 ** 9
        else:
            hl = HELD
        uo = [-1] * cx["M"]
        uo[m] = 0
        if not np.any(st.minute < hl):
            out[cid] = KV_MIN
            continue
        ft = F.kv_floor_tok(st, uo, 1, gp["floor_q"], held=hl)[0]
        Cg = d["G"] * C_REF * np_["m_h"] * np_["e_par"]
        ct = F.kv_conc_tok(st, uo, 1, st.models, gp, gp["size_q"], held=hl, C=Cg)[0]
        R = max(ft, ct) * S["pop"]["bpt"][m] * d["dup"] / d["G"] / G30
        out[cid] = min(max(R, KV_MIN), d["L0"])
    return out


def reserve_of(cx, np_, r):
    tab = cx["tab"]
    if r == "ALL":
        return {c: d["L0"] for c, d in tab.items() if d["placeable"]}
    if r in ("SZ", "HELD"):
        key = ("sz", r, json.dumps(np_, sort_keys=True))
        if key not in _C:
            _C[key] = sz_reserve(cx, np_, held=(r == "HELD"))
        return _C[key]
    q = {"Q25": 0.25, "Q50": 0.5, "Q75": 0.75}[r]
    return {c: q * d["L0"] for c, d in tab.items() if d["placeable"]}


def mg_ref(cx, np_):
    key = ("mgref", json.dumps(np_, sort_keys=True))
    if key not in _C:
        _C[key] = giant_run(cx, np_, reserve_of(cx, np_, "ALL"), {}, 0.0)
    return _C[key]


def full_ref(cx, np_):
    """One reference list (prereg §3): tail = dedicated H100; MG = arm A (R = L0, sigma 0); NOT-PLACEABLE MG -> +inf."""
    rw, rd = tail_ref(cx, np_)
    rw, rd = list(rw), [list(x) for x in rd]
    gw, gd = mg_ref(cx, np_)
    for m, c in gw.items():
        rw[m] = c
        for d in range(NDAYS):
            rd[d][m] = gd[m][d]
    return rw, rd


# ------------------------------------------------------------------------------------------------ tail group runs
def run_group(cx, units_idx, homes, K, card_gib, C, phys, models_extra=()):
    """One v2 simulate() call for a uniform card class: units S['units'][units_idx] on K cards; returns {model: Counter}
    week and day, plus stats. models_extra = MG models whose cells ride along as NO-CARD (I-REPRO stats parity)."""
    S, tr, M = cx["S"], cx["tr"], cx["M"]
    cfg = dict(S["cfg"])
    cfg["h2d"], cfg["d2h"], cfg["protect"] = phys["h2d"], phys["d2h"], phys["protect"]
    sub_units = [S["units"][u] for u in units_idx]
    new_of = {u: j for j, u in enumerate(units_idx)}
    unit_of = [new_of.get(S["unit_of"][m], -1) if S["unit_of"][m] >= 0 else -1 for m in range(M)]
    keep_m = [m for m in range(M) if unit_of[m] >= 0] + list(models_extra)
    st = relabel_days(subtrace(tr, keep_m), M)
    uo8 = unit_of * (1 + NDAYS)
    rs = dict(H=phys["resize_h"], restart_s=phys["restart_s"], thr=phys["resize_thr"]) if cfg["mech"] == "resize" else None
    out, met = V2.simulate(st, sub_units, uo8, ("card", homes), K, cfg, 0, None, card_gib=card_gib, phys=phys,
                           kvtok=[S["kvtok"][u] for u in units_idx], copies=None, C=C, mech=cfg["mech"],
                           shed_tau=cfg["shed_tau"], bpt=[S["bpt"][u] for u in units_idx], resize=rs)
    w, dy = split_days(out, M)
    return {m: w[m] for m in keep_m}, {m: [dy[d][m] for d in range(NDAYS)] for m in keep_m}, met


def tail_units_info(cx):
    S = cx["S"]
    return [dict(cid=U["cid"], f=U["f"], w=U["w"], host=S["hneed"][u][1]) for u, U in enumerate(S["units"])]


def mg_groups(cx):
    return [(cid, d["G"]) for cid, d in cx["tab"].items() if d["placeable"]]


def assemble(cx, parts, gw, gd):
    """Per-model week/day Counters from the group runs + the MG engines; checks every model is covered exactly once."""
    M = cx["M"]
    ow = [None] * M
    od = [[None] * M for _ in range(NDAYS)]
    mg = set(cx["mg"])
    for wk, dy in parts:
        for m, c in wk.items():
            if m in mg and m in gw:
                continue
            if ow[m] is not None:
                raise RuntimeError(f"model {m} served by two groups")
            ow[m] = c
            for d in range(NDAYS):
                od[d][m] = dy[m][d]
    for m in mg:
        if m in gw:
            ow[m] = gw[m]
            for d in range(NDAYS):
                od[d][m] = gd[m][d]
        elif ow[m] is None:                         # no MG engine (I-REPRO) and no ride-along run: NO-CARD, all +inf
            ow[m], dd = inf_counters(cx, m)
            for d in range(NDAYS):
                od[d][m] = dd[d]
    missing = [m for m in range(M) if ow[m] is None]
    if missing:
        raise RuntimeError(f"{len(missing)} models not covered (first {missing[:3]})")
    return ow, od


def inf_counters(cx, m):
    """+inf Counters (week, per day) for every held-out request of model m (a NO-CARD model with no engine)."""
    tr = cx["tr"]
    k = (tr.mid == m) & (tr.minute >= HELD)
    w = collections.Counter()
    dd = [collections.Counter() for _ in range(NDAYS)]
    for mi, n in zip(tr.minute[k], tr.n[k]):
        w[T.INF_KEY] += int(n)
        dd[(int(mi) - HELD) // DAY_MIN][T.INF_KEY] += int(n)
    return w, dd


def _stats_add(acc, met):
    for k, v in met.get("stats", {}).items():
        acc[k] = acc.get(k, 0) + v


def mg_detail(cx, gw, gd, R_of, hosted):
    det = {}
    for cid, d in cx["tab"].items():
        m = cx["cid_m"][cid]
        if not d["placeable"] or m not in gw:
            det[cid[:8]] = dict(placeable=False)
            continue
        reps = lodo([gw[m]], [[gd[m][x]] for x in range(NDAYS)])
        det[cid[:8]] = dict(G=d["G"], R=R_of.get(cid), L0=d["L0"], hosted=bool(hosted.get(cid)),
                            n=sum(gw[m].values()), p95=W._jsonable(T._pct(gw[m], 0.95)),
                            p95_reps=[W._jsonable(T._pct(r[0], 0.95)) for r in reps])
    return det


def run_cell(spec, K, seed, np_, root, cache, tag=""):
    """One row: arm SPEC at K full tail cards (A/B/C) or one WAKE cell. D has its own runner."""
    kind = spec.split(":")
    arm = kind[0]
    if arm in ("A", "B", "C"):
        stack = kind[1]
        if stack not in STACKS:
            raise ValueError(f"unknown stack {stack}")
        r = kind[2] if arm != "A" else "ALL"
        if r not in RESERVES + ("ALL",):
            raise ValueError(f"unknown reserve {r}")
    elif arm == "WAKE":
        return run_wake(kind[1], K, seed, np_, root, cache, tag)
    else:
        raise ValueError(f"unknown arm {spec}")
    cx = ctx(root, cache, np_, stack)
    S, tph, tab = cx["S"], cx["tph"], cx["tab"]
    bh = tph["big_card_gib"]
    Ch = C_REF * tph["cref_mult"] * tph["big_mult"]
    R_of = reserve_of(cx, np_, r)
    groups = mg_groups(cx)
    sumG = sum(G for _, G in groups)
    row = dict(spec=spec, arm=arm, stack=stack, r=r, K=K, seed=seed, tag=tag, nphys=np_, gpus=sumG + K, sum_G=sumG,
               not_placeable=[c[:8] for c, d in tab.items() if not d["placeable"]])
    row["nodes"] = len(pack_nodes(groups, K))
    t0 = time.time()
    uinfo = tail_units_info(cx)
    hosted = {}
    stats = {}
    parts = []
    try:
        if arm == "A":
            if K == 0:
                raise ValueError("A needs K >= 1")
            cards = [[bh, tph["h_tot"]] for _ in range(K)]
            homes = hetero_spread(uinfo, cards)
            wk, dy, met = run_group(cx, list(range(len(uinfo))), homes, K, bh, Ch, tph, models_extra=cx["mg"])
            _stats_add(stats, met); parts.append((wk, dy))
            row["placed"] = {"full": len(uinfo)}
        elif arm == "B":
            cards, owner = b_cards(tab, groups, R_of, K, bh, tph["h_tot"])
            homes = hetero_spread(uinfo, cards)
            for k in range(len(cards)):             # physical invariant (§7)
                if cards[k][0] < -1e-9:
                    raise RuntimeError(f"card {k} over-committed")
            byown = collections.defaultdict(list)
            for u, h in enumerate(homes):
                byown[owner[h]].append(u)
            first = {}
            for k, o in enumerate(owner):
                first.setdefault(o, k)
            placed = {}
            for o, us in sorted(byown.items(), key=lambda x: (x[0] is None, x[0] or "")):
                base = first[o]
                lh = [homes[u] - base for u in us]
                if o is None:
                    wk, dy, met = run_group(cx, us, lh, K, bh, Ch, tph, models_extra=cx["mg"])
                    placed["full"] = len(us)
                else:
                    hosted[o] = True
                    room = tab[o]["L0"] - R_of[o]
                    wk, dy, met = run_group(cx, us, lh, tab[o]["G"], room, Ch * np_["phi"], tph)
                    placed[o[:8]] = len(us)
                _stats_add(stats, met); parts.append((wk, dy))
            row["placed"] = placed
        else:                                        # C: node-scope pooling
            nodes_slots = pack_nodes(groups, K)
            nodes = []
            for nd in nodes_slots:
                gl = []
                for s in nd:
                    if s[0] == "mg":
                        gl.append([max(0.0, tab[s[1]]["L0"] - R_of[s[1]]), tph["h_tot"]])
                    else:
                        gl.append([bh, tph["h_tot"]])
                nodes.append(dict(free=sum(g[0] for g in gl), gpus=gl))
            cap = [nd["free"] for nd in nodes]
            homes = node_place(uinfo, nodes)
            for n, nd in enumerate(nodes):           # physical invariant (§7)
                if nd["free"] < -1e-9 or any(g[0] < -1e-9 for g in nd["gpus"]):
                    raise RuntimeError(f"node {n} over-committed")
            pc = tail_phys(np_, stack, {"tpot_q": tph["tpot_q"] * (1 + np_["beta"]),
                                        "tpot_sat": tph["tpot_sat"] * (1 + np_["beta"])})
            ckey = []
            for n, nd in enumerate(nodes_slots):
                cn = sum(Ch * (np_["phi"] if s[0] == "mg" else 1.0) for s in nd)
                ckey.append((round(cap[n], 9), round(cn, 6)))
            for u, (n, gi) in enumerate(homes):
                s = nodes_slots[n][gi]
                if s[0] == "mg":
                    hosted[s[1]] = True
            bykey = collections.defaultdict(list)
            for n in range(len(nodes_slots)):
                bykey[ckey[n]].append(n)
            placed = {}
            full_done = False
            for (cg, cn), ns in sorted(bykey.items()):
                idx = {n: j for j, n in enumerate(ns)}
                us = [u for u, (n, _) in enumerate(homes) if n in idx]
                if not us:
                    continue
                lh = [idx[homes[u][0]] for u in us]
                extra = cx["mg"] if not full_done else ()
                full_done = True
                wk, dy, met = run_group(cx, us, lh, len(ns), cg, cn, pc, models_extra=extra)
                _stats_add(stats, met); parts.append((wk, dy))
                placed[f"{cg:.2f}x{len(ns)}"] = len(us)
            row["placed"] = placed
    except ValueError as e:
        row.update(infeasible=str(e), elapsed_s=round(time.time() - t0, 1))
        return row
    gw, gd = giant_run(cx, np_, R_of, hosted, np_["sigma"])
    ow, od = assemble(cx, parts, gw, gd)
    rw, rd = full_ref(cx, np_)
    row["stats"] = stats
    row["slo"] = score_all(ow, od, rw, rd)
    tail_m = [m for m in range(cx["M"]) if m not in set(cx["mg"])]
    row["slo_tail"] = score_all([ow[m] for m in tail_m], [[od[d][m] for m in tail_m] for d in range(NDAYS)],
                                [rw[m] for m in tail_m], [[rd[d][m] for m in tail_m] for d in range(NDAYS)])
    row["mg"] = mg_detail(cx, gw, gd, R_of, hosted)
    row["heldout_req"] = int(sum(sum(c.values()) for c in ow))
    row["heldout_req_reps"] = [int(sum(sum(c.values()) for c in rep)) for rep in lodo(ow, od)]
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------------ WAKE (§4 item 6)
def run_wake(src, K, seed, np_, root, cache, tag=""):
    """v2 BIG_BAL_LEN on K full H100s, box 8, h2d/d2h = PCIe (53.8/53.9) or NVLink (350, upper bound). Tail-only scoring."""
    if src not in ("PCIE", "NVL"):
        raise ValueError(f"unknown wake source {src}")
    h2d, d2h = (H2D_PCIE, D2H_PCIE) if src == "PCIE" else (NVL_P2P, NVL_P2P)
    tph = tail_phys(np_, None, {"h2d": h2d, "d2h": d2h})
    S = V2.setup2("BIG_BAL_LEN", tph, root, cache)
    S = dict(S, box_g=8)
    tr = S["tr"]
    M = len(tr.models)
    row = dict(spec=f"WAKE:{src}", arm="WAKE", src=src, K=K, seed=seed, tag=tag, nphys=np_, gpus=K)
    t0 = time.time()
    try:
        if K == 0:
            raise ValueError("WAKE needs K >= 1")
        homes, cands, info = V2.place_arm2(S, K, tph)
    except ValueError as e:
        row.update(infeasible=str(e), elapsed_s=round(time.time() - t0, 1))
        return row
    cfg = dict(S["cfg"])
    cfg["h2d"], cfg["d2h"], cfg["protect"] = tph["h2d"], tph["d2h"], tph["protect"]
    st = relabel_days(tr, M)
    out, met = V2.simulate(st, S["units"], list(S["unit_of"]) * (1 + NDAYS), ("card", homes), K, cfg, seed, None,
                           card_gib=S["card_gib"], phys=tph, kvtok=S["kvtok"], copies=cands if cfg["copies"] else None,
                           C=S["C"], mech=cfg["mech"], shed_tau=cfg["shed_tau"], bpt=S["bpt"], resize=None)
    ow, od = split_days(out, M)
    cx = dict(S=S, tr=tr, M=M, tph=tph)
    rw, rd = tail_ref(cx, np_)
    tail_m = [m for m in range(M) if S["unit_of"][m] >= 0]
    row["stats"] = met["stats"]
    row["slo_tail"] = score_all([ow[m] for m in tail_m], [[od[d][m] for m in tail_m] for d in range(NDAYS)],
                                [rw[m] for m in tail_m], [[rd[d][m] for m in tail_m] for d in range(NDAYS)])
    row["heldout_req"] = int(sum(sum(c.values()) for c in ow))
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------------ D (§4 item 5)
def d_classes(cx):
    by = collections.defaultdict(list)
    for cid, d in cx["tab"].items():
        if d["placeable"]:
            by[d["G"]].append(cid)
    return {G: sorted(v, key=lambda c: (-(cx["tab"][c]["train_req"] or 0), c)) for G, v in by.items()}


def d_host_cap(G, np_):
    """prereg §4 D: host RAM per group = (G / 8 nodes, min 1/8) x node RAM."""
    return max(G, 1) / 8.0 * np_["node_ram"]


def run_d(G, Kc, seed, np_, root, cache, tag=""):
    """One D cell: the G-class's MG models round-robin over Kc groups of G GPUs, v2 sleep path, per-model Counters."""
    cx = ctx(root, cache, np_, "POOL_LEN")
    tab, tr, M, S = cx["tab"], cx["tr"], cx["M"], cx["S"]
    cls = d_classes(cx).get(G)
    if not cls:
        raise ValueError(f"no MG class with G={G}")
    n_c = len(cls)
    if not 1 <= Kc <= n_c:
        raise ValueError(f"Kc must be in 1..{n_c}")
    homes = [k % Kc for k in range(n_c)]
    row = dict(spec="D", arm="D", G=G, Kc=Kc, n_c=n_c, seed=seed, tag=tag, nphys=np_, gpus=G * Kc,
               members=[c[:8] for c in cls])
    bh = b_h(np_)
    card = G * bh
    s = np_["s_g"] * G
    nh = collections.Counter(homes)
    units = []
    kt = []
    bpt = []
    host = collections.Counter()
    for k, cid in enumerate(cls):
        d = tab[cid]
        m = cx["cid_m"][cid]
        kv = card - d["W"] - G * np_["o_g"] - s * (nh[homes[k]] - 1)
        if kv < KV_MIN * G - 1e-9:
            row["infeasible"] = f"GPU: {cid[:8]} leaves {kv:.2f} GiB of KV with {nh[homes[k]] - 1} sleeping group-mates"
            return row
        host[homes[k]] += d["W"] * np_["r_pin"] + G * F.PHYS0["o_proc"]
        b = S["pop"]["bpt"][m] * d["dup"]
        units.append(dict(cid=cid, rep=m, members=[m], tb=tr.models[m][1], prec=np_["fmt"], w=d["W"],
                          f=d["W"] + G * np_["o_g"] + kv, shape=(tr.models[m][1], np_["fmt"])))
        kt.append(kv * G30 / b)
        bpt.append(b)
    for g, h in host.items():
        if h > d_host_cap(G, np_) + 1e-9:
            row["infeasible"] = f"host RAM: group {g} needs {h:.1f} GiB > {d_host_cap(G, np_):.1f}"
            return row
    cfg, _ = V2.arm_cfg2("ALL")
    cfg = dict(cfg, copies=False, topo="direct", h2d=G * H2D_PCIE, d2h=G * D2H_PCIE)
    cfg["protect"] = cx["tph"]["protect"]
    gp = giant_phys(np_, cx["tph"])
    unit_of = [-1] * M
    for k, cid in enumerate(cls):
        unit_of[cx["cid_m"][cid]] = k
    keep = [cx["cid_m"][c] for c in cls]
    st = relabel_days(subtrace(tr, keep), M)
    t0 = time.time()
    out, met = V2.simulate(st, units, unit_of * (1 + NDAYS), ("card", homes), Kc, cfg, seed, None, card_gib=card, s=s,
                           phys=gp, kvtok=kt, copies=None, C=G * C_REF * np_["m_h"] * np_["e_par"], mech=None,
                           shed_tau=None, bpt=bpt, resize=None)
    ow, od = split_days(out, M)
    rw, rd = mg_ref(cx, np_)
    ref_w = [rw.get(m, collections.Counter()) for m in keep]
    ref_d = [[rd[m][d] if m in rd else collections.Counter() for m in keep] for d in range(NDAYS)]
    row["slo"] = score_all([ow[m] for m in keep], [[od[d][m] for m in keep] for d in range(NDAYS)], ref_w, ref_d)
    row["stats"] = met["stats"]
    row["p95"] = {c[:8]: W._jsonable(T._pct(ow[cx["cid_m"][c]], 0.95)) for c in cls}
    row["heldout_req"] = int(sum(sum(ow[m].values()) for m in keep))
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------------ instrument checks (§7)
def check_repro(root, cache, v2dir):
    """I-REPRO: B_h 88, m_h 1.1, no MG engines: arm A reproduces v2's committed BIG_POOL_LEN / PH-STATIC rows."""
    want = {}
    for f in sorted(os.listdir(v2dir)):
        if f.endswith(".jsonl") and (f.startswith("BIG_POOL_LEN__point__s0") or f.startswith("BIG_RESIZE__PH-STATIC__s0")):
            for l in open(os.path.join(v2dir, f)):
                if l.strip():
                    r = json.loads(l)
                    want[(r["arm"], r.get("tag", ""), r["K"])] = r
    plan_ = [("POOL_LEN", "BIG_POOL_LEN", "", (56, 68, 192)), ("STATIC", "BIG_RESIZE", "PH-STATIC", (68, 192))]
    np_ = nphys_with({"bh": 88.0, "nomg": 1.0})
    res = []
    for stack, a2, tg, ks in plan_:
        for K in ks:
            r2 = want.get((a2, tg, K))
            if r2 is None:
                res.append(dict(stack=stack, K=K, ok=False, why="v2 row missing"))
                continue
            r = run_cell(f"A:{stack}", K, 0, np_, root, cache)
            ok = ("infeasible" not in r and r["stats"] == r2["stats"] and r["heldout_req"] == r2["heldout_req"]
                  and all(abs((r["slo"][s]["M"] or 0) - (r2["slo"][s]["M_frac"] or 0)) < 1e-12
                          and abs((r["slo"][s]["R"] or 0) - (r2["slo"][s]["R_ratio"] or 0)) < 1e-12
                          and r["slo"][s]["eligible"] == r2["slo"][s]["eligible"]
                          and r["slo"][s]["feasible"] == r2["slo"][s]["slo_feasible_models"] for s in ("2.0", "10.0")))
            res.append(dict(stack=stack, K=K, ok=bool(ok), got={s: r["slo"][s]["M"] for s in ("2.0", "10.0")}
                            if "slo" in r else r.get("infeasible"),
                            want={s: r2["slo"][s]["M_frac"] for s in ("2.0", "10.0")}))
    return {"I_REPRO": res, "I_REPRO_pass": bool(res) and all(x["ok"] for x in res)}


def check_day(root, cache):
    """I-DAY: relabelled simulate's week == the un-relabelled simulate (one A cell); dedicated_by_day week == v1 dedicated."""
    np_ = nphys_with()
    cx = ctx(root, cache, np_, "POOL_LEN")
    S, tr, M, tph = cx["S"], cx["tr"], cx["M"], cx["tph"]
    K = 256
    uinfo = tail_units_info(cx)
    homes = hetero_spread(uinfo, [[tph["big_card_gib"], tph["h_tot"]] for _ in range(K)])
    Ch = C_REF * tph["cref_mult"] * tph["big_mult"]
    wk, dy, _ = run_group(cx, list(range(len(uinfo))), homes, K, tph["big_card_gib"], Ch, tph, models_extra=cx["mg"])
    cfg = dict(S["cfg"])
    cfg["h2d"], cfg["d2h"], cfg["protect"] = tph["h2d"], tph["d2h"], tph["protect"]
    out, _ = V2.simulate(tr, S["units"], S["unit_of"], ("card", homes), K, cfg, 0, None, card_gib=tph["big_card_gib"],
                         phys=tph, kvtok=S["kvtok"], copies=None, C=Ch, mech=cfg["mech"], shed_tau=None, bpt=S["bpt"],
                         resize=None)
    sim_ok = all(wk[m] == out[m] for m in range(M))
    days_ok = all(sum((dy[m][d] for d in range(NDAYS)), collections.Counter()) == wk[m] for m in wk)
    kt = [((tph["big_card_gib"] - mm[4] - OVH) * G30 / S["pop"]["bpt"][k]) if tph["big_card_gib"] - mm[4] - OVH > 0
          else -1.0 for k, mm in enumerate(tr.models)]
    v1 = F.dedicated(tr, kt, tph, None, C=Ch)
    mine, mday = dedicated_by_day(tr, kt, tph, Ch)
    ded_ok = all(v1[m] == mine[m] for m in range(M))
    dday_ok = all(sum((mday[d][m] for d in range(NDAYS)), collections.Counter()) == mine[m] for m in range(M))
    tot = int(sum(sum(c.values()) for c in mine))
    return {"I_DAY": dict(sim_week_equal=sim_ok, sim_days_sum=days_ok, dedicated_week_equal=ded_ok, dedicated_days_sum=dday_ok,
                          heldout_req=tot), "I_DAY_pass": sim_ok and days_ok and ded_ok and dday_ok and tot == 3687246}


# ------------------------------------------------------------------------------------------------ summarize (§5)
def _rows(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            out += [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
    return out


def _ntag(np_):
    keys = sorted(k for k in np_ if k not in ("fmt", "layout"))
    return json.dumps({k: np_[k] for k in keys}, sort_keys=True)


def _reps(r, slo, key="slo"):
    if "infeasible" in r:
        return None
    return r[key][str(slo)]["M_reps"]


def summarize(d):
    rows = _rows(d)
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    dr = collections.defaultdict(list)
    wk = collections.defaultdict(lambda: collections.defaultdict(dict))
    seen = set()
    for r in rows:                                    # QA r1: a duplicate row (an appended re-run) is an error, never a silent overwrite
        key = (r["spec"], r.get("tag", ""), json.dumps(r["nphys"], sort_keys=True), r.get("K"), r.get("G"), r.get("Kc"), r["seed"])
        if key in seen:
            raise ValueError(f"duplicate row {key[:2]} K={key[3]} seed={key[6]}")
        seen.add(key)
    for r in rows:
        if r["arm"] == "D":
            dr[(r["nphys"]["fmt"], r["nphys"]["layout"], _ntag(r["nphys"]))].append(r)
        elif r["arm"] == "WAKE":
            wk[(r["src"], r.get("tag", ""))][r["K"]][r["seed"]] = r
        else:
            by[(r["nphys"]["fmt"], r["nphys"]["layout"], _ntag(r["nphys"]), r["stack"], r.get("tag", ""))][(r["arm"], r["r"])][
                r["K"]] = r
    res = {"ABC": [], "D": [], "WAKE": {}}
    for key, arms in sorted(by.items()):
        fmt, lay, nt, stack, tag = key
        A = arms.get(("A", "ALL"))
        one = dict(fmt=fmt, layout=lay, stack=stack, tag=tag, nphys=json.loads(nt), arms={})
        if not A or K_TOP not in A or "infeasible" in A[K_TOP]:
            one["note"] = "control A missing or infeasible at K_top"
            res["ABC"].append(one)
            continue
        sumG = A[K_TOP]["sum_G"]
        for slo in T.SLOS:
            th = min(A[K_TOP]["slo"][str(slo)]["M_reps"])
            one.setdefault("theta", {})[str(slo)] = th

            def km(rowsk):
                pm_all, per = {}, [dict() for _ in range(NDAYS)]
                for K, rr in rowsk.items():
                    reps = _reps(rr, slo)
                    for x in range(NDAYS):
                        per[x][K] = reps is not None and reps[x] is not None and reps[x] >= th - 1e-12
                    pm_all[K] = all(per[x][K] for x in range(NDAYS))
                return k_match_stable(pm_all), [k_match_stable(p) for p in per]
            ka, ka_r = km(A)
            ga_r = [sumG + k for k in ka_r]
            one["arms"].setdefault("A:ALL", {})[str(slo)] = dict(K_match=ka, gpus=sumG + ka, gpus_reps=ga_r)
            n_fix = sumG + ka
            mA = A.get(ka, {})
            for (arm, rr_), rowsk in sorted(arms.items()):
                if arm == "A":
                    continue
                kx, kx_r = km(rowsk)
                gx_r = [sumG + k for k in kx_r]
                kfix = max([k for k in rowsk if k <= n_fix - sumG] or [None], key=lambda v: -1 if v is None else v)
                mx = _reps(rowsk[kfix], slo) if kfix is not None else None
                mtop = _reps(rowsk[K_TOP], slo) if K_TOP in rowsk else None
                one["arms"].setdefault(f"{arm}:{rr_}", {})[str(slo)] = dict(
                    K_match=kx, gpus=sumG + kx, gpus_reps=gx_r, q1=verdict_gpus(ga_r, gx_r),
                    delta_gpus=(sumG + ka) - (sumG + kx),
                    q2_at_N=dict(N=n_fix, K=kfix, verdict=verdict_m(_reps(mA, slo) if mA else None, mx)),
                    q2_at_top=verdict_m(_reps(A[K_TOP], slo), mtop),
                    M_at_N=(rowsk[kfix]["slo"][str(slo)]["M"] if kfix is not None and "slo" in rowsk[kfix] else None))
        res["ABC"].append(one)
    for key, rs in sorted(dr.items()):
        fmt, lay, nt = key
        byG = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in rs:
            byG[r["G"]][r["Kc"]][r["seed"]] = r
        out = dict(fmt=fmt, layout=lay, nphys=json.loads(nt), classes={})
        tot_d = tot_a = 0
        tot_d_r_lo = tot_d_r_hi = 0
        for G, ks in sorted(byG.items()):
            n_c = next(iter(next(iter(ks.values())).values()))["n_c"]
            if n_c not in ks:
                out["classes"][str(G)] = dict(n_c=n_c, note="K_c = n_c (D's own control) missing")
                continue
            full = ks[n_c]
            cls = {}
            for slo in T.SLOS:
                def cnt(rr):
                    if "infeasible" in rr:
                        return None
                    return rr["slo"][str(slo)]["meet_reps"]
                th_l = [v for s_ in full for v in (cnt(full[s_]) or [-1])]
                th = min(th_l) if th_l else None
                reps = sorted({(s_, x) for ss in ks.values() for s_ in ss for x in range(NDAYS)})

                def stable_k(pred):
                    best = None
                    for K in sorted(ks, reverse=True):
                        if pred(K):
                            best = K
                        else:
                            break
                    return best

                def ok_rep(K, s_, x):
                    v = cnt(ks[K][s_]) if s_ in ks[K] else None
                    return th is not None and v is not None and v[x] >= th
                best = stable_k(lambda K: all(ok_rep(K, s_, x) for s_, x in reps))
                per = [stable_k(lambda K, s_=s_, x=x: ok_rep(K, s_, x)) for s_, x in reps]
                g_reps = [G * (k if k is not None else n_c + 1) for k in per]
                if th is None or th <= 0:
                    verdict = ("DEGENERATE (theta_c <= 0: no eligible giant of the class meets this SLO even alone, or the control "
                               "K_c = n_c is infeasible)")
                else:
                    verdict = verdict_gpus([G * n_c], g_reps)
                cls[str(slo)] = dict(theta=th, Kc_match=best, gpus=None if best is None else G * best, gpus_dedicated=G * n_c,
                                     gpus_reps=[min(g_reps), max(g_reps)] if g_reps else None, verdict=verdict)
            out["classes"][str(G)] = dict(n_c=n_c, members=full[min(full)]["members"], slo=cls)
        res["D"].append(out)
    for (src, tag), byk in sorted(wk.items()):
        res["WAKE"][f"{src}|{tag}"] = {str(K): {str(s): (None if "infeasible" in r else r["slo_tail"]["10.0"]["M"])
                                                for s, r in ss.items()} for K, ss in sorted(byk.items())}
    pc = wk.get(("PCIE", "main")) or wk.get(("PCIE", "")); nv = wk.get(("NVL", "main")) or wk.get(("NVL", ""))
    if pc and nv:
        q5 = {}
        top = max(pc)
        for slo in T.SLOS:
            def reps(ss):
                v = []
                for s_, r in ss.items():
                    if "infeasible" in r:
                        return None
                    v += r["slo_tail"][str(slo)]["M_reps"]
                return v
            if top not in pc or reps(pc[top]) is None:
                q5[str(slo)] = "NOT READ"
                continue
            th = min(reps(pc[top]))

            def kmw(byk):
                pm = {}
                for K, ss in byk.items():
                    v = reps(ss)
                    pm[K] = v is not None and all(x is not None and x >= th - 1e-12 for x in v)
                return k_match_stable(pm, top=top)
            q5[str(slo)] = dict(theta=th, K_pcie=kmw(pc), K_nvl=kmw(nv), at_top=verdict_m(reps(pc[top]), reps(nv.get(top, {}))
                                                                                          if top in nv else None))
        res["Q5"] = q5
    return res


# ------------------------------------------------------------------------------------------------ self-test (§8)
def self_test():
    ok, fails, n = True, [], 0

    def chk(name, cond, detail=""):
        nonlocal ok, n
        n += 1
        if not cond:
            ok = False
            fails.append(f"FAIL {name} {detail}")

    def near(a, b, tol=1e-6):
        return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))

    np0 = nphys_with()
    bh = b_h(np0)
    chk("GH0 B_h", near(bh, 71.24326171875, 1e-12), bh)
    chk("GH0 B_NE_FP8", near(B_NE["fp8"], 1.2162577581951746, 1e-12))
    chk("GH0 B_NE_INT4", near(B_NE["int4"], 2.002323778256731, 1e-12))
    chk("GH0 KIMI_E", KIMI_E == 1014686023680)
    anchor = {"cid": ANCHOR, "name": "kimi", "moe": {"experts": 384}}
    W4, E4, NE4, moe4 = mg_weights(anchor, "int4")
    chk("GH1 int4 W", near(W4, 553.4221989139915, 1e-12), W4)
    lay4 = mg_layout(W4, E4, NE4, moe4, "DCP", bh, 6.0)
    chk("GH1 int4 G16 L0", lay4 is not None and lay4[0] == 16 and near(lay4[2], 30.65437428662554, 1e-12), lay4)
    chk("GH1 int4 Q50 KV tokens", near(kv_tokens(16, 0.5 * lay4[2], 70272, 1.0), 3747140.683071494, 1e-9))
    chk("GH1 int4 Q50 tail room x16", near(16 * (lay4[2] - 0.5 * lay4[2]), 245.23499429300432, 1e-12))
    W8, E8, NE8, moe8 = mg_weights(anchor, "fp8")
    lay8 = mg_layout(W8, E8, NE8, moe8, "DCP", bh, 6.0)
    chk("GH1 fp8 G16 L0", lay8 is not None and lay8[0] == 16 and near(lay8[2], 5.336462065670645, 1e-12), lay8)
    chk("GH1 fp8 TP16 KV tokens", near(kv_tokens(16, lay8[2], 70272, kv_dup("TP", 16, 1)), 81540.05168630474, 1e-9))
    chk("GH2 DPA fp8 NOT-PLACEABLE", mg_layout(W8, E8, NE8, True, "DPA", bh, 6.0) is None)
    d4 = mg_layout(W4, E4, NE4, True, "DPA", bh, 6.0)
    chk("GH2 DPA int4 L0", d4 is not None and d4[0] == 16 and near(d4[2], 10.160906554758554, 1e-12), d4)
    gh3 = {"7b546c7d": (358458391872, 160, 8, 1536, 89, 39292828992, {"fp8": (8, 22.937345286442564), "int4": (4, 10.755963219249253)}),
           "aa605b6d": (235107904512, 128, 8, 1536, 94, 22205033472, {"fp8": (4, 10.086637101925817), "int4": (4, 31.766090146404622)}),
           "a4d621aa": (753910024032, 256, 8, 2048, 75, 51783534432, {"fp8": (16, 20.982884649370682), "int4": (8, 10.991086558507732)}),
           "9b227eb9": (120412337472, 128, 4, 2880, 36, 9333732672, {"fp8": (2, 8.57976505298923), "int4": (2, 29.84754579691014)})}
    for cid, (P, Ex, k, mi, ml, A, want) in gh3.items():
        row = {"cid": cid, "params_total_R2": P, "params_active_R2": A,
               "moe": {"experts": Ex, "topk": k, "moe_inter": mi, "moe_layers": ml}}
        for fmt in FMTS:
            W_, E_, NE_, mo = mg_weights(row, fmt)
            L = mg_layout(W_, E_, NE_, mo, "DCP", bh, 6.0)
            chk(f"GH3 {cid} {fmt}", L is not None and L[0] == want[fmt][0] and near(L[2], want[fmt][1], 1e-9), (L, want[fmt]))
    chk("GH4 Kimi TP16", kv_dup("TP", 16, 1) == 16.0)
    chk("GH4 Qwen235 TP8", kv_dup("TP", 8, 4) == 2.0)
    chk("GH4 GLM TP8", kv_dup("TP", 8, 8) == 1.0)
    chk("GH4 Qwen235 TP4", kv_dup("TP", 4, 4) == 1.0)
    chk("GH4 DCP", kv_dup("DCP", 16, 1) == 1.0 and kv_dup("DPA", 16, 1) == 1.0)
    try:
        hetero_spread([dict(cid="x", f=6.0, host=0.0)], [[5.0, 99.0], [5.0, 99.0], [3.24, 99.0]])
        chk("GH5 adversarial raises", False, "placed")
    except ValueError:
        chk("GH5 adversarial raises", True)
    chk("GH5 f4 -> card 0", hetero_spread([dict(cid="y", f=4.0, host=0.0)], [[5.0, 99.0], [5.0, 99.0], [3.24, 99.0]]) == [0])
    chk("GH5 f20 -> full card", hetero_spread([dict(cid="z", f=20.0, host=0.0)], [[15.3, 99.0], [71.24, 99.0]]) == [1])
    cards = [[10.0, 99.0], [10.0, 99.0]]
    hetero_spread([dict(cid="a", f=6.0, host=0.0), dict(cid="b", f=6.0, host=0.0)], cards)
    chk("GH5 never over-commits", all(c[0] >= -1e-9 for c in cards))
    # GH6 straggler: through dedicated_by_day, tpot multiplier only where hosted
    models = [("g", 100.0, 3.1, "fp8", 1.0, 1.0)]
    tr = T.Trace.explicit_cells(models, [(HELD * 60 + 1.0, 0, 100.0, 1000.0)])
    p = dict(F.PHYS0, tpot_q=GIANT_TPOT, tpot_sat=GIANT_TPOT * TPOT_SAT_RATIO, kappa=0.5)
    w0, _ = dedicated_by_day(tr, [1e9], p, [1e9])
    w1, _ = dedicated_by_day(tr, [1e9], p, [1e9], tpot_mult=[1.3])
    chk("GH6 straggler unhosted TPOT", near(GIANT_TPOT, 0.02631578947368421, 1e-12) and w0[0] == w1[0],
        "TTFT has no decode term; KV residence does")
    # residence check: KV-bound case where TPOT moves the KV wait
    trk = T.Trace.explicit_cells(models, [(HELD * 60 + 1.0 + 0.001 * j, 0, 100.0, 1000.0) for j in range(20)])
    a0, _ = dedicated_by_day(trk, [3000.0], p, [1e9])
    a1, _ = dedicated_by_day(trk, [3000.0], p, [1e9], tpot_mult=[1.3])
    chk("GH6 straggler moves a KV-bound TTFT", T._pct(a1[0], 0.95) > T._pct(a0[0], 0.95), (T._pct(a0[0], .95), T._pct(a1[0], .95)))
    chk("GH6 hosting TPOT", near(GIANT_TPOT * 1.3, 0.034210526315789476, 1e-12))
    # GH5b / GH6b: B cards and the straggler multiplier as pure functions (prereg mutants)
    tabg = {"g1": dict(placeable=True, L0=30.65, G=2), "g2": dict(placeable=True, L0=5.0, G=1)}
    bc, bo = b_cards(tabg, [("g1", 2), ("g2", 1)], {"g1": 15.325, "g2": 6.0}, 2, 71.24, 99.0)
    chk("GH5b B cards", [round(c[0], 6) for c in bc] == [15.325, 15.325, 0.0, 71.24, 71.24] and bo == ["g1", "g1", "g2", None, None],
        bc)
    mu = tpot_mults(tabg, {"g1": 0, "g2": 1}, {"g1": True}, 0.3, 3)
    chk("GH6b straggler only where hosted", mu == [1.3, 1.0, 1.0], mu)
    # GH7 LODO
    k1, k20, k05 = T._key(1.0), T._key(20.0), T._key(0.5)
    X = [collections.Counter({k1: 30}), collections.Counter({k1: 30}), collections.Counter({k20: 30})]
    Y = [collections.Counter({k1: 30}) for _ in range(3)]
    Rf = [collections.Counter({k05: 30}) for _ in range(3)]
    dayX = [[X[d], Y[d]] for d in range(3)] + [[collections.Counter(), collections.Counter()] for _ in range(4)]
    dayR = [[Rf[d], Rf[d]] for d in range(3)] + [[collections.Counter(), collections.Counter()] for _ in range(4)]
    wkX = [sum(X, collections.Counter()), sum(Y, collections.Counter())]
    wkR = [sum(Rf, collections.Counter()), sum(Rf, collections.Counter())]
    sc = score_all(wkX, dayX, wkR, dayR)["10.0"]
    chk("GH7 LODO week M", near(sc["M"], 0.5, 1e-12), sc["M"])
    chk("GH7 LODO reps", [round(v, 6) for v in sc["M_reps"][:3]] == [0.5, 0.5, 1.0], sc["M_reps"])
    # GH8 decision rules
    chk("GH8 FEWER", verdict_gpus([100, 104, 104], [90, 92, 96]) == "FEWER-GPUS")
    chk("GH8 UNRESOLVED", verdict_gpus([100, 104], [100, 108]) == "UNRESOLVED")
    chk("GH8 MORE", verdict_gpus([100, 104], [108, 112]) == "MORE-GPUS")
    chk("GH8 K_match stable", k_match_stable({0: False, 4: True, 8: False, 12: True, 16: True}, 16, 4) == 12)
    chk("GH8 K_match never", k_match_stable({0: False, 4: False}, 4, 4) == 8)
    chk("GH8 LIFTS", verdict_m([0.8, 0.81], [0.83, 0.85]) == "LIFTS" and verdict_m([0.8, 0.81], [0.7, 0.75]) == "LOWERS"
        and verdict_m([0.8, 0.81], [0.805, 0.9]) == "UNATTRIBUTABLE" and verdict_m([0.8], None) == "INFEASIBLE")
    # GH9 node packing
    nds = pack_nodes([("k16", 16), ("g8", 8), ("a4", 4), ("b4", 4), ("c2", 2)], 10)
    chk("GH9 nodes", len(nds) == 6, len(nds))
    chk("GH9 16-group = 2 whole nodes", all(s[1] == "k16" for s in nds[0] + nds[1]) and len(nds[0]) == len(nds[1]) == 8)
    chk("GH9 4+4 share a node", len(nds) > 3 and sorted({str(s[1]) for s in nds[3]}) == ["a4", "b4"])
    chk("GH9 full fill partial first", len(nds) == 6 and sum(1 for s in nds[4] if s[0] == "full") == 6 and len(nds[5]) == 4)
    chk("GH9 GPUs", sum(len(x) for x in nds) == 44)
    # GH10 D host RAM
    glm = {"cid": "7b546c7d", "params_total_R2": 358458391872, "params_active_R2": 39292828992,
           "moe": {"experts": 160, "topk": 8, "moe_inter": 1536, "moe_layers": 89}}
    Wg = mg_weights(glm, "fp8")[0]
    per = Wg * 1.27 + 8 * F.PHYS0["o_proc"]
    chk("GH10 GLM358 host per homed", near(per, 472.6281109522436, 1e-9), per)
    chk("GH10 GLM358 max per GCP node", math.floor(d_host_cap(8, np0) / per) == 3)
    perk = W8 * 1.27 + 16 * F.PHYS0["o_proc"]
    chk("GH10 Kimi fp8 max per 2-node group", math.floor(d_host_cap(16, np0) / perk) == 2, perk)
    # GH11 C placement relaxes the per-GPU KV-floor fit
    U = dict(cid="u", w=6.0, f=12.0, host=0.0)
    try:
        hetero_spread([U], [[10.0, 99.0], [10.0, 99.0]])
        chk("GH11 B refuses", False)
    except ValueError:
        chk("GH11 B refuses", True)
    chk("GH11 C places on GPU 0", node_place([U], [dict(free=20.0, gpus=[[10.0, 99.0], [10.0, 99.0]])]) == [(0, 0)])
    try:
        node_place([dict(cid="v", w=9.5, f=12.0, host=0.0)], [dict(free=20.0, gpus=[[10.0, 99.0], [10.0, 99.0]])])
        chk("GH11 C refuses weights over any GPU", False)
    except ValueError:
        chk("GH11 C refuses weights over any GPU", True)
    # GN1 dedicated_by_day == v1 dedicated (scalar C, no multiplier) on a random explicit trace
    rng = np.random.default_rng(7)
    mods = [("a", 8.0, 8.0, "fp8", 8.56, 9.38), ("b", 4.0, 4.0, "fp8", 4.3, 5.1)]
    reqs = []
    for j in range(400):
        t = (HELD - 3) * 60.0 + float(rng.random()) * 3 * DAY_MIN * 60.0 + 3 * 60.0 * 0
        reqs.append((t + (DAY_MIN * 60.0 * float(rng.integers(0, 3))), int(rng.integers(0, 2)),
                     float(rng.integers(50, 4000)), float(rng.integers(10, 3000))))
    trr = T.Trace.explicit_cells(mods, reqs)
    pp = dict(F.PHYS0, tpot_sat=0.06)
    v1 = F.dedicated(trr, [6000.0, 3000.0], pp, None, C=C_REF)
    mine, mday = dedicated_by_day(trr, [6000.0, 3000.0], pp, C_REF)
    chk("GN1 dedicated fork == v1", all(v1[m] == mine[m] for m in range(2)), [(sum(v1[m].values()), sum(mine[m].values())) for m in range(2)])
    chk("GN1 days sum to week", all(sum((mday[d][m] for d in range(NDAYS)), collections.Counter()) == mine[m] for m in range(2)))
    chk("GN1 scored something", sum(sum(c.values()) for c in mine) > 0)
    want_day = collections.Counter()
    for t, m_, i_, o_ in reqs:
        mi_ = int(t // 60)
        if mi_ >= HELD:
            want_day[(mi_ - HELD) // DAY_MIN] += 1
    got_day = {d: sum(sum(mday[d][m].values()) for m in range(2)) for d in range(NDAYS)}
    chk("GN1b requests land on their own day", all(got_day[d] == want_day.get(d, 0) for d in range(NDAYS)) and len(want_day) >= 2,
        (got_day, dict(want_day)))
    # GN2 relabel_days + split_days == an un-relabelled v2 simulate (all-awake POOL on 1 card)
    units = [dict(cid="a", rep=0, members=[0], tb=8.0, prec="fp8", w=8.56, f=8.56 + OVH + 1.0, shape=(8.0, "fp8")),
             dict(cid="b", rep=1, members=[1], tb=4.0, prec="fp8", w=4.3, f=4.3 + OVH + 1.0, shape=(4.0, "fp8"))]
    cfg, _ = V2.arm_cfg2("POOL_SZ")
    cfg = dict(cfg, h2d=22.6, d2h=23.8, protect=60.0)
    kv = [1.0 * G30 / 147456.0, 1.0 * G30 / 147456.0]
    o1, _ = V2.simulate(trr, units, [0, 1], ("card", [0, 0]), 1, cfg, 0, None, score_from=HELD, min0=(HELD - 3 * DAY_MIN),
                        card_gib=30.0, phys=pp, kvtok=kv, copies=None, C=C_REF, mech="pool", bpt=[147456.0, 147456.0])
    t8 = relabel_days(trr, 2)
    o8, _ = V2.simulate(t8, units, [0, 1] * 8, ("card", [0, 0]), 1, cfg, 0, None, score_from=HELD, min0=(HELD - 3 * DAY_MIN),
                        card_gib=30.0, phys=pp, kvtok=kv, copies=None, C=C_REF, mech="pool", bpt=[147456.0, 147456.0])
    wk8, dy8 = split_days(o8, 2)
    chk("GN2 relabel == plain simulate", all(wk8[m] == o1[m] for m in range(2)), [(sum(o1[m].values()), sum(wk8[m].values())) for m in range(2)])
    chk("GN2 days sum", all(sum((dy8[d][m] for d in range(NDAYS)), collections.Counter()) == wk8[m] for m in range(2)))
    # GN3 subtrace keeps only the given models' cells, ids unchanged
    s1 = subtrace(trr, [1])
    chk("GN3 subtrace", len(s1.n) > 0 and set(int(x) for x in s1.mid) == {1} and s1.models is trr.models)
    chk("GN3b subtrace keeps explicit offsets aligned", s1.explicit is not None and len(s1.explicit) == len(s1.n)
        and all(len(e) == int(n) for e, n in zip(s1.explicit, s1.n)))
    # GN4 MG weights: dense keeps census w; INT4 does not touch dense
    dense = {"cid": "d", "weights_gib_fp8": 75.5, "moe": None}
    chk("GN4 dense W", mg_weights(dense, "fp8")[0] == 75.5 and not moe_ok(dense))
    chk("GN4 dense DPA -> DCP", mg_layout(75.5, 0.0, 75.5, False, "DPA", bh, 6.0)[3] == "DCP")
    # GN5 tail physics: point m_h keeps v2's TPOT; m_h 2.0 scales it by 1.1/2.0 (amendment 1)
    tp1 = tail_phys(np0)
    tp2 = tail_phys(nphys_with({"m_h": 2.0}))
    chk("GN5 m_h point = v2 TPOT", tp1["tpot_q"] == V2.PHYS0["tpot_q"] and tp1["big_mult"] == 1.1)
    chk("GN5 m_h 2.0", near(tp2["tpot_q"], V2.PHYS0["tpot_q"] * 0.55, 1e-12) and tp2["big_mult"] == 2.0)
    # CLI golden (prereg §8): --giant-table through the real entry point prints the anchor's GH1 numbers
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(root, V2.CENSUS_JSON)):
        try:
            pr = subprocess.run([sys.executable, os.path.abspath(__file__), "--giant-table", "--root", root],
                                capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:            # QA r1: a hang reports FAIL, never a crash
            pr = subprocess.CompletedProcess([], 124, "", "timeout")
        lines = [json.loads(x) for x in pr.stdout.splitlines() if x.strip()]
        an = [x for x in lines if x["cid"] == ANCHOR]
        chk("CLI giant-table rc 0", pr.returncode == 0, pr.stderr[-300:])
        chk("CLI anchor GH1", len(an) == 1 and an[0]["G"] == 16 and abs(an[0]["L0"] - 30.654374) < 1e-6
            and abs(an[0]["W"] - 553.422199) < 1e-6 and an[0]["layout"] == "DCP", an)
        chk("CLI MG count 24", len(lines) == 24, len(lines))
    else:
        chk("CLI census present under the root", False, root)
    for f in fails:
        print(f)
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'} {n - len(fails)}/{n}")
    return ok


# ------------------------------------------------------------------------------------------------ CLI
def giant_table_rows(np_, root):
    rows, _ = V2.load_census(root)
    bh = b_h(np_)
    mg = [c for c, r in rows.items() if float(r.get("weights_gib_fp8") or 0.0) > 0 and bh - float(r["weights_gib_fp8"]) - OVH <= 0]
    tab = mg_table(rows, mg, np_)
    return [dict(d, fmt=np_["fmt"] if moe_ok(rows[c]) else "fp8", B_h=bh) for c, d in sorted(tab.items(), key=lambda x: -x[1]["W"])]


def _parse_sets(sets):
    over = {}
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set needs k=v, got {s}")
        k, v = s.split("=", 1)
        over[k] = v
    return over


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--giant-table", action="store_true")
    ap.add_argument("--check", choices=("REPRO", "DAY"))
    ap.add_argument("--v2rows")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--ks")
    ap.add_argument("--kgrid", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--set", action="append")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    a = ap.parse_args()
    try:
        if a.self_test:
            return 0 if self_test() else 1
        if a.giant_table:
            for r in giant_table_rows(nphys_with(_parse_sets(a.set)), a.root):
                print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
            return 0
        if a.check:
            if not a.cache:
                raise ValueError("--check needs --cache")
            if a.check == "REPRO":
                if not a.v2rows or not os.path.isdir(a.v2rows):
                    raise ValueError("--check REPRO needs --v2rows DIR")
                res = check_repro(a.root, a.cache, a.v2rows)
                print(json.dumps(res))
                return 0 if res["I_REPRO_pass"] else 1
            res = check_day(a.root, a.cache)
            print(json.dumps(res))
            return 0 if res["I_DAY_pass"] else 1
        if a.run:
            if not (a.arm and a.out and a.cache):
                raise ValueError("--run needs --arm, --out and --cache")
            np_ = nphys_with(_parse_sets(a.set))
            seeds = [int(x) for x in a.seeds.split(",")]
            if a.arm != "D":                          # QA r1: validate before the output file is touched
                if a.kgrid:
                    ks = list(K_GRID)
                elif a.ks:
                    ks = [int(x) for x in a.ks.split(",")]
                else:
                    raise ValueError("--run needs --ks or --kgrid")
            with open(a.out, "a") as fh:
                if a.arm == "D":
                    cx = ctx(a.root, a.cache, np_, "POOL_LEN")
                    for G, cls in sorted(d_classes(cx).items()):
                        for Kc in range(1, len(cls) + 1):
                            for s in seeds:
                                fh.write(json.dumps(run_d(G, Kc, s, np_, a.root, a.cache, a.tag)) + "\n"); fh.flush()
                    return 0
                for K in ks:
                    for s in seeds:
                        fh.write(json.dumps(run_cell(a.arm, K, s, np_, a.root, a.cache, a.tag)) + "\n"); fh.flush()
            return 0
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
