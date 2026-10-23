#!/usr/bin/env python3
"""gpu_score.py: the frozen decision rules of (private report) (section 5) over a fetched run.

Reads <run>/rows + <run>/logs as gpu_driver.py writes them; prints one VERDICT line per question and writes --out.
  BETA     beta_dec = max over cells {Q, S} and passes of beta_anch(PUSH, f 0.5, lender BUSY); bands HOLDS (<= 1) /
           SHRINKS (<= 3) / NOT-ESTABLISHED (> 3); UNRESOLVED-AT-EDGE when the pass range crosses a band edge. Reported beside
           it, never deciding: f 0.875 (conservative), IDLE, PULL / PULLS, the H100 projection, the lender penalty.
  ITEM10   eta = fetched expert bytes / (delta TPOT x measured pinned H2D), median over attributable cells; I-FORM first.
  ITEM12   lambda(B) = measured distinct experts per MoE layer per decode step / uniform n(1 - (1 - k/n)^B); verdict at B 16.
  ITEM9    p_same (per variant) and p_other (abort, wait) against 0.05 s / 0.8 s.
  ITEM8    ENGINE-SHEDS (never-timeout clients) and CANCEL-FREES (10 s cancel: zombie tokens, drain).
  DCP      TPOT ratio DCP2 / DCP1 per cell (cross-check only).
Output contract: exit 0 + VERDICT lines + --out JSON written (overwritten) when the run dir parses; exit 1 if DONE is
missing or a required file is unreadable (a FAILED read is never scored as a number); exit 2 = usage. A stub run (DONE
written by MG_STUB) is refused unless --allow-stub. `--self-test` = exit 0 + "SELF-TEST PASS n/n".
"""
import argparse
import json
import math
import os
import statistics
import sys

GRID = ((0.0, 144.0), (1.0, 104.0), (3.0, 8.0))   # int4 STATIC S10 node-pool win in GPUs at beta 0 / 1 / 3
HBM_SPEC = {"A100_40": 1555.0, "A100_80": 2039.0, "H100_SXM": 3350.0}   # GB/s, vendor specs (projection only)
NVLINK_SPEC = {"A100": 300.0, "H100_SXM": 450.0}                       # GB/s per direction (projection only)


# ----------------------------------------------------------------------------------------------- pure rules
def band(b):
    if b is None:
        return "NO-DATA"
    return "HOLDS" if b <= 1.0 else ("SHRINKS" if b <= 3.0 else "NOT-ESTABLISHED")


def win_gpus(b):
    """Piecewise-linear read of three grid points (an assumption: the sim's response is stepwise in K)."""
    if b is None:
        return None
    if b < 0:
        return ">=144 (below the grid)"
    for (b0, w0), (b1, w1) in zip(GRID, GRID[1:]):
        if b <= b1:
            return round(w0 + (w1 - w0) * (b - b0) / (b1 - b0), 1)
    return "<=8 (beyond the grid at beta 3)"


def edge_verdict(lo, hi):
    if lo is None or hi is None:
        return "NO-DATA"
    a, b = band(lo), band(hi)
    return a if a == b else f"UNRESOLVED-AT-EDGE({a}->{b})"


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def uniform_distinct(n, k, B):
    return n * (1.0 - (1.0 - k / n) ** B)


def eta(bytes_, dt_s, h2d_gbs):
    if not dt_s or dt_s <= 0 or not h2d_gbs:
        return None
    return bytes_ / (dt_s * h2d_gbs * 1e9)


def offload_effective(mem0, mem_l, logged_gib):
    """I-OFF2: the engine's GPU model memory must fall by >= 0.8 x the logged offloaded GiB vs the L_off 0 control (a UVA
    offload that MoE post-processing undid would log the GiB at wrap time yet keep the weights on the GPU). None = unread."""
    if mem0 is None or mem_l is None or not logged_gib:
        return None
    return (mem0 - mem_l) >= 0.8 * logged_gib


def prop_fit(xs, ys):
    """Through-origin slope + R^2, and an intercept fit (c, a). Returns dict."""
    n = len(xs)
    if n < 3:
        return None
    a0 = sum(x * y for x, y in zip(xs, ys)) / max(1e-30, sum(x * x for x in xs))
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys) or 1e-30
    r2 = 1.0 - sum((y - a0 * x) ** 2 for x, y in zip(xs, ys)) / ss_tot
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs) or 1e-30
    a1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return {"slope0": a0, "r2_0": r2, "intercept": my - a1 * mx, "slope1": a1}


def distinct_per_step(reqs, first_dense, n_exp, top_k):
    """reqs: list of arrays [steps][layers][top_k]. Returns (mean distinct experts per MoE layer per step, check, n_steps).
    I-ROUTE: every row holds top_k distinct ids in [0, n_exp); a violation returns (None, why, 0)."""
    arrs = [r for r in reqs if r]
    if not arrs:
        return None, "no routed data", 0
    steps = min(len(a) for a in arrs)
    tot, cnt = 0, 0
    for a in arrs:
        for row in a[:steps]:
            for ly in row[first_dense:]:
                if len(ly) != top_k or len(set(ly)) != top_k or min(ly) < 0 or max(ly) >= n_exp:
                    return None, f"I-ROUTE bad row {ly}", 0
    nl = len(arrs[0][0])
    for k in range(steps):
        for ly in range(first_dense, nl):
            s = set()
            for a in arrs:
                s.update(a[k][ly])
            tot += len(s)
            cnt += 1
    return (tot / cnt if cnt else None), "ok", steps


def reuse_1step(arr, first_dense):
    """Fraction of a request's experts at step k already used at step k-1 (same layer), mean over steps and layers."""
    hits, cnt = 0, 0
    for k in range(1, len(arr)):
        for ly in range(first_dense, len(arr[k])):
            prev, cur = set(arr[k - 1][ly]), arr[k][ly]
            hits += sum(1 for e in cur if e in prev)
            cnt += len(cur)
    return hits / cnt if cnt else None


# ----------------------------------------------------------------------------------------------- io
def rows(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def jload(path):
    with open(path) as f:
        return json.load(f)


def opt(fn, *a):
    try:
        return fn(*a)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------- sections
def anchor_tpot(run, plan):
    """Per cell: per pass median TPOT of COMPLETE requests; a pass with < B complete rows is INFRA never slow."""
    out = {}
    rs = opt(rows, f"{run}/rows/anchor.jsonl") or []
    for c in plan["anchor"]["cells"]:
        per, bad = [], []
        for p in range(plan["anchor"]["passes"]):
            rr = [r for r in rs if r.get("cell") == c["name"] and r.get("pass") == p]
            ok = [r["tpot"] for r in rr if r.get("complete") and r.get("tpot") is not None and not r.get("err")]
            if len(ok) < c["B"]:
                bad.append(p)
            else:
                per.append(med(ok))
        out[c["name"]] = {"tpot_s": med(per) if not bad else None, "per_pass": per, "infra_passes": bad,
                          "spread": [min(per), max(per)] if per else None}
    return out


def beta_section(run, plan, anchor):
    res = {}
    for lender in ("IDLE", "BUSY"):
        rs = opt(rows, f"{run}/rows/beta_{lender}.jsonl")
        if rs is None:
            res[lender] = {"status": "NO-DATA"}
            continue
        sec = {}
        for c in plan["beta"]["cells"]:
            loc = {r["pass"]: r for r in rs if r["cell"] == c["name"] and r["mode"] == "LOCAL"}
            tl = [loc[p]["gpu_ms_p50"] for p in sorted(loc) if loc[p].get("ok")]
            sL = ((max(tl) - min(tl)) / med(tl)) if len(tl) >= 2 and med(tl) else None
            cell = {"T_local_ms": tl, "local_spread_rel": sL, "launch_bound": any(
                (r.get("host_ms") or 0) >= 0.8 * (r.get("gpu_ms_p50") or 1e9) for r in rs if r["cell"] == c["name"])}
            for m in plan["beta"]["modes"]:
                for f in plan["beta"]["f"]:
                    bb, ba, bad = [], [], []
                    for p in range(plan["beta"]["passes"]):
                        rr = [r for r in rs if r["cell"] == c["name"] and r["mode"] == m and r["f"] == f and r["pass"] == p]
                        if not rr or not rr[0].get("ok") or p not in loc or not loc[p].get("ok"):
                            bad.append(p)
                            continue
                        t, t0 = rr[0]["gpu_ms_p50"], loc[p]["gpu_ms_p50"]
                        bb.append(t / t0 - 1.0)
                        a = anchor.get(c["name"], {}).get("tpot_s")
                        ba.append(((t - t0) / 1e3 / a) if a else None)
                    key = f"{m}@{f}"
                    if bad:
                        cell[key] = {"status": f"INSUFFICIENT passes missing/invalid {bad}"}
                        continue
                    mb = med(bb)
                    cell[key] = {"beta_bench": [round(x, 4) for x in bb], "beta_anch": [round(x, 4) if x is not None else None
                                                                                       for x in ba],
                                 "beta_anch_lo": min(ba) if None not in ba else None,
                                 "beta_anch_hi": max(ba) if None not in ba else None,
                                 "unattributable": sL is not None and abs(mb) <= sL}
            sec[c["name"]] = cell
        res[lender] = sec
    return res


def beta_verdicts(beta, run):
    def pick(lender, mode, f):
        his, los = [], []
        for c in ("Q", "S"):
            e = (beta.get(lender) or {}).get(c, {}).get(f"{mode}@{f}")
            if not e or "beta_anch_hi" not in e or e["beta_anch_hi"] is None:
                return None, None
            his.append(e["beta_anch_hi"])
            los.append(e["beta_anch_lo"])
        return max(los), max(his)
    lo, hi = pick("BUSY", "PUSH", 0.5)
    env = opt(lambda p: open(p, errors="replace").read(), f"{run}/logs/env.txt") or ""
    card = "A100_80" if "80GB" in env else "A100_40"
    tau_push = HBM_SPEC["H100_SXM"] / HBM_SPEC[card]
    tau_pull = (HBM_SPEC["H100_SXM"] / NVLINK_SPEC["H100_SXM"]) / (HBM_SPEC[card] / NVLINK_SPEC["A100"])
    unat = [c for c in ("Q", "S") if ((beta.get("BUSY") or {}).get(c, {}).get("PUSH@0.5") or {}).get("unattributable")]
    out = {"deciding": {"lo": lo, "hi": hi, "verdict": edge_verdict(lo, hi), "win_gpus_at_hi": win_gpus(hi),
                        "unattributable_cells": unat},
           "card": card, "tau_push": round(tau_push, 3), "tau_pull": round(tau_pull, 3)}
    for name, (ld, m, f) in {"conservative_f0.875": ("BUSY", "PUSH", 0.875), "idle_f0.5": ("IDLE", "PUSH", 0.5),
                             "pull_f0.5": ("BUSY", "PULL", 0.5), "pulls_f0.5": ("BUSY", "PULLS", 0.5)}.items():
        l2, h2 = pick(ld, m, f)
        out[name] = {"lo": l2, "hi": h2, "verdict": edge_verdict(l2, h2)}
    if hi is not None:
        out["h100_projection_push"] = {"beta": round(hi * tau_push, 3), "band": band(hi * tau_push)}
    if out["pull_f0.5"]["hi"] is not None:
        out["h100_projection_pull"] = {"beta": round(out["pull_f0.5"]["hi"] * tau_pull, 3),
                                       "band": band(out["pull_f0.5"]["hi"] * tau_pull)}
    bands = {out["deciding"]["verdict"], out["conservative_f0.875"]["verdict"],
             out.get("h100_projection_push", {}).get("band", "NO-DATA")}
    out["agreement"] = "AGREE" if len(bands) == 1 else f"MIXED {sorted(bands)} -> escalate, never self-resolve"
    return out


def lender_penalty(run):
    ls = opt(rows, f"{run}/rows/lender_busy.jsonl")
    win = opt(jload, f"{run}/logs/beta_busy_windows.json")
    bs = opt(rows, f"{run}/rows/beta_BUSY.jsonl")
    if not (ls and win and bs):
        return {"status": "NO-DATA"}
    mid = [((r["t_first"] + r["t_last"]) / 2, r["tpot"]) for r in ls if r.get("tpot") and r.get("t_first")]
    t0s = [r["t0"] for r in bs if r.get("t0")]
    t_b0 = win.get("t_bench_start") or (min(t0s) if t0s else None)   # the reference window ends at the bench
    if t_b0 is None:                                                  # QA r2: a reported metric never fails the score
        return {"status": "NO-DATA (no bench start time)"}
    ref = med([t for m, t in mid if win["t_settle_start"] <= m < t_b0])
    during = med([t for m, t in mid if t_b0 <= m <= win.get("t_bench_end", 1e18)])
    by = {}
    for r in bs:
        if r.get("t0") and r.get("t1"):
            by.setdefault(f"{r['mode']}@{r['f']}", []).extend(t for m, t in mid if r["t0"] <= m <= r["t1"])
    loc = med(by.get("LOCAL@0.0", []))
    per = {k: (round(med(v) / loc - 1.0, 4) if (v and loc) else None) for k, v in by.items() if k != "LOCAL@0.0"}
    return {"ref_tpot_s": ref, "during_bench_tpot_s": during,
            "overall_penalty": (during / ref - 1.0) if (ref and during) else None, "per_mode_f_vs_local": per,
            "lender_alive": win.get("lender_alive"), "note": "midpoint attribution, reported only"}


def item12_section(run, plan):
    m = plan["models"]["dsv2"]
    meta = opt(jload, f"{run}/rows/routing_meta.json") or {}
    res = {"meta_ok": meta.get("ok"), "B": {}}
    for B in plan["item12"]["Bs"]:
        d = opt(jload, f"{run}/rows/routing_B{B}.json")
        if not d:
            res["B"][B] = {"status": "NO-DATA"}
            continue
        arrs = [r.get("experts") for r in d["reqs"]]
        if any(a is None for a in arrs):
            res["B"][B] = {"status": "CAPABILITY-ABSENT (routed_experts None)"}
            continue
        D, why, steps = distinct_per_step(arrs, m["first_dense"], m["n_experts"], m["top_k"])
        U = uniform_distinct(m["n_experts"], m["top_k"], len(arrs))
        e = {"n_req": len(arrs), "steps": steps, "check": why, "D": D, "U": U, "lambda": (D / U) if D else None}
        if B == 1 and D is not None and abs(D - m["top_k"]) > 1e-9:
            e["check"] = f"I-ROUTE B=1 distinct {D} != top_k {m['top_k']}"
            e["lambda"] = None
        if arrs and why == "ok":
            e["reuse_1step"] = med([reuse_1step(a, m["first_dense"]) for a in arrs])
        res["B"][B] = e
    l16 = (res["B"].get(16) or {}).get("lambda")
    res["verdict"] = ("NO-DATA" if l16 is None else "LOCALITY" if l16 <= 0.75 else "PARTIAL" if l16 <= 0.95
                      else "UNIFORM-HOLDS")
    return res


def item10_section(run, plan, probe, i12):
    m, it = plan["models"]["dsv2"], plan["item10"]
    rs = opt(rows, f"{run}/rows/offload.jsonl") or []
    meta = {x["L_off"]: x for x in (opt(jload, f"{run}/rows/offload_meta.json") or [])}
    h2d = (probe or {}).get("h2d_gbs_0")
    cells, ctrl = {}, {}
    for l_off in it["L_off"]:
        for B in it["Bs"]:
            per, bad = [], []
            for p in range(it["passes"]):
                ok = [r["tpot"] for r in rs if r.get("L_off") == l_off and r.get("B") == B and r.get("pass") == p
                      and r.get("complete") and r.get("tpot") is not None]
                (per.append(med(ok)) if len(ok) >= B else bad.append(p))
            cells[(l_off, B)] = {"tpot": med(per) if not bad else None, "per_pass": per, "infra": bad}
    for B in it["Bs"]:
        ctrl[B] = cells[(0, B)]
    xs, ys, etas, attrib, flags = [], [], [], 0, []
    for (l_off, B), c in cells.items():
        if l_off == 0 or c["tpot"] is None or ctrl[B]["tpot"] is None:
            continue
        mm = meta.get(l_off, {})
        lg = mm.get("logged_offload_gib")
        planned = l_off * m["layer_expert_gib"]
        if lg is None or abs(lg - planned) > 0.02:
            flags.append(f"I-OFF L_off {l_off}: logged {lg} GiB vs planned {planned}")
        eff = offload_effective(meta.get(0, {}).get("model_mem_gib"), mm.get("model_mem_gib"), lg)
        c["offload_effective"] = eff
        if eff is False:                                           # the weights stayed on the GPU: no fetch to time
            flags.append(f"I-OFF2 L_off {l_off}: model memory did not fall by 0.8 x {lg} GiB (offload undone?)")
            c["attributable"] = False
            continue
        if eff is None:
            flags.append(f"I-OFF2 L_off {l_off}: model memory unread (offload not proven)")
        layers = round((lg if lg is not None else planned) / m["layer_expert_gib"])
        dB = (i12["B"].get(B) or {}).get("D") if i12 else None
        src = "measured" if dB else "UNIFORM-FALLBACK"
        if not dB:
            dB = uniform_distinct(m["n_experts"], m["top_k"], B) if B > 1 else float(m["top_k"])
        byt = layers * dB * m["expert_bytes"]
        dt = c["tpot"] - ctrl[B]["tpot"]
        sp = ctrl[B]["per_pass"]
        noise = (max(sp) - min(sp)) if len(sp) >= 2 else 0.0
        c.update(bytes=byt, dt=dt, D_src=src, attributable=dt > noise)
        if dt > noise:
            attrib += 1
            xs.append(byt)
            ys.append(dt)
            c["eta"] = eta(byt, dt, h2d)
            etas.append(c["eta"])
    fit = prop_fit(xs, ys)
    form_ok = bool(fit) and fit["r2_0"] >= 0.9 and fit["intercept"] <= 0.5 * (med(ys) or 0)
    em = med(etas)
    if not h2d:                                                    # QA r1: no H2D read -> no eta, never a TypeError
        v = "INSUFFICIENT (no pinned H2D measurement in logs/probe.json)"
    elif attrib < 6:
        v = f"INSUFFICIENT ({attrib} attributable cells < 6)"
    elif em is None:
        v = "INSUFFICIENT (no eta computed)"
    elif not form_ok:
        v = "ROOFLINE-FORM-FAILS"
    else:
        v = "ROOFLINE-HOLDS" if em >= 0.8 else ("ROOFLINE-OPTIMISTIC" if em >= 0.4 else "BELOW-SWEEP")
        if any(e > 1.2 for e in etas):
            v += "+OVERLAP"
    both = (probe or {}).get("h2d_both_ratio")
    return {"cells": {f"{k[0]}x{k[1]}": v2 for k, v2 in cells.items()}, "h2d_gbs": h2d, "eta_median": em, "fit": fit,
            "verdict": v, "flags": flags, "item11_two_gpu_ratio": both,
            "item11_read": None if both is None else ("2 GPUs pull at full rate; an 8-GPU cap is untested" if both >= 0.9
                                                      else "host DMA already shared at 2 GPUs: an 8-GPU cap likely binds")}


def item9_section(run, plan):
    out = {}
    same = opt(rows, f"{run}/rows/item9_same.jsonl") or []
    mets = opt(jload, f"{run}/rows/item9_same_metrics.json") or {}
    for var in [v["name"] for v in plan["item9"]["same"]["variants"]]:
        idle = [r["ttft"] for r in same if r.get("variant") == var and r.get("phase") == "idle" and r.get("ttft") is not None]
        busy = [r["ttft"] for r in same if r.get("variant") == var and r.get("phase") == "busy" and r.get("ttft") is not None]
        n_want = plan["item9"]["same"]["n_live"]
        if len(idle) < n_want or len(busy) < n_want:
            out[f"same_{var}"] = {"status": f"INSUFFICIENT idle {len(idle)} busy {len(busy)} of {n_want}"}
            continue
        p_med, p_hi = med(busy) - med(idle), max(busy) - med(idle)
        mm = mets.get(var, {})
        dpre = None
        if mm.get("m_busy1") and mm.get("m_busy0"):
            dpre = mm["m_busy1"]["vllm:num_preemptions_total"] - mm["m_busy0"]["vllm:num_preemptions_total"]
        out[f"same_{var}"] = {"ttft_idle_med": med(idle), "ttft_busy_med": med(busy), "p_med": p_med, "p_hi": p_hi,
                              "preemptions_during_live": dpre,
                              "verdict": "HOLDS (<= 0.05 s)" if p_hi <= 0.05 else "EXCEEDS 0.05 s: re-run at p_hi"}
    oth = opt(rows, f"{run}/rows/item9_other.jsonl") or []
    for mode in plan["item9"]["other"]["modes"]:
        rr = [r for r in oth if r.get("mode") == mode]
        ps = [r["p_s"] for r in rr if r.get("p_s") is not None and not r.get("err")]
        lost = [sum((b.get("tokens") or 0) for b in r.get("batch", []) if b.get("err")) for r in rr if not r.get("err")]
        if len(ps) < plan["item9"]["other"]["reps"]:
            out[f"other_{mode}"] = {"status": f"INSUFFICIENT {len(ps)} of {plan['item9']['other']['reps']}",
                                    "errs": [r.get("err") for r in rr if r.get("err")]}
            continue
        out[f"other_{mode}"] = {"p_med": med(ps), "p_hi": max(ps), "batch_tokens_lost_med": med(lost),
                                "verdict": "HOLDS (<= 0.8 s)" if max(ps) <= 0.8 else "EXCEEDS 0.8 s: re-run at p_hi"}
    return out


def item8_section(run):
    out = {}
    for pol in ("Tinf", "T10"):
        rr = opt(rows, f"{run}/rows/item8_{pol}.jsonl")
        ms = opt(rows, f"{run}/rows/item8_metrics_{pol}.jsonl")
        meta = opt(jload, f"{run}/rows/item8_{pol}_meta.json")
        if not (rr and ms and meta):
            out[pol] = {"status": "NO-DATA"}
            continue
        ms = [m for m in ms if "err" not in m]
        if not ms:                                                 # QA r1: every metrics read failed (engine died?)
            out[pol] = {"status": "NO-DATA (no readable metrics sample)", "n": len(rr), "engine_alive": meta.get("alive")}
            continue
        first, last = ms[0], ms[-1]
        at_done = min(ms, key=lambda m: abs(m["t"] - meta["t_clients_done"]))
        zombie = last["vllm:generation_tokens_total"] - at_done["vllm:generation_tokens_total"]
        drained = [m["t"] for m in ms if m["t"] >= meta["t_clients_done"]
                   and m["vllm:num_requests_running"] + m["vllm:num_requests_waiting"] == 0]
        e = {"n": len(rr), "status_counts": {}, "max_waiting": max(m["vllm:num_requests_waiting"] for m in ms),
             "preemptions": last["vllm:num_preemptions_total"] - first["vllm:num_preemptions_total"],
             "engine_abort_count": last.get("success_abort", 0.0) - first.get("success_abort", 0.0),
             "zombie_tokens_after_clients": zombie,
             "drain_after_clients_s": (drained[0] - meta["t_clients_done"]) if drained else None,
             "engine_alive": meta.get("alive"), "unfinished_clients": meta.get("unfinished_clients", 0),
             "rows_missing": max(0, meta.get("n", len(rr)) - len(rr))}
        for r in rr:
            k = r["status"] if not str(r["status"]).startswith("err") else "err"
            e["status_counts"][k] = e["status_counts"].get(k, 0) + 1
        if pol == "Tinf":
            sheds = sum(1 for r in rr if str(r["status"]).startswith("err") or (r["status"] == "ok" and not r.get("complete")))
            e["engine_sheds"] = sheds
        out[pol] = e
    ti, t10 = out.get("Tinf", {}), out.get("T10", {})
    v = {}
    if "engine_sheds" in ti:
        v["ENGINE-SHEDS"] = "YES" if ti["engine_sheds"] > 0 else "NO"
        if ti["engine_sheds"] == 0 and ti["rows_missing"] > 0:     # a client with no row is unknown, never 'not shed'
            v["ENGINE-SHEDS"] = f"NO-WITH-{ti['rows_missing']}-UNKNOWN (clients still waiting at the join cap)"
    if "zombie_tokens_after_clients" in t10:
        cancelled = t10["status_counts"].get("cancelled", 0)
        frees = t10["zombie_tokens_after_clients"] == 0 and (t10["drain_after_clients_s"] is not None
                                                              and t10["drain_after_clients_s"] <= 2.0)
        v["CANCEL-FREES"] = ("VACUOUS (0 cancelled: no contention)" if cancelled == 0 else ("YES" if frees else "NO"))
    out["verdict"] = v
    return out


def dcp_section(run, plan):
    rs = opt(rows, f"{run}/rows/dcp.jsonl") or []
    meta = opt(jload, f"{run}/rows/dcp_meta.json") or []
    out = {"backends": {m["dcp"]: m.get("backend") for m in meta}, "cells": {}}
    for c in plan["dcp"]["cells"]:
        t = {}
        for d in (1, 2):
            t[d] = med([med([r["tpot"] for r in rs if r.get("dcp") == d and r.get("cell") == c["name"] and r.get("pass") == p
                             and r.get("tpot") is not None]) for p in range(plan["dcp"]["passes"])])
        out["cells"][c["name"]] = {"tpot_dcp1": t[1], "tpot_dcp2": t[2],
                                   "ratio": (t[2] / t[1]) if (t[1] and t[2]) else None}
    return out


def score(run, plan, allow_stub=False):
    done = open(f"{run}/DONE").read().strip()
    stub = "stub=True" in (opt(lambda p: open(p).read(), f"{run}/logs/drive.log") or "")
    if stub and not allow_stub:
        raise ValueError("STUB run refused (pass --allow-stub for the CPU smoke)")
    probe = opt(jload, f"{run}/logs/probe.json")
    anchor = anchor_tpot(run, plan)
    beta = beta_section(run, plan, anchor)
    i12 = item12_section(run, plan)
    return {"run": run, "DONE": done, "stub": stub, "gates": {k: opt(jload, f"{run}/logs/{k}.json")
                                                              for k in ("g6_serving", "nvlink_gate", "net_gate")},
            "skips": opt(jload, f"{run}/logs/skips.json"), "probe": probe, "anchor": anchor, "beta": beta,
            "beta_verdict": beta_verdicts(beta, run), "lender_penalty": lender_penalty(run), "item12": i12,
            "item10": item10_section(run, plan, probe, i12), "item9": item9_section(run, plan), "item8": item8_section(run),
            "dcp": dcp_section(run, plan)}


def summary_lines(s):
    bv = s["beta_verdict"]
    L = [f"DONE: {s['DONE']}  stub={s['stub']}  skips={[x['arm'] for x in (s['skips'] or [])]}",
         f"VERDICT BETA (PUSH f0.5 BUSY, cells Q+S, upper bound): {bv['deciding']['verdict']} hi={bv['deciding']['hi']} "
         f"lo={bv['deciding']['lo']} win~{bv['deciding']['win_gpus_at_hi']} GPUs | f0.875 {bv['conservative_f0.875']['verdict']} "
         f"| IDLE {bv['idle_f0.5']['verdict']} | PULL {bv['pull_f0.5']['verdict']} | PULLS {bv['pulls_f0.5']['verdict']} "
         f"| H100-proj {bv.get('h100_projection_push')} | {bv['agreement']}",
         f"REPORT lender penalty overall: {s['lender_penalty'].get('overall_penalty')}",
         f"VERDICT ITEM10: {s['item10']['verdict']} eta_med={s['item10']['eta_median']} flags={s['item10']['flags']}",
         f"REPORT ITEM11 (2-GPU slice): {s['item10']['item11_read']} ratio={s['item10']['item11_two_gpu_ratio']}",
         f"VERDICT ITEM12: {s['item12']['verdict']} lambda16={(s['item12']['B'].get(16) or {}).get('lambda')}",
         f"VERDICT ITEM9: " + " | ".join(f"{k}: {v.get('verdict', v.get('status'))} p_hi={v.get('p_hi')}"
                                       for k, v in s["item9"].items()),
         f"VERDICT ITEM8/14: {s['item8'].get('verdict')}",
         f"REPORT DCP: {s['dcp']['cells']} backends={s['dcp']['backends']}"]
    return L


# ----------------------------------------------------------------------------------------------- self-test
def self_test():
    fails, n = [], [0]

    def ck(name, cond, detail=""):
        n[0] += 1
        print(f"  {'PASS' if cond else 'FAIL'} {name}{'' if cond else ' :: ' + str(detail)}")
        if not cond:
            fails.append(name)

    # S1 (hand): bands + interpolation: beta 0.5 -> 144 - 40 x 0.5 = 124; beta 2 -> 104 - 48 x 1 = 56
    ck("S1 bands/win", band(0.5) == "HOLDS" and win_gpus(0.5) == 124.0 and band(2.0) == "SHRINKS" and win_gpus(2.0) == 56.0
       and band(3.5) == "NOT-ESTABLISHED" and str(win_gpus(3.5)).startswith("<=8") and str(win_gpus(-0.2)).startswith(">=144"))
    # S2 ADVERSARIAL: a pass range 0.9..1.1 straddles the HOLDS edge; a mean-based decider would say HOLDS (mean 1.0)
    ck("S2 edge straddle -> UNRESOLVED", edge_verdict(0.9, 1.1) == "UNRESOLVED-AT-EDGE(HOLDS->SHRINKS)", edge_verdict(0.9, 1.1))
    # S3 (hand): beta from rows: local 10 ms, push 13 ms, anchor TPOT 12 ms -> beta_bench 0.3, beta_anch 3/12 = 0.25
    plan = {"anchor": {"cells": [{"name": "Q", "B": 1, "L": 64}], "passes": 1},
            "beta": {"cells": [{"name": "Q", "B": 1, "L": 64}], "modes": ["PUSH"], "f": [0.5], "passes": 1}}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/rows")
        with open(f"{d}/rows/beta_BUSY.jsonl", "w") as f:
            f.write(json.dumps({"cell": "Q", "mode": "LOCAL", "f": 0.0, "pass": 0, "ok": True, "gpu_ms_p50": 10.0}) + "\n")
            f.write(json.dumps({"cell": "Q", "mode": "PUSH", "f": 0.5, "pass": 0, "ok": True, "gpu_ms_p50": 13.0}) + "\n")
        b = beta_section(d, plan, {"Q": {"tpot_s": 0.012}})["BUSY"]["Q"]["PUSH@0.5"]
        ck("S3 beta_bench 0.3 / beta_anch 0.25", abs(b["beta_bench"][0] - 0.3) < 1e-9 and abs(b["beta_anch"][0] - 0.25) < 1e-4, b)
        # S10 ADVERSARIAL: plan says 2 passes, only pass 0 exists -> INSUFFICIENT, never a value from fewer passes
        plan["beta"]["passes"] = 2
        b2 = beta_section(d, plan, {"Q": {"tpot_s": 0.012}})["BUSY"]["Q"]["PUSH@0.5"]
        ck("S10 missing pass -> INSUFFICIENT", str(b2.get("status", "")).startswith("INSUFFICIENT"), b2)
        # S9 ADVERSARIAL: an anchor pass whose rows errored (engine died) is INFRA, not a slow TPOT
        with open(f"{d}/rows/anchor.jsonl", "w") as f:
            f.write(json.dumps({"cell": "Q", "pass": 0, "tpot": None, "complete": False, "err": "err:ConnectionRefused"}) + "\n")
        an = anchor_tpot(d, plan)["Q"]
        ck("S9 dead-engine anchor pass -> INFRA, tpot None", an["tpot_s"] is None and an["infra_passes"] == [0], an)
    # S4 (hand): eta = 13 x 6 x 17,301,504 B = 1,349,517,312 B over (0.0675 s x 25 GB/s = 1.6875 GB) = 0.79971
    e = eta(13 * 6 * 17301504, 0.0675, 25.0)
    ck("S4 eta 0.79971", abs(e - 0.79971) < 1e-5, e)
    # S5 (hand): U(16) for n 64, k 6 = 64 x (1 - 0.90625^16); 0.90625^16 = 0.2069992 -> 50.752
    ck("S5 uniform U(16) = 50.752", abs(uniform_distinct(64, 6, 16) - 50.752) < 1e-3, uniform_distinct(64, 6, 16))
    # S6 (hand): two requests, one step, one MoE layer: {1..6} u {4..9} = 9 distinct; U(2) = 64 x (1 - 0.90625^2) = 11.4375
    reqs = [[[[0] * 6, [1, 2, 3, 4, 5, 6]]], [[[0] * 6, [4, 5, 6, 7, 8, 9]]]]
    D, why, steps = distinct_per_step(reqs, 1, 64, 6)
    ck("S6 distinct 9, U 11.4375", D == 9 and why == "ok" and abs(uniform_distinct(64, 6, 2) - 11.4375) < 1e-9, (D, why))
    # S7 ADVERSARIAL: a routed row with a duplicate id or an id >= n_experts fails I-ROUTE (bad decode / misalignment)
    bad = [[[[0] * 6, [1, 1, 3, 4, 5, 6]]]]
    bad2 = [[[[0] * 6, [1, 2, 3, 4, 5, 64]]]]
    ck("S7 I-ROUTE rejects dup / out-of-range", distinct_per_step(bad, 1, 64, 6)[0] is None
       and distinct_per_step(bad2, 1, 64, 6)[0] is None)
    # S8 (hand): through-origin fit of y = 2x exactly -> slope 2, R^2 1, intercept 0
    fit = prop_fit([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    ck("S8 prop fit", abs(fit["slope0"] - 2) < 1e-12 and abs(fit["r2_0"] - 1) < 1e-12 and abs(fit["intercept"]) < 1e-12, fit)
    # S12 (hand): I-OFF2: control 29.26 GiB, L_off 13 engine 15.85 -> fell 13.41 >= 0.8 x 13.41 = 10.73 -> effective;
    #   ADVERSARIAL: 29.20 (fell 0.06) -> undone; unread memory -> None (never 'effective')
    ck("S12 I-OFF2", offload_effective(29.26, 15.85, 13.41) is True and offload_effective(29.26, 29.20, 13.41) is False
       and offload_effective(None, 15.85, 13.41) is None)
    # S13 ADVERSARIAL (QA r1): no probe.json + 9 attributable offload cells -> INSUFFICIENT, never a TypeError; ITEM8 with
    #   only error metric rows -> NO-DATA, never an IndexError
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/rows")
        p10 = {"models": {"dsv2": {"layer_expert_gib": 1.03125, "n_experts": 64, "top_k": 6, "expert_bytes": 17301504}},
               "item10": {"L_off": [0, 13, 20, 26], "Bs": [1, 4, 16], "passes": 1}}
        with open(f"{d}/rows/offload.jsonl", "w") as f:
            for lo in (0, 13, 20, 26):
                for B in (1, 4, 16):
                    for i in range(B):
                        f.write(json.dumps({"L_off": lo, "B": B, "pass": 0, "complete": True,
                                            "tpot": 0.01 + 0.004 * lo}) + "\n")
        json.dump([{"L_off": lo, "logged_offload_gib": lo * 1.03125, "model_mem_gib": 29.26 - lo * 1.03125}
                   for lo in (0, 13, 20, 26)], open(f"{d}/rows/offload_meta.json", "w"))
        try:
            v10 = item10_section(d, p10, None, None)["verdict"]
        except TypeError as e:
            v10 = f"TypeError {e}"
        with open(f"{d}/rows/item8_Tinf.jsonl", "w") as f:
            f.write(json.dumps({"status": "ok", "complete": True}) + "\n")
        with open(f"{d}/rows/item8_metrics_Tinf.jsonl", "w") as f:
            f.write(json.dumps({"t": 1.0, "err": "URLError"}) + "\n")
        json.dump({"t_clients_done": 1.0, "n": 1}, open(f"{d}/rows/item8_Tinf_meta.json", "w"))
        try:
            v8 = item8_section(d)["Tinf"].get("status")
        except IndexError as e:
            v8 = f"IndexError {e}"
    ck("S13 no H2D -> INSUFFICIENT; all-error metrics -> NO-DATA", str(v10).startswith("INSUFFICIENT (no pinned H2D")
       and str(v8).startswith("NO-DATA"), (v10, v8))
    # S11 (hand): reuse: steps [1..6] then [4..9] on one layer -> 3 of 6 reused = 0.5
    ck("S11 reuse 0.5", reuse_1step([[[0] * 6, [1, 2, 3, 4, 5, 6]], [[0] * 6, [4, 5, 6, 7, 8, 9]]], 1) == 0.5)
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n[0] - len(fails)}/{n[0]}")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--plan")
    ap.add_argument("--out")
    ap.add_argument("--allow-stub", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.run and a.plan and a.out):
        print("USAGE gpu_score.py --run DIR --plan PLAN --out JSON [--allow-stub] | --self-test", file=sys.stderr)
        return 2
    try:
        s = score(a.run, json.load(open(a.plan)), a.allow_stub)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:      # QA r1: any parse failure is loud, exit 1
        print(f"SCORE-FAIL {type(e).__name__}: {e}")
        return 1
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1, default=str)
    for ln in summary_lines(s):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
