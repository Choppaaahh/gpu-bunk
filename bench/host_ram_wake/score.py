#!/usr/bin/env python3
""" Host-RAM wake scorer. Committed WITH the prereg, before any rental. Reads <run>/rows/*.json, prints a JSON verdict.

usage: score.py <run_dir> | --self-test
Rules (prereg section 5, verbatim in code):
  D1 COOL verdict on d8b (Qwen3-8B-FP8): P = nearest-rank p95 of l1[].pagein_to_first_token_s (n=5 -> the max).
     P <= 1.0 s -> COOL-INTERACTIVE ; 1.0 < P <= 5.0 -> COOL-BATCH-ONLY ; P > 5.0 -> COOL-NO-BETTER-THAN-COLD.
     Also COOL-NO-BETTER-THAN-COLD if P >= median warm-cache cold start (launch_to_first_token_s) of the same model.
     Fewer than 3 valid l1 rows -> INSUFFICIENT.
  D2 colocation guard: A's TPOT p95 over requests overlapping any B page-in window / A's baseline TPOT p95.
     ratio <= 1.25 -> NON-DISRUPTIVE ; > 1.25 -> DISRUPTIVE ; fewer than 5 overlapping requests -> UNRESOLVED.
     TTFT ratio reported alongside, not decision-bearing.
  G  host gate: >= 3 burn samples in P0/P1 with SM clock >= 1500 MHz and power >= 200 W, burn >= 150 TFLOPS.
     Gate FAIL -> every verdict is reported as INFRA:<verdict> and is not relayed as a result.
  PCIE flag (not a gate): median 9.44 GB pinned H2D GB/s < 0.6 x theoretical for the recorded gen/width -> PCIE-DEGRADED.
"""
import json, os, statistics, sys

PCIE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}  # per lane, GB/s


def p95(xs):
    xs = sorted(xs)
    if not xs:
        return None
    k = max(0, -(-95 * len(xs) // 100) - 1)  # nearest rank
    return xs[k]


def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 4), "median": round(statistics.median(xs), 4),
            "p95": round(p95(xs), 4), "max": round(max(xs), 4)}


def gate(env):
    ok = 0
    for s in env.get("burn_samples", []):
        f = [x.strip() for x in s.split(",")]
        try:
            pst, clk, pw = f[0], float(f[1].split()[0]), float(f[2].split()[0])
        except Exception:
            continue
        if pst in ("P0", "P1") and clk >= 1500 and pw >= 200:
            ok += 1
    return ok >= 3 and env.get("burn_tflops", 0) >= 150


def pcie_flag(env, copy):
    try:
        gen, width = [int(x) for x in copy["pcie_during_copy"][-1].split(",")]
    except Exception:
        return "PCIE-UNKNOWN"
    rows = [r["GBps"] for r in copy.get("h2d_pinned", []) if abs(r["gb"] - 9.44) < 0.01]
    if not rows:
        return "PCIE-UNKNOWN"
    theo = PCIE_GBPS[gen] * width
    return ("PCIE-DEGRADED" if statistics.median(rows) < 0.6 * theo else "PCIE-OK") + f" gen{gen}x{width} theo={theo:.1f}"


def model_summary(m):
    if not m or "cold" not in m:
        return {"infra": (m or {}).get("infra", "missing")}
    cyc = m.get("cycles", {})
    l1 = [r for r in cyc.get("l1", []) if r.get("pagein_to_first_token_s") is not None]
    return {
        "bytes": m["download"]["bytes"],
        "cold_cache_launch_to_ft": spread([r["launch_to_first_token_s"] for r in m["cold"]]),
        "warm_cache_launch_to_ft": spread([r["launch_to_first_token_s"] for r in m["warm"]]),
        "warm_cache_ready": spread([r["ready_s"] for r in m["warm"]]),
        "awake_ttft": spread(cyc.get("awake_ttft", [])),
        "l1_sleep": spread([r["sleep_s"] for r in l1]),
        "l1_wake": spread([r["wake_s"] for r in l1]),
        "l1_pagein_to_ft": spread([r["pagein_to_first_token_s"] for r in l1]),
        "l1_identical": sum(1 for r in l1 if r.get("identical")),
        "l2_reload_to_ft": spread([r["reload_to_first_token_s"] for r in cyc.get("l2", [])]),
        "l2_identical": sum(1 for r in cyc.get("l2", []) if r.get("identical")),
        "n_l1": len(l1),
    }


def d1(ms):
    if not ms or "l1_pagein_to_ft" not in ms or not ms["l1_pagein_to_ft"] or ms["n_l1"] < 3:
        return "INSUFFICIENT"
    P = ms["l1_pagein_to_ft"]["p95"]
    v = "COOL-INTERACTIVE" if P <= 1.0 else ("COOL-BATCH-ONLY" if P <= 5.0 else "COOL-NO-BETTER-THAN-COLD")
    wc = ms.get("warm_cache_launch_to_ft")
    if wc and P >= wc["median"]:
        v = "COOL-NO-BETTER-THAN-COLD"
    return v


def d2(co):
    if not co or "events" not in co:
        return {"verdict": "UNRESOLVED", "why": "no coloc rows"}
    wins = [(e["wake_start"], e["first_token_at"]) for e in co["events"]]
    ok = [r for r in co["a_rows"] if "tpot" in r]
    base = [r for r in ok if r["t0"] >= co["t_start"] + 5 and r["t_end"] <= co["baseline_end"]]
    over = [r for r in ok if any(r["t0"] <= b and r["t_end"] >= a for a, b in wins)]
    out = {"n_base": len(base), "n_overlap": len(over), "n_err": sum(1 for r in co["a_rows"] if "err" in r),
           "base_tpot": spread([r["tpot"] for r in base]), "overlap_tpot": spread([r["tpot"] for r in over]),
           "base_ttft": spread([r["ttft"] for r in base]), "overlap_ttft": spread([r["ttft"] for r in over]),
           "b_pagein_to_ft_under_load": spread([e["pagein_to_first_token_s"] for e in co["events"]])}
    if len(over) < 5 or len(base) < 5:
        out["verdict"] = "UNRESOLVED"
        return out
    rt = out["overlap_tpot"]["p95"] / out["base_tpot"]["p95"]
    out["tpot_p95_ratio"] = round(rt, 3)
    out["ttft_p95_ratio"] = round(out["overlap_ttft"]["p95"] / out["base_ttft"]["p95"], 3)
    out["verdict"] = "NON-DISRUPTIVE" if rt <= 1.25 else "DISRUPTIVE"
    return out


def score(run):
    R = lambda n: json.load(open(os.path.join(run, "rows", f"{n}.json"))) if os.path.exists(os.path.join(run, "rows", f"{n}.json")) else None
    env, copy = R("env") or {}, R("copy") or {}
    g = gate(env)
    out = {"gate_pass": g, "pcie": pcie_flag(env, copy), "models": {}}
    for k in ("d8b", "small", "moe"):
        out["models"][k] = model_summary(R(f"m_{k}"))
    v1 = d1(out["models"]["d8b"])
    v2 = d2(R("coloc"))
    out["D1"] = v1 if g else f"INFRA:{v1}"
    out["D2"] = v2 if g else dict(v2, verdict=f"INFRA:{v2['verdict']}")
    out["copy_h2d_pinned"] = {gb: spread([r["GBps"] for r in copy.get("h2d_pinned", []) if r["gb"] == gb])
                              for gb in sorted({r["gb"] for r in copy.get("h2d_pinned", [])})}
    return out


def self_test():
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/rows")
    env = {"burn_samples": ["P1, 2700 MHz, 560 W, 5, 16"] * 4, "burn_tflops": 200}
    copy = {"pcie_during_copy": ["4, 16"], "h2d_pinned": [{"gb": 9.44, "GBps": 24.0}] * 5}
    mk = lambda ps, wc: {"download": {"bytes": 9}, "cold": [{"launch_to_first_token_s": 90}] * 3,
                         "warm": [{"launch_to_first_token_s": wc, "ready_s": wc}] * 3,
                         "cycles": {"awake_ttft": [0.02] * 5, "l1": [{"sleep_s": 1, "wake_s": p, "pagein_to_first_token_s": p, "identical": True} for p in ps], "l2": []}}
    json.dump(env, open(f"{d}/rows/env.json", "w")); json.dump(copy, open(f"{d}/rows/copy.json", "w"))
    cases = [([0.4, 0.5, 0.6, 0.7, 0.9], 30, "COOL-INTERACTIVE"), ([0.4, 0.5, 0.6, 0.7, 1.2], 30, "COOL-BATCH-ONLY"),
             ([6, 6, 6, 6, 6], 30, "COOL-NO-BETTER-THAN-COLD"), ([2, 2, 2, 2, 2], 1.5, "COOL-NO-BETTER-THAN-COLD"),
             ([0.5, 0.5], 30, "INSUFFICIENT")]
    for ps, wc, want in cases:
        json.dump(mk(ps, wc), open(f"{d}/rows/m_d8b.json", "w"))
        got = score(d)["D1"]
        assert got == want, (ps, wc, got, want)
    co = {"t_start": 0, "baseline_end": 30, "events": [{"wake_start": 40, "first_token_at": 41, "pagein_to_first_token_s": 1}],
          "a_rows": [{"t0": 6 + i, "t_end": 7 + i, "ttft": 0.05, "tpot": 0.010} for i in range(10)] +
                    [{"t0": 39.5 + i * 0.1, "t_end": 40.5, "ttft": 0.05, "tpot": t} for i, t in enumerate([0.011, 0.012, 0.012, 0.011, 0.012])]}
    json.dump(co, open(f"{d}/rows/coloc.json", "w"))
    assert score(d)["D2"]["verdict"] == "NON-DISRUPTIVE", score(d)["D2"]
    co["a_rows"][-1]["tpot"] = 0.030
    json.dump(co, open(f"{d}/rows/coloc.json", "w"))
    assert score(d)["D2"]["verdict"] == "DISRUPTIVE"
    env["burn_tflops"] = 50
    json.dump(env, open(f"{d}/rows/env.json", "w"))
    assert score(d)["D1"].startswith("INFRA:")
    assert pcie_flag({}, {"pcie_during_copy": ["4, 16"], "h2d_pinned": [{"gb": 9.44, "GBps": 10.0}]}).startswith("PCIE-DEGRADED")
    print("SELF-TEST PASS (8 goldens)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        print(json.dumps(score(sys.argv[1]), indent=1))
