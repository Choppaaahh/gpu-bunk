#!/usr/bin/env python3
""" gpu-bunk: real multi-model traffic from a public trace (Chutes, "A Year in LLM Serving", arXiv 2608.13573,
CC BY 4.0). Report-only, $0, CPU. Answers: is there a long tail of low-demand models, and do models take turns
(anti-correlated demand at intra-day scale), or is traffic a few hot models with co-phased peaks?

Modes
  --fetch   (needs pyarrow; run it with a venv python) streams ONLY the row groups of the public 91 GB parquet that
            overlap [--start, --start + --days) over anonymous S3 range reads, reads 4 columns (chute_id, started_at,
            it, ot), and WRITES per-(minute, model) bins to <out-dir>/bins_<tag>.csv.gz plus fetch_<tag>.json (bytes,
            row groups, rows, ETag). No raw rows are written except the first 200 in-window rows to <out-dir>/sample_<tag>.csv.
  --analyze reads bins + raw/chute_models.csv, WRITES <out>.json (and prints a short summary). Pure stdlib + numpy.
  --export-v2  WRITES a demand_summary.json in the shape scripts/placement_v2.py --plan --demand reads
            (placement_set rows: slug, params_total_b, params_active_b, share, ...). v2 regenerates SYNTHETIC
            arrivals from the shares: only the share vector and the sizes cross over, never the real arrival times.
  --self-test goldens on a fixture shaped from real downloaded rows (see FIXTURE below).

Cost model (CONDITIONAL, stated, never measured on these models):
  work = input + output tokens per request (v1 sim convention, MEAN_WORK). Card capacity for a model with A active
  params (B) = C_REF * ACTIVE_REF_B / A, C_REF = 18,777 work-tok/s = Qwen2.5-3B fp8 on one RTX 5090 (Stage B
  post-calibration, (private report)). Cards needed by a model =
  its p95 5-min work rate / its capacity. Models whose size cannot be parsed from the name are reported under the
  three uniform scenarios (3B / 8B / 32B active) and excluded from the parsed-size column.

Output contract: exit 0 = success; 1 = self-test FAIL; 2 = bad input. Every mode that writes says so above.
Top pitfalls (observed while building): the trace's timestamps start at 1970-01-01 (normalized), so hour-of-day is
  trace-relative, not a wall clock; chute_models.csv maps 4,558 of the trace's 9,174 chute_ids (unmapped ids keep
  their uuid and count in every share); pyarrow is not on the system python (use the scratch venv for --fetch only).
"""
import argparse, csv, gzip, hashlib, json, math, os, re, sys, time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S3_BUCKET_KEY = "harvardsys-datasets/2026_chutes_anonymized/chutes_trace.parquet"
PUBLISHED = {"bytes": 91044147663, "rows": 6122413756,
             "sha256_full_file": "74833c1469577c5d1f1ef9d7393d165dab31d29950f1d76fe906d5726594e9ec"}
C_REF = 18777.01470132433      # Stage B post-calibration C_long_x qwen (work tok/s per 5090, fp8, 3B)
ACTIVE_REF_B = 3.1
SCENARIOS_B = (3.0, 8.0, 32.0)
TOPK_CORR = 30


# ----------------------------------------------------------------------------------------------------- fetch
def fetch(start, days, out_dir, tag, sample_rows=200, max_bytes=5.0e9):
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    import datetime as dt
    os.makedirs(out_dir, exist_ok=True)
    t0 = dt.datetime.fromisoformat(start)
    t1 = t0 + dt.timedelta(days=days)
    origin = dt.datetime(1970, 1, 1)
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    info = s3.get_file_info(S3_BUCKET_KEY)
    f = s3.open_input_file(S3_BUCKET_KEY)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    idx = {n: i for i, n in enumerate(names)}
    cols = ["chute_id", "started_at", "it", "ot"]
    ts_type = pf.schema_arrow.field("started_at").type
    if str(ts_type) != "timestamp[us]":                 # review: minute binning assumes microseconds
        print(f"refuse: started_at is {ts_type}, expected timestamp[us]", file=sys.stderr)
        return 2
    groups, est = [], 0
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(idx["started_at"]).statistics
        if st is None or not st.has_min_max:          # review: no stats -> read it (conservative)
            lo, hi = t0, t0
        else:
            lo, hi = st.min, st.max
        if hi >= t0 and lo < t1:
            groups.append(g)
            est += sum(md.row_group(g).column(idx[c]).total_compressed_size for c in cols)
    if est > max_bytes:
        print(f"refuse: {est/1e9:.2f} GB > cap {max_bytes/1e9:.2f} GB", file=sys.stderr)
        return 2
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    rows_read = rows_kept = 0
    sample = []
    for k, g in enumerate(groups):
        tb = pf.read_row_group(g, columns=cols)
        rows_read += tb.num_rows
        ts = tb.column("started_at")
        mask = pc.and_(pc.greater_equal(ts, pa.scalar(t0, pa.timestamp("us"))),
                       pc.less(ts, pa.scalar(t1, pa.timestamp("us"))))
        tb = tb.filter(mask)
        rows_kept += tb.num_rows
        if len(sample) < sample_rows and tb.num_rows:
            sl = tb.slice(0, sample_rows - len(sample)).to_pylist()
            sample += [{"chute_id": r["chute_id"], "started_at": r["started_at"].isoformat(),
                        "it": r["it"], "ot": r["ot"]} for r in sl]
        us = pc.cast(tb.column("started_at"), pa.int64())
        minute = pc.divide(us, pa.scalar(60_000_000, pa.int64()))   # integer division on int64
        it = pc.fill_null(tb.column("it"), 0.0)
        ot = pc.fill_null(tb.column("ot"), 0.0)
        t2 = pa.table({"m": minute, "c": tb.column("chute_id"), "it": it, "ot": ot})
        agg = t2.group_by(["m", "c"]).aggregate([("it", "count"), ("it", "sum"), ("ot", "sum")])
        for m, c, n, si, so in zip(*(agg.column(x).to_pylist() for x in ("m", "c", "it_count", "it_sum", "ot_sum"))):
            b = bins[(m, c)]
            b[0] += n; b[1] += si or 0.0; b[2] += so or 0.0
        if k % 10 == 0:
            print(f"rg {k+1}/{len(groups)} rows_kept={rows_kept}", file=sys.stderr, flush=True)
    bpath = os.path.join(out_dir, f"bins_{tag}.csv.gz")
    with gzip.open(bpath, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["minute", "chute_id", "n", "it", "ot"])
        for (m, c), (n, si, so) in sorted(bins.items(), key=lambda z: (z[0][0], str(z[0][1]))):
            w.writerow([m, c, n, round(si, 1), round(so, 1)])
    spath = os.path.join(out_dir, f"sample_{tag}.csv")
    with open(spath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["chute_id", "started_at", "it", "ot"])
        w.writeheader(); w.writerows(sample)
    man = {"source": "s3://" + S3_BUCKET_KEY, "https": "https://harvardsys-datasets.s3.us-east-1.amazonaws.com/"
           "2026_chutes_anonymized/chutes_trace.parquet", "license": "CC BY 4.0 (repo README, HarvardMadSys/chutes_workload)",
           "window_trace_time": [t0.isoformat(), t1.isoformat()], "days": days,
           "remote_size": info.size, "published_bytes": PUBLISHED["bytes"], "size_matches_published": info.size == PUBLISHED["bytes"],
           "footer_rows": md.num_rows, "rows_match_published": md.num_rows == PUBLISHED["rows"],
           "row_groups_total": md.num_row_groups, "row_groups_read": [groups[0], groups[-1]] if groups else None,
           "n_row_groups_read": len(groups), "compressed_bytes_read_est": est, "columns": cols,
           "rows_read": rows_read, "rows_kept": rows_kept, "bins": len(bins),
           "sha256_bins": sha256_file(bpath), "sha256_sample": sha256_file(spath),
           "full_file_sha_note": "the published sha256 covers the whole 91 GB file; a range read cannot be checked "
                                 "against it. Size + footer row count are checked instead.",
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(os.path.join(out_dir, f"fetch_{tag}.json"), "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("rows_kept", "bins", "compressed_bytes_read_est", "size_matches_published",
                                          "rows_match_published")}))
    return 0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------- sizes
SIZE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
MOE_RE = re.compile(r"(\d+(?:\.\d+)?)[bB][-_ ]?[aA](\d+(?:\.\d+)?)[bB]")
KNOWN_HUGE = ("deepseek-v3", "deepseek-r1", "deepseek-v3.1", "deepseek-v3.2", "kimi-k2", "glm-4.5", "glm-4.6", "glm-4.7",
              "glm-5", "minimax-m2", "qwen3-coder-480b", "qwen3-235b", "hermes-4-405b", "longcat")


def parse_size(name):
    """(total_B, active_B, src) from a model name, or (None, None, reason). MoE 'NNNB-AxB' -> active x."""
    n = (name or "").lower()
    m = MOE_RE.search(n)
    if m:
        return float(m.group(1)), float(m.group(2)), "moe-from-name"
    sizes = [float(x) for x in SIZE_RE.findall(n)]
    sizes = [s for s in sizes if 0.1 <= s <= 2000]
    if sizes:
        s = max(sizes)
        return s, s, "dense-from-name"
    if any(k in n for k in KNOWN_HUGE):
        return None, None, "frontier-moe-size-not-in-name"
    return None, None, "unparsed"


# ----------------------------------------------------------------------------------------------------- analysis
def load_bins(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = []
    with op(path, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["minute"]), r["chute_id"], int(r["n"]), float(r["it"]), float(r["ot"])))
    return rows


def load_names(path):
    out = {}
    if path and os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                out[r["chute_id"]] = r["name"]
    return out


def series(rows, width_min):
    """{model: {bin: [n, work, out]}} and bin range."""
    import numpy as np
    m0 = min(r[0] for r in rows); m1 = max(r[0] for r in rows)
    m0 -= m0 % 1440          # review: day-aligned origin so bin-of-day is exact for every width
    nb = (m1 - m0) // width_min + 1
    s = {}
    for m, c, n, it, ot in rows:
        a = s.get(c)
        if a is None:
            a = s[c] = np.zeros((3, nb))
        b = (m - m0) // width_min
        a[0, b] += n; a[1, b] += it + ot; a[2, b] += ot
    return s, m0, nb


def cv(x):
    import numpy as np
    mu = float(np.mean(x))
    return None if mu == 0 else float(np.std(x) / mu)


def pearson(a, b):
    import numpy as np
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0 or sb == 0:
        return None
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def rolling_mean(x, k):
    import numpy as np
    if k <= 1:
        return x.copy()
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x)
    h = k // 2
    for i in range(len(x)):
        lo, hi = max(0, i - h), min(len(x), i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def analyze(rows, names, topk=TOPK_CORR):
    import numpy as np
    s1, m0, nb1 = series(rows, 1)
    s5, _, nb5 = series(rows, 5)
    models = sorted(s1)
    tot_req = sum(float(s1[c][0].sum()) for c in models)
    tot_work = sum(float(s1[c][1].sum()) for c in models)
    days = nb1 / 1440.0
    per = {}
    for c in models:
        a1, a5 = s1[c], s5[c]
        name = names.get(c, c)
        tb, ab, src = parse_size(name)
        rate5 = a5[1] / 300.0                       # work tok/s per 5-min bin
        p95 = float(np.percentile(rate5, 95)); mean = float(rate5.mean()); peak = float(rate5.max())
        cards = {f"{int(b)}B": p95 / (C_REF * ACTIVE_REF_B / b) for b in SCENARIOS_B}
        cards_parsed = None if ab is None else p95 / (C_REF * ACTIVE_REF_B / ab)
        per[c] = {"name": name, "req": int(a1[0].sum()), "work_tok": float(a1[1].sum()), "out_tok": float(a1[2].sum()),
                  "req_share": float(a1[0].sum()) / tot_req, "work_share": float(a1[1].sum()) / tot_work,
                  "active_bins_1m": int((a1[0] > 0).sum()), "cv_1m": cv(a1[0]), "cv_5m": cv(a5[0]),
                  "work_rate_mean": mean, "work_rate_p95_5m": p95, "work_rate_peak_5m": peak,
                  "params_total_b": tb, "params_active_b": ab, "size_src": src,
                  "cards_p95_scen": cards, "cards_p95_parsed": cards_parsed}
    order = sorted(models, key=lambda c: -per[c]["work_tok"])
    # GPU-time hotness proxy: work / capacity, parsed sizes only (CONDITIONAL); unparsed -> scenario 32B for big names?
    gpu = {c: per[c]["work_tok"] * per[c]["params_active_b"] / ACTIVE_REF_B / C_REF for c in models
           if per[c]["params_active_b"] is not None}
    gsum = sum(gpu.values()) or 1.0
    for c in gpu:
        per[c]["gpu_share_parsed"] = gpu[c] / gsum
    # concentration
    ws = sorted((per[c]["work_share"] for c in models), reverse=True)
    rs = sorted((per[c]["req_share"] for c in models), reverse=True)
    conc = {f"top{k}_work": float(sum(ws[:k])) for k in (1, 5, 10, 20, 50, 100)}
    conc.update({f"top{k}_req": float(sum(rs[:k])) for k in (1, 5, 10, 20, 50, 100)})
    # long tail
    lt = {}
    for key in [f"{int(b)}B" for b in SCENARIOS_B]:
        below = [c for c in models if per[c]["cards_p95_scen"][key] < 1.0]
        tiny = [c for c in models if per[c]["cards_p95_scen"][key] < 0.1]
        bB = float(key[:-1])
        pooled = sum(s5[c][1] for c in below) / 300.0 if below else None
        cv_t = sorted(per[c]["cv_5m"] for c in below if per[c]["cv_5m"] is not None)
        peak_below = [c for c in models if per[c]["work_rate_peak_5m"] < C_REF * ACTIVE_REF_B / bB]
        act1 = sum((s1[c][0] > 0).astype(np.int32) for c in below) if below else None
        act5 = sum((s5[c][0] > 0).astype(np.int32) for c in below) if below else None
        lt_extra = {"models_PEAK_below_1_card": len(peak_below),
                    "work_share_PEAK_below_1_card": float(sum(per[c]["work_share"] for c in peak_below)),
                    "concurrent_active_tail_1m": None if act1 is None else
                        {"mean": float(act1.mean()), "p50": float(np.percentile(act1, 50)), "p95": float(np.percentile(act1, 95)), "max": int(act1.max())},
                    "concurrent_active_tail_5m": None if act5 is None else
                        {"mean": float(act5.mean()), "p50": float(np.percentile(act5, 50)), "p95": float(np.percentile(act5, 95)), "max": int(act5.max())},
                    "pooled_p95_cards_below": None if pooled is None else float(np.percentile(pooled, 95)) / (C_REF * ACTIVE_REF_B / bB),
                    "median_cv_5m_below": cv_t[len(cv_t) // 2] if cv_t else None}
        lt[key] = {"models_below_1_card": len(below), "frac_models_below_1_card": len(below) / len(models),
                   "work_share_below_1_card": float(sum(per[c]["work_share"] for c in below)),
                   "models_below_0.1_card": len(tiny), "work_share_below_0.1_card": float(sum(per[c]["work_share"] for c in tiny)),
                   "sum_cards_p95_below": float(sum(per[c]["cards_p95_scen"][key] for c in below)),
                   "sum_cards_p95_all": float(sum(per[c]["cards_p95_scen"][key] for c in models)), **lt_extra}
    parsed = [c for c in models if per[c]["cards_p95_parsed"] is not None]
    below_p = [c for c in parsed if per[c]["cards_p95_parsed"] < 1.0]
    lt["parsed_size"] = {"n_models_parsed": len(parsed), "work_share_parsed": float(sum(per[c]["work_share"] for c in parsed)),
                         "models_below_1_card": len(below_p),
                         "frac_parsed_below_1_card": (len(below_p) / len(parsed)) if parsed else None,
                         "work_share_below_1_card": float(sum(per[c]["work_share"] for c in below_p)),
                         "sum_cards_p95_below": float(sum(per[c]["cards_p95_parsed"] for c in below_p)),
                         "n_below_ceil_cards_dedicated": len(below_p),
                         "cards_if_pooled_ceil": int(math.ceil(sum(per[c]["cards_p95_parsed"] for c in below_p))),
                         "pooled_p95_cards_below_GPU_weighted": (float(np.percentile(sum(s5[c][1] * per[c]["params_active_b"] / ACTIVE_REF_B
                                                                                      for c in below_p) / 300.0, 95)) / C_REF) if below_p else None}
    # correlation at 5-min (top-k by work, active >= 50% of 5-min bins)
    top = [c for c in order if (s5[c][0] > 0).mean() >= 0.5][:topk]
    hr = 12   # 12 x 5 min = 1 h rolling mean for the intra-day deviation series
    raw, dev, ddev, bnd = [], [], [], []
    resid = {}
    day_bins = 288
    band = {}
    for c in top:
        x = s5[c][1]
        r1h = rolling_mean(x, hr)
        resid[c] = x - r1h
        band[c] = r1h - rolling_mean(x, 24 * hr)      # hours-scale band: 1-h mean minus 24-h mean (intra-day)
    # daily-profile removal: subtract each model's mean profile by bin-of-day (diurnal), keep the rest
    prof_resid = {}
    for c in top:
        x = s5[c][1]
        bod = (np.arange(nb5) + (m0 // 5)) % day_bins
        prof = np.array([x[bod == k].mean() if (bod == k).any() else 0.0 for k in range(day_bins)])
        prof_resid[c] = x - prof[bod]
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            r0 = pearson(s5[a][1], s5[b][1]); r2 = pearson(prof_resid[a], prof_resid[b])
            # rolling windows are truncated at the series ends; trim half a window so edge artifacts do not set the r
            r1 = pearson(resid[a][hr // 2:-(hr // 2)], resid[b][hr // 2:-(hr // 2)])
            tb = 12 * hr
            r3 = pearson(band[a][tb:-tb], band[b][tb:-tb]) if len(band[a]) > 2 * tb + 2 else None
            pairs.append({"a": per[a]["name"], "b": per[b]["name"], "r_raw": r0, "r_minus_1h_mean": r1,
                          "r_minus_daily_profile": r2, "r_band_1h_24h": r3})
            for lst, v in ((raw, r0), (dev, r1), (ddev, r2), (bnd, r3)):
                if v is not None:
                    lst.append(v)

    def dist(v):
        if not v:
            return None
        v = sorted(v)
        q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
        return {"n": len(v), "mean": float(np.mean(v)), "median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": v[0], "max": v[-1],
                "frac_below_-0.2": sum(1 for x in v if x < -0.2) / len(v), "frac_below_-0.5": sum(1 for x in v if x < -0.5) / len(v)}
    # diurnal phase: hour (trace-relative) of each top model's mean daily peak
    phase = {}
    for c in top:
        x = s5[c][1]
        bod = (np.arange(nb5) + (m0 // 5)) % day_bins
        prof = np.array([x[bod == k].mean() if (bod == k).any() else 0.0 for k in range(day_bins)])
        ph = prof.reshape(24, 12).mean(1)
        phase[per[c]["name"]] = {"peak_hour": int(ph.argmax()), "trough_hour": int(ph.argmin()),
                                 "peak_to_trough": float(ph.max() / ph.min()) if ph.min() > 0 else None}
    tot = sum(s5[c][1] for c in models)
    bod = (np.arange(nb5) + (m0 // 5)) % day_bins
    tprof = np.array([tot[bod == k].mean() if (bod == k).any() else 0.0 for k in range(day_bins)]).reshape(24, 12).mean(1)  # QA(glm) guard
    plat_peak = int(tprof.argmax())
    near = sum(1 for v in phase.values() if min(abs(v["peak_hour"] - plat_peak), 24 - abs(v["peak_hour"] - plat_peak)) <= 3)
    sums = {"n_models": len(models), "days": days, "minutes": nb1, "total_requests": int(tot_req), "total_work_tok": tot_work,
            "concentration": conc, "long_tail": lt,
            "burstiness_cv_5m_top": {per[c]["name"]: per[c]["cv_5m"] for c in order[:15]},
            "burstiness_cv_1m_top": {per[c]["name"]: per[c]["cv_1m"] for c in order[:15]},
            "corr_top_models": [per[c]["name"] for c in top],
            "corr_raw_5m": dist(raw), "corr_minus_1h_mean_5m": dist(dev), "corr_minus_daily_profile_5m": dist(ddev),
            "corr_band_1h_24h_5m": dist(bnd),
            "most_negative_pairs_band": sorted([p for p in pairs if p["r_band_1h_24h"] is not None],
                                               key=lambda p: p["r_band_1h_24h"])[:8],
            "most_negative_pairs_raw": sorted([p for p in pairs if p["r_raw"] is not None], key=lambda p: p["r_raw"])[:8],
            "most_negative_pairs_minus_daily": sorted([p for p in pairs if p["r_minus_daily_profile"] is not None],
                                                      key=lambda p: p["r_minus_daily_profile"])[:8],
            "platform_peak_hour_trace_relative": plat_peak,
            "platform_peak_to_trough": float(tprof.max() / tprof.min()) if tprof.min() > 0 else None,
            "top_models_peak_within_3h_of_platform_peak": f"{near}/{len(phase)}", "phase": phase,
            "cost_model": f"cards = p95 5-min work rate / (C_REF {C_REF:.0f} * {ACTIVE_REF_B} / active_B); CONDITIONAL"}
    return {"summary": sums, "per_model": {c: per[c] for c in order}}


def export_v2(an, out_path, k=5, max_total_b=40.0, window=None):
    """5 hottest models (by work) whose size parses and total <= max_total_b (v2's --plan enumerates 4 and 8 cards over
    top4 / all5, so the set must stay at 5). share = work share within the set (v2 reads 'share')."""
    pm = an["per_model"]
    cand = [c for c in pm if pm[c]["params_total_b"] is not None and pm[c]["params_total_b"] <= max_total_b]
    cand = cand[:k]
    ssum = sum(pm[c]["work_tok"] for c in cand) or 1.0
    rows = [{"slug": pm[c]["name"], "chute_id": c, "params_total_b": pm[c]["params_total_b"],
             "params_active_b": pm[c]["params_active_b"], "size_src": pm[c]["size_src"],
             "share": pm[c]["work_tok"] / ssum, "share_of_trace_work": pm[c]["work_share"],
             "cards_p95_parsed": pm[c]["cards_p95_parsed"]} for c in cand]
    d = {"source": "Chutes trace (arXiv 2608.13573, CC BY 4.0) via scripts/real_trace_analysis.py --export-v2",
         "window": window, "share_unit": "input+output tokens (work), within the 5-model set", "placement_set": rows,
         "note": "v2 --plan regenerates synthetic arrivals (IND / DIURNAL) from these shares; real arrival times do not cross over"}
    json.dump(d, open(out_path, "w"), indent=1)
    return d


# ----------------------------------------------------------------------------------------------------- ServeGen
def servegen(dirpath):
    """ServeGen (alibaba/ServeGen, Apache-2.0) per-client 600-s windows: col0 = window start (s from 00:00 day 0),
    col1 = rate (req/s). Per model: sum of client rates per window (the release holds a SAMPLE of clients, so the sum
    is a lower bound on the model's rate). Returns per-model series stats + pairwise correlation."""
    import numpy as np
    per = defaultdict(dict)
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith("-trace.csv"):
            continue
        model = fn.split("__chunk-")[0]
        with open(os.path.join(dirpath, fn)) as fh:
            for line in fh:
                p = line.strip().split(",")
                if len(p) < 2 or not p[0]:
                    continue
                t, r = int(float(p[0])), float(p[1] or 0.0)
                per[model][t] = per[model].get(t, 0.0) + r
    ts = sorted(set(t for m in per for t in per[m]))
    X = {m: np.array([per[m].get(t, 0.0) for t in ts]) for m in per}
    bod = np.array([(t // 600) % 144 for t in ts])
    out = {"n_windows": len(ts), "window_s": 600, "days": len(ts) * 600 / 86400.0, "models": {}}
    prof_res = {}
    for m, x in X.items():
        prof = np.array([x[bod == k].mean() for k in range(144)])
        prof_res[m] = x - prof[bod]
        hr = prof.reshape(24, 6).mean(1)
        out["models"][m] = {"mean_rate_rps": float(x.mean()), "p95_rate_rps": float(np.percentile(x, 95)),
                            "cv_10m": cv(x), "peak_hour_local": int(hr.argmax()), "trough_hour_local": int(hr.argmin()),
                            "peak_to_trough": float(hr.max() / hr.min()) if hr.min() > 0 else None}
    tot = sum(x.mean() for x in X.values())
    for m in X:
        out["models"][m]["req_share"] = float(X[m].mean() / tot)
    ms = sorted(X)
    out["pairs"] = [{"a": a, "b": b, "r_raw": pearson(X[a], X[b]), "r_minus_daily_profile": pearson(prof_res[a], prof_res[b])}
                    for i, a in enumerate(ms) for b in ms[i + 1:]]
    return out


# ----------------------------------------------------------------------------------------------------- self-test
# FIXTURE: chute_ids, names and per-request token counts below are copied from real rows of
# (private report) (Chutes, trace time 1970-10-28T00:00:00.02-.04):
#   0d7184a2-... (Qwen/Qwen3-32B) it 3072 ot 232 ; 722df757-... (deepseek-ai/DeepSeek-V3.2-Exp) it 3787 ot 13.
# Only the arrival grid is re-laid on 2 days so every golden is hand-computable.
def self_test():
    import numpy as np
    ok = True

    def chk(name, got, want, tol=1e-9):
        nonlocal ok
        good = (got is None and want is None) or (got is not None and want is not None and abs(got - want) <= tol)
        print(("PASS " if good else "FAIL ") + f"{name}: got {got} want {want}")
        ok &= good
    A, B = "0d7184a2-32a3-53e0-9607-058c37edaab5", "722df757-203b-58df-b54b-22130fd1fc53"
    names = {A: "Qwen/Qwen3-32B", B: "deepseek-ai/DeepSeek-V3.2-Exp"}
    rows = []
    # A: 2 req every minute for 2 days (it 3072 ot 232 each -> work 3304/req)
    # B: 1 req/min only in the first 12 h of each day (it 3787 ot 13 -> work 3800/req)
    for m in range(2880):
        rows.append((m, A, 2, 2 * 3072.0, 2 * 232.0))
        if (m % 1440) < 720:
            rows.append((m, B, 1, 3787.0, 13.0))
    an = analyze(rows, names, topk=2)
    s, pm = an["summary"], an["per_model"]
    chk("n_models", s["n_models"], 2)
    chk("total_requests", s["total_requests"], 2 * 2880 + 1440)
    chk("A req_share", pm[A]["req_share"], 5760 / 7200)
    chk("A work_share", pm[A]["work_share"], 5760 * 3304 / (5760 * 3304 + 1440 * 3800))
    chk("A cv_1m (constant)", pm[A]["cv_1m"], 0.0)
    chk("B cv_1m (on half)", pm[B]["cv_1m"], 1.0, 1e-12)          # 0/1 half-half: sd/mean = 0.5/0.5
    chk("A p95 work rate", pm[A]["work_rate_p95_5m"], 10 * 3304 / 300.0, 1e-9)
    chk("A cards 32B scen", pm[A]["cards_p95_scen"]["32B"], (33040 / 300) / (C_REF * ACTIVE_REF_B / 32), 1e-12)
    chk("A cards parsed (32B dense)", pm[A]["cards_p95_parsed"], (33040 / 300) / (C_REF * ACTIVE_REF_B / 32), 1e-12)
    chk("B size unparsed (frontier MoE)", pm[B]["params_active_b"], None)
    chk("A size parsed 32B", pm[A]["params_active_b"], 32.0)
    chk("corr raw undefined (A constant)", s["corr_raw_5m"], None)
    # anti-phase pair: C on in first half day, D in second half -> r_raw = -1
    rows2 = [(m, "c", 1, 1.0, 0.0) for m in range(2880) if (m % 1440) < 720] + \
            [(m, "d", 1, 1.0, 0.0) for m in range(2880) if (m % 1440) >= 720]
    an2 = analyze(rows2, {"c": "x/c-8B", "d": "x/d-8B"}, topk=2)
    chk("anti-phase r_raw", an2["summary"]["corr_raw_5m"]["median"], -1.0, 1e-9)
    chk("anti-phase band r < -0.9 (hours-scale take-turns visible)", float(an2["summary"]["corr_band_1h_24h_5m"]["median"] < -0.9), 1.0)
    # co-phased pair with independent sub-hour noise: band r must be > 0.9 (diurnal co-phase), 1h-residual r near 0
    rng = np.random.default_rng(7)
    base = [1 + (m % 1440 < 720) for m in range(2880)]
    rows3 = []
    for nm in ("e", "f"):
        for m in range(2880):
            n = int(base[m] * 10 + rng.integers(0, 5))
            rows3.append((m, nm, n, float(n), 0.0))     # work = 1 token per request
    an3 = analyze(rows3, {"e": "x/e-8B", "f": "x/f-8B"}, topk=2)["summary"]
    chk("co-phase band r > 0.9", float(an3["corr_band_1h_24h_5m"]["median"] > 0.9), 1.0)
    chk("pooled tail p95 3B (anti-phase c+d = flat 5 req/5min*1 tok)", an2["summary"]["long_tail"]["3B"]["pooled_p95_cards_below"],
        (5 / 300.0) / (C_REF * ACTIVE_REF_B / 3.0), 1e-15)
    # M3: rolling mean at the edge divides by the window actually covered
    chk("rolling_mean edge", float(rolling_mean(np.array([0.0, 0.0, 3.0]), 3)[2]), 1.5)
    # M2: opposite slow trends, no intra-day structure -> raw r ~ -1 but band r must NOT read as take-turns
    lv = np.random.default_rng(3).uniform(0, 500, size=(2, 72))      # independent random level per hour per model
    rows4 = [(m, "g", 1, 1.0 + m + lv[0][m // 60], 0.0) for m in range(4320)] + \
            [(m, "h", 1, 1.0 + 4320 - m + lv[1][m // 60], 0.0) for m in range(4320)]
    s4 = analyze(rows4, {"g": "x/g-8B", "h": "x/h-8B"}, topk=2)["summary"]
    chk("trend pair raw r < -0.9", float(s4["corr_raw_5m"]["median"] < -0.9), 1.0)
    chk("trend pair band r > -0.5 (drift is not take-turns)", float(s4["corr_band_1h_24h_5m"]["median"] > -0.5), 1.0)
    # M4: a model whose 5-min peak (200k tok/min -> 3333 tok/s) sits between the 32B (1819) and 3B (19403) capacities
    s5_ = analyze([(m, "k", 1, 200000.0, 0.0) for m in range(1440)], {"k": "x/k"}, topk=1)["summary"]["long_tail"]
    chk("peak-below 3B counts it", s5_["3B"]["models_PEAK_below_1_card"], 1)
    chk("peak-below 32B excludes it", s5_["32B"]["models_PEAK_below_1_card"], 0)
    chk("concurrent tail 1m (anti-phase: exactly 1 active)", an2["summary"]["long_tail"]["3B"]["concurrent_active_tail_1m"]["max"], 1)
    chk("peak-below count", an2["summary"]["long_tail"]["3B"]["models_PEAK_below_1_card"], 2)
    chk("long tail 3B: both below 1 card", an2["summary"]["long_tail"]["3B"]["models_below_1_card"], 2)
    # parse_size adversarial: MoE + date-like digits
    chk("moe parse active", parse_size("Qwen/Qwen3-30B-A3B-Instruct-2507")[1], 3.0)
    chk("no size for deepseek", parse_size("deepseek-ai/DeepSeek-V3.2-TEE")[0], None)
    chk("2407 not a size", parse_size("unsloth/Mistral-Nemo-Instruct-2407")[0], None)
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fetch", action="store_true"); ap.add_argument("--start"); ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tag", default="w1"); ap.add_argument("--out-dir", default="data/traces/derived")
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--bins"); ap.add_argument("--names",
                    default="data/traces/raw/chute_models.csv")
    ap.add_argument("--out"); ap.add_argument("--export-v2"); ap.add_argument("--window")
    ap.add_argument("--servegen", help="dir of ServeGen *-trace.csv (WRITES --out json)")
    a = ap.parse_args()
    for k in ("out_dir", "bins", "names", "out", "export_v2", "servegen"):   # review: caller-relative paths
        v = getattr(a, k)
        if v:
            setattr(a, k, os.path.abspath(v) if os.path.exists(v) or k in ("out", "export_v2", "out_dir") else
                    os.path.join(REPO, v))
    os.chdir(REPO)
    if a.self_test:
        return self_test()
    if a.servegen:
        if not os.path.isdir(a.servegen):
            print("--servegen needs a dir", file=sys.stderr); return 2
        res = servegen(a.servegen)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return 0
    if a.fetch:
        if not a.start:
            print("--fetch needs --start", file=sys.stderr); return 2
        try:                                                   # review: clean exit 2
            __import__("datetime").datetime.fromisoformat(a.start)
        except ValueError:
            print(f"--start not ISO: {a.start}", file=sys.stderr); return 2
        return fetch(a.start, a.days, a.out_dir, a.tag)
    if a.analyze:
        if not a.bins or not os.path.exists(a.bins):
            print("--analyze needs an existing --bins", file=sys.stderr); return 2
        rows = load_bins(a.bins)
        if not rows:                                           # review: empty bins = bad input
            print("--bins has no rows", file=sys.stderr); return 2
        an = analyze(rows, load_names(a.names))
        an["summary"]["bins_file"] = os.path.relpath(a.bins, REPO); an["summary"]["bins_sha256"] = sha256_file(a.bins); an["summary"]["window"] = a.window
        if a.out:
            json.dump(an, open(a.out, "w"), indent=1)
        if a.export_v2:
            export_v2(an, a.export_v2, window=a.window)
        s = an["summary"]
        print(json.dumps({k: s[k] for k in ("n_models", "days", "total_requests", "concentration", "long_tail",
                                            "corr_raw_5m", "corr_minus_daily_profile_5m", "corr_band_1h_24h_5m", "top_models_peak_within_3h_of_platform_peak")}, indent=1))
        return 0
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
