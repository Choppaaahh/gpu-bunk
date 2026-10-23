#!/usr/bin/env python3
"""tailpool_sim.py — pooled long-tail serving on K cards, real Chutes arrivals + model-load latency.

Prereg (frozen before any number): (private report). Every definition below
is that file's; on any divergence the prereg wins.

Inputs: minute bins ((private report) gitignored, sha in MANIFEST.json)
and per-model demand (derived/chutes_w1_analysis.json). Reuses `parse_size` (which applies KNOWN_HUGE), `C_REF`,
`ACTIVE_REF_B` from scripts/real_trace_analysis.py. Memory constants are 's measured ones.

Output contract:
  --self-test              exit 0 + "SELF-TEST PASS n/n" on the hand-computed goldens; exit 1 + FAIL lines otherwise.
  --run ...                one JSON object per run on stdout (summary); exit 0. Bad input (model larger than a card,
                           missing file, bad sha) = exit 2 with one "ERROR:" line on stderr. READ-ONLY on the trace;
                           writes only --cache (an npz of parsed bins) and --out when given.
  --grid PRIMARY|SEEDS|SENS|S2Z|DED  runs a prereg arm set and appends one JSON line per run to --out (WRITES --out).
                           SEEDS/SENS share PRIMARY's loop; the axis comes from --seeds/--ks/--cold-s/--scenario/--host-gib-per-card.
  --summarize DIR          read-only: prereg D1-D4 over every *.jsonl in DIR, one JSON object on stdout.
Drop-in for: --cool-s <measured page-in seconds> (host-RAM COOL tier), --cold-s <cold/snapshot seconds>.

Top pitfalls (observed while building): within-minute order matters (cells are processed by first-arrival time, not
file order); the same seed must give byte-identical summaries (checked in --self-test).
"""
import argparse, bisect, collections, csv, gzip, hashlib, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_trace_analysis import parse_size, C_REF, ACTIVE_REF_B  # noqa: E402  (reused as-is)

import numpy as np  # noqa: E402

CARD_GIB = 28.22          # 5090 budget, G4 receipt
OVERHEAD_GIB = 0.82       # engine overhead, (median of 12 warm 5090 starts)
GIB_PER_B = {"fp8": 1.07, "int4": 0.60}   # (int4 UNMEASURED there)
KV_GIB = 1.0              # prereg §1, CONDITIONAL
HOST_GIB_PER_CARD = 128.0  # prereg §1, CONDITIONAL
PROTECT_S = 60.0
LFU_HALF_LIFE_S = 6 * 3600.0
PRED_HORIZON_MIN = 30
PRED_BACKOFF_N = 5
MIN0 = 432000             # trace day 300
HELDOUT_MIN = 442080      # trace day 307
INF_KEY = 10 ** 12
SLOS = (2.0, 10.0)
K_GRID = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
COOL_GRID = (0.5, 2.0, 5.0, 15.0, 30.0)
BINS_SHA = "0d60071280ac4afe180bf6cd76183786c9d58d68704818625928233b09518f7e"
TRACE_DIR = "data/traces"


# ------------------------------------------------------------------------------------------------ models / scenarios
def footprint(total_b, prec, kv=KV_GIB):
    w = total_b * GIB_PER_B[prec]
    return w, w + OVERHEAD_GIB + kv


def scenario_sizes(name, scen):
    """(total_B, active_B, prec) for a model name under a footprint scenario, or None if UNPOOLABLE-by-size-unknown."""
    if scen == "U3":
        return 3.0, 3.0, "fp8"
    if scen == "U32":
        return 32.0, 32.0, "int4"
    if scen == "P8":
        tb, ab, src = parse_size(name)
        if src == "frontier-moe-size-not-in-name":
            return None
        if tb is None:
            return 8.0, 8.0, "fp8"
        return tb, ab, "fp8"
    raise ValueError(f"unknown scenario {scen}")


def tail_set(per_model, scen, card_gib=CARD_GIB):
    """prereg §1 tail definition. Returns ({cid: (total, active, prec, weights, foot)}, counts)."""
    tail, cnt = {}, collections.Counter()
    for cid, r in per_model.items():
        sz = scenario_sizes(r["name"], scen)
        if sz is None:
            cnt["unpoolable_frontier"] += 1
            continue
        tb, ab, prec = sz
        cards = r["work_rate_p95_5m"] / (C_REF * ACTIVE_REF_B / ab)
        if cards >= 1:
            cnt["head_ge_1_card"] += 1
            continue
        w, foot = footprint(tb, prec)
        if foot > card_gib:
            cnt["unpoolable_too_big"] += 1
            continue
        tail[cid] = (tb, ab, prec, w, foot)
        cnt["tail"] += 1
    return tail, dict(cnt)


# ------------------------------------------------------------------------------------------------ trace
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_bins_np(path, cache):
    """All rows as numpy arrays (minute, model-index, n, it, ot) + id list. Verifies the MANIFEST sha first."""
    got = sha256_file(path)   # review: verify the source on EVERY load, cache or not
    if got != BINS_SHA:
        raise ValueError(f"bins sha mismatch: {got} != {BINS_SHA}")
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if "src_sha" not in z.files or str(z["src_sha"]) != BINS_SHA:
            raise ValueError(f"cache {cache} was not built from bins sha {BINS_SHA}; delete it and re-run")
        return z["minute"], z["midx"], z["n"], z["it"], z["ot"], list(z["ids"])
    ids, idx = [], {}
    mi, mx, nn, it, ot = [], [], [], [], []
    with gzip.open(path, "rt") as fh:
        for r in csv.DictReader(fh):
            c = r["chute_id"]
            j = idx.get(c)
            if j is None:
                j = idx[c] = len(ids)
                ids.append(c)
            mi.append(int(r["minute"])); mx.append(j); nn.append(int(r["n"]))
            it.append(float(r["it"])); ot.append(float(r["ot"]))
    arr = (np.array(mi, np.int64), np.array(mx, np.int32), np.array(nn, np.int64), np.array(it), np.array(ot))
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, minute=arr[0], midx=arr[1], n=arr[2], it=arr[3], ot=arr[4], ids=np.array(ids),
                 src_sha=np.array(BINS_SHA))
    return arr + (ids,)


class Trace:
    """Cells of the tail models only, sorted by minute. Model ids are re-indexed 0..M-1 over the tail."""

    def __init__(self, minute, mid, n, it, ot, models):
        o = np.argsort(minute, kind="stable")
        self.minute = np.asarray(minute)[o]
        self.mid = np.asarray(mid)[o]
        self.n = np.asarray(n)[o]
        self.it = np.asarray(it, dtype=float)[o]
        self.ot = np.asarray(ot, dtype=float)[o]
        self.models = models      # list of (cid, total, active, prec, weights, foot)
        self.explicit = None      # goldens: per-cell sorted arrival offsets (seconds within the minute)

    @classmethod
    def from_bins(cls, arrays, tail):
        minute, midx, n, it, ot, ids = arrays
        keep_ids = [i for i, c in enumerate(ids) if c in tail]
        remap = np.full(len(ids), -1, np.int64)
        models = []
        for new, old in enumerate(keep_ids):
            remap[old] = new
            models.append((ids[old],) + tuple(tail[ids[old]]))
        m = remap[midx]
        k = m >= 0
        key = minute[k].astype(np.int64) * 100000 + m[k]
        if len(np.unique(key)) != len(key):   # QA 2026-10-13: dedicated/pool assume one cell per (minute, model)
            raise ValueError("duplicate (minute, model) rows in the bins")
        return cls(minute[k], m[k], n[k], it[k], ot[k], models)

    @classmethod
    def explicit_cells(cls, models, reqs):
        """Goldens: reqs = [(abs_seconds, model_index, in_tokens, out_tokens)] -> one cell per (minute, model)."""
        cells = collections.OrderedDict()
        for t, m, i, o in sorted(reqs):
            key = (int(t // 60), m)
            c = cells.setdefault(key, [[], 0.0, 0.0])
            c[0].append(t - 60 * key[0]); c[1] += i; c[2] += o
        keys = list(cells)
        tr = cls([k[0] for k in keys], [k[1] for k in keys], [len(cells[k][0]) for k in keys],
                 [cells[k][1] for k in keys], [cells[k][2] for k in keys], models)
        # argsort(stable) of already-sorted minutes keeps order; rebuild explicit in the sorted order
        order = np.argsort(np.array([k[0] for k in keys]), kind="stable")
        tr.explicit = [sorted(cells[keys[i]][0]) for i in order]
        return tr


def arrivals(trace, seed):
    """Per-cell first and last arrival offsets (s) + the processing order (by minute, then first arrival)."""
    if trace.explicit is not None:
        first = np.array([a[0] for a in trace.explicit], float)
        last = np.array([a[-1] for a in trace.explicit], float)
    else:
        rng = np.random.default_rng([seed, 2683])
        n = trace.n.astype(float)
        v1 = rng.random(len(n)); v2 = rng.random(len(n))
        first = 60.0 * (1.0 - (1.0 - v1) ** (1.0 / n))           # min of n U[0,60)
        rest = np.maximum(n - 1.0, 1.0)
        last = np.where(n > 1, first + (60.0 - first) * v2 ** (1.0 / rest), first)   # max of the other n-1
    order = np.lexsort((first, trace.minute))
    return first, last, order


def cell_times(trace, i, first, last, seed):
    """All n arrival offsets for cell i (sorted). Deterministic per (seed, cell)."""
    if trace.explicit is not None:
        return np.asarray(trace.explicit[i], float)
    n = int(trace.n[i])
    if n == 1:
        return np.array([first[i]])
    if n == 2:
        return np.array([first[i], last[i]])
    rng = np.random.default_rng([seed, 2683, i])
    mid = np.sort(first[i] + (last[i] - first[i]) * rng.random(n - 2))
    return np.concatenate(([first[i]], mid, [last[i]]))


# ------------------------------------------------------------------------------------------------ predictive policy
class Predictor:
    """P(active again within 30 min | idle g min), fitted on TRAIN-week gaps between active minutes (prereg §1)."""

    def __init__(self, trace, train_end_min):
        M = len(trace.models)
        per = [[] for _ in range(M)]
        k = trace.minute < train_end_min
        for m, mi in zip(trace.mid[k], trace.minute[k]):
            per[int(m)].append(int(mi))
        self.gaps = []
        pooled = []
        for lst in per:
            u = sorted(set(lst))
            g = np.diff(np.array(u, np.int64)) if len(u) > 1 else np.array([], np.int64)
            g = np.sort(g)
            self.gaps.append(g)
            pooled.append(g)
        self.pooled = np.sort(np.concatenate(pooled)) if pooled else np.array([], np.int64)
        self.cache = {}

    @staticmethod
    def _est(g_arr, g):
        n_gt = len(g_arr) - bisect.bisect_right(g_arr, g)
        n_in = bisect.bisect_right(g_arr, g + PRED_HORIZON_MIN) - bisect.bisect_right(g_arr, g)
        return n_gt, (n_in + 1.0) / (n_gt + 2.0)

    def score(self, m, g):
        key = (m, g)
        v = self.cache.get(key)
        if v is None:
            n_gt, p = self._est(self.gaps[m], g)
            if n_gt < PRED_BACKOFF_N:
                _, p = self._est(self.pooled, g)
            v = self.cache[key] = p
        return v


# ------------------------------------------------------------------------------------------------ pooled simulator
def _pct(counter, q):
    """Nearest-rank percentile over {key(0.01 s units or INF_KEY): count}. Returns seconds (inf allowed)."""
    tot = sum(counter.values())
    if tot == 0:
        return None
    need = math.ceil(q * tot)
    acc = 0
    for k in sorted(counter):
        acc += counter[k]
        if acc >= need:
            return math.inf if k >= INF_KEY else k / 100.0
    return None


def _key(v):
    return INF_KEY if v == math.inf else int(round(v * 100.0))


def simulate_pool(trace, K, policy, cool_s, cold_s, seed, card_gib=CARD_GIB, host_gib_per_card=HOST_GIB_PER_CARD,
                  score_from_min=HELDOUT_MIN, predictor=None, preload=(), min0=MIN0, cool_gbps=None, cool_fixed=0.02):
    """cool_gbps set (amendment 1, measured) -> COOL page-in = cool_fixed + weight bytes / cool_gbps GB/s per model,
    and cool_s is ignored. Returns (per-model Counter of TTFT keys over scored cells, stats dict). Deterministic given seed."""
    models = trace.models
    M = len(models)
    for m in models:
        if m[5] > card_gib:
            raise ValueError(f"model {m[0]} footprint {m[5]:.2f} GiB > card {card_gib} GiB")
    if policy not in ("LRU", "LFU", "PRED"):
        raise ValueError(f"unknown policy {policy}")
    if policy == "PRED" and predictor is None:
        raise ValueError("PRED needs a predictor fitted on TRAIN")
    first, last, order = arrivals(trace, seed)
    card_free = [card_gib] * K
    card_res = [set() for _ in range(K)]
    where = [-1] * M
    ready = [0.0] * M
    last_arr = [-1e18] * M
    last_min = [-10 ** 9] * M
    lfu_log = [-1e300] * M      # log2(score) + t/HL at the last update, so ranking is decay-free
    host = collections.OrderedDict()
    host_used = 0.0
    host_cap = K * host_gib_per_card
    backlog = [0.0] * K
    cap_min = C_REF * 60.0
    out = [collections.Counter() for _ in range(M)]
    st = collections.Counter()

    for m in preload:
        c = max(range(K), key=lambda j: card_free[j])
        card_free[c] -= models[m][5]; card_res[c].add(m); where[m] = c; ready[m] = -1e18

    def host_insert(m):
        nonlocal host_used
        if m in host:
            host.move_to_end(m); return
        w = models[m][4]
        if w > host_cap:
            return
        while host_used + w > host_cap and host:
            v, wv = host.popitem(last=False)
            host_used -= wv; st["host_evictions"] += 1
        host[m] = w; host_used += w

    def keep_score(m, now, now_min):
        if policy == "LRU":
            return (last_arr[m], m)
        if policy == "LFU":
            return (lfu_log[m], last_arr[m], m)
        return (predictor.score(m, max(0, now_min - last_min[m])), last_arr[m], m)

    def place(m, now, now_min):
        need = models[m][5]
        best = max(range(K), key=lambda j: (card_free[j], -j))
        if card_free[best] >= need - 1e-9:
            return best
        cand = []
        for c in range(K):
            for r in card_res[c]:
                if ready[r] > now or last_arr[r] > now - PROTECT_S:
                    continue
                cand.append((keep_score(r, now, now_min), r, c))
        cand.sort()
        tried = set()
        for _, r, c in cand:
            if c in tried:
                continue
            tried.add(c)
            mine = sorted((s, rr) for s, rr, cc in cand if cc == c)
            free = card_free[c]
            vict = []
            for s, rr in mine:
                if free >= need - 1e-9:
                    break
                vict.append(rr); free += models[rr][5]
            if free >= need - 1e-9:
                for rr in vict:
                    card_res[c].discard(rr); card_free[c] += models[rr][5]; where[rr] = -1
                    st["evictions"] += 1
                    host_insert(rr)
                return c
        return -1

    i_pos = 0
    N = len(order)
    prev_mi = None
    while i_pos < N:
        mi = int(trace.minute[order[i_pos]])
        if prev_mi is not None and mi - prev_mi > 1:   # minutes with no tail cell still drain every card
            for c in range(K):
                backlog[c] = max(0.0, backlog[c] - cap_min * (mi - prev_mi - 1))
        prev_mi = mi
        j = i_pos
        while j < N and int(trace.minute[order[j]]) == mi:
            j += 1
        W = [0.0] * K
        pend = []
        base = (mi - min0) * 60.0
        for p in range(i_pos, j):
            i = int(order[p])
            m = int(trace.mid[i])
            n = int(trace.n[i])
            t0 = base + float(first[i])
            tb, ab, prec, w, foot = models[m][1:]
            c = where[m]
            if c < 0:
                L = cool_s if m in host else cold_s
                if cool_gbps and m in host:
                    L = cool_fixed + w * 1.073741824 / cool_gbps
                c = place(m, t0, mi)
                if c >= 0:
                    if m in host:
                        st["cool_loads"] += 1; host.move_to_end(m)
                    else:
                        st["cold_loads"] += 1; host_insert(m)
                    card_free[c] -= foot; card_res[c].add(m); where[m] = c; ready[m] = t0 + L
            # policy bookkeeping (requested even if unserved)
            if policy == "LFU":
                cur = lfu_log[m]
                now_l = t0 / LFU_HALF_LIFE_S
                # add 1 to the decayed score: log2(2^(cur-now_l) + 1) + now_l
                lfu_log[m] = now_l + math.log2(2.0 ** (cur - now_l) + 1.0) if cur > -1e299 else now_l
            last_arr[m] = base + float(last[i])
            last_min[m] = mi
            pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF
            if c < 0:
                st["unserved_req"] += n
                pend.append((m, -1, n, pre, None)); continue
            W[c] += (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
            if ready[m] <= t0:
                pend.append((m, c, n, pre, None))
            else:
                ts = cell_times(trace, i, first, last, seed) + base
                pend.append((m, c, n, pre, np.maximum(0.0, ready[m] - ts)))
        q = [0.0] * K
        for c in range(K):
            q[c] = (backlog[c] + 0.5 * max(0.0, W[c] - cap_min)) / C_REF
            backlog[c] = max(0.0, backlog[c] + W[c] - cap_min)
        if mi >= score_from_min:
            for m, c, n, pre, waits in pend:
                if c < 0:
                    out[m][INF_KEY] += n; continue
                if waits is None:
                    out[m][_key(q[c] + pre)] += n
                else:
                    for wv in waits:
                        out[m][_key(float(wv) + q[c] + pre)] += 1
        i_pos = j
    return out, dict(st)


def simulate_dedicated(trace, score_from_min=HELDOUT_MIN):
    """One always-resident card per model: TTFT = own-card queue + prefill. No randomness."""
    M = len(trace.models)
    out = [collections.Counter() for _ in range(M)]
    backlog = [0.0] * M
    last_min = [None] * M
    cap_min = C_REF * 60.0
    # cells sorted by minute; one cell per (minute, model) so each cell is that model's whole minute
    for i in range(len(trace.n)):
        m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        ab = trace.models[m][2]
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        if last_min[m] is not None:   # the own card drains over the model's idle minutes too
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        last_min[m] = mi
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / C_REF
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        if mi >= score_from_min:
            out[m][_key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF)] += n
    return out


def simulate_s2z(trace, t_snap, t_idle, seed, score_from_min=HELDOUT_MIN, min0=MIN0):
    """Scale-to-zero: own card on demand, released t_idle after the last arrival; each start costs t_snap."""
    M = len(trace.models)
    first, last, order = arrivals(trace, seed)
    alive_until = [-1e18] * M
    ready = [0.0] * M
    start_t = [0.0] * M
    backlog = [0.0] * M
    last_min = [None] * M
    cap_min = C_REF * 60.0
    out = [collections.Counter() for _ in range(M)]
    intervals = []
    st = collections.Counter()
    for p in range(len(order)):
        i = int(order[p]); m = int(trace.mid[i]); n = int(trace.n[i]); mi = int(trace.minute[i])
        base = (mi - min0) * 60.0
        t0 = base + float(first[i])
        ab = trace.models[m][2]
        if last_min[m] is not None:   # drain over idle minutes first
            backlog[m] = max(0.0, backlog[m] - cap_min * (mi - last_min[m] - 1))
        # T3 2026-10-13 (R ratio read > 1): a card with queued work is never released, so it is no cold start
        if t0 > alive_until[m] and backlog[m] <= 0.0:
            if alive_until[m] > -1e17:
                intervals.append((start_t[m], alive_until[m]))
            start_t[m] = t0; ready[m] = t0 + t_snap; st["cold_starts"] += 1
        alive_until[m] = base + float(last[i]) + t_idle
        last_min[m] = mi
        Wm = (trace.it[i] + trace.ot[i]) * (ab / ACTIVE_REF_B)
        q = (backlog[m] + 0.5 * max(0.0, Wm - cap_min)) / C_REF
        backlog[m] = max(0.0, backlog[m] + Wm - cap_min)
        pre = (trace.it[i] / n) * (ab / ACTIVE_REF_B) / C_REF
        if mi >= score_from_min:
            if ready[m] <= t0:
                out[m][_key(q + pre)] += n
            else:
                ts = cell_times(trace, i, first, last, seed) + base
                for wv in np.maximum(0.0, ready[m] - ts):
                    out[m][_key(float(wv) + q + pre)] += 1
    for m in range(M):
        if alive_until[m] > -1e17:
            intervals.append((start_t[m], alive_until[m]))
    return out, intervals, dict(st)


def concurrency(intervals, from_s, to_s):
    """Per-minute max concurrent cards over [from_s, to_s): (p99, max, mean)."""
    nmin = int((to_s - from_s) // 60)
    if nmin <= 0:
        raise ValueError("empty concurrency window")
    diff = np.zeros(nmin + 1)
    for a, b in intervals:
        if b <= from_s or a >= to_s:
            continue
        lo = int(max(0, (a - from_s) // 60)); hi = int(min(nmin - 1, (b - from_s) // 60))
        diff[lo] += 1; diff[hi + 1] -= 1
    cnt = np.cumsum(diff)[:nmin]
    return float(np.percentile(cnt, 99)), float(cnt.max()), float(cnt.mean())


# ------------------------------------------------------------------------------------------------ scoring
def score(out, ded, slo, min_req=20):
    """prereg §1 (M) and (R) for one SLO."""
    elig = feas = meet = 0
    req_ok = req_feas = 0
    n_infeas_models = 0
    for m in range(len(out)):
        d = ded[m]
        nd = sum(d.values())
        lim = _key(slo)
        req_feas += sum(v for k, v in d.items() if k <= lim)
        req_ok += sum(v for k, v in out[m].items() if k <= lim)
        if nd < min_req:
            continue
        elig += 1
        if _pct(d, 0.95) > slo:
            n_infeas_models += 1
            continue
        feas += 1
        if _pct(out[m], 0.95) <= slo:
            meet += 1
    Mf = meet / feas if feas else None
    Rr = req_ok / req_feas if req_feas else None
    return {"slo": slo, "eligible": elig, "slo_feasible_models": feas, "infeasible_even_dedicated": n_infeas_models,
            "meet": meet, "M_frac": Mf, "R_ratio": Rr, "req_ok": req_ok, "req_feasible": req_feas,
            "serves": bool(Mf is not None and Rr is not None and Mf >= 0.95 and Rr >= 0.99)}


def model_pcts(out):
    return [(_pct(c, 0.95), _pct(c, 0.99), sum(c.values())) for c in out]


# ------------------------------------------------------------------------------------------------ prereg D1-D4
def _rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def min_k(rows, policy, cool, slo, cold=30.0, scen="P8", host=HOST_GIB_PER_CARD, seeds=(0, 1, 2)):
    """prereg §1 min-K: smallest grid K where Serves holds under every seed in `seeds` (all must have run).
    Returns (min_k or None, seed0_min_k, by_k {K: {seed: (serves, miss_frac)}}, flag)."""
    by = collections.defaultdict(dict)
    for d in rows:
        if (d.get("arm") == "POOL" and d["policy"] == policy and d["cool_s"] == cool and d["cold_s"] == cold
                and d["scenario"] == scen and d["host_gib_per_card"] == host):
            v = d["slo"][str(slo)]
            by[d["K"]][d["seed"]] = (v["serves"], 1.0 - v["R_ratio"])
    s0 = next((K for K in sorted(by) if by[K].get(0, (False,))[0]), None)
    mk = next((K for K in sorted(by) if all(sd in by[K] for sd in seeds) and all(by[K][sd][0] for sd in seeds)), None)
    flag = ""
    if s0 is not None and mk is None:
        top = max(K_GRID)
        flag = ("NOT_SERVED_LE_%d_ALL_SEEDS" % top if all(sd in by.get(top, {}) for sd in seeds) else "NEEDS_STEP_UP")
    elif mk is not None and any(not all(sd in by[K] for sd in seeds) and by[K].get(0, (False,))[0]
                                for K in sorted(by) if K < mk):
        flag = "SEED0_PASS_BELOW_UNCHECKED"
    return mk, s0, dict(by), flag


def summarize(d):
    rows = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            rows += _rows(os.path.join(d, f))
    out = {"n_rows": len(rows), "D1": {}, "D2": {}, "D3": {}, "sens": {}}
    pols = ("LRU", "LFU", "PRED")
    for slo in SLOS:
        for cool in COOL_GRID:
            for pol in pols:
                mk, s0, by, flag = min_k(rows, pol, cool, slo)
                out["D1"][f"{pol}|{cool}|{slo}"] = {"min_k": mk, "seed0_min_k": s0, "flag": flag}
            # D2: smallest min-K; ties by 3-seed mean (R) miss at that K vs the larger seed range
            cand = []
            for pol in pols:
                mk, s0, by, flag = min_k(rows, pol, cool, slo)
                if mk is None:
                    cand.append((math.inf, pol, None, None)); continue
                ms = [by[mk][sd][1] for sd in (0, 1, 2)]
                cand.append((mk, pol, sum(ms) / 3.0, max(ms) - min(ms)))
            cand.sort(key=lambda x: (x[0], x[2] if x[2] is not None else 1.0))
            best, second = cand[0], cand[1]
            if best[0] == math.inf:
                verdict = "NONE_SERVES"
            elif best[0] < second[0]:
                verdict = best[1]
            else:
                gap = second[2] - best[2]
                verdict = best[1] if gap > max(best[3], second[3]) else "TIE"
            out["D2"][f"{cool}|{slo}"] = {"verdict": verdict,
                                         "cands": [(c[1], c[0] if c[0] != math.inf else None, c[2], c[3]) for c in cand]}
    wins = collections.Counter(v["verdict"] for v in out["D2"].values())
    top = [p for p in pols if wins[p] >= 6]
    out["D2_overall"] = {"wins": dict(wins), "winner": top[0] if top else "NO SINGLE WINNER"}
    w = top[0] if top else "LRU"
    for slo in SLOS:
        a = out["D1"][f"{w}|{COOL_GRID[0]}|{slo}"]["min_k"]
        b = out["D1"][f"{w}|{COOL_GRID[-1]}|{slo}"]["min_k"]
        served = [out["D1"][f"{w}|{c}|{slo}"]["min_k"] is not None for c in COOL_GRID]
        ratio = (b / a) if (a and b) else None
        sens = (ratio is not None and ratio >= 1.5) or (len(set(served)) > 1)
        out["D3"][str(slo)] = {"policy": w, "min_k_cool_0.5": a, "min_k_cool_30": b, "ratio": ratio,
                               "verdict": "PAGE-IN-SENSITIVE" if sens else "INSENSITIVE (at grid resolution)"}
    for tag, kw in (("cold10", {"cold": 10.0}), ("cold60", {"cold": 60.0}), ("U3", {"scen": "U3"}),
                    ("U32", {"scen": "U32"}), ("host64", {"host": 64.0})):
        for slo in SLOS:
            for pol in pols:
                mk, s0, by, flag = min_k(rows, pol, 2.0, slo, seeds=(0,), **kw)
                out["sens"][f"{tag}|{pol}|{slo}"] = s0
    out["A1_measured"] = {}
    for slo in SLOS:
        cand = []
        for pol in pols:
            mk, s0, by, flag = min_k(rows, pol, "measured", slo, cold=55.0)
            out["A1_measured"][f"{pol}|{slo}"] = {"min_k": mk, "seed0_min_k": s0, "flag": flag}
            if mk is None:
                cand.append((math.inf, pol, None, None)); continue
            ms = [by[mk][sd][1] for sd in (0, 1, 2)]
            cand.append((mk, pol, sum(ms) / 3.0, max(ms) - min(ms)))
        cand.sort(key=lambda x: (x[0], x[2] if x[2] is not None else 1.0))
        b, c2 = cand[0], cand[1]
        v = ("NONE_SERVES" if b[0] == math.inf else b[1] if b[0] < c2[0]
             else (b[1] if (c2[2] - b[2]) > max(b[3], c2[3]) else "TIE"))
        out["A1_measured"][f"D2|{slo}"] = {"verdict": v, "cands": [(x[1], None if x[0] == math.inf else x[0], x[2], x[3])
                                                                  for x in cand]}
        for tag, kw in (("A2_cold187", {"cold": 187.0}), ("A3_host32", {"cold": 55.0, "host": 32.0})):
            for pol in pols:
                out["A1_measured"][f"{tag}|{pol}|{slo}"] = min_k(rows, pol, "measured", slo, seeds=(0,), **kw)[1]
    ded = [r for r in rows if r.get("arm") == "DEDICATED"]
    out["dedicated"] = {k: v for k, v in ded[0].items() if k != "per_model"} if ded else None
    s2z = collections.defaultdict(list)
    for r in rows:
        if r.get("arm") == "S2Z":
            s2z[(r["t_snap"], r["t_idle"])].append(r)
    out["S2Z"] = {f"{k[0]}|{k[1]}": {"cards_p99_max_over_seeds": max(x["cards_p99"] for x in v),
                                     "cards_max": max(x["cards_max"] for x in v),
                                     "cards_mean": sum(x["cards_mean"] for x in v) / len(v),
                                     "serves": {str(s): [x["slo"][str(s)]["serves"] for x in v] for s in SLOS},
                                     "M": {str(s): [round(x["slo"][str(s)]["M_frac"], 4) for x in v] for s in SLOS},
                                     "R": {str(s): [round(x["slo"][str(s)]["R_ratio"], 4) for x in v] for s in SLOS},
                                     "n_seeds": len(v)} for k, v in sorted(s2z.items())}
    return out


# ------------------------------------------------------------------------------------------------ self-test
def self_test():
    fails, n = [], 0

    def chk(name, got, want, tol=1e-6):
        nonlocal n
        n += 1
        ok = (got == want) if not isinstance(want, float) else (got is not None and abs(got - want) <= tol)
        if isinstance(want, list):
            ok = len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))
        if not ok:
            fails.append(f"FAIL {name}: got {got!r} want {want!r}")

    def vals(counter):
        out = []
        for k in sorted(counter):
            out += [k / 100.0] * counter[k]
        return out

    A3 = lambda cid: (cid, 3.0, 3.0, "fp8") + footprint(3.0, "fp8")
    chk("footprint 3B fp8", A3("A")[5], 5.03, 1e-9)
    # G1
    tr = Trace.explicit_cells([A3("A")], [(0, 0, 0, 0), (10, 0, 0, 0), (100, 0, 0, 0)])
    o, s = simulate_pool(tr, 1, "LRU", 2.0, 30.0, 0, score_from_min=0, min0=0)
    chk("G1 ttft", sorted(vals(o[0])), [0.0, 20.0, 30.0])
    chk("G1 cold loads", s.get("cold_loads", 0), 1); chk("G1 cool loads", s.get("cool_loads", 0), 0)
    # G2 (both LRU and LFU)
    for pol in ("LRU", "LFU"):
        tr = Trace.explicit_cells([A3("A"), A3("B")], [(0, 0, 0, 0), (100, 1, 0, 0), (200, 0, 0, 0), (300, 1, 0, 0)])
        o, s = simulate_pool(tr, 1, pol, 2.0, 30.0, 0, card_gib=8.0, score_from_min=0, min0=0)
        chk(f"G2 {pol} A", vals(o[0]), [2.0, 30.0]); chk(f"G2 {pol} B", vals(o[1]), [2.0, 30.0])
        chk(f"G2 {pol} cold", s.get("cold_loads", 0), 2); chk(f"G2 {pol} cool", s.get("cool_loads", 0), 2)
        chk(f"G2 {pol} evictions", s.get("evictions", 0), 3)
    # G3 policy split
    reqs = [(0, 0, 0, 0), (100, 0, 0, 0), (200, 0, 0, 0), (300, 1, 0, 0), (400, 2, 0, 0), (500, 0, 0, 0)]
    for pol, want in (("LRU", [0.0, 0.0, 2.0, 30.0]), ("LFU", [0.0, 0.0, 0.0, 30.0])):
        tr = Trace.explicit_cells([A3("A"), A3("B"), A3("C")], reqs)
        o, s = simulate_pool(tr, 1, pol, 2.0, 30.0, 0, card_gib=11.0, score_from_min=0, min0=0)
        chk(f"G3 {pol} A", vals(o[0]), want)
    # G4 adversarial: 32B fp8 does not fit; unknown scenario; loud
    big = ("X", 32.0, 32.0, "fp8") + footprint(32.0, "fp8")
    chk("G4 32B fp8 footprint", big[5], 36.06, 1e-9)
    try:
        simulate_pool(Trace.explicit_cells([big], [(0, 0, 0, 0)]), 1, "LRU", 2, 30, 0, score_from_min=0, min0=0)
        fails.append("FAIL G4: model larger than a card did not raise"); n += 1
    except ValueError:
        n += 1
    try:
        Trace.explicit_cells([A3("A")], [(0, 1, 0, 0)])  # model index 1 does not exist
        simulate_pool(Trace.explicit_cells([A3("A")], [(0, 1, 0, 0)]), 1, "LRU", 2, 30, 0, score_from_min=0, min0=0)
        fails.append("FAIL G4b: request for a model with no footprint did not raise"); n += 1
    except (ValueError, IndexError):
        n += 1
    # G5 queue (3.1B active, preloaded)
    A31 = ("A", 3.1, 3.1, "fp8") + footprint(3.1, "fp8")
    big_out = 2 * C_REF * 60
    tr = Trace.explicit_cells([A31], [(0, 0, 0, big_out), (61, 0, 0, 0)])
    o, s = simulate_pool(tr, 1, "LRU", 2.0, 30.0, 0, preload=(0,), score_from_min=0, min0=0)
    chk("G5 queue", vals(o[0]), [30.0, 60.0], 0.011)
    ded = simulate_dedicated(tr, score_from_min=0)
    chk("G5 dedicated queue", vals(ded[0]), [30.0, 60.0], 0.011)
    # G6 prefill
    tr = Trace.explicit_cells([A31], [(0, 0, C_REF, 0)])
    o, s = simulate_pool(tr, 1, "LRU", 2.0, 30.0, 0, preload=(0,), score_from_min=0, min0=0)
    chk("G6 prefill", vals(o[0]), [1.0], 0.006)
    # G7 scale-to-zero
    tr = Trace.explicit_cells([A3("A")], [(0, 0, 0, 0), (100, 0, 0, 0), (500, 0, 0, 0)])
    o, iv, s = simulate_s2z(tr, 5.0, 300.0, 0, score_from_min=0, min0=0)
    chk("G7 s2z ttft", sorted(vals(o[0])), [0.0, 5.0, 5.0])
    chk("G7 cold starts", s.get("cold_starts", 0), 2)
    chk("G7 max concurrent", concurrency(iv, 0, 1200)[1], 1.0)
    # G9 (amendment 1): page-in by bytes, 8B fp8 -> 0.02 + 8.56 * 1.073741824 / 22.6 = 0.4267 -> 0.43
    B8 = lambda cid: (cid, 8.0, 8.0, "fp8") + footprint(8.0, "fp8")
    tr = Trace.explicit_cells([B8("A"), B8("B")], [(0, 0, 0, 0), (100, 1, 0, 0), (200, 0, 0, 0)])
    o, s = simulate_pool(tr, 1, "LRU", 99.0, 55.0, 0, card_gib=12.0, score_from_min=0, min0=0, cool_gbps=22.6)
    chk("G9 measured page-in", vals(o[0]), [0.43, 55.0], 1e-9)
    # G7b: queued work keeps the card alive past T_idle (no cold start, no backlog reset)
    tr = Trace.explicit_cells([A31], [(0, 0, 0, 3 * C_REF * 60), (90, 0, 0, 0)])
    o, iv, s = simulate_s2z(tr, 5.0, 30.0, 0, score_from_min=0, min0=0)
    chk("G7b s2z busy card", sorted(vals(o[0])), [65.0, 120.0], 0.011)
    chk("G7b cold starts", s.get("cold_starts", 0), 1)
    # G8 percentile
    c = collections.Counter({_key(float(v)): 1 for v in range(1, 21)})
    chk("G8 p95", _pct(c, 0.95), 19.0); chk("G8 p99", _pct(c, 0.99), 20.0)
    # PRED: predictor fitted on gaps; a model with gaps of 1 scores higher at g=1 than one with gaps of 1000
    mins = list(range(10)) + [1000 * k for k in range(10)]
    tr = Trace(mins, [0] * 10 + [1] * 10, [1] * 20, [0] * 20, [0] * 20, [A3("A"), A3("B")])
    pr = Predictor(tr, 10 ** 9)
    chk("PRED hot model g=0", pr.score(0, 0), 10 / 11)     # 9 gaps of 1: (9+1)/(9+2)
    chk("PRED cold model g=0", pr.score(1, 0), 1 / 11)     # 9 gaps of 1000: (0+1)/(9+2)
    chk("PRED back-off when <5 gaps exceed g", pr.score(0, 5), Predictor._est(pr.pooled, 5)[1])
    # determinism: same seed -> identical; random arrival generator respects n
    tr = Trace([0, 0, 1], [0, 1, 0], [5, 3, 1], [10, 10, 10], [5, 5, 5], [A3("A"), A3("B")])
    r1 = simulate_pool(tr, 1, "LRU", 2.0, 30.0, 7, card_gib=8.0, score_from_min=0, min0=0)
    r2 = simulate_pool(tr, 1, "LRU", 2.0, 30.0, 7, card_gib=8.0, score_from_min=0, min0=0)
    chk("determinism", json.dumps([dict(x) for x in r1[0]]) == json.dumps([dict(x) for x in r2[0]]), True)
    chk("request conservation", sum(sum(x.values()) for x in r1[0]), 9)
    # score(): (M)/(R) arithmetic on a 2-model fixture
    ded = [collections.Counter({_key(0.1): 20}), collections.Counter({_key(0.1): 20})]
    pool = [collections.Counter({_key(0.1): 20}), collections.Counter({_key(0.1): 18, _key(5.0): 2})]
    sc = score(pool, ded, 2.0)
    chk("score M", sc["M_frac"], 0.5); chk("score R", sc["R_ratio"], 38 / 40); chk("score serves", sc["serves"], False)
    # min_k: seed-0 pass at K=4 but seed 1 fails there; all 3 pass at K=8 -> min_k 8
    mk_rows = []
    for K, res in ((4, (True, False, True)), (8, (True, True, True))):
        for sd, ok in enumerate(res):
            mk_rows.append({"arm": "POOL", "policy": "LRU", "cool_s": 2.0, "cold_s": 30.0, "scenario": "P8",
                            "host_gib_per_card": HOST_GIB_PER_CARD, "K": K, "seed": sd,
                            "slo": {"2.0": {"serves": ok, "R_ratio": 0.995}}})
    chk("min_k all-seeds rule", min_k(mk_rows, "LRU", 2.0, 2.0)[0], 8)
    chk("min_k seed0", min_k(mk_rows, "LRU", 2.0, 2.0)[1], 4)
    for f in fails:
        print(f)
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n - len(fails)}/{n}")
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ CLI
def build(scen, root, cache):
    per_model = json.load(open(os.path.join(root, TRACE_DIR, "derived/chutes_w1_analysis.json")))["per_model"]
    tail, cnt = tail_set(per_model, scen)
    arrays = load_bins_np(os.path.join(root, TRACE_DIR, "derived/bins_w1.csv.gz"), cache)
    tr = Trace.from_bins(arrays, tail)
    return tr, cnt


def run_one(tr, cnt, ded, K, policy, cool, cold, seed, scen, host_gib, predictor, detail=False, cool_gbps=None):
    t = time.time()
    out, st = simulate_pool(tr, K, policy, cool, cold, seed, host_gib_per_card=host_gib, predictor=predictor,
                            cool_gbps=cool_gbps)
    res = {"arm": "POOL", "scenario": scen, "K": K, "policy": policy,
           "cool_s": "measured" if cool_gbps else cool, "cool_gbps": cool_gbps, "cold_s": cold, "seed": seed,
           "host_gib_per_card": host_gib, "tail_counts": cnt, "stats": st,
           "slo": {str(s): score(out, ded, s) for s in SLOS}, "elapsed_s": round(time.time() - t, 1),
           "heldout_req": int(sum(sum(c.values()) for c in out))}
    if detail:
        res["per_model"] = {tr.models[m][0]: v for m, v in enumerate(model_pcts(out))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--grid", choices=["PRIMARY", "SEEDS", "SENS", "S2Z", "DED"])
    ap.add_argument("--scenario", default="P8", choices=["P8", "U3", "U32"])
    ap.add_argument("--policy", default="LRU", choices=["LRU", "LFU", "PRED"])
    ap.add_argument("--policies", default="LRU,LFU,PRED")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ks", default=",".join(map(str, K_GRID)))
    ap.add_argument("--cool-s", type=float, default=2.0, help="host-RAM COOL page-in seconds (drop-in)")
    ap.add_argument("--cools", default=",".join(map(str, COOL_GRID)))
    ap.add_argument("--cool-gbps", type=float, default=None,
                    help="amendment 1: page-in = 0.02 s + weight bytes / this GB/s (measured 22.6); overrides --cool-s/--cools")
    ap.add_argument("--snaps", default="2,5,10,30", help="S2Z snapshot cold-start seconds")
    ap.add_argument("--cold-s", type=float, default=30.0, help="cold / snapshot load seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--host-gib-per-card", type=float, default=HOST_GIB_PER_CARD)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/tailpool/bins_w1.npz"))
    ap.add_argument("--out", help="append JSON lines here (WRITES)")
    ap.add_argument("--detail", action="store_true", help="include per-model p95/p99 in each row")
    ap.add_argument("--summarize", metavar="DIR", help="read every *.jsonl in DIR, print prereg D1-D4 as JSON (read-only)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.summarize:
        print(json.dumps(summarize(a.summarize), indent=1, default=str))
        return 0
    try:
        tr, cnt = build(a.scenario, a.root, a.cache)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    ded = simulate_dedicated(tr)
    pred = Predictor(tr, HELDOUT_MIN)
    sink = open(a.out, "a") if a.out else None

    def emit(r):
        line = json.dumps(r, sort_keys=True)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n"); sink.flush()

    try:
        if a.run:
            emit(run_one(tr, cnt, ded, a.k, a.policy, a.cool_s, a.cold_s, a.seed, a.scenario, a.host_gib_per_card,
                         pred, a.detail, a.cool_gbps))
        elif a.grid == "DED":
            emit({"arm": "DEDICATED", "scenario": a.scenario, "cards": len(tr.models), "tail_counts": cnt,
                  "heldout_req": int(sum(sum(c.values()) for c in ded)),
                  "slo": {str(s): score(ded, ded, s) for s in SLOS},
                  "per_model": {tr.models[m][0]: v for m, v in enumerate(model_pcts(ded))} if a.detail else None})
        elif a.grid in ("PRIMARY", "SEEDS", "SENS"):
            for seed in [int(x) for x in a.seeds.split(",")]:
                for cool in ([None] if a.cool_gbps else [float(x) for x in a.cools.split(",")]):
                    for pol in a.policies.split(","):
                        for K in [int(x) for x in a.ks.split(",")]:
                            emit(run_one(tr, cnt, ded, K, pol, cool, a.cold_s, seed, a.scenario,
                                         a.host_gib_per_card, pred, a.detail, a.cool_gbps))
        elif a.grid == "S2Z":
            span = (HELDOUT_MIN - MIN0) * 60.0
            for seed in [int(x) for x in a.seeds.split(",")]:
                for t_idle in (60.0, 300.0):
                    for t_snap in [float(x) for x in a.snaps.split(",")]:
                        out, iv, st = simulate_s2z(tr, t_snap, t_idle, seed)
                        p99, mx, mean = concurrency(iv, span, 2 * span)
                        emit({"arm": "S2Z", "scenario": a.scenario, "t_snap": t_snap, "t_idle": t_idle, "seed": seed,
                              "cards_p99": p99, "cards_max": mx, "cards_mean": mean, "stats": st,
                              "slo": {str(s): score(out, ded, s) for s in SLOS},
                              "heldout_req": int(sum(sum(c.values()) for c in out))})
        else:
            ap.print_help()
            return 2
    finally:
        if sink:
            sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
