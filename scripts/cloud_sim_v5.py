#!/usr/bin/env python3
"""cloud_sim_v5.py — settle 2 s question EXACTLY. Every request's static TTFT is recorded with its identity
(model, held-out minute, arrival rank), so an elastic TTFT is static + its own wake delay and p95 / p99 are OBSERVED, not bounded
over every pairing; HOLD-EDGE (HE) never parks a slot holding a model whose TRAIN p99 headroom is smaller than its wake cost; an
error-budget (EB) gate ("≥ 99 % of requests ≤ S, fleet-wide") stands beside the per-model (PM) gate; p is swept on v4's frozen grid.

Prereg (frozen before this file existed): (private report). On any divergence the prereg
wins. FORK BY IMPORT: evaluate, job path, static_row and overlay4 run UNCHANGED (row["v4"] must equal
rows, I-FIXV4). The three static writers (fabric_sim_v2.simulate, node_sim.dedicated_by_day,
cloud_sim.dedicated_rep) are forked IN-PROCESS by exact-once source insertions that only APPEND to a recorder; no file of
theirs is edited, and I-PERREQ proves the record equals every scored Counter.

Output contract:
  --self-test                       exit 0 + "SELF-TEST PASS n/n" (v4's GV1-GV5 + v3's GE1-GE7 + v5's GX1-GX5); exit 1 + FAIL lines.
  --premise --cell D1|D2 --root R --cache C
                                    TRAIN-ONLY read (seed 0): HE / HE-Q hold sets per (S, p), TRAIN p99 headroom, TRAIN share over S;
                                    one JSON object on stdout; exit 0. Prints no held-out number.
  --run --cell D1|D2|CP [--seeds 0,1,2] [--tag T] --root R --cache C --out F
                                    WRITES (appends) ONE JSON line to F: static row + "v4" (overlay4, unchanged) + "v5"
                                    (276 variants, exact composition, in-job I-checks); exit 0. A failed in-job I-check exits 1.
  --ifix NEW_DIR V4_DIR             read-only: I-FIXV4, I-REPLAY, I-PMONO5, I-GATE-CONSIST, I- in-job checks; one JSON on stdout;
                                    exit 0 PASS / 1 FAIL.
  --summarize DIR                   read-only: prereg §5 verdicts over DIR/*.jsonl; one JSON object on stdout; exit 0.
  Bad input (unknown cell, missing file) or a held-out manifest mismatch = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed):
  1. **Static keys are per CELL, not per request** (premise read): a fluid queue keys every request of a (model, minute) cell
     alike, except where the static run's own wake or shed splits the cell by arrival rank — identity is (model, minute, rank).
  2. **The static sims draw arrival offsets on their own subtrace** (`T.arrivals` draws per trace position; the tail run is seed 0), so a rank's static wait and its overlay delay come from different draws — the rank is the shared identity (I-SPLIT counts
     the delayed requests where that matters).
  3. **`ssh host '... &'` holds the session open** — launch jobs with `ssh -n` + `setsid -f`.
"""
import argparse, collections, inspect, json, math, os, sys, textwrap, time, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cloud_sim as CS       # noqa: E402  (not edited)
import cloud_sim_v2 as V     # noqa: E402  (not edited)
import cloud_sim_v3 as V3    # noqa: E402  (not edited)
import cloud_sim_v4 as V4    # noqa: E402  (not edited)
import fabric_sim_v2 as FV2  # noqa: E402  (not edited; forked in-process)
import node_sim as NS        # noqa: E402  (not edited; forked in-process)
import tailpool_sim as T     # noqa: E402  (not edited)

import numpy as np  # noqa: E402

HELD, NDAYS, DAY_MIN, WEEK_MIN = CS.HELD, CS.NDAYS, CS.DAY_MIN, CS.WEEK_MIN
SLOS = CS.SLOS
NREPS = V3.NREPS
BAR = V3.BAR
INF = V3.INF
P_GRID = V4.P_GRID                   # frozen in prereg §3; unchanged (prereg §2)
P_POINT = V4.P_POINT
H2D_TAIL = V4.H2D_TAIL               # 22.6 GB/s MEASURED
WARM = V4.WARM
EB_THR = 0.99                        # prereg §5: "≥ 99 % of requests ≤ S, fleet-wide", on every replicate
POL5 = ("KA-1", "KA-5", "KA-60")
DEC_ARMS5 = ("BASE", "HE", "HE-S")      # HE-S added by the pre-data addendum (TRAIN premise read)
CELLS = V4.CELLS
TIMING5 = set(V3.TIMING) | {"overlay5_s"}


def _vname(pol, arm, p, S=None):
    return f"{pol}|{arm}|p{p:g}" if S is None else f"{pol}|{arm}|S{S:g}|p{p:g}"


# variant -> (policy, p, tail h2d, hold kind None|BIG22|HE|HE-Q, S the HE set is sized for)
VARIANTS5 = {"STATIC": ("STATIC", None, None, None, None)}
for _pol in POL5:
    for _p in P_GRID:
        VARIANTS5[_vname(_pol, "BASE", _p)] = (_pol, _p, H2D_TAIL, None, None)
for _pol in POL5:
    for _S in SLOS:
        for _p in P_GRID:
            VARIANTS5[_vname(_pol, "HE", _p, _S)] = (_pol, _p, H2D_TAIL, "HE", _S)
            VARIANTS5[_vname(_pol, "HE-Q", _p, _S)] = (_pol, _p, H2D_TAIL, "HE-Q", _S)
            VARIANTS5[_vname(_pol, "HE-S", _p, _S)] = (_pol, _p, H2D_TAIL, "HE-S", _S)
for _pol in ("KA-1", "KA-5"):
    for _arm in ("HOLD", "LR53"):
        for _p in P_GRID:
            VARIANTS5[_vname(_pol, _arm, _p)] = (_pol, _p, V4.ARMS[_arm][0], V4.ARMS[_arm][1], None)
V4_SHARED = {v: ("KA-60" if v == "KA-60|BASE|p0.79" else v) for v in VARIANTS5
             if v == "STATIC" or v == "KA-60|BASE|p0.79" or (v.split("|")[0] in ("KA-1", "KA-5") and v.split("|")[1] in ("BASE", "HOLD", "LR53"))}


# ------------------------------------------------------------------------------------------------ the per-request recorder (prereg §0)
class Rec:
    """static keys by identity. const: id(day Counter) -> {minute: (n, key)}; split: id -> {minute: [n, [(lo, hi, key | [keys])]]};
    train: id(anchor Counter) -> Counter of TRAIN keys; link: id(day Counter) -> id(its TRAIN anchor). keep holds strong refs."""

    def __init__(self):
        self.const, self.split, self.train, self.link, self.keep = {}, {}, {}, {}, {}
        self.n_train_unready = 0

    def _k(self, obj):
        k = id(obj)
        if k not in self.keep:
            self.keep[k] = obj
        return k

    def seg(self, obj, mi, n, lo, hi, key):
        k, mi, n, lo, hi = self._k(obj), int(mi), int(n), int(lo), int(hi)
        if hi <= lo:
            return
        cd = self.const.setdefault(k, {})
        sd = self.split.get(k)
        if lo == 0 and hi == n and not isinstance(key, list) and mi not in cd and (sd is None or mi not in sd):
            cd[mi] = (n, int(key))
            return
        sd = self.split.setdefault(k, {})
        if mi in cd:
            n0, k0 = cd.pop(mi)
            sd[mi] = [n0, [(0, n0, k0)]]
        ent = sd.setdefault(mi, [n, []])
        if ent[0] != n:
            raise RuntimeError(f"I-PERREQ: cell minute {mi} recorded with n {ent[0]} and {n}")
        ent[1].append((lo, hi, [int(x) for x in key] if isinstance(key, list) else int(key)))

    def train_add(self, obj, key, cnt):
        self.train.setdefault(self._k(obj), collections.Counter())[int(key)] += int(cnt)

    def link_days(self, day):
        for d in range(len(day)):
            for m in range(len(day[d])):
                self.link[self._k(day[d][m])] = self._k(day[0][m])

    def link_relabeled(self, out):
        L = len(out)
        if L % (1 + NDAYS):
            return
        M0 = L // (1 + NDAYS)
        for m in range(M0):
            a = self._k(out[m])
            for d in range(NDAYS):
                self.link[self._k(out[m + M0 * (1 + d)])] = a

    def fv2_train(self, out, m, n, na, c, e, q, extra, pre, kw, trace, i, first, last, seed, base, td):
        o = out[m]
        if c < 0:
            self.train_add(o, INF, n)
            return
        if na < n:
            self.train_add(o, INF, n - na)
        if na == 0:
            return
        if e is None:
            self.train_add(o, T._key(q[c] + extra + pre + kw), na)
        else:
            ts = T.cell_times(trace, i, first, last, seed) + base
            td.append((o, ts[:na] if na < n else ts, q[c], extra, pre, e, kw))

    def fv2_train_deferred(self, o, ts, qc, ex, pre, e, kw, phi):
        if e.ready is None:
            self.n_train_unready += 1
            self.train_add(o, INF, len(ts))
            return
        for wv in np.maximum(0.0, e.ready - ts):
            wv = float(wv)
            if phi > 0.0:
                hid = min(phi * pre, wv)
                self.train_add(o, T._key(wv + qc + ex + (pre - hid) + kw), 1)
            else:
                self.train_add(o, T._key(wv + qc + ex + pre + kw), 1)

    # -- reads
    def keys_of(self, obj, mi):
        """(static keys by arrival rank, split?) for cell (obj's model-day, minute mi), or (None, False) if never recorded."""
        k = id(obj)
        c = self.const.get(k, {}).get(int(mi))
        if c is not None:
            return np.full(c[0], c[1], np.int64), False
        e = self.split.get(k, {}).get(int(mi))
        if e is None:
            return None, False
        return expand(e[0], e[1]), True

    def cells_of(self, obj):
        k = id(obj)
        out = {mi: n for mi, (n, _) in self.const.get(k, {}).items()}
        for mi, (n, _) in self.split.get(k, {}).items():
            if mi in out:
                raise RuntimeError(f"I-PERREQ: minute {mi} both const and split")
            out[mi] = n
        return out

    def counter_of(self, obj):
        k = id(obj)
        c = collections.Counter()
        for n, key in self.const.get(k, {}).values():
            c[key] += n
        for n, segs in self.split.get(k, {}).values():
            c.update(expand(n, segs).tolist())
        return c

    def train_counter(self, day_obj):
        a = self.link.get(id(day_obj))
        return None if a is None else self.train.get(a, collections.Counter())


def expand(n, segs):
    """a split cell's segments -> keys by arrival rank; overlaps and gaps raise (I-PERREQ)."""
    arr = np.full(n, -1, np.int64)
    for lo, hi, key in segs:
        if (arr[lo:hi] != -1).any():
            raise RuntimeError("I-PERREQ: overlapping segments in one cell")
        if isinstance(key, list):
            if len(key) != hi - lo:
                raise RuntimeError("I-PERREQ: segment length != its key count")
            arr[lo:hi] = np.asarray(key, np.int64)
        else:
            arr[lo:hi] = key
    if (arr == -1).any():
        raise RuntimeError("I-PERREQ: a rank of a recorded cell has no key")
    return arr


REC = None

_FV2_EDITS = [
    ("    deferred = []\n", "    deferred = []\n    _v5d, _v5td = [], []\n"),
    ("                    out[m][T.INF_KEY] += n - na\n                    week[T.INF_KEY] += n - na\n",
     "                    out[m][T.INF_KEY] += n - na\n                    week[T.INF_KEY] += n - na\n"
     "                    _V5REC.seg(out[m], mi, n, na, n, T.INF_KEY)\n"),
    ("            if mi < score_from or na == 0:\n                continue\n",
     "            if mi < score_from or na == 0:\n                if mi < score_from:\n"
     "                    _V5REC.fv2_train(out, m, n, na, c, e, q, extra, pre, kvw.get(u, 0.0), trace, i, first, last, seed, base,"
     " _v5td)\n                continue\n"),
    ("            if c < 0:\n                out[m][T.INF_KEY] += n\n                week[T.INF_KEY] += n\n                continue\n",
     "            if c < 0:\n                out[m][T.INF_KEY] += n\n                week[T.INF_KEY] += n\n"
     "                _V5REC.seg(out[m], mi, n, 0, n, T.INF_KEY)\n                continue\n"),
    ("                key = T._key(q[c] + extra + pre + kw)\n                out[m][key] += na\n                week[key] += na\n",
     "                key = T._key(q[c] + extra + pre + kw)\n                out[m][key] += na\n                week[key] += na\n"
     "                _V5REC.seg(out[m], mi, n, 0, na, key)\n"),
    ("                deferred.append((m, ts[:na] if na < n else ts, q[c], extra, pre, e, kw))\n",
     "                deferred.append((m, ts[:na] if na < n else ts, q[c], extra, pre, e, kw))\n"
     "                _v5d.append((out[m], mi, n))\n"),
    ("    for m, ts, qc, extra, pre, e, kw in deferred:\n",
     "    for _v5k, (m, ts, qc, extra, pre, e, kw) in enumerate(deferred):\n        _v5keys = []\n"),
    ("            out[m][key] += 1\n            week[key] += 1\n",
     "            out[m][key] += 1\n            week[key] += 1\n            _v5keys.append(key)\n"
     "        _V5REC.seg(_v5d[_v5k][0], _v5d[_v5k][1], _v5d[_v5k][2], 0, len(_v5keys), _v5keys)\n"),
    ("    held_svc = [",
     "    for _o, _ts, _qc, _ex, _pre, _e, _kw in _v5td:\n        _V5REC.fv2_train_deferred(_o, _ts, _qc, _ex, _pre, _e, _kw, phi)\n"
     "    _V5REC.link_relabeled(out)\n    held_svc = ["),
]
_DED_EDITS = [
    ("            if mi >= score_from_min:\n                out[m][T.INF_KEY] += n\n"
     "                day[(mi - HELD) // DAY_MIN][m][T.INF_KEY] += n\n            continue\n",
     "            if mi >= score_from_min:\n                out[m][T.INF_KEY] += n\n"
     "                day[(mi - HELD) // DAY_MIN][m][T.INF_KEY] += n\n"
     "                _V5REC.seg(day[(mi - HELD) // DAY_MIN][m], mi, n, 0, n, T.INF_KEY)\n            else:\n"
     "                _V5REC.train_add(day[0][m], T.INF_KEY, n)\n            continue\n"),
    ("        if mi >= score_from_min:\n            k_ = T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm + kw)\n"
     "            out[m][k_] += n\n            day[(mi - HELD) // DAY_MIN][m][k_] += n\n    return out, day\n",
     "        if mi >= score_from_min:\n            k_ = T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm + kw)\n"
     "            out[m][k_] += n\n            day[(mi - HELD) // DAY_MIN][m][k_] += n\n"
     "            _V5REC.seg(day[(mi - HELD) // DAY_MIN][m], mi, n, 0, n, k_)\n        else:\n"
     "            _V5REC.train_add(day[0][m], T._key(q + (trace.it[i] / n) * (ab / ACTIVE_REF_B) / Cm + kw), n)\n"
     "    _V5REC.link_days(day)\n    return out, day\n"),
]


def fork(fn, edits):
    """fn's source with each (old, new) applied exactly once, compiled and bound to fn's OWN module globals (prereg §0)."""
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new in edits:
        k = src.count(old)
        if k != 1:
            raise RuntimeError(f"PATCH {fn.__module__}.{fn.__name__}: anchor found {k} times: {old[:70]!r}")
        src = src.replace(old, new)
    tmp = {}
    exec(compile(src, f"<v5-fork {fn.__module__}.{fn.__name__}>", "exec"), dict(fn.__globals__), tmp)
    g = types.FunctionType(tmp[fn.__name__].__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    g.__kwdefaults__ = fn.__kwdefaults__
    return g


def install():
    """swap the three static writers for their recording forks; -> the recorder. Idempotent per process."""
    global REC
    if REC is not None:
        return REC
    REC = Rec()
    for mod in (FV2, NS, CS):
        mod._V5REC = REC
    FV2.simulate = fork(FV2.simulate, _FV2_EDITS)
    NS.dedicated_by_day = fork(NS.dedicated_by_day, _DED_EDITS)
    CS.dedicated_rep = fork(CS.dedicated_rep, _DED_EDITS)
    return REC


# ------------------------------------------------------------------------------------------------ exact composition (prereg §4)
def pair_cell(s, dc):
    """one cell: static keys by rank s, delays by rank dc (cs) -> [(static, elastic)] for the delayed ranks (INF stays INF)."""
    s = np.asarray(s, np.int64)
    dc = np.asarray(dc, np.int64)
    if len(s) != len(dc):
        raise RuntimeError(f"I-PERREQ: cell has {len(s)} static keys and {len(dc)} delays")
    e = np.where(s >= INF, INF, s + dc)
    k = dc > 0
    return list(zip(s[k].tolist(), e[k].tolist()))


def compose_exact(days, moves):
    """day Counters + per-day moves [(static, elastic)] -> (cstats-shaped exact stats, exact p99 list) per LODO replicate."""
    nd = []
    for d in range(NDAYS):
        c = collections.Counter(days[d])
        for s, e in moves[d]:
            c[s] -= 1
            if c[s] < 0:
                raise RuntimeError("I-PERREQ: a move takes a key the day Counter does not hold")
            c[e] += 1
        nd.append(+c)
    nw = sum(nd, collections.Counter())
    return CS.cstats(nw, nd), V4.static_pq(nw, nd, 0.99)


def lodo_max(day_vals):
    return [max([day_vals[x] for x in range(NDAYS) if x != d] + [0]) for d in range(NDAYS)]


def lodo_sum(day_vals):
    t = sum(day_vals)
    return [int(t - day_vals[d]) for d in range(NDAYS)]


# ------------------------------------------------------------------------------------------------ HOLD-EDGE (prereg §3)
def wake_costs(sl, m, gb_tail):
    """(own page-in, serial page-in of every engine on the slot) for model m on slot sl."""
    gb = gb_tail if sl["kind"] == "tail" else sl["gbps"]
    own = V3.page_in(sl["units"][sl["unit_of"][m]], gb)
    ser = sum(V3.page_in(w, gb) for w in sl["units"].values())
    return own, ser


def he_hold(slots, ptrain, S, p, kind, gb_tail=H2D_TAIL):
    """HE / HE-Q(S, p): slot held iff some model m on it has TRAIN n >= 1 and S - p99_TRAIN(m) < wake cost (strict).
    HE-S (addendum): the same with own cost, protecting only models with 0 <= headroom (a model already over S has none to keep)."""
    out = []
    for sl in slots:
        h = False
        for m in sl["models"]:
            nt, p99 = ptrain.get(m, (0, None))
            if nt < 1 or p99 is None:
                continue
            own, ser = wake_costs(sl, m, gb_tail)
            cost = p + (ser if kind == "HE-Q" else own)
            if S - p99 < cost and (kind != "HE-S" or S - p99 >= 0):
                h = True
                break
        out.append(h)
    return out


def he_check(slots, ptrain, S, p, kind, hset, gb_tail=H2D_TAIL):
    """I-HE, written apart from he_hold: no PARKED slot hosts a model violating the rule. -> violations."""
    bad = 0
    for sl, f in zip(slots, hset):
        if f:
            continue
        for m in sl["models"]:
            nt, p99 = ptrain.get(m, (0, None))
            if nt >= 1 and p99 is not None:
                gb = gb_tail if sl["kind"] == "tail" else sl["gbps"]
                c = p + (sum(V3.page_in(w, gb) for w in sl["units"].values()) if kind == "HE-Q" else
                         V3.page_in(sl["units"][sl["unit_of"][m]], gb))
                if not (S - p99 >= c or (kind == "HE-S" and S - p99 < 0)):
                    bad += 1
    return bad


def train_p99(rec, sc, hosted):
    """model -> (TRAIN requests, nearest-rank TRAIN p99 s | inf | None) from the static run that scores it."""
    out = {}
    for m in hosted:
        tc = rec.train_counter(sc[m][1][0])
        if tc is None:
            raise RuntimeError(f"I-PERREQ: model {m} has no TRAIN link")
        n = int(sum(tc.values()))
        out[m] = (n, T._pct(tc, 0.99) if n else None)
    return out


# ------------------------------------------------------------------------------------------------ gates (prereg §5)
def eb_share(ok, n):
    tot = sum(n)
    return None if tot == 0 else sum(ok) / float(tot)


def pm_gate(p95_pass, mww, p99_pass, S):
    if not (len(p95_pass) == len(mww) == len(p99_pass) == NREPS):
        raise ValueError(f"I-REPS: {len(p95_pass)} / {len(mww)} / {len(p99_pass)} PM replicates, need {NREPS}")
    return "PM-HOLDS" if all(p95_pass) and all(x <= S for x in mww) and all(p99_pass) else "PM-FAILS"


def eb_gate(shares, thr=EB_THR):
    if len(shares) != NREPS:
        raise ValueError(f"I-REPS: {len(shares)} EB replicates, need {NREPS}")
    return "EB-HOLDS" if all(x is not None and x >= thr for x in shares) else "EB-FAILS"


def eb_reading(static_eb, static_ebe):
    """prereg §5 vacuity rule: which EB reading decides at (cell, S)."""
    if static_eb == "EB-HOLDS":
        return "EB"
    if static_ebe == "EB-HOLDS":
        return "EB-ELIG"
    return "NONE"


def both(pm, eb, reading):
    ebh = True if reading == "NONE" else eb == "EB-HOLDS"
    return pm == "PM-HOLDS" and ebh


def p_meet(rows, grid=P_GRID):
    """rows: p -> dict(pm, eb, bar) (eb already per the reading, True/False). -> P_MEET_{BOTH,PM,EB} + P_BREAK with the part."""
    out = {}
    for nm, need in (("both", ("pm", "eb")), ("pm", ("pm",)), ("eb", ("eb",))):
        k = 0
        while k < len(grid):
            r = rows[grid[k]]
            if not (r["bar"] == "UNDER-BAR" and all(r[x] for x in need)):
                break
            k += 1
        brk = None
        if k < len(grid):
            r = rows[grid[k]]
            brk = [grid[k], [x for x in (("BAR",) if r["bar"] != "UNDER-BAR" else ()) + tuple(y.upper() for y in need if not r[y])]]
        out[nm] = dict(p_meet=grid[k - 1] if k else None, p_break=brk)
    return out


# ------------------------------------------------------------------------------------------------ the v5 overlay
def run_variant(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs, checks, pol_cache):
    """v4's wake loop for one variant on one seed (copied verbatim in semantics; I-DELAY proves it against row["v4"])."""
    busy_d = np.zeros(NDAYS, np.int64)
    held_d = np.zeros(NDAYS, np.int64)
    wake_d = np.zeros(NDAYS, np.int64)
    held_tot = np.zeros(WEEK_MIN, np.int64)
    cell_delay = {}
    for s_i, (sl, b) in enumerate(zip(slots, bs)):
        key = (pol, s_i, bool(hset[s_i]))
        if key not in pol_cache:
            if not hset[s_i] and not sl["host_ok"] and pol != "STATIC":
                checks["I_HOST_parked_blocked"] += 1
                pol_cache[key] = V3.held_static(b)
            else:
                pol_cache[key] = V4.slot_policy(pol, b, khs[s_i], bool(hset[s_i]))
        held, wake = pol_cache[key]
        if (b & ~held).any():
            checks["I_HELD"] = False
            raise RuntimeError(f"I-HELD: busy minute not held ({pol}, slot {s_i})")
        g = sl["g"]
        held_tot += g * held.astype(np.int64)
        busy_d += g * b.reshape(NDAYS, DAY_MIN).sum(axis=1)
        held_d += g * held.reshape(NDAYS, DAY_MIN).sum(axis=1)
        wk_t = np.flatnonzero(wake)
        wake_d += np.bincount(wk_t // DAY_MIN, minlength=NDAYS)[:NDAYS]
        if not len(wk_t):
            continue
        mins, ids = slot_cells[s_i]
        kind = "tail" if sl["kind"] == "tail" else "trigger"
        gb = (h2d if sl["kind"] == "tail" else sl["gbps"]) if h2d is not None else None
        for t in wk_t:
            te = V3.window_end(pol, held, int(t))
            a_, z_ = np.searchsorted(mins, t, "left"), np.searchsorted(mins, te, "right")
            cl = [(int(mins[q]), float(first[ids[q]]), int(tr.mid[ids[q]]), sl["unit_of"][int(tr.mid[ids[q]])], ids[q])
                  for q in range(a_, z_)]
            if not cl or cl[0][0] != t:
                checks["I_WAKE"] = False
            for cid, m, mi, dd in V3.window_delays(kind, cl, int(t), p if p is not None else 0.0,
                                                   (lambda u, _sl=sl, _gb=gb: V3.page_in(_sl["units"][u], _gb)), False, offs):
                prev = cell_delay.get(cid)
                cell_delay[cid] = dd if prev is None else np.maximum(prev, dd)
    return busy_d, held_d, wake_d, held_tot, cell_delay


def _le(a, b):
    """a <= b for nearest-rank values (None = empty replicate, both None or skip)."""
    if a is None or b is None:
        return a is None and b is None
    return a <= b + 1e-12


def overlay5(wd, row, rec, snap, r, seeds, R, variants=None):
    variants = variants or list(VARIANTS5)
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
    V4v = row["v4"]["variants"]
    sk = {str(S): T._key(S) for S in SLOS}
    res = {v: dict(idle_el=[], held_gpuh=[], peak_held=[], busy_day=[], held_day=[], wakes=[], wake_hit=[], mww=[], n_split=[],
                   hold_slots=[], hold_gpus=[],
                   x={str(S): dict(p95_pass=[], p99_pass=[], p99_scope=[], eb=[], eb_elig=[], mct_scoped=[], mct_all=[],
                                   over_by_wake=[]) for S in SLOS}, edge={str(S): {} for S in SLOS})
           for v in variants}
    checks = dict(I_PERREQ=True, I_CELLUNIQ=True, I_DELAY=True, I_INSIDE=True, I_EXACT0=True, I_HE=True, I_HELD=True, I_WAKE=True,
                  I_HOST_parked_blocked=0, inside_compared=0, inside_violations=0, delay_compared=0, n_train_unready=0, peri_cells=0,
                  split_cells=0)
    keyc = tr.mid[kk].astype(np.int64) * (WEEK_MIN + HELD + 10 ** 6) + tr.minute[kk].astype(np.int64)
    if len(np.unique(keyc)) != len(kk):
        checks["I_CELLUNIQ"] = False
        raise RuntimeError("I-CELLUNIQ: (model, minute) is not unique over held-out cells")
    he_info = {}
    stat_eb = {}
    for si, seed in enumerate(seeds):
        fi = rec["fi"][si]
        slots = V3.build_slots(wd, fi)
        big22 = V4.big_set(slots, H2D_TAIL)
        bs = [V3.slot_busy(sl, fi) for sl in slots]
        tot = np.zeros(WEEK_MIN, np.int64)
        for sl, b in zip(slots, bs):
            tot += sl["g"] * b.astype(np.int64)
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
        # I-PERREQ: the record reproduces every scored day Counter and covers every held-out cell once
        for m in hosted:
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
            z = [[] for _ in range(NDAYS)]
            ex0, p0 = compose_exact(sc[m][1], z)
            s0 = st_stats[m]
            if ex0["n"] != s0["n"] or ex0["p95"] != s0["p95"] or ex0["ok"] != s0["ok"] or p0 != st_p99[m]:
                checks["I_EXACT0"] = False
                raise RuntimeError("I-EXACT0: zero-move exact composition != the static stats")
        ptr = train_p99(R, sc, hosted)
        checks["n_train_unready"] = R.n_train_unready
        hsets = {}
        for v in variants:
            hk, hS, hp = VARIANTS5[v][3], VARIANTS5[v][4], VARIANTS5[v][1]
            if hk in ("HE", "HE-Q", "HE-S") and (hk, hS, hp) not in hsets:
                hs = he_hold(slots, ptr, hS, hp, hk)
                if he_check(slots, ptr, hS, hp, hk, hs):
                    checks["I_HE"] = False
                    raise RuntimeError(f"I-HE: {hk} S{hS} p{hp} parks a slot that violates the rule")
                hsets[(hk, hS, hp)] = hs
                he_info.setdefault(f"{hk}|S{hS:g}|p{hp:g}", []).append(
                    dict(seed=seed, slots=int(sum(hs)), gpus=int(sum(sl["g"] for sl, f in zip(slots, hs) if f)),
                         kinds=dict(collections.Counter(sl["kind"] for sl, f in zip(slots, hs) if f))))
        rep_req = [int(sum(int(rd[:, m].sum() - rd[x, m]) for m in hosted)) for x in range(NDAYS)]
        pol_cache = {}
        for v in variants:
            pol, p, h2d, hk, hS = VARIANTS5[v]
            hset = (big22 if hk == "BIG22" else hsets[(hk, hS, p)] if hk in ("HE", "HE-Q", "HE-S") else [False] * len(slots))
            busy_d, held_d, wake_d, held_tot, cell_delay = run_variant(pol, p, h2d, hset, slots, bs, slot_cells, khs, tr, first, offs,
                                                                       checks, pol_cache)
            o = res[v]
            o["idle_el"].extend(V3.lodo(busy_d, held_d))
            H_ = held_d.sum()
            o["held_gpuh"].extend([(H_ - held_d[d]) / 60.0 for d in range(NDAYS)])
            o["wakes"].extend([int(wake_d.sum() - wake_d[d]) for d in range(NDAYS)])
            V4.peak_check(held_tot, tot)
            o["peak_held"].extend(V4.peak_reps(held_tot, WARM))
            o["busy_day"].append([int(x) for x in busy_d])
            o["held_day"].append([int(x) for x in held_d])
            o["hold_slots"].append(int(sum(hset)))
            o["hold_gpus"].append(int(sum(sl["g"] for sl, f in zip(slots, hset) if f)))
            # exact pairing by identity (model, minute, rank)
            moves = collections.defaultdict(lambda: [[] for _ in range(NDAYS)])
            dl = collections.defaultdict(lambda: [[] for _ in range(NDAYS)])
            hit_d = np.zeros(NDAYS, np.int64)
            dmax = np.zeros(NDAYS, np.int64)
            nspl = np.zeros(NDAYS, np.int64)
            for cid, dd in cell_delay.items():
                dc = np.round(dd * 100.0).astype(np.int64)
                if not (dc > 0).any():
                    continue
                m, mi = int(tr.mid[cid]), int(tr.minute[cid])
                d = (mi - HELD) // DAY_MIN
                if m not in sc:
                    raise RuntimeError(f"a delay on a model that is not hosted: {m}")
                s_, split = R.keys_of(sc[m][1][d], mi)
                if s_ is None:
                    checks["I_PERREQ"] = False
                    raise RuntimeError(f"I-PERREQ: delayed cell {cid} has no static record")
                moves[m][d].extend(pair_cell(s_, dc))
                pos = dc[dc > 0]
                dl[m][d].extend(pos.tolist())
                hit_d[d] += len(pos)
                dmax[d] = max(dmax[d], int(pos.max()))
                if split:
                    nspl[d] += len(pos)
            o["wake_hit"].extend(lodo_sum(hit_d))
            o["mww"].extend(V4.mww_reps(dmax))
            o["n_split"].extend(lodo_sum(nspl))
            ost = dict(st_stats)
            ex99 = {}
            for m in moves:
                ost[m], ex99[m] = compose_exact(sc[m][1], moves[m])
                # I-INSIDE: the exact values sit inside v3 / v4's bounds over every pairing
                dlm = [np.asarray(x, np.int64) for x in dl[m]]
                bd = V3.compose(sc[m][0], sc[m][1], dlm)
                b99 = V4.compose_p99(sc[m][0], sc[m][1], dlm)
                for d in range(NDAYS):
                    checks["inside_compared"] += 1
                    ok = (_le(bd["best"]["p95"][d], ost[m]["p95"][d]) and _le(ost[m]["p95"][d], bd["worst"]["p95"][d]) and
                          _le(b99["best"][d], ex99[m][d]) and _le(ex99[m][d], b99["worst"][d]) and
                          all(bd["worst"]["ok"][s][d] <= ost[m]["ok"][s][d] <= bd["best"]["ok"][s][d] for s in sk))
                    if not ok:
                        checks["inside_violations"] += 1
                        checks["I_INSIDE"] = False
                        raise RuntimeError(f"I-INSIDE: {v} model {tr.models[m][0][:8]} replicate {d}: exact outside the bounds")
                for x in SLOS:
                    s = str(x)
                    for d in range(NDAYS):
                        sp = st_p99[m][d]
                        if ost[m]["n"][d] >= V4.P99_FLOOR and sp is not None and sp <= x and b99["worst"][d] > x:
                            k8 = tr.models[m][0][:8]
                            ed = o["edge"][s].setdefault(k8, [0.0, 0.0, 9e9])
                            ed[0] = max(ed[0], ex99[m][d] if ex99[m][d] != math.inf else 1e9)
                            ed[1] = max(ed[1], b99["worst"][d] if b99["worst"][d] != math.inf else 1e9)
                            ed[2] = min(ed[2], b99["best"][d] if b99["best"][d] != math.inf else 1e9)
            for x in SLOS:
                s = str(x)
                xs = o["x"][s]
                sr = CS.score_reps(hosted, ost, rst, allm, x)
                xs["p95_pass"].extend(sr["pass"])
                okd = np.zeros(NDAYS, np.int64)
                mct_s = np.zeros(NDAYS, np.int64)
                mct_a = np.zeros(NDAYS, np.int64)
                over_d = np.zeros(NDAYS, np.int64)
                for m, per in moves.items():
                    for d in range(NDAYS):
                        for s_k, e_k in per[d]:
                            mct_a[d] = max(mct_a[d], min(e_k, 10 ** 9))
                            if s_k <= sk[s]:
                                mct_s[d] = max(mct_s[d], min(e_k, 10 ** 9))
                                if e_k > sk[s]:
                                    over_d[d] += 1
                for d in range(NDAYS):
                    mods = [(ost[m]["n"][d], st_p99[m][d], ex99[m][d] if m in ex99 else st_p99[m][d],
                             ex99[m][d] if m in ex99 else st_p99[m][d]) for m in hosted]
                    pw, _, ns, _ = V4.p99_rep_pass(mods, x)
                    xs["p99_pass"].append(pw)
                    xs["p99_scope"].append(ns)
                    xs["eb"].append(eb_share([ost[m]["ok"][s][d] for m in hosted], [ost[m]["n"][d] for m in hosted]))
                    fe = [m for m in hosted if rst[m]["n"][d] >= 20 and rst[m]["p95"][d] <= x]
                    xs["eb_elig"].append(eb_share([ost[m]["ok"][s][d] for m in fe], [ost[m]["n"][d] for m in fe]))
                xs["mct_scoped"].extend([y / 100.0 for y in lodo_max(mct_s)])
                xs["mct_all"].extend([y / 100.0 for y in lodo_max(mct_a)])
                xs["over_by_wake"].extend(lodo_sum(over_d))
        if si == 0:
            stat_eb = {str(x): eb_share([st_stats[m]["ok"][str(x)][0] for m in hosted], [st_stats[m]["n"][0] for m in hosted])
                       for x in SLOS}
    out = {}
    for v in variants:
        o = res[v]
        o["idle_el"] = [None if x is None else round(x, 6) for x in o["idle_el"]]
        out[v] = o
        # I-DELAY: v5's own delay pass = v4's for every shared variant
        w4 = V4_SHARED.get(v)
        if w4 is not None and w4 in V4v:
            checks["delay_compared"] += 1
            a = V4v[w4]
            for k in ("idle_el", "held_gpuh", "peak_held", "mww", "wake_hit", "wakes"):
                if V3._strip(o[k]) != V3._strip(a[k]):
                    checks["I_DELAY"] = False
                    raise RuntimeError(f"I-DELAY: {v}.{k} != v4's {w4}.{k}")
    return dict(variants=out, checks=checks, he_info=he_info, static_eb_rep0=stat_eb, overlay5_s=round(time.time() - t_start, 1),
                consts=dict(EB_THR=EB_THR, P_GRID=list(P_GRID), POL5=list(POL5), H2D_TAIL=H2D_TAIL, WARM=WARM, n_variants=len(variants)))


def run_cell(cell, seeds, tag, root, cache, out):
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell}")
    R = install()
    wd, row, rec, snap, r = V3.static_row(cell, seeds, tag, root, cache)
    if abs(wd.tph["h2d"] - H2D_TAIL) > 1e-9:
        raise RuntimeError(f"prereg §2: the cloud sim's tail h2d is {wd.tph['h2d']}, not {H2D_TAIL}")
    row["cell"] = cell
    row["v4"] = V4.overlay4(wd, row, rec, snap, r, seeds)
    row["v5"] = overlay5(wd, row, rec, snap, r, seeds, R)
    with open(out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


def premise(cell, root, cache):
    """TRAIN-only read (seed 0): HE / HE-Q hold sets, TRAIN p99 headroom, TRAIN share over S. Prints no held-out number."""
    if cell not in ("D1", "D2"):
        raise ValueError("--premise takes D1 or D2")
    R = install()
    wd, row, rec, snap, r = V3.static_row(cell, (0,), "premise", root, cache)
    cx = wd.cx
    hosted = [wd.am] + [cx["cid_m"][c] for c in snap["admitted"]]
    sc = V3.static_counters(wd, rec, snap, r, 0)
    ptr = train_p99(R, sc, hosted)
    slots = V3.build_slots(wd, rec["fi"][0])
    out = dict(cell=cell, seed=0, hosted=len(hosted), slots=len(slots), gpus=int(sum(sl["g"] for sl in slots)),
               train_n0_models=sum(1 for m in hosted if ptr[m][0] == 0), n_train_unready=R.n_train_unready, hold={}, train_over_s={},
               headroom={})
    for S in SLOS:
        tot = over = 0
        for m in hosted:
            tc = R.train_counter(sc[m][1][0])
            tot += sum(tc.values())
            over += sum(c for k_, c in tc.items() if k_ > T._key(S))
        out["train_over_s"][f"S{S:g}"] = dict(requests=int(tot), over=int(over), share_over=round(over / tot, 6) if tot else None)
        hr = sorted(S - ptr[m][1] for m in hosted if ptr[m][0] >= 1 and ptr[m][1] is not None)
        out["headroom"][f"S{S:g}"] = dict(n=len(hr), lt_0=sum(1 for h in hr if h < 0), lt_0p5=sum(1 for h in hr if h < 0.5),
                                          lt_1=sum(1 for h in hr if h < 1.0), lt_2=sum(1 for h in hr if h < 2.0),
                                          min=round(hr[0], 3) if hr else None)
        for kind in ("HE", "HE-Q", "HE-S"):
            for p in P_GRID:
                hs = he_hold(slots, ptr, S, p, kind)
                out["hold"][f"{kind}|S{S:g}|p{p:g}"] = dict(slots=int(sum(hs)), gpus=int(sum(sl["g"] for sl, f in zip(slots, hs) if f)),
                                                           kinds=dict(collections.Counter(sl["kind"] for sl, f in zip(slots, hs) if f)))
    out["big22"] = dict(slots=int(sum(V4.big_set(slots, H2D_TAIL))))
    print(json.dumps(out))
    return 0


# ------------------------------------------------------------------------------------------------ instrument checks over rows
def _strip5(x):
    if isinstance(x, dict):
        return {k: _strip5(v) for k, v in x.items() if k not in TIMING5}
    if isinstance(x, list):
        return [_strip5(v) for v in x]
    return x


def mono5(row):
    """I-PMONO5: fixed-hold-set arms (BASE, v4 HOLD / LR53): exact PM pass counts and every EB share non-increasing in p."""
    X = row["v5"]["variants"]
    bad = []
    for pol in POL5:
        for arm in ("BASE", "HOLD", "LR53"):
            names = [_vname(pol, arm, p) for p in P_GRID]
            if names[0] not in X:
                continue
            for a, b in zip(names, names[1:]):
                if any(y < x for x, y in zip(X[a]["mww"], X[b]["mww"])):
                    bad.append(f"mww {a}->{b}")
                for s in X[a]["x"]:
                    xa, xb = X[a]["x"][s], X[b]["x"][s]
                    for k in ("p95_pass", "p99_pass"):
                        if any(y and not x for x, y in zip(xa[k], xb[k])):
                            bad.append(f"{k} {a}->{b} S{s}")
                    for k in ("eb", "eb_elig"):
                        if any(x is not None and y is not None and y > x + 1e-12 for x, y in zip(xa[k], xb[k])):
                            bad.append(f"{k} {a}->{b} S{s}")
    return bad


def consist(row):
    """I-GATE-CONSIST: v4's bounded GATE vs the exact PM gate on shared names (HOLDS => HOLDS, FAILS => FAILS). -> (bad, table)."""
    X, V4v = row["v5"]["variants"], row["v4"]["variants"]
    bad, tab = [], collections.Counter()
    for v, w4 in V4_SHARED.items():
        if v == "STATIC" or w4 not in V4v or v not in X:
            continue
        for x in SLOS:
            s = str(x)
            g4 = V4v[w4]["gate4"][s]
            pm = pm_gate(X[v]["x"][s]["p95_pass"], X[v]["mww"], X[v]["x"][s]["p99_pass"], x)
            tab[f"S{x:g}|{g4}->{pm}"] += 1
            if (g4 == "GATE-HOLDS" and pm != "PM-HOLDS") or (g4 == "GATE-FAILS" and pm != "PM-FAILS"):
                bad.append(f"{v} S{s}: {g4} -> {pm}")
    return bad, dict(tab)


def ifix(new_dir, v4_dir):
    rows = V3._rows(new_dir)
    by = {(r["tag"], r["cell"]): r for r in rows}
    olds = V3._rows(v4_dir)
    res, ok = {}, True
    for o in olds:
        key = (o["tag"], o["cell"])
        n = by.get(key)
        if n is None:
            res[f"I_FIXV4_{key[0]}_{key[1]}"] = "MISSING"
            ok = False
            continue
        dfs = [k for k in sorted(set(o) - TIMING5) if _strip5(n.get(k)) != _strip5(o.get(k))]
        if "v4" in dfs:
            dfs.remove("v4")
            nv, ov = _strip5(n["v4"]), _strip5(o["v4"])
            dfs += [f"v4.{k}" for k in sorted(set(nv) | set(ov)) if nv.get(k) != ov.get(k)]
            if any(d == "v4.variants" for d in dfs):
                dfs += [f"v4.variants.{vv}" for vv in sorted(set(nv["variants"]) | set(ov["variants"]))
                        if nv["variants"].get(vv) != ov["variants"].get(vv)][:8]
        res[f"I_FIXV4_{key[0]}_{key[1]}"] = dict(n_diffs=len(dfs), diffs=dfs[:10])
        ok = ok and not dfs
    res["I_FIXV4_rows"] = len(olds)
    ok = ok and len(olds) == 9
    for (tag, cell), r in sorted(by.items()):
        ch = r["v5"]["checks"]
        res[f"checks_{tag}_{cell}"] = ch
        ok = ok and all(ch[k] for k in ("I_PERREQ", "I_CELLUNIQ", "I_DELAY", "I_INSIDE", "I_EXACT0", "I_HE", "I_HELD", "I_WAKE"))
        ok = ok and ch["inside_violations"] == 0 and ch["inside_compared"] > 0 and ch["delay_compared"] == len(V4_SHARED)
        bad = mono5(r)
        res[f"mono5_{tag}_{cell}"] = bad[:8]
        cb, ct = consist(r)
        res[f"consist_{tag}_{cell}"] = dict(bad=cb[:8], table=ct)
        ok = ok and not bad and not cb
    d1 = by.get(("main", "D1"))
    if d1 is None:
        ok = False
        res["I_ECOLD_TAIL"] = "MISSING"
    else:
        ec = d1["v4"]["variants"]["E-COLD"]["tail_gate"]
        res["I_ECOLD_TAIL"] = ec
        ok = ok and all(v == "TAIL-FAILS" for v in ec.values())
    for cell in ("D1", "D2"):
        a = by.get(("main", cell))
        for t in ("rep_b", "rep_c"):
            b = by.get((t, cell))
            if a is None or b is None:
                res[f"I_REPLAY_{cell}_{t}"] = "MISSING"
                ok = False
                continue
            sa, sb = _strip5(dict(a, tag=None)), _strip5(dict(b, tag=None))
            diff = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
            res[f"I_REPLAY_{cell}_{t}"] = dict(diffs=diff[:5])
            ok = ok and not diff
    res["pass"] = bool(ok)
    return res


# ------------------------------------------------------------------------------------------------ summarize (prereg §5)
def _rng(v):
    return V3._rng(v)


def gates_of(o, x):
    s = str(x)
    xs = o["x"][s]
    return dict(pm=pm_gate(xs["p95_pass"], o["mww"], xs["p99_pass"], x), eb=eb_gate(xs["eb"]), ebe=eb_gate(xs["eb_elig"]),
                bar=V3.bar_v(o["idle_el"]), idle=_rng(o["idle_el"]), eb_rng=_rng(xs["eb"]), ebe_rng=_rng(xs["eb_elig"]),
                mww=_rng(o["mww"]), mct_scoped=_rng(xs["mct_scoped"]), over_by_wake=_rng(xs["over_by_wake"]),
                n_split=_rng(o["n_split"]), wake_hit=_rng(o["wake_hit"]), hold_gpus=_rng(o["hold_gpus"]))


def cell_v5(row):
    X = row["v5"]["variants"]
    out = dict(static={}, reading={}, q1={}, qmain={}, q2={}, q3={}, q4={}, desc={})
    for x in SLOS:
        s = f"S{x:g}"
        st = gates_of(X["STATIC"], x)
        out["static"][s] = st
        rd_ = eb_reading(st["eb"], st["ebe"])
        out["reading"][s] = rd_
        st_both = both(st["pm"], st["ebe"] if rd_ == "EB-ELIG" else st["eb"], rd_)
        best = {"both": None, "pm": None, "eb": None}
        for pol in POL5:
            for arm in ("BASE", "HE", "HE-S", "HE-Q"):
                rows = {}
                per = {}
                for p in P_GRID:
                    v = _vname(pol, arm, p, None if arm == "BASE" else x)
                    g = gates_of(X[v], x)
                    ebv = g["ebe"] if rd_ == "EB-ELIG" else g["eb"]
                    rows[p] = dict(pm=g["pm"] == "PM-HOLDS", eb=(True if rd_ == "NONE" else ebv == "EB-HOLDS"), bar=g["bar"])
                    per[f"{p:g}"] = dict(pm=g["pm"], eb=g["eb"], ebe=g["ebe"], bar=g["bar"], idle=g["idle"], eb_rng=g["eb_rng"],
                                         ebe_rng=g["ebe_rng"], mww=g["mww"], mct_scoped=g["mct_scoped"], hold_gpus=g["hold_gpus"],
                                         over_by_wake=g["over_by_wake"], n_split=g["n_split"])
                pm_ = p_meet(rows)
                tgt = out["q1"] if arm in DEC_ARMS5 else out["desc"]
                sent = (f"{row['cell']} {s} {pol}|{arm}: MEETS both gates under the bar for p <= "
                        f"{pm_['both']['p_meet'] if pm_['both']['p_meet'] is not None else 'NONE'} (grid); PM-only <= "
                        f"{pm_['pm']['p_meet']}; EB-only <= {pm_['eb']['p_meet']}; at P_BREAK: {pm_['both']['p_break']}")
                tgt[f"{s}|{pol}|{arm}"] = dict(**{f"p_meet_{k}": v_["p_meet"] for k, v_ in pm_.items()},
                                              **{f"p_break_{k}": v_["p_break"] for k, v_ in pm_.items()}, by_p=per, sentence=sent)
                if arm in DEC_ARMS5:
                    for k in best:
                        pmv = pm_[k]["p_meet"]
                        if pmv is not None and (best[k] is None or pmv > best[k][0]):
                            best[k] = (pmv, f"{pol}|{arm}")
                    # peak busy is policy-independent: read it off row["v4"]'s STATIC (overlay5 does not store it; summarize-only fix)
                    out["q4"][f"{s}|{pol}|{arm}"] = V4.rented(dict(X[_vname(pol, arm, P_POINT, None if arm == "BASE" else x)],
                                                                   peak_busy=row["v4"]["variants"]["STATIC"]["peak_busy"]), row["N"])
        ans = {}
        for k in ("both", "pm", "eb"):
            if best[k] is not None:
                ans[k] = f"YES for p <= {best[k][0]:g} ({best[k][1]})"
            else:
                need = {"both": st_both, "pm": st["pm"] == "PM-HOLDS",
                        "eb": True if rd_ == "NONE" else (st["ebe"] if rd_ == "EB-ELIG" else st["eb"]) == "EB-HOLDS"}[k]
                ans[k] = "NO at any grid p; " + ("HELD-HOME-ONLY" if need else "NOT MET EVEN HELD-HOME")
        out["qmain"][s] = dict(answer=ans, static_pm=st["pm"], static_eb=st["eb"], static_ebe=st["ebe"], reading=rd_)
        for pp in (0.0, P_POINT):
            cands = []
            for v, spec in VARIANTS5.items():
                if spec[1] != pp or (spec[4] is not None and spec[4] != x):
                    continue
                g = gates_of(X[v], x)
                ebv = g["ebe"] if rd_ == "EB-ELIG" else g["eb"]
                if both(g["pm"], ebv, rd_):
                    cands.append((max(X[v]["idle_el"]), min(X[v]["idle_el"]), v))
            c = min(cands) if cands else None
            out["q2"][f"{s}|p{pp:g}"] = ("NO VARIANT HOLDS BOTH GATES" if c is None else
                                         dict(closest=c[2], idle=[round(c[1], 4), round(c[0], 4)],
                                              decides=c[2].split("|")[1] in DEC_ARMS5 and c[2].split("|")[0] in POL5))
        cb, ct = consist(row)
        edges = {}
        for v in V4_SHARED:
            if v in X and v != "STATIC":
                for k8, e in X[v]["edge"][str(x)].items():
                    a = edges.setdefault(k8, [0.0, 0.0, 9e9, 0])
                    a[0], a[1], a[2], a[3] = max(a[0], e[0]), max(a[1], e[1]), min(a[2], e[2]), a[3] + 1
        out["q3"][s] = dict(transitions={k: v for k, v in ct.items() if k.startswith(s + "|")},
                            edge_models={k: dict(exact_max=round(e[0], 2), worst_max=round(e[1], 2), best_min=round(e[2], 2),
                                                 variants=e[3]) for k, e in sorted(edges.items())})
    out["he_info_seed0"] = {k: v[0] for k, v in row["v5"]["he_info"].items()}
    return out


def verdict_key5(c):
    k = {}
    for a, x in c["q1"].items():
        k[f"q1|{a}"] = [x["p_meet_both"], x["p_meet_pm"], x["p_meet_eb"]]
    for s, x in c["qmain"].items():
        k[f"qmain|{s}"] = x["answer"]
    return k


def summarize(d):
    rows = V3._rows(d)
    by = {(r["tag"], r["cell"]): r for r in rows}
    out = dict(cells={}, seedshift={})
    for cell in ("D1", "D2"):
        r = by.get(("main", cell))
        if r:
            out["cells"][cell] = cell_v5(r)
            out["cells"][cell]["N"] = r["N"]
        ss = by.get(("seedshift", cell))
        if r and ss:
            a, b = verdict_key5(out["cells"][cell]), verdict_key5(cell_v5(ss))
            out["seedshift"][cell] = dict(UNATTRIBUTABLE={k: [a[k], b.get(k)] for k in a if a[k] != b.get(k)}, n_compared=len(a))
    return out


# ------------------------------------------------------------------------------------------------ self-test (prereg §7)
def self_test():
    rc4 = V4.self_test()
    res = []

    def chk(name, got, exp):
        res.append((name, got == exp, got, exp))
    R = NREPS
    # GX1: exact pairing tells slow from fast (one model, the 200 requests on day 1; replicate 0 drops day 0 and keeps them)
    days = [collections.Counter() for _ in range(NDAYS)]
    days[1] = collections.Counter({50: 197, 190: 3})
    slow = [[] for _ in range(NDAYS)]
    fast = [[] for _ in range(NDAYS)]
    for _ in range(3):
        slow[1].extend(pair_cell([190], [30]))
        fast[1].extend(pair_cell([50], [30]))
    _, p99s = compose_exact(days, slow)
    _, p99f = compose_exact(days, fast)
    _, p990 = compose_exact(days, [[] for _ in range(NDAYS)])
    chk("GX1 static p99", p990[0], 1.9)
    chk("GX1 wakes on the slow requests: exact p99 2.20 (fails S2)", p99s[0], 2.2)
    chk("GX1 wakes on fast requests: exact p99 1.90 (passes)", p99f[0], 1.9)
    arr = np.array([50] * 197 + [190] * 3, np.int64)
    chk("GX1 v4 bounds (WORST, BEST) cs", V4.pq_bounds(arr, [30, 30, 30], 0.99), (220, 190))
    chk("GX1 tail p99 pass: slow / fast", [V4.p99_rep_pass([(200, 1.9, p99s[0], p99s[0])], 2.0)[0],
                                          V4.p99_rep_pass([(200, 1.9, p99f[0], p99f[0])], 2.0)[0]], [False, True])
    chk("GX1 replicate 1 drops the day: empty", [p99s[1], p99f[1]], [None, None])
    # GX2: rank identity through cells (split, one-key, shed)
    chk("GX2 split cell", pair_cell(expand(3, [(0, 3, [120, 90, 60])]), [30, 20, 10]), [(120, 150), (90, 110), (60, 70)])
    rk = Rec()
    obj = collections.Counter()
    rk.seg(obj, 7, 4, 0, 4, 50)
    s4, sp4 = rk.keys_of(obj, 7)
    chk("GX2 one-key cell", [[int(e) for _, e in pair_cell(s4, [0, 25, 0, 0])], sp4, (s4 + np.array([0, 25, 0, 0])).tolist()],
        [[75], False, [50, 75, 50, 50]])
    rk.seg(obj, 9, 3, 2, 3, INF)
    rk.seg(obj, 9, 3, 0, 2, 40)
    s3, sp3 = rk.keys_of(obj, 9)
    chk("GX2 shed cell: [0.40, 0.40, INF] + 0.10 each", [s3.tolist(), sp3, [e for _, e in pair_cell(s3, [10, 10, 10])]],
        [[40, 40, INF], True, [50, 50, INF]])
    chk("GX2 the record's Counter", rk.counter_of(obj), collections.Counter({50: 4, 40: 2, INF: 1}))
    try:
        expand(3, [(0, 2, 40)])
        gap = False
    except RuntimeError:
        gap = True
    chk("GX2 a rank with no key raises", gap, True)
    # GX3: HOLD-EDGE
    tl = lambda units, models: dict(kind="tail", units=units, unit_of={m: u for m, u in models}, gbps=None,  # noqa: E731
                                    models=[m for m, _ in models], g=1)
    chk("GX3 page-ins", [round(V3.page_in(10, 22.6), 6), round(V3.page_in(2, 22.6), 6), round(V3.page_in(20, 22.6), 6)],
        [0.495107, 0.115021, 0.970214])
    s1 = tl({"a": 10.0, "b": 2.0}, [(1, "a"), (2, "b")])
    s2 = tl({"c": 10.0}, [(3, "c")])
    s3_ = tl({"d": 10.0, "e": 20.0}, [(4, "d"), (5, "e")])
    pt = {1: (500, 1.70), 2: (500, 0.40), 3: (500, 0.40), 4: (500, 1.00), 5: (500, 0.20)}
    chk("GX3 HE S2 p0: slot1 held, slot2 not", he_hold([s1, s2], pt, 2.0, 0.0, "HE"), [True, False])
    chk("GX3 HE S10 p0: slot1 not held", he_hold([s1], pt, 10.0, 0.0, "HE"), [False])
    chk("GX3 HE S2 p1.5: slot2 held", he_hold([s2], pt, 2.0, 1.5, "HE"), [True])
    chk("GX3 slot3 S2 p0.5: HE not held, HE-Q held", [he_hold([s3_], pt, 2.0, 0.5, "HE"), he_hold([s3_], pt, 2.0, 0.5, "HE-Q")],
        [[False], [True]])
    chk("GX3 TRAIN p99 INF -> held; TRAIN n 0 -> not held",
        [he_hold([s2], {3: (5, math.inf)}, 10.0, 0.0, "HE"), he_hold([s2], {3: (0, None)}, 2.0, 30.0, "HE")], [[True], [False]])
    chk("GX3 I-HE: a parked violator is counted", [he_check([s1, s2], pt, 2.0, 0.0, "HE", [False, False]),
                                                   he_check([s1, s2], pt, 2.0, 0.0, "HE", [True, False])], [1, 0])
    # GX3b (addendum, hand-computed pre-data): HE-S protects 0 <= headroom < cost only
    s6 = tl({"f": 10.0}, [(6, "f")])
    chk("GX3b HE-S: m1 h 0.30 < 0.495 -> held; h -0.50 (TRAIN p99 2.50) -> HE held, HE-S not; INF -> HE-S not",
        [he_hold([s1], pt, 2.0, 0.0, "HE-S"), he_hold([s6], {6: (500, 2.50)}, 2.0, 0.0, "HE"), he_hold([s6], {6: (500, 2.50)}, 2.0, 0.0, "HE-S"),
         he_hold([s6], {6: (500, math.inf)}, 2.0, 0.0, "HE-S")], [[True], [True], [False], [False]])
    chk("GX3b I-HE accepts a parked HE-S model with negative headroom", he_check([s6], {6: (500, 2.50)}, 2.0, 0.0, "HE-S", [False]), 0)
    hh, ww = V4.slot_policy("KA-1", np.array([0, 1, 0, 0, 1, 0], bool), None, True)
    chk("GX3 a held slot never wakes", [int(hh.sum()), int(ww.sum())], [6, 0])
    # GX4: adversarial — passes the per-model gate, > 1 % over S fleet-wide -> EB-FAILS
    dA = [collections.Counter() for _ in range(NDAYS)]
    dA[1] = collections.Counter({50: 48, 300: 2})
    dB = [collections.Counter() for _ in range(NDAYS)]
    dB[1] = collections.Counter({50: 100})
    sa, _ = compose_exact(dA, [[] for _ in range(NDAYS)])
    sb, _ = compose_exact(dB, [[] for _ in range(NDAYS)])
    chk("GX4 model A p95 0.50 s, ok 48 of 50", [sa["p95"][0], sa["ok"]["2.0"][0], sa["n"][0]], [0.5, 48, 50])
    pm4 = pm_gate([True] * R, [0.9] * R, [V4.p99_rep_pass([(50, 3.0, 3.0, 3.0), (100, 0.5, 0.5, 0.5)], 2.0)[0]] * R, 2.0)
    sh = eb_share([sa["ok"]["2.0"][0], sb["ok"]["2.0"][0]], [sa["n"][0], sb["n"][0]])
    chk("GX4 PM-HOLDS, share 0.986667, EB-FAILS, BOTH fails",
        [pm4, round(sh, 6), eb_gate([sh] * R), both(pm4, eb_gate([sh] * R), "EB")], ["PM-HOLDS", 0.986667, "EB-FAILS", False])
    chk("GX4 B at n 200: 0.992 -> EB-HOLDS", eb_gate([eb_share([48, 200], [50, 200])] * R), "EB-HOLDS")
    dAi = [collections.Counter() for _ in range(NDAYS)]
    dAi[1] = collections.Counter({50: 48, INF: 2})
    si_, _ = compose_exact(dAi, [[] for _ in range(NDAYS)])
    chk("GX4 INF counts as over S", eb_gate([eb_share([si_["ok"]["2.0"][0], 100], [50, 100])] * R), "EB-FAILS")
    chk("GX4 one bad replicate fails EB (not a mean)", eb_gate([0.995] * (R - 1) + [0.98]), "EB-FAILS")
    # GX5: P_MEET with a p-dependent BAR; the vacuity rule
    U, A = "UNDER-BAR", "ABOVE-BAR"
    rows = {p: dict(pm=True, eb=True, bar=U) for p in P_GRID}
    rows[0.5] = dict(pm=True, eb=True, bar=A)
    pmx = p_meet(rows)
    chk("GX5 P_MEET 0.25, break at 0.5 on BAR", [pmx["both"]["p_meet"], pmx["both"]["p_break"]], [0.25, [0.5, ["BAR"]]])
    rows2 = {p: dict(pm=False, eb=True, bar=U) for p in P_GRID}
    chk("GX5 NONE when p 0 fails PM", [p_meet(rows2)["both"]["p_meet"], p_meet(rows2)["eb"]["p_meet"]], [None, 30.0])
    chk("GX5 vacuity: static EB fails, EB-ELIG holds -> EB-ELIG; both fail -> NONE",
        [eb_reading("EB-FAILS", "EB-HOLDS"), eb_reading("EB-HOLDS", "EB-FAILS"), eb_reading("EB-FAILS", "EB-FAILS")],
        ["EB-ELIG", "EB", "NONE"])
    chk("GX5 BOTH with reading NONE is PM alone", [both("PM-HOLDS", "EB-FAILS", "NONE"), both("PM-HOLDS", "EB-FAILS", "EB")],
        [True, False])
    # the fork patches apply exactly once on the live sources (no install)
    try:
        fork(FV2.simulate, _FV2_EDITS)
        fork(NS.dedicated_by_day, _DED_EDITS)
        fork(CS.dedicated_rep, _DED_EDITS)
        pok = True
    except RuntimeError as e:
        print(f"FAIL fork: {e}")
        pok = False
    chk("fork anchors apply exactly once (FV2.simulate, NS.dedicated_by_day, CS.dedicated_rep)", pok, True)
    chk("variants: 276, shared with v4: 68", [len(VARIANTS5), len(V4_SHARED)], [276, 68])
    n_ok = sum(1 for r_ in res if r_[1])
    for name, ok, got, exp in res:
        if not ok:
            print(f"FAIL {name}: got {got!r} expected {exp!r}")
    allok = n_ok == len(res) and rc4 == 0
    print(f"SELF-TEST {'PASS' if allok else 'FAIL'} {n_ok}/{len(res)} (v5) + v4 rc {rc4}")
    return 0 if allok else 1


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--self-test", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--premise", action="store_true")
    m.add_argument("--ifix", nargs=2, metavar=("NEW_DIR", "V4_DIR"))
    m.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--cell")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
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
        return run_cell(a.cell, seeds, a.tag, a.root, a.cache, a.out)
    except (ValueError, V.ManifestError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"CHECK-FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
