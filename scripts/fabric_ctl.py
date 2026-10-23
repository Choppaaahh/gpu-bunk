#!/usr/bin/env python3
"""fabric_ctl.py -- the gpu-bunk NODE CONTROLLER in fabric mode (bite 2: B3 + B6).

WHAT. One process per node that runs the whole fabric on stock vLLM 0.30.0:
  * FabricCore -- the DECISION core: `cw_engine` rule (cowake_handoff.py L95-271) as an incremental
    state machine. Park = place each unit on the most-free parked GPU of the node (ties -> lowest index; parking slots in
    SLOT order, units largest first); reclaim = evict the reclaimed card's guests largest first and re-home each to the
    most-free other parked GPU (excluding the reclaimed card and the guest's owner), else drop it to host-only; wake = read
    the owner's guest where the core last placed it. B6 (I-NEST-E2E, `inest`) proves that this object, fed the sim's own
    schedule minute by minute, reproduces cw_engine's records with 0 diffs. The controller below drives THIS object.
  * FabricController -- the actuators + router: KA-K park timers (no request in flight and no arrival for K s), park =
    holder alloc -> `bb_push_to_peer` -> holder attach(verify) -> owner `/sleep?level=1&mode=wait` -> batch engine
    `/wake_up` + its closed-loop load; reclaim = batch `/sleep?level=1&mode=abort` (the stock yield) || re-homes holder ->
    holder || owner restore after the yield from the EARLIEST source (a landed copy, else the reclaimed card's copy = the
    NO-WAIT handoff, else host) -> `/wake_up?tags=weights` -> `/wake_up?tags=kv_cache` -> `/is_sleeping` must read false
    -> forward. One OpenAI-compatible endpoint; one event log with monotonic + wall clocks.
  * Modes: `fabric` (A1 BB-NB), `host` (A2 BB-HOST: CAP 0, every wake from host), `ded` (A0: engines always awake).
  * Backends: `stub` (bite 1's CPU stub engines + holders on a file fabric -- everything here is proven on these only) and
    `vllm` (the box; UNPROVEN until S0).

WHAT IT CANNOT SHOW ON A CPU: CUDA IPC across processes, peer-context HBM NVLink rates, CUDA_VISIBLE_DEVICES
reordering inside vLLM (the vllm launcher sets <own GPU>,<the rest> for owners and holders; tail-server pins one GPU,
tail_server.py L288-290), real yield p. Those are S0 list.

Output contract:
  serve --config C   serves until SIGTERM/SIGINT or POST /fabric/stop; prints exactly one stdout line
                     `BB-FABRIC-READY port=<n>` when routing; WRITES <log_dir>/{events,requests,sampler,boot}.jsonl and one
                     log file per child; exit 0 on a clean stop, 1 + one `BOOT-FAIL <reason>` stderr line if boot fails,
                     2 on a bad config.
  inest --fixture F  read-only: FabricCore vs cw_engine (B6, I-NEST-E2E) on 300 toys x 3 modes + the fixture (node 4 and
                     all nodes); one JSON line on stdout; exit 0 iff 0 diffs everywhere AND the fixture's own checks
                     reproduce, else exit 1. Bad input exit 2.
  --self-test        core goldens CG1-CG4 in process + controller goldens R1-R4 through `serve` children on the CPU stub;
                     exit 0 + `SELFTEST-OK n/n`; exit 1 + `FAIL` lines + `SELFTEST-FAIL k/n`. Writes only a tempdir.
  t3 [--out F]       a stub node run (8 slots, sim-shaped sizes at 1/1024, real-time KA timers); exit 0 + one JSON on
                     stdout (WRITES --out when given); exit 1 + `T3-FAIL` on stderr.

Clocks: keepalive_s, tick_s, sample_s and request_timeout_s are REAL seconds; time_scale only maps real time to the trace
  minutes the core records (minute = floor((now - origin) x time_scale / 60)). A time-compressed replay sets
  keepalive_s = 300 / compression itself (QA 2026-10-21 asked; by design, so one knob cannot silently move another).

Top pitfalls (observed):
  1. **`/sleep` defaults to mode=abort and drops in-flight requests** -- owners always sleep with mode=wait; only
     the batch engine's yield uses abort.
  2. **A `/wake_up?tags=X` 200 is not a wake** (abstract.py L351-360 warn and return) -- every wake ends with a
     `/is_sleeping` read that must say false.
  3. **The DRAFT prereg §1 says 'largest first' across engines parking in one tick; cw_engine orders by SLOT** (L244-256)
     -- this controller follows cw_engine (I-NEST-E2E); golden CG3 shows the two placements differ.
  4. **Restoring before the owner card's own batch yield returns fails out of memory** (bite 1 §2) -- the restore waits
     for the yield; the re-homes run concurrently with both.
"""

import argparse
import collections
import gzip
import hashlib
import http.client
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

GIB = 1 << 30
MIB = 1 << 20
DAY = 1440
MODES = ("fabric", "host", "ded")
SOURCES = ("HOLDER", "RECLAIMED-CARD", "NEW-HOME", "HOST")
TIERS = ("HOT", "WARM-WAKE", "COOL-WAKE", "JOIN")
KVCACHED_ENV = ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH")     #: never let kvcached autopatch a stock engine
HERE = os.path.dirname(os.path.abspath(__file__))


# ================================================================================================ the decision core
class CoreError(RuntimeError):
    pass


class FabricCore:
    """cw_engine's state engine, one controller tick at a time. `step(t, becomes, wakes, parks)` applies one batch of
    transitions at trace minute t in cw_engine's order (evictions of every reclaimed card, then wakes, then parks); lender
    eligibility reads the held state AFTER this batch's transitions, as cw_engine reads held_of[L][t]. `finish(n_min)`
    returns cw_engine's record dict (rec, k, copies, stats, x, cps, wake_chain, reclaims)."""

    def __init__(self, slots, node_of, units_of, cap, evict="REHOME", avail="SAME", bounce=None, charge=False, day=DAY):
        if avail not in ("SAME", "NEXT") or evict not in ("DROP", "REHOME"):
            raise ValueError(f"bad engine mode {avail} / {evict}")
        self.slots = sorted(slots)
        self.node_of = {s: node_of[s] for s in self.slots}
        self.units_of = {s: dict(units_of.get(s, {})) for s in self.slots}
        self.cap, self.evict, self.avail, self.bounce, self.charge, self.day = float(cap), evict, avail, bounce, charge, day
        self.by_node = collections.defaultdict(list)
        for s in self.slots:
            self.by_node[self.node_of[s]].append(s)
        self.nodes = sorted(self.by_node)
        self.held = {s: True for s in self.slots}          # cw_engine L105: the minute before 0 reads as held
        self.guests = {s: {} for s in self.slots}
        self.free = {s: float(cap) for s in self.slots}
        self.loc, self.rec, self.k, self.copies = {}, {}, collections.Counter(), []
        self.st = dict(evicted=0, rehomed=0, pushes=0, park_uncached=0, guest_gib_min=0.0, lender_gpu_min=0)
        self.tot_guest = 0.0
        self.g_day = {n: [] for n in self.nodes}
        self.c_day = {n: [] for n in self.nodes}
        self.l_day = {n: [] for n in self.nodes}
        self.lvl = {n: 0.0 for n in self.nodes}
        self.nre = {}
        self.inflight = set()
        self._inflight_t = None
        self.x = dict(inflight_fallback=0, dropped_at_evict=0, nb1_dropped=0, rehome_items=[],
                      bounce_hist=collections.Counter(), cap_ok=True)
        self.cps, self.chain, self.wake_chain, self.reclaims = [], {}, {}, []
        self.wake_minutes = collections.defaultdict(set)
        self.t_last = 0
        self.ctl = collections.Counter()                    # controller-only reconciliation counts (never in x)

    # -- integration between ticks (cw_engine L138-148, L120-131, L260-268)
    @staticmethod
    def _grow(lst, d, zero):
        while len(lst) <= d:
            lst.append(zero)

    def _integrate(self, a, b):
        if b <= a:
            return
        for n in self.nodes:
            v = self.lvl[n]
            if v == 0.0:
                continue
            m = a
            while m < b:
                d = m // self.day
                e = min(b, (d + 1) * self.day)
                self._grow(self.g_day[n], d, 0.0)
                self.g_day[n][d] += v * (e - m)
                m = e
        for s in self.slots:
            if self.held[s]:
                continue
            n = self.node_of[s]
            self.st["lender_gpu_min"] += b - a
            m = a
            while m < b:
                d = m // self.day
                e = min(b, (d + 1) * self.day)
                self._grow(self.l_day[n], d, 0)
                self.l_day[n][d] += e - m
                m = e

    # -- cw_engine's helpers, verbatim in behaviour
    def lender_for(self, s_home, w, exclude):
        best = None
        for L in self.by_node[self.node_of[s_home]]:
            if L in exclude or self.held[L] or self.free[L] < w - 1e-12:
                continue
            if best is None or self.free[L] > self.free[best] + 1e-12:
                best = L
        return best

    def _place(self, L, key, w, t):
        self.guests[L][key] = w
        self.free[L] -= w
        self.loc[key] = L
        self.k[(L, t)] += 1
        self.copies.append((L, t, w))
        self.tot_guest += w
        self.st["pushes"] += 1
        self.lvl[self.node_of[L]] += w
        if self.free[L] < -1e-9:
            self.x["cap_ok"] = False
            raise CoreError(f"I-CAP: lender {L} over CAP at minute {t}")

    def _unplace(self, L, key, w):
        del self.guests[L][key]
        self.free[L] += w
        self.tot_guest -= w
        self.lvl[self.node_of[L]] -= w

    def _evictions(self, becomes, t):
        out = []
        for L in sorted(becomes):
            rc = dict(L=L, t=t, items=[])
            self.reclaims.append(rc)
            out.append(rc)
            for key, w in sorted(self.guests[L].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                self._unplace(L, key, w)
                self.loc[key] = None
                self.st["evicted"] += 1
                if self.evict == "REHOME":
                    if self.bounce is not None and self.nre.get(key, 0) >= self.bounce:
                        self.x["nb1_dropped"] += 1
                        self.x["dropped_at_evict"] += 1
                        self.x["bounce_hist"][self.nre.get(key, 0)] += 1
                        rc["items"].append((key, w, None, None))
                        self.chain[key] = []
                        continue
                    L2 = self.lender_for(L, w, {L, key[0]})
                    if L2 is not None:
                        self._place(L2, key, w, t)
                        self.st["rehomed"] += 1
                        self.nre[key] = self.nre.get(key, 0) + 1
                        self.x["rehome_items"].append((L, t, w, L2))
                        ci = len(self.cps)
                        prv = self.chain[key][-1] if self.chain.get(key) else None
                        self.cps.append(dict(kind="R", key=key, src=L, dst=L2, t=t, w=w, prev=prv, ri=len(self.reclaims) - 1))
                        self.chain.setdefault(key, []).append(ci)
                        rc["items"].append((key, w, L2, ci))
                        if self.avail == "NEXT":
                            self.inflight.add(key)
                        if self.charge:
                            d = t // self.day
                            self._grow(self.c_day[self.node_of[L]], d, 0.0)
                            self.c_day[self.node_of[L]][d] += w
                    else:
                        self.x["dropped_at_evict"] += 1
                        self.x["bounce_hist"][self.nre.get(key, 0)] += 1
                        rc["items"].append((key, w, None, None))
                        self.chain[key] = []
                else:
                    self.x["dropped_at_evict"] += 1
                    self.x["bounce_hist"][self.nre.get(key, 0)] += 1
                    rc["items"].append((key, w, None, None))
                    self.chain[key] = []
            if self.guests[L]:
                raise CoreError("I-NB-EVICT: guests left on a held lender")
        return out

    def _wakes(self, wakes, t):
        out = []
        for s in sorted(wakes):
            self.wake_minutes[s].add(t)
            for u, w in self.units_of[s].items():
                key = (s, u)
                L = self.loc.get(key)
                ch = tuple(self.chain.get(key, ()))
                self.wake_chain[(s, t, u)] = ch
                self.chain[key] = []
                if L is not None and key in self.inflight:
                    self.rec[(s, t, u)] = None
                    self._unplace(L, key, w)
                    self.x["inflight_fallback"] += 1
                    self.x["bounce_hist"][self.nre.get(key, 0)] += 1
                    self.loc[key] = None
                    out.append(dict(key=key, w=w, src=None, chain=ch, inflight=True))
                    continue
                self.rec[(s, t, u)] = L
                if L is not None:
                    self.k[(L, t)] += 1
                    self.copies.append((L, t, w))
                    self._unplace(L, key, w)
                    self.x["bounce_hist"][self.nre.get(key, 0)] += 1
                self.loc[key] = None
                out.append(dict(key=key, w=w, src=L, chain=ch, inflight=False))
        return out

    def _parks(self, parks, t):
        out = []
        for s in sorted(parks):
            for u, w in sorted(self.units_of[s].items(), key=lambda kv: (-kv[1], str(kv[0]))):
                self.nre[(s, u)] = 0
                L = self.lender_for(s, w, {s})
                if L is None:
                    self.loc[(s, u)] = None
                    self.st["park_uncached"] += 1
                    self.chain[(s, u)] = []
                    out.append(dict(key=(s, u), w=w, dst=None))
                    continue
                self._place(L, (s, u), w, t)
                self.cps.append(dict(kind="P", key=(s, u), src=s, dst=L, t=t, w=w, prev=None, ri=None))
                self.chain[(s, u)] = [len(self.cps) - 1]
                out.append(dict(key=(s, u), w=w, dst=L))
        return out

    # -- the public tick
    def step(self, t, becomes=(), wakes=(), parks=()):
        becomes, wakes, parks = set(becomes), set(wakes), set(parks)
        bad = [s for s in becomes | wakes | parks if s not in self.held]
        if bad:
            raise CoreError(f"unknown slots {sorted(bad)}")
        if not wakes <= becomes:
            raise CoreError(f"I-NB-EVENTS: slots {sorted(wakes - becomes)} wake at {t} without becoming held")
        if becomes & parks:
            raise CoreError(f"slots {sorted(becomes & parks)} both become held and park at {t}")
        for s in becomes:
            if self.held[s]:
                raise CoreError(f"slot {s} becomes held at {t} but is already held")
        for s in parks:
            if not self.held[s]:
                raise CoreError(f"slot {s} parks at {t} but is already parked")
        if int(t) != t or t < self.t_last:
            raise CoreError(f"minute {t} is not an integer >= the last tick {self.t_last}")
        t = int(t)
        self.st["guest_gib_min"] += self.tot_guest * (t - self.t_last)
        self._integrate(self.t_last, t)
        self.t_last = t
        if self._inflight_t != t:
            self.inflight.clear()
            self._inflight_t = t
        for s in becomes:
            self.held[s] = True
        for s in parks:
            self.held[s] = False
        ev = self._evictions(becomes, t)
        wk = self._wakes(wakes, t)
        pk = self._parks(parks, t)
        return dict(t=t, evictions=ev, wakes=wk, parks=pk)

    def drop_guest(self, key, why):
        """Controller reconciliation (never in the sim): a copy the core placed did not happen physically (push or re-home
        failed). Unplace it so later placements see the real free room; the owner's next wake reads host."""
        L = self.loc.get(key)
        if L is None or key not in self.guests.get(L, {}):
            return False
        self._unplace(L, key, self.guests[L][key])
        self.loc[key] = None
        self.chain[key] = []
        self.ctl["drop:" + why] += 1
        return True

    def cowake_events(self):
        """ cowake_events: re-home items (not drops) whose owner wakes in the reclaim minute."""
        return [(key[0], rc["L"], rc["t"], w) for rc in self.reclaims for key, w, L2, ci in rc["items"]
                if ci is not None and rc["t"] in self.wake_minutes[key[0]]]

    def is_cowake(self, key, t):
        return any(rc["t"] == t and k_ == key and ci is not None for rc in self.reclaims for k_, w, L2, ci in rc["items"])

    def finish(self, n_min):
        self.st["guest_gib_min"] += self.tot_guest * (n_min - self.t_last)
        self._integrate(self.t_last, n_min)
        self.t_last = n_min
        ndays = -(-n_min // self.day)
        for dd, zero in ((self.g_day, 0.0), (self.c_day, 0.0), (self.l_day, 0)):
            for n in self.nodes:
                if len(dd[n]) > ndays:
                    raise CoreError("day records past n_min")
                self._grow(dd[n], ndays - 1, zero)
        x = dict(self.x)
        x.update(g_day=self.g_day, c_day=self.c_day, l_day=self.l_day, ndays=ndays, day=self.day, charge=bool(self.charge))
        return dict(rec=self.rec, k=self.k, copies=self.copies, stats=self.st, x=x, cps=self.cps,
                    wake_chain=self.wake_chain, reclaims=self.reclaims)


def core_replay(held_of, wake_of, units_of, node_of, cap, evict, n_min, avail="SAME", bounce=None, charge=False, day=DAY):
    """Feed FabricCore the sim's per-minute schedule exactly as cw_engine walks it: becomes = held & ~prev, wakes = wake_of,
    parks = ~held & prev (prev of minute 0 = held). Same signature as cw_engine; returns (records, core)."""
    slots = sorted(held_of)
    core = FabricCore(slots, node_of, units_of, cap, evict, avail, bounce, charge, day)
    ev = collections.defaultdict(lambda: (set(), set(), set()))
    for s in slots:
        h = [bool(v) for v in held_of[s]]
        w = [bool(v) for v in wake_of[s]]
        if len(h) != n_min or len(w) != n_min:
            raise CoreError(f"slot {s}: schedule length != n_min {n_min}")
        prev = True
        for t in range(n_min):
            if h[t] and not prev:
                ev[t][0].add(s)
            elif prev and not h[t]:
                ev[t][2].add(s)
            if w[t]:
                ev[t][1].add(s)
            prev = h[t]
    for t in sorted(ev):
        b, wk, pk = ev[t]
        core.step(t, b, wk, pk)
    return core.finish(n_min), core


# ================================================================================================ B6: I-NEST-E2E
FIELDS = ("rec", "k", "copies", "stats", "x", "cps", "wake_chain", "reclaims", "cowakes")


def _norm(rec_, cow):
    d = {f: rec_[f] for f in FIELDS if f != "cowakes"}
    d["k"] = dict(d["k"])
    d["cowakes"] = sorted(cow)
    return d


def compare_engines(cw, held_of, wake_of, units_of, node_of, cap, evict, n_min, avail="SAME", bounce=None, charge=False):
    """-> {field: equal?} for cw_engine vs FabricCore on one schedule (CW = the module)."""
    a = cw.cw_engine(held_of, wake_of, units_of, node_of, cap, evict, n_min, avail, bounce, charge)
    b, core = core_replay(held_of, wake_of, units_of, node_of, cap, evict, n_min, avail, bounce, charge)
    na, nb = _norm(a, cw.cowake_events(a, wake_of, n_min)), _norm(b, core.cowake_events())
    return {f: na[f] == nb[f] for f in FIELDS}, na, nb


def _toys(n=300, seed=2938):
    rng = random.Random(seed)
    for _ in range(n):
        ns, nmin = rng.randint(3, 9), rng.randint(30, 120)
        held, units, nodes = {}, {}, {}
        for s in range(ns):
            h, cur = [], rng.random() < 0.5
            for _t in range(nmin):
                if rng.random() < 0.15:
                    cur = not cur
                h.append(cur)
            held[s] = h
            units[s] = {f"u{j}": rng.uniform(0.5, 30.0) for j in range(rng.randint(0, 3))}
            nodes[s] = rng.randint(0, 2)
        wake = {s: [bool(h[t] and not (h[t - 1] if t else True)) for t in range(nmin)] for s, h in held.items()}
        yield held, wake, units, nodes, rng.uniform(10, 40), nmin


def load_fixture(path):
    raw = open(path, "rb").read()
    d = json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)

    def key(u):
        return int(u) if str(u).lstrip("-").isdigit() else u
    n = int(d["n_min"])
    held, wake, units, nodes = {}, {}, {}, {}
    for s in d["tail"]:
        h = [False] * n
        for a, b in d["held_runs"][str(s)]:
            for t in range(a, b):
                h[t] = True
        w = [False] * n
        for t in d["wake_min"][str(s)]:
            w[t] = True
        held[s], wake[s] = h, w
        units[s] = {key(u): float(x) for u, x in d["units_of"][str(s)].items()}
        nodes[s] = int(d["node_of"][str(s)])
    return d, hashlib.sha256(raw).hexdigest(), held, wake, units, nodes


def inest(fixture, cw=None, n_toys=300):
    if cw is None:
        sys.path.insert(0, HERE)
        import cowake_handoff as cw  # noqa:
    out = dict(source="fabric_ctl.inest", trust_tier="agent-reasoning", toys=dict(n=0, ok=0, bad=[]))
    for i, (held, wake, units, nodes, cap, nmin) in enumerate(_toys(n_toys)):
        for kw in (dict(avail="SAME", charge=True), dict(avail="NEXT", charge=True), dict()):
            eq, _, _ = compare_engines(cw, held, wake, units, nodes, cap, "REHOME", nmin, **kw)
            out["toys"]["n"] += 1
            if all(eq.values()):
                out["toys"]["ok"] += 1
            elif len(out["toys"]["bad"]) < 5:
                out["toys"]["bad"].append(dict(toy=i, mode=kw, diff=[f for f, v in eq.items() if not v]))
    if fixture:
        d, sha, held, wake, units, nodes = load_fixture(fixture)
        out["fixture"] = dict(path=fixture, sha256=sha, n_tail=len(held), n_min=d["n_min"])
        res = {}
        n4 = [s for s in held if nodes[s] == 4]
        for scope, ss in (("node4", n4), ("all", sorted(held))):
            sub = {s: held[s] for s in ss}
            eq, na, nb = compare_engines(cw, sub, {s: wake[s] for s in ss}, {s: units[s] for s in ss},
                                         {s: nodes[s] for s in ss}, d["cap"], d["evict"], d["n_min"], d["avail"], None,
                                         d["charge"])
            res[scope] = dict(fields=eq, diffs=[f for f, v in eq.items() if not v],
                              counts=dict(pushes=nb["stats"]["pushes"], rehomed=nb["stats"]["rehomed"],
                                          evicted=nb["stats"]["evicted"], park_uncached=nb["stats"]["park_uncached"],
                                          cps=len(nb["cps"]), reclaims=len(nb["reclaims"]), cowakes=len(nb["cowakes"])))
            if scope == "node4":
                w0, w1 = 7620, 7800
                cow = nb["cowakes"]
                res[scope]["checks"] = dict(
                    node4_slots=n4, week_wakes=sum(sum(wake[s]) for s in n4), week_cowakes=len(cow),
                    week_reclaims=len(nb["reclaims"]), win_wakes=sum(sum(wake[s][w0:w1]) for s in n4),
                    win_cowakes=sum(1 for e in cow if w0 <= e[2] < w1),
                    win_cowakes_big=sum(1 for e in cow if w0 <= e[2] < w1 and e[3] >= 13.0),
                    win_reclaims=sum(1 for rc in nb["reclaims"] if w0 <= rc["t"] < w1),
                    win_rehome_items=sum(len(rc["items"]) for rc in nb["reclaims"] if w0 <= rc["t"] < w1))
                res[scope]["checks_match_fixture"] = res[scope]["checks"] == d["checks"]
                # same-tick park ordering (prereg note): minutes on node 4 with >= 2 parks and >= 1 placed guest
                multi = 0
                for t in range(d["n_min"]):
                    pk = [s for s in n4 if (not held[s][t]) and (held[s][t - 1] if t else True)]
                    multi += len(pk) >= 2
                res[scope]["minutes_with_2plus_parks"] = multi
            if scope == "all":
                res[scope]["counts_match_fixture"] = res[scope]["counts"] == d["ref_counts"]
        out["fixture_results"] = res
    ok = out["toys"]["ok"] == out["toys"]["n"] and out["toys"]["n"] > 0
    if fixture:
        r = out["fixture_results"]
        ok = ok and not r["node4"]["diffs"] and not r["all"]["diffs"] and r["node4"]["checks_match_fixture"] \
            and r["all"]["counts_match_fixture"]
    out["pass"] = bool(ok)
    return out


# ================================================================================================ HTTP helpers
def http_json(url, body=None, method="POST", timeout=60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:                                        # review: a malformed 200 body is data, not a crash
                return r.status, (json.loads(raw) if raw else None)
            except json.JSONDecodeError:
                return r.status, {"raw": raw[:300].decode(errors="replace"), "decode_error": True}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, (json.loads(raw) if raw else None)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:300].decode(errors="replace")}


def wait_health(url, deadline_s, proc=None):
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < deadline_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{url}: process exited rc={proc.returncode} before health")
        try:
            st, _ = http_json(url + "/health", method="GET", timeout=2.0)
            if st == 200:
                return time.monotonic() - t0
            last = st
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:120]
        time.sleep(0.1)
    raise RuntimeError(f"{url}: not healthy in {deadline_s}s (last {last})")


class EventLog:
    """One JSONL file per stream; every row carries mono + wall clocks (the scorer's D_w uses mono)."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.dir = log_dir
        self.lock = threading.Lock()
        self.fh = {}
        self.counts = collections.Counter()
        self.closed = False

    def write(self, stream, row):
        row = dict(row)
        row.setdefault("mono", time.monotonic())
        row.setdefault("wall", time.time())
        line = json.dumps(row, default=str, sort_keys=True) + "\n"
        with self.lock:
            if self.closed:                             # shutdown only; counted, never silent
                self.counts[("dropped_after_close", row.get("ev"))] += 1
                return
            f = self.fh.get(stream)
            if f is None:
                f = self.fh[stream] = open(os.path.join(self.dir, f"{stream}.jsonl"), "a", buffering=1)
                f.write(json.dumps(dict(ev="log_open", source="fabric_ctl", trigger="bite 2",
                                        trust_tier="agent-reasoning", mono=time.monotonic(), wall=time.time())) + "\n")
            f.write(line)
            self.counts[(stream, row.get("ev"))] += 1

    def close(self):
        with self.lock:
            self.closed = True
            for f in self.fh.values():
                f.close()
            self.fh = {}

    def dropped(self):
        with self.lock:
            return {str(k[1]): v for k, v in self.counts.items() if k[0] == "dropped_after_close"}


# ================================================================================================ children
def _cvd(gpu, gpus):
    return ",".join(str(g) for g in [gpu] + [g for g in gpus if g != gpu])


class Child:
    def __init__(self, name, cmd, env, logpath):
        self.name, self.cmd, self.env, self.logpath = name, cmd, env, logpath
        self.proc = None

    def start(self):
        with open(self.logpath, "ab") as lf:            # review: no fd leak if Popen raises
            self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=lf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        return self

    @property
    def pid(self):
        return None if self.proc is None else self.proc.pid

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace=5.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:                                        # review: an unreapable child must not abort the sweep
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"CHILD-UNREAPED {self.name} pid={self.proc.pid}", file=sys.stderr, flush=True)


def child_env(cvd=None):
    env = {k: v for k, v in os.environ.items() if k not in KVCACHED_ENV}
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if cvd is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = cvd
    return env


def build_cmds(cfg, slot, role):
    """-> (cmd, env) for role in {owner, batch, holder} of one slot; the only backend-specific code."""
    be, gpus, gpu = cfg["backend"], cfg["gpus"], slot["gpu"]
    ext = cfg.get("ext_file") or os.path.join(HERE, "warm_wake_ext.py")
    host = cfg.get("host", "127.0.0.1")
    if role == "holder":
        cvd = _cvd(gpu, gpus)
        if be == "stub":
            cmd = [sys.executable, ext, "--serve-holder", "--stub-fabric", cfg["stub"]["fabric_dir"], "--stub-cvd", cvd,
                   "--port", str(slot["holder_port"]), "--cap-mib", str(cfg["cap_bytes"] / MIB)]
            env = child_env(None)
        else:
            cmd = [sys.executable, ext, "--serve-holder", "--port", str(slot["holder_port"]),
                   "--cap-gib", str(cfg["cap_bytes"] / GIB)]
            env = child_env(cvd)
        if cfg.get("allow_corrupt"):
            cmd.append("--allow-corrupt")
        return cmd, env
    spec = slot[role]
    port = spec["port"]
    if be == "stub":
        st = spec["stub"]
        cvd = _cvd(gpu, gpus) if role == "owner" else str(gpu)
        cmd = [sys.executable, cfg["stub"].get("stub_file") or os.path.join(HERE, "warm_wake_stub.py"),
               "--serve-engine", "--fabric", cfg["stub"]["fabric_dir"], "--cvd", cvd, "--port", str(port),
               "--weights-bytes", ",".join(str(int(x)) for x in st["weights_bytes"]),
               "--kv-bytes", ",".join(str(int(x)) for x in st.get("kv_bytes", [])), "--seed", str(st.get("seed", 0)),
               "--ext-file", ext]
        return cmd, child_env(None)
    v = spec["vllm"]
    if role == "owner":                       # owners open neighbours' IPC handles: see every GPU, own first
        env = child_env(_cvd(gpu, gpus))
        env["PYTHONPATH"] = os.path.dirname(ext) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        extra = ["--worker-extension-cls", "warm_wake_ext.WarmWakeExt"]
    else:                                     # the batch engine never touches IPC: pin it to its card
        env = child_env(str(gpu))
        extra = []
    cmd = ["vllm", "serve", v["repo"], "--revision", v["revision"], "--host", host, "--port", str(port),
           "--served-model-name", spec.get("served", slot["name"] if role == "owner" else f"{slot['name']}-batch"),
           "--enable-sleep-mode"] + extra + list(v.get("flags", []))
    return cmd, env


def validate_cfg(cfg):
    for k in ("mode", "backend", "gpus", "slots", "log_dir"):
        if k not in cfg:
            raise ValueError(f"config: missing {k}")
    if cfg["mode"] not in MODES or cfg["backend"] not in ("stub", "vllm"):
        raise ValueError(f"config: mode {cfg['mode']} / backend {cfg['backend']}")
    names, gpus_used = set(), set()
    for s in cfg["slots"]:
        for k in ("slot", "gpu", "name", "owner"):
            if k not in s:
                raise ValueError(f"config slot: missing {k}")
        if s["name"] in names or s["gpu"] in gpus_used:
            raise ValueError(f"config: duplicate name/gpu {s['name']} / {s['gpu']}")
        if s["gpu"] not in cfg["gpus"]:
            raise ValueError(f"config: slot {s['slot']} gpu {s['gpu']} not in gpus")
        names.add(s["name"])
        gpus_used.add(s["gpu"])
        if cfg["mode"] == "fabric" and "holder_port" not in s:
            raise ValueError(f"config slot {s['slot']}: holder_port required in fabric mode")
    if cfg["mode"] == "fabric" and int(cfg.get("cap_bytes", 0)) <= 0:
        raise ValueError("config: fabric mode needs cap_bytes > 0")
    be = cfg["backend"]                                 # review: nested backend keys fail HERE (exit 2), not at spawn
    if be == "stub" and not all(k in (cfg.get("stub") or {}) for k in ("fabric_dir", "gpu_cap_bytes")):
        raise ValueError("config: stub backend needs stub.fabric_dir + stub.gpu_cap_bytes")
    for sl in cfg["slots"]:
        for role in ("owner", "batch"):
            sp = sl.get(role)
            if sp is None:
                continue
            if "port" not in sp:
                raise ValueError(f"config slot {sl['slot']} {role}: missing port")
            if be == "stub" and not (sp.get("stub") or {}).get("weights_bytes"):
                raise ValueError(f"config slot {sl['slot']} {role}: stub.weights_bytes required")
            if be == "vllm" and not all(k in (sp.get("vllm") or {}) for k in ("repo", "revision")):
                raise ValueError(f"config slot {sl['slot']} {role}: vllm.repo + vllm.revision required")
    if cfg.get("host", "127.0.0.1") not in ("127.0.0.1", "localhost", "::1") and not cfg.get("allow_non_loopback"):
        raise ValueError("config: the router binds loopback only (a rented box has a public IP); "
                         "set allow_non_loopback to override")
    for k, dflt in (("keepalive_s", 300), ("tick_s", 0.25), ("sample_s", 1.0), ("request_timeout_s", 600)):
        if float(cfg.get(k, dflt)) <= 0:                # review: every real-second clock
            raise ValueError(f"config: {k} must be > 0")
    if float(cfg.get("time_scale", 1.0)) <= 0:
        raise ValueError("config: time_scale must be > 0")
    return cfg


# ================================================================================================ batch load
class BatchLoad:
    """The parked GPU's sold work: `streams` closed-loop completions (prompt_tokens / max_tokens, ignore_eos) against the
    batch engine. Counts every streamed token per GPU (the sampler's SOLD read) and the tokens of aborted requests."""

    def __init__(self, ctl, slot):
        self.ctl, self.slot = ctl, slot
        bl = ctl.cfg.get("batch_load", {})
        self.n = int(bl.get("streams", 8))
        self.p, self.m = int(bl.get("prompt_tokens", 512)), int(bl.get("max_tokens", 512))
        self.pace = float(bl.get("pace_s", 0.0))
        self.running = threading.Event()
        self.stop_all = threading.Event()
        self.lock = threading.Lock()
        self.tokens = 0                                 # streamed chunks with text (liveness: SOLD reads its delta)
        self.usage_tokens = 0                           # exact completion_tokens from each request's usage chunk
        self.aborted_tokens = 0
        self.requests = 0
        self.aborted = 0
        self.errors = 0
        self.last_error = None
        self.rng = random.Random(int(bl.get("seed", 0)) * 1000 + slot["slot"])
        self.threads = []

    def start_threads(self):
        for i in range(self.n):
            th = threading.Thread(target=self._loop, args=(i,), daemon=True, name=f"batch{self.slot['slot']}-{i}")
            th.start()
            self.threads.append(th)

    def _loop(self, i):
        port = self.slot["batch"]["port"]
        host = self.ctl.cfg.get("host", "127.0.0.1")
        while not self.stop_all.is_set():
            if not self.running.wait(0.2):
                continue
            with self.lock:
                ids = [self.rng.randint(1000, 100000) for _ in range(self.p)]
            body = {"model": self.slot["batch"].get("served", f"{self.slot['name']}-batch"), "prompt": ids,
                    "max_tokens": self.m, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
            got, fin, conn = 0, None, None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=600)
                conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    r.read()
                    conn.close()                        # QA 2026-10-21 (gemini): close on non-200 too
                    raise RuntimeError(f"batch status {r.status}")
                for ln in r:
                    if not ln.startswith(b"data: ") or ln.startswith(b"data: [DONE]"):
                        continue
                    ob = json.loads(ln[6:])
                    if ob.get("usage"):
                        with self.lock:
                            self.usage_tokens += int(ob["usage"].get("completion_tokens") or 0)
                    for ch in ob.get("choices") or []:
                        if ch.get("text") is not None:
                            got += 1
                            with self.lock:
                                self.tokens += 1
                        fin = ch.get("finish_reason") or fin
                conn.close()
                with self.lock:
                    self.requests += 1
                    if fin == "abort":
                        self.aborted += 1
                        self.aborted_tokens += got
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.errors += 1
                    first = self.last_error is None
                    self.last_error = f"{type(e).__name__}: {e}"[:200]
                if first or self.errors % 100 == 0:     # review: never a silent counter
                    self.ctl.ev("batch_error", slot=self.slot["slot"], n=self.errors, err=self.last_error)
                time.sleep(0.2)
            finally:
                if conn is not None:                    # QA 2026-10-21: closed on every path
                    conn.close()
            if self.pace:
                time.sleep(self.pace)

    def snapshot(self):
        with self.lock:
            return dict(tokens=self.tokens, usage_tokens=self.usage_tokens, aborted_tokens=self.aborted_tokens,
                        requests=self.requests,
                        aborted=self.aborted, errors=self.errors, last_error=self.last_error,
                        running=self.running.is_set())


# ================================================================================================ the controller
class SlotState:
    def __init__(self, spec):
        self.spec = spec
        self.s = spec["slot"]
        self.name = spec["name"]
        self.state = "BOOT"
        self.inflight = 0
        self.last_arr = 0.0
        self.last_done = 0.0
        self.awake_since = 0.0
        self.wake_pending = None        # the r_w request id waiting for a park to finish
        self.wake_id = None
        self.wake_seq = 0
        self.park_seq = 0
        self.w_bytes = None
        self.dead_why = None


class FabricController:
    def __init__(self, cfg):
        self.cfg = validate_cfg(cfg)
        self.mode = cfg["mode"]
        self.K = float(cfg.get("keepalive_s", 300.0))
        self.tick = float(cfg.get("tick_s", 0.25))
        self.scale = float(cfg.get("time_scale", 1.0))
        self.hold = float(cfg.get("rehome_hold_s", 0.0))
        self.park_hold = float(cfg.get("park_hold_s", 0.0))     # test knob (default 0): stretch a park before its sleep
        self.req_timeout = float(cfg.get("request_timeout_s", 600.0))
        self.host = cfg.get("host", "127.0.0.1")
        self.log = EventLog(cfg["log_dir"])
        self.mu = threading.RLock()
        self.cv = threading.Condition(self.mu)
        self.slots = {sp["slot"]: SlotState(sp) for sp in cfg["slots"]}
        self.by_name = {st.name: st for st in self.slots.values()}
        self.children = {}
        self.batch = {}
        self.core = None
        self.origin = time.monotonic()
        self.guests = {}                # key -> {"copies": [copy...], "gid": str, "nbytes": int, "manifest_id": str}
        self.wake_source = {}           # wake_id -> source (a request row's tier = WARM-WAKE / COOL-WAKE from it)
        self.req_seq = itertools.count(1)
        self.stopping = threading.Event()
        self.spawn_mu = threading.Lock()                # spawn+register vs stop: no child can appear after the sweep
        self.threads = []
        self.srv = None

    # -- clocks
    def minute(self, now=None):
        now = time.monotonic() if now is None else now
        return int(max(0.0, now - self.origin) * self.scale // 60)

    def ev(self, ev, **kw):
        row = dict(ev=ev, minute=self.minute())
        row.update(kw)                                  # a caller's own minute (the wake's) wins; never a kw clash
        self.log.write("events", row)

    def _url(self, slot, role):
        sp = self.slots[slot].spec
        port = sp["holder_port"] if role == "holder" else sp[role]["port"]
        return f"http://{self.host}:{port}"

    # -- boot
    def boot(self):
        cfg = self.cfg
        if cfg["backend"] == "stub":
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            import importlib.util
            sf = cfg["stub"].get("stub_file") or os.path.join(HERE, "warm_wake_stub.py")
            spec = importlib.util.spec_from_file_location("bb_stub_for_ctl", sf)
            stub = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stub)
            stub.StubFabric.create(cfg["stub"]["fabric_dir"], ngpu=len(cfg["gpus"]),
                                   cap_bytes=int(cfg["stub"]["gpu_cap_bytes"]))
        t0 = time.monotonic()
        if self.mode == "fabric":                       # QA 2026-10-21: host mode places no guests, so it runs no holders
            for s, st in self.slots.items():
                self._spawn(s, "holder")
            for s in self.slots:
                wait_health(self._url(s, "holder"), cfg.get("boot_timeout_s", 120), self.children[(s, "holder")].proc)
        errs = []

        def boot_gpu(s):
            try:
                self._boot_slot(s)
            except Exception as e:  # noqa: BLE001
                errs.append(f"slot {s}: {e}")
        ths = [threading.Thread(target=boot_gpu, args=(s,), daemon=True) for s in self.slots]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if errs:
            raise RuntimeError("; ".join(errs)[:600])
        units = {s: {st.name: st.w_bytes / GIB} for s, st in self.slots.items()}
        cap = 0.0 if self.mode != "fabric" else int(cfg["cap_bytes"]) / GIB
        self.core = FabricCore(list(self.slots), {s: 0 for s in self.slots}, units, cap, "REHOME", "SAME", None, True)
        now = time.monotonic()
        self.origin = now
        with self.mu:
            for st in self.slots.values():
                st.state, st.awake_since, st.last_done = "AWAKE", now, now
        self.log.write("boot", dict(ev="boot_done", s=round(now - t0, 3), mode=self.mode,
                                    w_bytes={st.name: st.w_bytes for st in self.slots.values()},
                                    pids={f"{k[0]}:{k[1]}": c.pid for k, c in self.children.items()}))
        parked = [s for s, st in self.slots.items() if st.spec.get("initial", "awake") == "parked"]
        if parked and self.mode != "ded":
            self.park_batch(parked, why="initial", wait=True)

    def _spawn(self, s, role):
        cmd, env = build_cmds(self.cfg, self.slots[s].spec, role)
        name = f"{self.slots[s].name}_{role}"
        with self.spawn_mu:
            if self.stopping.is_set():                  # never spawn a child once a stop began (it would be orphaned)
                raise RuntimeError("stopping")
            c = Child(name, cmd, env, os.path.join(self.cfg["log_dir"], f"{name}.log")).start()
            self.children[(s, role)] = c
        self.log.write("boot", dict(ev="spawn", slot=s, role=role, pid=c.pid, cmd=cmd[:6]))
        return c

    def _boot_slot(self, s):
        cfg, st = self.cfg, self.slots[s]
        bt = cfg.get("boot_timeout_s", 1800 if cfg["backend"] == "vllm" else 120)
        c = self._spawn(s, "owner")
        wait_health(self._url(s, "owner"), bt, c.proc)
        self._complete(s, "owner", [1, 2, 3], 2)                      # warm-up request (one throwaway)
        if self.mode != "ded":                                        #: the first sleep pays the pinned allocation
            self._sleep(s, "owner", "wait")
            self._wake_all(s, "owner")
            inv = self._rpc(s, "bb_inventory", {})
            by = (inv or {}).get("tags", {}).get("weights", {})
            st.w_bytes = int(by.get("bytes", 0)) if isinstance(by, dict) else 0
            if not st.w_bytes:
                raise RuntimeError(f"bb_inventory gave no weights bytes: {str(inv)[:200]}")
            if inv.get("shape_problem"):
                raise RuntimeError(f"bb_inventory shape_problem: {inv['shape_problem']}")
        else:
            st.w_bytes = 1
        if self.mode != "ded" and st.spec.get("batch"):
            b = self._spawn(s, "batch")
            wait_health(self._url(s, "batch"), bt, b.proc)
            self._complete(s, "batch", [1, 2, 3], 2)
            self._sleep(s, "batch", "wait")
            self.batch[s] = BatchLoad(self, st.spec)
            self.batch[s].start_threads()

    # -- engine calls
    def _complete(self, s, role, prompt, n):
        served = self.slots[s].spec[role].get("served", self.slots[s].name if role == "owner" else
                                              f"{self.slots[s].name}-batch")
        st, js = http_json(self._url(s, role) + "/v1/completions",
                           {"model": served, "prompt": prompt, "max_tokens": n}, timeout=600)
        if st != 200:
            raise RuntimeError(f"{role} {s} completion -> {st} {str(js)[:200]}")
        return js

    def _sleep(self, s, role, mode):
        t0 = time.monotonic()
        st, js = http_json(f"{self._url(s, role)}/sleep?level=1&mode={mode}", timeout=900)
        if st != 200:
            raise RuntimeError(f"{role} {s} sleep -> {st} {str(js)[:200]}")
        return time.monotonic() - t0

    def _is_sleeping(self, s, role):
        st, js = http_json(self._url(s, role) + "/is_sleeping", method="GET", timeout=60)
        if st != 200 or not isinstance(js, dict) or "is_sleeping" not in js:
            raise RuntimeError(f"{role} {s} is_sleeping -> {st} {js}")
        return bool(js["is_sleeping"])

    def _wake_all(self, s, role):
        t0 = time.monotonic()
        st, _ = http_json(self._url(s, role) + "/wake_up", timeout=900)
        if st != 200 or self._is_sleeping(s, role):
            raise RuntimeError(f"{role} {s} wake_up did not wake")
        return time.monotonic() - t0

    def _wake_tag(self, s, tag):
        t0 = time.monotonic()
        st, js = http_json(f"{self._url(s, 'owner')}/wake_up?tags={tag}", timeout=900)
        if st != 200:
            raise RuntimeError(f"owner {s} wake_up?tags={tag} -> {st} {js}")
        return time.monotonic() - t0

    def _rpc(self, s, method, spec, timeout=900):
        st, js = http_json(self._url(s, "owner") + "/collective_rpc",
                           {"method": method, "args": [json.dumps(spec)], "timeout": timeout}, timeout=timeout + 30)
        if st != 200 or not isinstance(js, dict):
            raise RuntimeError(f"rpc {method} on {s} -> {st} {str(js)[:200]}")
        return (js.get("results") or [None])[0]

    def _holder(self, L, path, body, timeout=900):
        try:                                            # review: a dead holder must not kill a healthy owner
            st, js = http_json(self._url(L, "holder") + path, body, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"CONN:{type(e).__name__}:{e}"[:200]}
        if st != 200 or not isinstance(js, dict):
            return {"ok": False, "reason": f"HTTP_{st}"}
        return js

    # -- park
    def park_batch(self, slots, why="ka", wait=False):
        with self.mu:
            due = sorted(s for s in slots if self.slots[s].state == "AWAKE" and self.slots[s].inflight == 0)
            if not due or self.mode == "ded":
                return []
            t = self.minute()
            plan = self.core.step(t, parks=due)
            for s in due:
                st = self.slots[s]
                st.state, st.park_seq = "PARKING", st.park_seq + 1
            planned = {p["key"][0] for p in plan["parks"]}
            todo = list(plan["parks"]) + [dict(key=(s, None), w=0.0, dst=None) for s in due if s not in planned]
            self.ev("park_batch", slots=due, why=why, placements=[(p["key"][0], p["dst"]) for p in plan["parks"]])
        ths = []
        for p in todo:                                  # QA r3: a due slot with no unit still parks (never stuck PARKING)
            th = threading.Thread(target=self._do_park, args=(p,), daemon=True, name=f"park{p['key'][0]}")
            th.start()
            ths.append(th)
        if wait:
            for th in ths:
                th.join()
        return [(p["key"][0], p["dst"]) for p in plan["parks"]]

    def _do_park(self, p):
        key, L = p["key"], p["dst"]
        s = key[0]
        st = self.slots[s]
        t0 = time.monotonic()
        rec = dict(slot=s, lender=L, gid=None, ok=None)
        try:
            if L is not None:
                gid = f"{st.name}-p{st.park_seq}"
                rec["gid"] = gid
                g = dict(gid=gid, nbytes=st.w_bytes, manifest_id=None, copies=[], ready=threading.Event(), ok=False)
                with self.mu:
                    self.guests[key] = g
                ok, why = self._push(s, L, g)
                rec.update(ok=ok, why=why, push_s=round(time.monotonic() - t0, 6))
                if not ok:
                    with self.mu:
                        self.core.drop_guest(key, "push")
                        self.guests.pop(key, None)
                    g["ready"].set()
                    self.ev("park_copy_failed", slot=s, lender=L, why=why)
            if self.park_hold > 0:
                time.sleep(self.park_hold)
            rec["owner_sleep_s"] = round(self._sleep(s, "owner", "wait"), 6)
            if not self._is_sleeping(s, "owner"):
                raise RuntimeError("owner still awake after sleep")
            if self.cfg.get("seal") and rec.get("ok"):
                rec["seal"] = self._rpc(s, "bb_seal_after_sleep", {"guest_id": rec["gid"]})
            if s in self.batch:
                rec["batch_wake_s"] = round(self._wake_all(s, "batch"), 6)
                self.batch[s].running.set()
        except Exception as e:  # noqa: BLE001
            if L is not None:                           # QA r1/r3: never leave a pushed or half-pushed guest behind
                with self.mu:
                    self.core.drop_guest(key, "park_failed")
                    g = self.guests.pop(key, None)
                if g is not None:
                    g["ready"].set()
                    self._holder(L, "/holder/free", {"guest_id": g["gid"], "force": True})
            self._infra(s, f"park: {e}")
            return
        with self.mu:
            st.state = "PARKED"
            rec["park_s"] = round(time.monotonic() - t0, 6)
            self.ev("park_done", **rec)
            pend, st.wake_pending = st.wake_pending, None
            if pend is not None:                        # QA 2026-10-21 (pair): atomic, so no other arrival can claim r_w
                self._start_wake_locked(s, pend[0], pend[1], during_park=True)
            self.cv.notify_all()

    def _push(self, s, L, g):
        st = self.slots[s]
        al = self._holder(L, "/holder/alloc", {"guest_id": g["gid"], "nbytes": g["nbytes"], "owner_id": st.name})
        if not al.get("ok"):
            return False, f"ALLOC:{al.get('reason')}"
        try:
            push = self._rpc(s, "bb_push_to_peer", {"guest": {"guest_id": g["gid"], "ipc_handle": al["ipc_handle"],
                                                              "nbytes": g["nbytes"]}, "owner_id": st.name})
        except Exception as e:  # noqa: BLE001
            push = {"ok": False, "reason": f"RPC:{e}"[:200]}
        if not (push or {}).get("ok"):
            self._holder(L, "/holder/free", {"guest_id": g["gid"], "force": True})
            return False, f"PUSH:{(push or {}).get('reason')}"
        at = self._holder(L, "/holder/attach", {"guest_id": g["gid"], "manifest": push["manifest"], "verify": True})
        if not at.get("ok"):
            self._holder(L, "/holder/free", {"guest_id": g["gid"], "force": True})
            return False, f"ATTACH:{at.get('reason')}"
        with self.mu:
            g["manifest_id"] = push.get("manifest_id")
            g["copies"].append(dict(card=L, handle=al["ipc_handle"], via="park", minute=self.minute(), landed=True))
            g["ok"] = True
        g["ready"].set()
        self.ev("push_done", slot=s, lender=L, gid=g["gid"], bytes=g["nbytes"], rate_gbs=push.get("rate_gbs"),
                verify_s=at.get("seconds"))
        return True, None

    # -- requests
    def acquire(self, name, rid):
        """-> (slot state, tier, wake_id, t_arr) once the engine can take the request; raises on DEAD / timeout."""
        t_arr = time.monotonic()
        with self.mu:
            st = self.by_name.get(name)
            if st is None:
                raise KeyError(name)
            st.last_arr = t_arr
            s0 = st.state
            if s0 == "DEAD":
                raise RuntimeError(f"INFRA: slot {st.s} is dead ({st.dead_why})")
            tier, wid = "HOT", None
            if s0 == "AWAKE":
                st.inflight += 1
                return st, tier, wid, t_arr, s0
            if s0 == "PARKED":
                wid = self._start_wake_locked(st.s, rid, t_arr)
                tier = "WAKE"
            elif s0 == "PARKING":
                if st.wake_pending is None:
                    st.wake_pending = (rid, t_arr)
                    tier = "WAKE"
                    self.ev("arrival_during_park", slot=st.s, rid=rid)
                else:
                    tier = "JOIN"
            else:
                tier = "JOIN"
            deadline = t_arr + self.req_timeout
            while st.state not in ("AWAKE", "DEAD"):
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"slot {st.s} not ready in {self.req_timeout}s")
                self.cv.wait(min(left, 1.0))
            if st.state == "DEAD":
                raise RuntimeError(f"INFRA: slot {st.s} died during its wake ({st.dead_why})")
            st.inflight += 1
            return st, tier, st.wake_id if tier != "HOT" else wid, t_arr, s0

    def release(self, st):
        with self.mu:
            st.inflight -= 1
            st.last_done = time.monotonic()
            self.cv.notify_all()

    # -- reclaim / wake
    def _start_wake_locked(self, s, rid, t_arr, during_park=False):
        st = self.slots[s]
        t = self.minute(t_arr)
        plan = self.core.step(max(t, self.core.t_last), becomes={s}, wakes={s})
        st.state = "WAKING"
        st.wake_seq += 1
        st.wake_id = f"{st.name}-w{st.wake_seq}"
        info = dict(wake_id=st.wake_id, slot=s, rid=rid, t_arr=t_arr, minute=plan["t"], during_park=during_park,
                    evictions=[(k[0], L2) for rc in plan["evictions"] for k, w, L2, ci in rc["items"]],
                    core_src=[w_["src"] for w_ in plan["wakes"]],
                    cowake=any(self.core.is_cowake(w_["key"], plan["t"]) for w_ in plan["wakes"]))
        self.ev("wake_start", **info)
        th = threading.Thread(target=self._do_wake, args=(s, plan, info), daemon=True, name=f"wake{s}")
        th.start()
        return st.wake_id

    def _do_wake(self, s, plan, info):
        st = self.slots[s]
        parts = dict(wake_id=info["wake_id"], slot=s)
        try:
            yth = None
            ydone = threading.Event()
            if s in self.batch:
                self.batch[s].running.clear()

                def yield_():
                    try:
                        parts["yield_s"] = round(self._sleep(s, "batch", "abort"), 6)
                    except Exception as e:  # noqa: BLE001
                        parts["yield_err"] = str(e)[:200]
                    ydone.set()
                yth = threading.Thread(target=yield_, daemon=True)
                yth.start()
            else:
                ydone.set()
            for rc in plan["evictions"]:                # drains run on, NO-WAIT (§3.2h); holder reader refcounts sync
                threading.Thread(target=self._drain, args=(rc,), daemon=True).start()
            ydone.wait()
            if "yield_err" in parts:
                raise RuntimeError(f"batch yield failed: {parts['yield_err']}")
            parts["t_yield_done"] = time.monotonic()
            src, why, rr = "HOST", "NO_GUEST", None
            for w_ in plan["wakes"]:
                src, why, rr = self._restore(s, w_["key"], parts)
            parts["restore_end"] = time.monotonic()
            if rr is not None and rr.get("served") == "NONE":
                raise RuntimeError(f"restore served NONE ({rr.get('fatal')})")
            parts["wake_weights_s"] = round(self._wake_tag(s, "weights"), 6)
            parts["wake_kv_s"] = round(self._wake_tag(s, "kv_cache"), 6)
            if self._is_sleeping(s, "owner"):
                raise RuntimeError("owner still sleeping after both wakes")
        except Exception as e:  # noqa: BLE001
            self._infra(s, f"wake: {e}")
            return
        now = time.monotonic()
        with self.mu:
            st.state, st.awake_since = "AWAKE", now
            self.wake_source[info["wake_id"]] = src
            self.ev("ready", source=src, reason=why, cowake=info["cowake"], t_arr=info["t_arr"], t_ready=now,
                    during_park=info["during_park"], rid=info["rid"], w_bytes=st.w_bytes,
                    D=round(now - info["t_arr"], 6), restore=rr and {k: rr.get(k) for k in
                                                                      ("served", "reason", "bytes", "seconds", "rate_gbs")},
                    **{k: v for k, v in parts.items() if k not in ("slot",)}, slot=s)
            self.cv.notify_all()

    def _restore(self, s, key, parts):
        """Earliest source: a landed copy (HOLDER, or NEW-HOME if it landed by a re-home this minute) -> the reclaimed
        card's copy while it is still moving (RECLAIMED-CARD, the NO-WAIT handoff) -> host. -> (source, reason, result)."""
        st = self.slots[s]
        with self.mu:
            g = self.guests.get(key)
        if g is None or self.mode != "fabric":
            return "HOST", "NO_GUEST", None
        if not g["ready"].wait(self.req_timeout):       # review: a stuck push is named, never folded into host
            self.ev("push_wait_timeout", slot=s, gid=g["gid"])
            return "HOST", "PUSH_WAIT_TIMEOUT", None
        tried = []
        while True:
            with self.mu:
                cands = [c for c in g["copies"] if c.get("landed") and not c.get("freed") and c["card"] not in tried]
                land = [c for c in cands if not c.get("moving_from_me")]
                mov = [c for c in cands if c.get("moving_from_me")]
                c = (land or mov or [None])[-1] if (land or mov) else None
                if c is None:
                    return "HOST", "NO_READABLE_COPY" if tried else "NO_GUEST", None
                if c.get("moving_from_me"):
                    label = "RECLAIMED-CARD"
                elif c["via"] == "rehome" and c["minute"] == self.minute():
                    label = "NEW-HOME"
                else:
                    label = "HOLDER"
                tried.append(c["card"])
            rb = self._holder(c["card"], "/holder/reader_begin", {"guest_id": g["gid"], "reader": parts["wake_id"]})
            if not rb.get("ok"):
                self.ev("reader_refused", slot=s, card=c["card"], reason=rb.get("reason"))
                continue
            try:
                rr = self._rpc(s, "bb_restore_from_peer", {"guest": {"guest_id": g["gid"], "ipc_handle": c["handle"],
                                                                     "nbytes": g["nbytes"],
                                                                     "manifest_id": g["manifest_id"]}})
            except Exception:
                self._holder(c["card"], "/holder/reader_end", {"guest_id": g["gid"], "reader": parts["wake_id"]})
                self._release_guest(key, g)             # review: a failed RPC must not leak the lender's buffer
                raise
            self._holder(c["card"], "/holder/reader_end", {"guest_id": g["gid"], "reader": parts["wake_id"]})
            parts["restore_card"] = c["card"]
            if not isinstance(rr, dict) or rr.get("served") not in ("PEER", "HOST_FALLBACK", "NONE"):
                self._release_guest(key, g)             # QA 2026-10-21 (pair): no result is a broken contract, not a wake
                raise RuntimeError(f"restore returned no verdict: {str(rr)[:200]}")
            if label == "RECLAIMED-CARD":
                self.ev("handoff_read", slot=s, card=c["card"], wake_id=parts["wake_id"])
            self._release_guest(key, g)
            served = (rr or {}).get("served")
            if served == "PEER":
                return label, None, rr
            return ("HOST" if served == "HOST_FALLBACK" else "NONE"), (rr or {}).get("reason"), rr

    def _release_guest(self, key, g):
        """the owner woke: every copy of its guest is garbage (a moving copy's landing is freed by its drain)."""
        with self.mu:
            if self.guests.get(key) is g:
                self.guests.pop(key, None)
            g["dead"] = True
            frees = [c for c in g["copies"] if c.get("landed") and not c.get("freed") and not c.get("moving_from_me")]
            for c in frees:
                c["freed"] = True
        for c in frees:
            self._free_when_idle(c["card"], g["gid"])

    def _free_when_idle(self, card, gid, timeout=60.0):
        t0 = time.monotonic()
        while True:
            r = self._holder(card, "/holder/free", {"guest_id": gid})
            if r.get("ok") or r.get("reason") == "NO_GUEST":
                return r
            if time.monotonic() - t0 > timeout:
                self.ev("free_timeout", card=card, gid=gid, reason=r.get("reason"))
                return self._holder(card, "/holder/free", {"guest_id": gid, "force": True})
            time.sleep(0.02)

    def _drain(self, rc):
        """§3.2h serial drain of one reclaimed card's guests, in the core's eviction order (never raises)."""
        try:
            self._drain_items(rc)
        except Exception as e:  # noqa: BLE001
            self.ev("drain_error", src=rc["L"], err=str(e)[:300])

    def _drain_items(self, rc):
        L = rc["L"]
        for key, w, L2, ci in rc["items"]:
            with self.mu:
                g = self.guests.get(key)
            if g is None:
                continue
            if not g["ready"].wait(self.req_timeout):
                self.ev("push_wait_timeout", key=key, src=L, gid=g["gid"])
                continue
            with self.mu:
                src = next((c for c in g["copies"] if c["card"] == L and c.get("landed") and not c.get("freed")), None)
            if src is None:
                continue
            t0 = time.monotonic()
            if L2 is None:
                with self.mu:
                    src["freed"] = True
                    if not g.get("dead") and self.guests.get(key) is g and not any(
                            c is not src and c.get("landed") and not c.get("freed") for c in g["copies"]):
                        self.guests.pop(key, None)
                self._free_when_idle(L, g["gid"])
                self.ev("rehome_drop", key=key, src=L, gid=g["gid"])
                continue
            with self.mu:
                src["moving_from_me"] = True
            ok, why, landed = self._rehome(L, L2, g)
            with self.mu:
                if landed is not None:
                    if g.get("dead"):
                        landed["freed"] = True
                    g["copies"].append(landed)
                src["freed"] = True
                if not ok:
                    self.core.drop_guest(key, "rehome")
            if landed is not None and g.get("dead"):
                self._free_when_idle(L2, g["gid"])
                self.ev("rehome_land_orphan_freed", key=key, dst=L2, gid=g["gid"])
            self._free_when_idle(L, g["gid"])
            self.ev("rehome_done" if ok else "rehome_failed", key=key, src=L, dst=L2, gid=g["gid"], why=why,
                    s=round(time.monotonic() - t0, 6))

    def _rehome(self, L, L2, g):
        al = self._holder(L2, "/holder/alloc", {"guest_id": g["gid"], "nbytes": g["nbytes"], "owner_id": "rehome"})
        if not al.get("ok"):
            return False, f"ALLOC:{al.get('reason')}", None
        rh = self._holder(L, "/holder/rehome_to", {"guest_id": g["gid"], "dest_handle": al["ipc_handle"],
                                                   "dest_nbytes": al["nbytes"]})
        if not rh.get("ok"):
            self._holder(L2, "/holder/free", {"guest_id": g["gid"], "force": True})
            return False, f"REHOME:{rh.get('reason')}", None
        if self.hold > 0:                                   # test knob: keep the copy 'in flight' (default 0)
            time.sleep(self.hold)
        at = self._holder(L2, "/holder/attach", {"guest_id": g["gid"], "manifest": rh["manifest"], "verify": True})
        if not at.get("ok"):
            self._holder(L2, "/holder/free", {"guest_id": g["gid"], "force": True})
            return False, f"ATTACH:{at.get('reason')}", None
        return True, None, dict(card=L2, handle=al["ipc_handle"], via="rehome", minute=self.minute(), landed=True)

    def _infra(self, s, why):
        st = self.slots[s]
        alive = self.children.get((s, "owner")).alive() if (s, "owner") in self.children else False
        with self.mu:
            st.state, st.dead_why = "DEAD", why[:300]
            self.ev("engine_dead", slot=s, why=why[:300], owner_alive=alive)
            self.cv.notify_all()

    # -- the KA loop + sampler
    def ka_loop(self):
        while not self.stopping.is_set():
            time.sleep(self.tick)
            if self.mode == "ded":
                continue
            try:                                        # review: the whole tick, not only the park call
                now = time.monotonic()
                with self.mu:
                    due = [s for s, st in self.slots.items() if st.state == "AWAKE" and st.inflight == 0
                           and now - max(st.last_arr, st.last_done, st.awake_since) >= self.K]
                if due:
                    self.park_batch(due, why="ka")
            except Exception as e:  # noqa: BLE001  QA 2026-10-21 (pair): one failed tick must not stop the timer
                print(f"KA-PARK-ERROR {e}"[:300], file=sys.stderr, flush=True)

    def sampler_loop(self):
        period = float(self.cfg.get("sample_s", 1.0))
        while not self.stopping.is_set():
            t0 = time.monotonic()
            try:
                self._sample_once()
            except Exception as e:  # noqa: BLE001  review: one bad read must not blind the run
                print(f"SAMPLER-ERROR {type(e).__name__}: {e}"[:300], file=sys.stderr, flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _sample_once(self):
        if True:
            with self.mu:
                rows = {st.s: dict(state=st.state, inflight=st.inflight) for st in self.slots.values()}
                if self.core is not None:               # HBM tax (prereg §5): guest bytes the core holds on each card
                    for L in rows:
                        rows[L]["guest_bytes"] = int(round(sum(self.core.guests[L].values()) * GIB))
            for s, b in self.batch.items():
                rows[s]["batch"] = b.snapshot()
            self.log.write("sampler", dict(ev="sample", slots=rows))

    def snapshot(self):
        with self.mu:
            return dict(mode=self.mode, minute=self.minute(), origin=self.origin,
                        slots={st.name: dict(slot=st.s, state=st.state, inflight=st.inflight, wake_id=st.wake_id,
                                             dead_why=st.dead_why, w_bytes=st.w_bytes) for st in self.slots.values()},
                        pids={f"{k[0]}:{k[1]}": c.pid for k, c in self.children.items()},
                        guests={f"{k[0]}:{k[1]}": dict(gid=g["gid"], copies=[{x: c.get(x) for x in
                                                                                ("card", "via", "landed", "freed",
                                                                                 "moving_from_me")} for c in g["copies"]])
                                for k, g in self.guests.items()},
                        core_guest_bytes={str(L): int(round(sum(self.core.guests[L].values()) * GIB))
                                          for L in self.core.guests} if self.core is not None else {},
                        core=None if self.core is None else dict(stats=dict(self.core.st), x_dropped=self.core.x[
                            "dropped_at_evict"], ctl=dict(self.core.ctl), cowakes=self.core.cowake_events(),
                            held={str(k): v for k, v in self.core.held.items()}),
                        batch={str(s): b.snapshot() for s, b in self.batch.items()})

    def stop(self):
        with self.spawn_mu:
            self.stopping.set()
        for b in self.batch.values():
            b.stop_all.set()
            b.running.clear()
        for c in list(self.children.values()):
            c.stop()
        self.log.close()
        time.sleep(0.2)
        dr = self.log.dropped()
        if dr:                                          # review: rows lost at shutdown are reported, not silent
            print(f"EVENTLOG-DROPPED-AFTER-CLOSE {json.dumps(dr)}", file=sys.stderr, flush=True)


# ================================================================================================ router
class ClientGone(Exception):
    """a write to the CLIENT failed (it hung up): never engine INFRA (bite 3 T3: 145 of 4,106 rows were mislabelled
    INFRA_MIDSTREAM because the replay client closes right after `data: [DONE]`)."""


def make_router(ctl, port=0):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if u.path == "/fabric/state":
                return self._send(200, ctl.snapshot())
            return self._send(404, {"detail": "Not Found"})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"detail": f"bad json: {e}"})
            if u.path == "/v1/completions":
                return self._completion(body, raw)
            if u.path.startswith("/fabric/"):           # review: admin needs the token when one is configured
                tok = ctl.cfg.get("admin_token")
                if tok and self.headers.get("X-BB-Admin") != tok:
                    return self._send(403, {"detail": "admin token required"})
            try:
                if u.path == "/fabric/park":
                    raw_sl = body.get("slots", [])
                    if not isinstance(raw_sl, list) or any(type(x) is not int for x in raw_sl):
                        raise ValueError("slots must be a list of ints")   # review: no 1.9 -> 1
                    sl = list(raw_sl)
                    if any(x not in ctl.slots for x in sl):
                        raise ValueError(f"unknown slots {sl}")
                    out = ctl.park_batch(sl, why="admin", wait=True)
                    return self._send(200, {"placements": out})
                if u.path == "/fabric/origin":
                    mono = float(body["mono"])
                    with ctl.mu:
                        ctl.origin = mono
                    ctl.ev("origin_set", origin=ctl.origin)
                    return self._send(200, {"ok": True, "origin": ctl.origin})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"detail": f"bad admin body: {e}"[:300]})
            if u.path == "/fabric/stop":
                self._send(200, {"ok": True})
                threading.Thread(target=ctl.srv.shutdown, daemon=True).start()
                return None
            return self._send(404, {"detail": "Not Found"})

        def _completion(self, body, raw):
            if not isinstance(body, dict):              # QA r3: a JSON array/scalar gets a 400, never a dropped socket
                return self._send(400, {"detail": "body must be a JSON object"})
            rid = body.pop("bb_rid", None) or f"r{next(ctl.req_seq)}"
            row = dict(ev="request", rid=rid, model=body.get("model"), t_arr=time.monotonic(), wall_arr=time.time())
            st = None
            try:
                st, tier, wid, t_arr, s0 = ctl.acquire(body.get("model"), rid)
            except KeyError:
                row.update(status="BAD_MODEL", code=404)
                ctl.log.write("requests", row)
                return self._send(404, {"detail": f"unknown model {body.get('model')}"})
            except Exception as e:  # noqa: BLE001
                row.update(status="INFRA", code=502, err=str(e)[:200])
                ctl.log.write("requests", row)
                return self._send(502, {"detail": str(e)[:300]})
            row.update(slot=st.s, state_at_arrival=s0, tier=tier, wake_id=wid, t_fwd=time.monotonic())
            rel = {"done": False}
            try:
                code = self._forward(st, body, row, rel)
            except ClientGone as e:
                code = None
                row.update(status="CLIENT_GONE", err=str(e)[:200])
                self.close_connection = True
            except Exception as e:  # noqa: BLE001
                code = None
                row.update(status="INFRA", err=str(e)[:200])
                eng = ctl.children.get((st.s, "owner"))
                if eng is not None and not eng.alive():
                    ctl._infra(st.s, f"engine exited rc={eng.proc.returncode} during a request")
                if row.get("headers_sent"):             # review: never a 2nd status line mid-stream; an
                    row["status"] = "INFRA_MIDSTREAM"   # abrupt close is the client's honest truncation signal
                    self.close_connection = True
                else:
                    try:
                        self._send(502, {"detail": str(e)[:300]})
                    except Exception:  # noqa: BLE001
                        self.close_connection = True
            finally:
                if not rel["done"]:
                    ctl.release(st)
                row["t_end"] = time.monotonic()
                if row.get("tier") == "WAKE":
                    src = ctl.wake_source.get(wid)
                    row["wake_source"] = src
                    row["tier"] = "WARM-WAKE" if src in ("HOLDER", "RECLAIMED-CARD", "NEW-HOME") else "COOL-WAKE"
                if code is not None:
                    row.setdefault("status", "OK" if code == 200 else f"HTTP_{code}")
                    row["code"] = code
                ctl.log.write("requests", row)
            return None

        def _forward(self, st, body, row, rel):
            def engine_done():                          # in flight = the ENGINE's work; it ends when its response is read,
                if not rel["done"]:                     # not when the client write finishes (a parked-while-busy race)
                    rel["done"] = True
                    row["t_engine_done"] = time.monotonic()
                    ctl.release(st)
            conn = http.client.HTTPConnection(ctl.host, st.spec["owner"]["port"], timeout=ctl.req_timeout)
            body = dict(body)
            body["model"] = st.spec["owner"].get("served", st.name)
            if body.get("stream"):
                so = dict(body.get("stream_options") or {})
                so["include_usage"] = True
                body["stream_options"] = so
            try:
                conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
                r = conn.getresponse()
                data = r.read() if (not body.get("stream") or r.status != 200) else None
            except Exception:
                conn.close()
                raise
            if data is not None:
                row["t_first_byte"] = time.monotonic()
                conn.close()
                engine_done()
                try:
                    js = json.loads(data)
                    row["usage"] = js.get("usage")
                except (json.JSONDecodeError, AttributeError):
                    pass
                self.send_response(r.status)
                self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                row["headers_sent"] = True
                self.wfile.write(data)
                return r.status
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            row["headers_sent"] = True                  # review: after this, an error closes the socket
            try:
                self._relay(r, row)
            finally:
                conn.close()                            # review: closed on every path
            engine_done()
            self._end_stream(row)
            return 200

        def _end_stream(self, row):
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):   # the client left after [DONE]: a normal completion
                row["client_closed_after_done"] = True
                self.close_connection = True

        def _relay(self, r, row):
            for ln in r:
                if "t_first_byte" not in row and ln.startswith(b"data: ") and b'"text"' in ln:
                    row["t_first_byte"] = time.monotonic()
                if ln.startswith(b"data: {") and b'"usage": {' in ln.replace(b'"usage":{', b'"usage": {'):
                    try:
                        ob = json.loads(ln[6:])
                        if ob.get("usage"):
                            row["usage"] = ob["usage"]
                    except json.JSONDecodeError:
                        pass
                try:
                    self.wfile.write(f"{len(ln):x}\r\n".encode() + ln + b"\r\n")
                except (BrokenPipeError, ConnectionResetError) as e:
                    raise ClientGone(f"client write failed: {e}") from e

    return ThreadingHTTPServer((ctl.host, int(port)), H)


def cmd_serve(cfg_path, port=None):
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        ctl = FabricController(cfg)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    def on_sig(*_a):
        if ctl.srv is None:                             # a stop DURING boot: interrupt it; the except below cleans up
            raise KeyboardInterrupt
        threading.Thread(target=ctl.srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_sig)               # installed BEFORE boot: a SIGTERM mid-boot orphaned every child
    signal.signal(signal.SIGINT, on_sig)
    try:
        ctl.boot()
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        print(f"BOOT-FAIL {type(e).__name__} {e}"[:800], file=sys.stderr, flush=True)
        ctl.stop()
        return 1
    ctl.srv = make_router(ctl, cfg.get("port", 0) if port is None else port)
    for fn in (ctl.ka_loop, ctl.sampler_loop):
        th = threading.Thread(target=fn, daemon=True)
        th.start()
    print(f"BB-FABRIC-READY port={ctl.srv.server_address[1]}", flush=True)
    try:
        ctl.srv.serve_forever()
    finally:
        ctl.stop()
    return 0


# ================================================================================================ stub rigs (tests + T3)
def free_ports(n):
    """n distinct ports the kernel says are free right now (bind 0, read, close). A random base collided with a
    concurrent run on a CPU server (2026-10-21: four mutants 'killed' by a boot failure, not by their golden)."""
    import socket
    socks, ports = [], []
    try:
        for _ in range(n):
            sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sk.bind(("0.0.0.0", 0))                     # free on EVERY address: a 127.0.0.1-only pick once returned
            # a port already held on another interface of the same host
            socks.append(sk)
            ports.append(sk.getsockname()[1])
    finally:
        for sk in socks:
            sk.close()
    return ports


def stub_cfg(root, n, mode="fabric", keepalive_s=3600.0, tick_s=0.05, cap_bytes=8 * MIB, w=(1 * MIB, 3 * MIB),
             kv=(2 * MIB,), bw=(2 * MIB,), bkv=(2 * MIB,), batch=True, **kw):
    pp = free_ports(3 * n + 1)
    slots = []
    for i in range(n):
        sl = dict(slot=i, gpu=i, name=f"s{i}", holder_port=pp[3 * i],
                  owner=dict(port=pp[3 * i + 1], stub=dict(weights_bytes=list(w), kv_bytes=list(kv), seed=i)))
        if batch:
            sl["batch"] = dict(port=pp[3 * i + 2], stub=dict(weights_bytes=list(bw), kv_bytes=list(bkv), seed=100 + i))
        slots.append(sl)
    cfg = dict(mode=mode, backend="stub", gpus=list(range(n)), slots=slots, log_dir=os.path.join(root, "logs"),
               keepalive_s=keepalive_s, tick_s=tick_s, cap_bytes=cap_bytes, time_scale=1.0, sample_s=0.5,
               request_timeout_s=60.0, batch_load=dict(streams=2, prompt_tokens=8, max_tokens=4, pace_s=0.05),
               stub=dict(fabric_dir=os.path.join(root, "fab"), gpu_cap_bytes=64 * MIB), port=pp[-1])
    cfg.update(kw)
    return cfg


def _ready_line(proc, tag, timeout):
    """read the child's stdout for `tag` with a hard deadline (no blocking readline past it)."""
    import select
    fd = proc.stdout.fileno()
    buf = b""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{tag}: child exited rc={proc.returncode}: {buf[-400:]!r}")
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            chunk = os.read(fd, 4096)
            if not chunk:
                continue
            buf += chunk
            for ln in buf.split(b"\n")[:-1]:            # complete lines only (review: a partial line raised)
                if ln.startswith(tag.encode()):
                    return int(ln.decode().split("port=")[1].split()[0])
            buf = buf[buf.rfind(b"\n") + 1:]             # QA r3: drop consumed lines
    raise RuntimeError(f"{tag} not seen in {timeout}s: {buf[-400:]!r}")


def _pdeathsig():
    """preexec for the test's `serve` child: SIGTERM it if the test process dies (the mutation tester's timeout SIGKILLs a hung
    mutant run; without this its serve and every stub child were orphaned and dirtied the next law's jail, 2026-10-21).
    Only for serve, spawned from the test's MAIN thread: the signal follows the spawning thread, and the engines are
    spawned from boot threads that exit."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


class Rig:
    """`serve` as a CHILD process (the real entry point) on a stub config."""

    def __init__(self, cfg, root):
        self.cfg, self.root = cfg, root
        os.makedirs(root, exist_ok=True)
        self.cfg_path = os.path.join(root, "fabric.json")
        with open(self.cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=1)
        self.err = open(os.path.join(root, "serve.stderr"), "wb")
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", "--config", self.cfg_path],
                                     stdout=subprocess.PIPE, stderr=self.err, start_new_session=True,
                                     preexec_fn=_pdeathsig if sys.platform.startswith("linux") else None)
        self.child_pids = []
        try:
            self.port = _ready_line(self.proc, "BB-FABRIC-READY", 120)
        except Exception:
            self.stop()                                 # never leave a booting serve (and its children) behind
            raise
        self.url = f"http://127.0.0.1:{self.port}"
        self.t_ready = time.monotonic()
        self.child_pids = [p for p in (self.state() or {}).get("pids", {}).values() if p]

    def req(self, model, max_tokens=8, prompt=(1, 2, 3, 4, 5), plp=True, timeout=60):
        body = {"model": model, "prompt": list(prompt), "max_tokens": max_tokens}
        if plp:
            body["prompt_logprobs"] = 1
        return http_json(self.url + "/v1/completions", body, timeout=timeout)

    def park(self, slots):
        return http_json(self.url + "/fabric/park", {"slots": list(slots)}, timeout=120)[1]

    def state(self):
        return http_json(self.url + "/fabric/state", method="GET")[1]

    def events(self, ev=None):
        p = os.path.join(self.cfg["log_dir"], "events.jsonl")
        rows = [json.loads(x) for x in open(p)] if os.path.exists(p) else []
        return [r for r in rows if ev is None or r.get("ev") == ev]

    def requests(self):
        p = os.path.join(self.cfg["log_dir"], "requests.jsonl")
        return [r for r in (json.loads(x) for x in open(p)) if r.get("ev") == "request"] if os.path.exists(p) else []

    def holder_used(self):
        out = {}
        for sl in self.cfg["slots"]:
            if "holder_port" in sl and self.cfg["mode"] == "fabric":
                out[sl["slot"]] = http_json(f"http://127.0.0.1:{sl['holder_port']}/holder/status", method="GET")[1]["used"]
        return out

    def stop(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        for pid in self.child_pids:                     # children run in their own sessions: reap any survivor
            try:                                        # (a SIGKILLed serve would orphan them; the mutation tester's sandbox saw one)
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not self.proc.stdout.closed:
            self.proc.stdout.close()
        if not self.err.closed:
            self.err.close()
        return self.proc.returncode


def _until(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _lp(js):
    return [None if x is None else list(x.values())[0]["logprob"] for x in js["choices"][0]["prompt_logprobs"]]


# ================================================================================================ self-test
def self_test(fixture=None, only=None):
    res = []
    skipped = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), detail))
        print(("ok    " if cond else "FAIL  ") + name + ("" if cond else f" - {str(detail)[:400]}"), flush=True)

    sys.path.insert(0, HERE)
    import cowake_handoff as cw

    def H(d):
        return {s: [bool(v) for v in h] for s, h in d.items()}

    def W(h):
        return {s: [bool(x[t] and not (x[t - 1] if t else True)) for t in range(len(x))] for s, x in h.items()}

    if only in (None, "core"):
        # CG1
        h1 = H({0: [0, 0, 1, 1, 1, 1], 1: [0, 0, 0, 1, 1, 1], 2: [0] * 6, 3: [1, 0, 0, 0, 0, 1]})
        r1, c1 = core_replay(h1, W(h1), {3: {"g": 10.0}}, {0: 0, 1: 0, 2: 0, 3: 0}, 30.0, "REHOME", 6, "SAME", None, True)
        chain = [(c["kind"], c["src"], c["dst"], c["t"]) for c in r1["cps"]]
        check("CG1 copies chain P 3->0 t1, R 0->1 t2, R 1->2 t3", chain == [("P", 3, 0, 1), ("R", 0, 1, 2), ("R", 1, 2, 3)],
              chain)
        check("CG1 reclaims + rec + wake_chain", [(rc["L"], rc["t"], rc["items"]) for rc in r1["reclaims"]] ==
              [(0, 2, [((3, "g"), 10.0, 1, 1)]), (1, 3, [((3, "g"), 10.0, 2, 2)]), (3, 5, [])]
              and r1["rec"] == {(3, 5, "g"): 2} and r1["wake_chain"][(3, 5, "g")] == (0, 1, 2), r1["reclaims"])
        check("CG1 stats + no co-wake", {k: r1["stats"][k] for k in ("pushes", "rehomed", "evicted", "park_uncached",
                                                                      "lender_gpu_min", "guest_gib_min")} ==
              dict(pushes=3, rehomed=2, evicted=2, park_uncached=0, lender_gpu_min=15, guest_gib_min=40.0)
              and c1.cowake_events() == [], (r1["stats"], c1.cowake_events()))
        # CG2
        h2 = H({0: [0, 0, 1, 1], 1: [0, 0, 0, 0], 2: [1, 0, 1, 1]})
        r2, c2 = core_replay(h2, W(h2), {2: {"g": 12.0}}, {0: 0, 1: 0, 2: 0}, 20.0, "REHOME", 4, "SAME", None, True)
        check("CG2 co-wake (2,0,2,12.0); rec (2,2,g)=1; k[(1,2)]=2",
              c2.cowake_events() == [(2, 0, 2, 12.0)] and r2["rec"] == {(2, 2, "g"): 1} and r2["k"][(1, 2)] == 2,
              (c2.cowake_events(), r2["rec"], dict(r2["k"])))
        r2n, _ = core_replay(h2, W(h2), {2: {"g": 12.0}}, {0: 0, 1: 0, 2: 0}, 20.0, "REHOME", 4, "NEXT", None, True)
        check("CG2 NEXT: rec None + inflight_fallback 1, same placements",
              r2n["rec"] == {(2, 2, "g"): None} and r2n["x"]["inflight_fallback"] == 1
              and [(c["src"], c["dst"]) for c in r2n["cps"]] == [(2, 0), (0, 1)], (r2n["rec"], r2n["x"]["inflight_fallback"]))
        # CG3
        h3 = H({0: [0, 0], 1: [1, 0], 2: [1, 0], 3: [0, 0]})
        r3, c3 = core_replay(h3, W(h3), {1: {"a": 8.0, "b": 15.0}, 2: {"c": 18.0}}, {s: 0 for s in range(4)}, 20.0,
                             "REHOME", 2, "SAME", None, True)
        pl = {c["key"][1]: c["dst"] for c in r3["cps"] if c["kind"] == "P"}
        check("CG3 slot order, units largest first: b->0, a->2, c->1", pl == {"b": 0, "a": 2, "c": 1}, pl)
        h3b = H({0: [0, 0, 0], 1: [1, 0, 1]})
        r3b, _ = core_replay(h3b, W(h3b), {1: {"x": 25.0}}, {0: 0, 1: 0}, 20.0, "REHOME", 3, "SAME", None, True)
        check("CG3b no lender: park_uncached 1, rec None", r3b["stats"]["park_uncached"] == 1
              and r3b["rec"] == {(1, 2, "x"): None}, (r3b["stats"], r3b["rec"]))
        # CG4 adversarial
        h4 = H({0: [0, 0, 0], 1: [1, 0, 0]})
        w4 = W(h4)
        w4[1][2] = True
        try:
            core_replay(h4, w4, {}, {0: 0, 1: 0}, 20.0, "REHOME", 3)
            check("CG4 a wake without becoming held raises I-NB-EVENTS", False, "accepted")
        except CoreError as e:
            check("CG4 a wake without becoming held raises I-NB-EVENTS", "I-NB-EVENTS" in str(e), e)
        try:
            cw.cw_engine({s: __import__("numpy").array(v) for s, v in h4.items()},
                         {s: __import__("numpy").array(v) for s, v in w4.items()}, {}, {0: 0, 1: 0}, 20.0, "REHOME", 3)
            check("CG4 cw_engine raises on the same schedule (parity)", False, "accepted")
        except RuntimeError as e:
            check("CG4 cw_engine raises on the same schedule (parity)", "I-NB-EVENTS" in str(e), e)
        cc = FabricCore([0, 1], {0: 0, 1: 0}, {}, 10.0)
        errs = []
        for f in (lambda: cc.step(0, becomes={0}, wakes={0}), lambda: (cc.step(0, parks={1}), cc.step(1, parks={1})),
                  lambda: cc.step(0, parks={7})):
            try:
                f()
                errs.append("accepted")
            except CoreError:
                errs.append("raised")
        check("CG4 direct step: wake an awake slot / re-park / unknown slot all raise", errs == ["raised"] * 3, errs)
        # B6a
        out = inest(None, cw, 300)
        check("B6a 300 toys x 3 modes == cw_engine (every field)", out["toys"]["ok"] == out["toys"]["n"] == 900,
              out["toys"])
        if not fixture:                                 # QA r3: a missing fixture is a counted SKIP, never silent
            skipped.append("B6b (no fixture)")
            print("SKIP  B6b fixture checks: no fixture file", flush=True)
        if fixture:
            o2 = inest(fixture, cw, 0)
            r = o2["fixture_results"]
            check("B6b fixture node 4: 0 diffs + checks reproduce", not r["node4"]["diffs"] and r["node4"][
                "checks_match_fixture"], (r["node4"]["diffs"], r["node4"]["checks"]))
            check("B6b fixture all 80 slots: 0 diffs + counts reproduce", not r["all"]["diffs"] and r["all"][
                "counts_match_fixture"], (r["all"]["diffs"], r["all"]["counts"]))
    if only in (None, "ctl"):
        root = tempfile.mkdtemp(prefix="bbfab_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        try:
            _ctl_goldens(check, root)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("controller goldens ran to completion", False, repr(e))
        if all(r[1] for r in res):                      # keep the tmpfs dir only when something failed (diagnosis)
            import shutil
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"kept {root} for diagnosis")
    n_ok = sum(1 for r in res if r[1])
    sk = f" ({len(skipped)} skipped: {'; '.join(skipped)})" if skipped else ""
    print(f"SELFTEST-{'OK' if n_ok == len(res) else 'FAIL'} {n_ok if n_ok == len(res) else len(res) - n_ok}/{len(res)}{sk}")
    return 0 if n_ok == len(res) else 1


def _ctl_goldens(check, root):
    # ---- R1 (real entry point)
    rig = Rig(stub_cfg(os.path.join(root, "r1"), 3), os.path.join(root, "r1"))
    try:
        st, a = rig.req("s0")
        lp = {}
        for nm in ("s1", "s2"):
            lp[nm] = _lp(rig.req(nm)[1])
        p1 = rig.park([1, 2])
        p2 = rig.park([0])
        check("R1 park batches: (1)->2, (2)->1 then (0)->1", p1 == {"placements": [[1, 2], [2, 1]]}
              and p2 == {"placements": [[0, 1]]}, (p1, p2))
        lp0 = lp.get("s0") or _lp(a)
        sb, b = rig.req("s1")
        # amendment A1 (post-freeze): wait for B's serial drain to land both guests before C, as the frozen expected
        # values assume (the first run sent C at once and C correctly read the still-moving copy: RECLAIMED-CARD)
        _until(lambda: len(rig.events("rehome_done")) >= 2, 10.0)
        sc, c = rig.req("s2")
        sa, a2 = rig.req("s0")
        _until(lambda: len(rig.requests()) >= 6, 5.0)
        rd = {e["slot"]: e for e in rig.events("ready")}
        rq = {r["rid"]: r for r in rig.requests()}
        tiers = [r.get("tier") for r in sorted(rq.values(), key=lambda r: r["t_arr"])]
        check("R1 B: HOLDER (gpu 2) PEER, no co-wake", rd.get(1, {}).get("source") == "HOLDER"
              and rd[1].get("restore_card") == 2 and rd[1]["restore"]["served"] == "PEER" and not rd[1]["cowake"], rd.get(1))
        check("R1 C: NEW-HOME (gpu 0) PEER, co-wake", rd.get(2, {}).get("source") == "NEW-HOME"
              and rd[2].get("restore_card") == 0 and rd[2]["cowake"], rd.get(2))
        check("R1 A: HOST, no restore, co-wake", rd.get(0, {}).get("source") == "HOST" and rd[0]["restore"] is None
              and rd[0]["cowake"], rd.get(0))
        sn = rig.state()
        cs = sn["core"]["stats"]
        check("R1 core: pushes 5, rehomed 2, evicted 3, dropped 1", (cs["pushes"], cs["rehomed"], cs["evicted"],
                                                                     sn["core"]["x_dropped"]) == (5, 2, 3, 1), sn["core"])
        check("R1 tiers HOT x3 (pre-park reads), WARM-WAKE, WARM-WAKE, COOL-WAKE + 200s + 8 tokens",
              [st, sb, sc, sa] == [200] * 4 and all((x or {}).get("usage", {}).get("completion_tokens") == 8
                                                    for x in (a, b, c, a2))
              and tiers == ["HOT", "HOT", "HOT", "WARM-WAKE", "WARM-WAKE", "COOL-WAKE"], (tiers, [st, sb, sc, sa]))
        check("R1 identity: prompt logprobs after each wake == before its park",
              _lp(b) == lp["s1"] and _lp(c) == lp["s2"] and _lp(a2) == lp0, (_lp(b), lp["s1"]))
        time.sleep(0.3)
        check("R1 every holder ends at used 0", set(rig.holder_used().values()) == {0}, rig.holder_used())
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R1 scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R1 serve exits 0 on SIGTERM", rc == 0, rc)
    # ---- R2a corrupt guest
    rig = Rig(stub_cfg(os.path.join(root, "r2a"), 3, allow_corrupt=True), os.path.join(root, "r2a"))
    try:
        lpb = _lp(rig.req("s1")[1])
        rig.park([1, 2])
        cr = http_json(f"http://127.0.0.1:{rig.cfg['slots'][2]['holder_port']}/holder/corrupt",
                       {"guest_id": "s1-p1", "offset": 1048576 + 12345})[1]
        sb, b = rig.req("s1")
        rd = {e["slot"]: e for e in rig.events("ready")}
        check("R2a corrupt guest -> DIGEST_MISMATCH -> HOST (never WARM), identity holds",
              cr.get("ok") and sb == 200 and rd.get(1, {}).get("source") == "HOST"
              and rd[1].get("reason") == "DIGEST_MISMATCH" and _lp(b) == lpb, (cr, rd.get(1)))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R2a scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R2a serve exits 0 after the scenario", rc == 0, rc)
    # ---- R2b dead engine
    rig = Rig(stub_cfg(os.path.join(root, "r2b"), 3), os.path.join(root, "r2b"))
    try:
        rig.park([1, 2])
        pid = rig.state()["pids"]["2:owner"]
        os.killpg(pid, signal.SIGKILL)
        time.sleep(0.2)
        t0 = time.monotonic()
        sc, c = rig.req("s2", timeout=30)
        dt = time.monotonic() - t0
        dead = [e for e in rig.events("engine_dead") if e["slot"] == 2]
        rows = [r for r in rig.requests() if r.get("slot") == 2 or r.get("model") == "s2"]
        sa, _ = rig.req("s0")
        check("R2b dead engine -> 502 within 15 s + engine_dead + INFRA row; others keep serving",
              sc == 502 and dt < 15 and dead and rows and rows[-1]["status"] == "INFRA" and sa == 200,
              (sc, round(dt, 2), dead, rows[-1:] if rows else None, sa))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R2b scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R2b serve exits 0 after the scenario", rc == 0, rc)
    # ---- R3 co-wake NO-WAIT handoff
    rig = Rig(stub_cfg(os.path.join(root, "r3"), 4, rehome_hold_s=1.5), os.path.join(root, "r3"))
    try:
        lpb = _lp(rig.req("s1")[1])
        p = rig.park([1, 2, 3])
        check("R3 park: (1)->2, (2)->1, (3)->1", p == {"placements": [[1, 2], [2, 1], [3, 1]]}, p)
        out = {}

        def rq(nm):
            out[nm] = rig.req(nm)
        tc = threading.Thread(target=rq, args=("s2",))
        tc.start()
        time.sleep(0.3)
        rq("s1")
        tc.join()
        time.sleep(2.0)
        rd = {e["slot"]: e for e in rig.events("ready")}
        hr = rig.events("handoff_read")
        check("R3 C: HOLDER (gpu 1), no co-wake", rd.get(2, {}).get("source") == "HOLDER" and rd[2].get("restore_card") == 1
              and not rd[2]["cowake"], rd.get(2))
        check("R3 B: RECLAIMED-CARD (holder 2) PEER, co-wake, 1 handoff read",
              rd.get(1, {}).get("source") == "RECLAIMED-CARD" and rd[1].get("restore_card") == 2 and rd[1]["cowake"]
              and len(hr) == 1 and _lp(out["s1"][1]) == lpb, (rd.get(1), hr))
        dr = rig.events("rehome_drop")
        orph = rig.events("rehome_land_orphan_freed")
        check("R3 D's guest dropped; orphan landing on 3 freed; every holder ends at used 0",
              len(dr) == 1 and dr[0]["src"] == 1 and len(orph) == 1 and orph[0]["dst"] == 3
              and set(rig.holder_used().values()) == {0}, (dr, orph, rig.holder_used()))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R3 scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R3 serve exits 0 after the scenario", rc == 0, rc)
    # ---- R4a DED
    cfg = stub_cfg(os.path.join(root, "r4a"), 2, mode="ded", keepalive_s=0.6, batch=False)
    for sl in cfg["slots"]:
        sl.pop("holder_port")
    rig = Rig(cfg, os.path.join(root, "r4a"))
    try:
        time.sleep(2.0)
        s0, _ = rig.req("s0", plp=False)
        _until(lambda: len(rig.requests()) >= 1, 5.0)
        rq_ = rig.requests()
        check("R4a DED: 0 parks after 2 s idle; request HOT", not rig.events("park_batch") and s0 == 200
              and rq_[-1]["tier"] == "HOT", (rig.events("park_batch"), rq_[-1:]))
        ap = rig.park([0])                              # (added) the guard in park_batch itself, not only ka_loop's
        check("R4a (added) DED: an admin park places nothing and parks nothing", ap == {"placements": []}
              and not rig.events("park_batch"), (ap, rig.events("park_batch")))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R4a scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R4a serve exits 0 after the scenario", rc == 0, rc)
    # ---- R4b KA timer
    rig = Rig(stub_cfg(os.path.join(root, "r4b"), 2, keepalive_s=0.6, tick_s=0.1), os.path.join(root, "r4b"))
    try:
        time.sleep(2.0)
        pb = rig.events("park_batch")
        with open(os.path.join(rig.cfg["log_dir"], "boot.jsonl")) as fh:
            boot = [json.loads(x) for x in fh]
        t_b = next(r["mono"] for r in boot if r.get("ev") == "boot_done")
        check("R4b KA: both park in ONE tick between boot+0.6 and boot+1.2 s: (0)->1, (1)->0",
              len(pb) == 1 and pb[0]["slots"] == [0, 1] and [tuple(x) for x in pb[0]["placements"]] == [(0, 1), (1, 0)]
              and 0.6 <= pb[0]["mono"] - t_b <= 1.2, (pb, t_b))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R4b scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R4b serve exits 0 after the scenario", rc == 0, rc)
    # ---- R4c HOST mode
    rig = Rig(stub_cfg(os.path.join(root, "r4c"), 2, mode="host"), os.path.join(root, "r4c"))
    try:
        p = rig.park([0, 1])
        s0, _ = rig.req("s0", plp=False)
        rd = {e["slot"]: e for e in rig.events("ready")}
        sn = rig.state()
        check("R4c HOST: parks uncached (2), wake source HOST", p == {"placements": [[0, None], [1, None]]}
              and sn["core"]["stats"]["park_uncached"] == 2 and rd.get(0, {}).get("source") == "HOST" and s0 == 200,
              (p, rd.get(0)))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R4c scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R4c serve exits 0 after the scenario", rc == 0, rc)
    # ---- R7 (added after the freeze): SIGTERM DURING boot leaves no child alive (the mutation tester's sandbox caught orphans)
    c7 = stub_cfg(os.path.join(root, "r7"), 3)
    os.makedirs(os.path.join(root, "r7"), exist_ok=True)
    c7p = os.path.join(root, "r7", "fabric.json")
    with open(c7p, "w") as fh:
        json.dump(c7, fh)
    bootf = os.path.join(c7["log_dir"], "boot.jsonl")

    def spawned():
        try:
            with open(bootf) as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
        except (OSError, json.JSONDecodeError):
            return []
        return [r_["pid"] for r_ in rows if r_.get("ev") == "spawn"]    # review: parse ev, never a substring

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:                         # review: conservative: a pid we cannot signal exists
            return True
        try:                                            # a zombie child of a dead parent reads as alive to kill(0)
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().split(")")[-1].split()[0] != "Z"
        except OSError:
            return True                                 # review: an unreadable /proc row fails the check, never passes it
    pr7 = subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", "--config", c7p], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, start_new_session=True)
    kids, rc7, e7 = [], None, b""
    try:
        _until(lambda: len(spawned()) >= 5, 30.0)
        kids = spawned()
        os.kill(pr7.pid, signal.SIGTERM)
        try:
            rc7 = pr7.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc7 = "HUNG"
        time.sleep(0.5)
        left = [p_ for p_ in kids if alive(p_)]
        check("R7 (added) SIGTERM mid-boot: exit 1 + BOOT-FAIL, 0 children left alive",
              rc7 == 1 and kids and not left, (rc7, kids, left))
    finally:                                            # review: nothing of R7 survives a failure inside it
        for p_ in [pr7.pid] + spawned():
            try:
                os.killpg(p_, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            _, e7 = pr7.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    check("R7 (added) stderr names BOOT-FAIL", b"BOOT-FAIL" in (e7 or b""), (e7 or b"")[-200:])
    # ---- R6 (added after the freeze): a request during an unfinished park is r_w of the wake that follows the park
    rig = Rig(stub_cfg(os.path.join(root, "r6"), 3, park_hold_s=1.0), os.path.join(root, "r6"))
    try:
        lpb = _lp(rig.req("s1")[1])
        rig.park([2])                                   # a parked lender first (no lender of its own: uncached)
        pt = threading.Thread(target=rig.park, args=([1],))
        pt.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        sb, b = rig.req("s1")
        pt.join()
        rd = [e for e in rig.events("ready") if e["slot"] == 1]
        adp = rig.events("arrival_during_park")
        check("R6 (added) mid-park arrival: one wake after the park, during_park, D includes the park remainder",
              sb == 200 and len(adp) == 1 and len(rd) == 1 and rd[0]["during_park"] and rd[0]["D"] >= 0.6
              and rd[0]["source"] == "HOLDER" and _lp(b) == lpb and time.monotonic() - t0 >= 0.6, (adp, rd))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R6 scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R6 (added) serve exits 0", rc == 0, rc)
    # ---- R8 (added after the freeze, from bite 3's T3): a client hang-up is never engine INFRA
    rig = Rig(stub_cfg(os.path.join(root, "r8"), 2, batch=False), os.path.join(root, "r8"))
    try:
        def stream(nmax, stop_after_first):
            c = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
            c.request("POST", "/v1/completions", json.dumps({"model": "s0", "prompt": [1, 2, 3], "max_tokens": nmax,
                                                              "stream": True}), {"Content-Type": "application/json"})
            r = c.getresponse()
            for ln in r:
                if stop_after_first and ln.startswith(b"data: {"):
                    break
                if ln.startswith(b"data: [DONE]"):
                    break
            c.close()                                   # hang up without reading the terminating chunk
        stream(8, False)
        stream(20000, True)
        _until(lambda: len(rig.requests()) >= 2, 10.0)
        time.sleep(0.5)
        rq = rig.requests()
        s0, _ = rig.req("s0", plp=False)
        st = {r["slot"] if "slot" in r else None for r in rq}
        check("R8 (added) hang-up after [DONE] = OK; mid-stream hang-up = CLIENT_GONE (never INFRA); the slot serves on",
              [r.get("status") for r in rq[:2]] == ["OK", "CLIENT_GONE"] and s0 == 200 and not rig.events("engine_dead"),
              ([r.get("status") for r in rq], s0, st))
    except Exception as e:  # noqa: BLE001
        check("R8 scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R8 (added) serve exits 0", rc == 0, rc)
    # ---- R8c/R8d (added): the same two paths at unit level, deterministic (a write into a just-closed socket usually
    # succeeds -- the RST comes later -- so R8 alone could not force them; mutant K11 survived R8 on a CPU server)
    import types

    class _Broken:
        def write(self, b):
            raise BrokenPipeError("peer closed")

        def flush(self):
            raise BrokenPipeError("peer closed")
    srv0 = make_router(types.SimpleNamespace(host="127.0.0.1"), 0)
    Hc = srv0.RequestHandlerClass
    srv0.server_close()
    h = Hc.__new__(Hc)
    h.wfile, h.close_connection = _Broken(), False
    row = {}
    try:
        h._end_stream(row)
        ok_end = row.get("client_closed_after_done") is True and h.close_connection is True
    except Exception as e:  # noqa: BLE001
        ok_end = repr(e)
    check("R8c (added) a broken socket after [DONE] = a normal completion (unit)", ok_end is True, ok_end)
    try:
        h._relay([b'data: {"choices": [{"text": "t"}]}\n'], {})
        got = "no exception"
    except ClientGone:
        got = "ClientGone"
    except Exception as e:  # noqa: BLE001
        got = repr(e)
    check("R8d (added) a broken socket mid-stream raises ClientGone, not a generic error (unit)", got == "ClientGone", got)
    # ---- R5 (added after the freeze, from review): admin token, malformed admin body, loopback-only config
    rig = Rig(stub_cfg(os.path.join(root, "r5"), 2, batch=False, admin_token="t0k"), os.path.join(root, "r5"))
    try:
        st1, _ = http_json(rig.url + "/fabric/park", {"slots": [0]})
        req = urllib.request.Request(rig.url + "/fabric/park", data=json.dumps({"slots": ["x"]}).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-BB-Admin": "t0k"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                st2 = r.status
        except urllib.error.HTTPError as e:
            st2 = e.code
        check("R5 (added) admin: 403 without the token, 400 on a malformed body with it", (st1, st2) == (403, 400),
              (st1, st2))
    except Exception as e:  # noqa: BLE001  review: one scenario's crash is its FAIL, the rest still run
        check("R5 scenario ran without an exception", False, repr(e))
    finally:
        rc = rig.stop()
    check("R5 (added) serve exits 0", rc == 0, rc)
    bad = stub_cfg(os.path.join(root, "r5b"), 1, host="0.0.0.0")
    os.makedirs(os.path.join(root, "r5b"), exist_ok=True)
    bp = os.path.join(root, "r5b", "fabric.json")
    with open(bp, "w") as fh:
        json.dump(bad, fh)
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "serve", "--config", bp], capture_output=True,
                        text=True, timeout=60)
    check("R5 (added) a non-loopback host is refused at config: exit 2 + ERROR", pr.returncode == 2
          and "loopback" in pr.stderr, (pr.returncode, pr.stderr[-200:]))
    # ---- R9 T3's tmpfs root: dropped on PASS, kept on FAIL (the file inside survives a FAIL only)
    for ok_flag, want_gone in ((True, True), (False, False)):
        t3root = tempfile.mkdtemp(prefix="bbfab_t3x_", dir=root)
        with open(os.path.join(t3root, "fabric.json"), "w") as fh:
            fh.write("{}")
        removed = _t3_finish_root(t3root, ok_flag)
        check(f"R9 T3 root with pass={ok_flag} is {'removed' if want_gone else 'kept'}",
              removed == want_gone and os.path.exists(t3root) == (not want_gone), (removed, ok_flag))
    return root


# ================================================================================================ T3
def _t3_finish_root(root, ok):
    """: T3's tmpfs root (stub fabric, logs, config: RAM) is removed on PASS and kept on FAIL for
    diagnosis, like the self-test's. Returns True when removed."""
    if not ok:
        return False
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    return True


def t3_run(out_path=None, seconds=40.0):
    """8 slots shaped like node 4 at 1/1024 (8B 8.79 MiB in 4 allocations; 14B 15.33 MiB; 24B 23.2 MiB; batch 15.26 MiB)
    with real-time KA timers (keep-alive 1.5 s) and a random request trace; checks every wake is fully stamped."""
    root = tempfile.mkdtemp(prefix="bbfab_t3_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    sz = {"8b": [2 * MIB, 2 * MIB, 2 * MIB, 3 * MIB], "14b": [4 * MIB, 4 * MIB, 4 * MIB, 4 * MIB],
          "24b": [6 * MIB, 6 * MIB, 6 * MIB, 6 * MIB]}
    kinds = ["8b", "8b", "8b", "8b", "8b", "14b", "24b", "8b"]
    cfg = stub_cfg(root, 8, keepalive_s=1.5, tick_s=0.1, cap_bytes=36 * MIB, bw=(8 * MIB, 8 * MIB), bkv=(16 * MIB,))
    cfg["stub"]["gpu_cap_bytes"] = 80 * MIB
    for sl, kd in zip(cfg["slots"], kinds):
        sl["owner"]["stub"]["weights_bytes"] = sz[kd]
        sl["owner"]["stub"]["kv_bytes"] = [4 * MIB]
    try:
        rig = Rig(cfg, root)
    except BaseException:                               # boot failure: Rig already stopped its children; the root is ours to drop
        _t3_finish_root(root, True)
        raise
    rng = random.Random(2967)
    res = dict(root=root, seconds=seconds, sent=0, codes=collections.Counter())
    try:
        t0 = time.monotonic()
        clock = threading.Lock()
        busy = {2: 0.6, 5: 0.5, 6: 0.15}                  # mean gaps (s): 6 stays HOT, 2 and 5 park and wake
        nxt = {s: t0 + rng.expovariate(1 / g) for s, g in busy.items()}
        ths = []
        while time.monotonic() - t0 < seconds:
            s = min(nxt, key=nxt.get)
            time.sleep(max(0.0, nxt[s] - time.monotonic()))
            if rng.random() < 0.25:                        # bursts of silence so 2 and 5 park
                nxt[s] = time.monotonic() + rng.uniform(1.8, 4.0)
            else:
                nxt[s] = time.monotonic() + rng.expovariate(1 / busy[s])

            def one(nm):
                try:
                    st_, _ = rig.req(nm, plp=False, timeout=60)
                except Exception as e:  # noqa: BLE001
                    st_ = f"EXC:{type(e).__name__}"
                with clock:                              # review: Counter += is not atomic across threads
                    res["codes"][st_] += 1
            th = threading.Thread(target=one, args=(f"s{s}",))
            th.start()
            ths.append(th)
            res["sent"] += 1
        for th in ths:
            th.join(timeout=90)
        res["hung_threads"] = sum(1 for th in ths if th.is_alive())
        time.sleep(1.0)
        ready = rig.events("ready")
        starts = rig.events("wake_start")
        need = ("t_arr", "t_ready", "D", "source", "t_yield_done", "restore_end", "wake_weights_s", "wake_kv_s")
        res.update(wakes=len(starts), ready=len(ready), dead=len(rig.events("engine_dead")),
                   unstamped=sum(1 for e in ready if any(e.get(k) is None for k in need)),
                   sources=collections.Counter(e.get("source") for e in ready),
                   cowakes=sum(1 for e in ready if e.get("cowake")),
                   handoff_reads=len(rig.events("handoff_read")), parks=len(rig.events("park_done")),
                   rehomes=len(rig.events("rehome_done")), drops=len(rig.events("rehome_drop")),
                   D_max=max((e["D"] for e in ready if e.get("D") is not None), default=None),
                   batch_tokens=sum(b["tokens"] for b in rig.state()["batch"].values()),
                   holders_used_end=rig.holder_used())
        # I-LEDGER: at quiescence every holder holds exactly the bytes the core placed on its card (no leak, no ghost)
        snap = rig.state()
        res["core_guest_bytes_end"] = snap["core_guest_bytes"]
        res["ledger_ok"] = all(int(res["holders_used_end"][k]) == int(snap["core_guest_bytes"].get(str(k), 0))
                               for k in res["holders_used_end"])
        with clock:                                     # review: a straggler thread may still be counting
            res["codes"] = dict(res["codes"])
        res["sources"] = dict(res["sources"])
        res["pass"] = (res["wakes"] > 0 and res["wakes"] == res["ready"] and res["unstamped"] == 0 and res["dead"] == 0
                       and res["hung_threads"] == 0 and res["ledger_ok"]
                       and set(res["codes"]) == {200} and res["sent"] == sum(res["codes"].values()))
    finally:
        res["serve_rc"] = rig.stop()
    res["pass"] = bool(res.get("pass")) and res["serve_rc"] == 0     # QA 2026-10-21: a bad shutdown is a fail
    out = json.dumps(res, default=str, sort_keys=True)
    if not _t3_finish_root(root, res["pass"]):
        print(f"kept {root} for diagnosis", file=sys.stderr)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(out + "\n")
    print(out)
    if not res.get("pass"):
        print("T3-FAIL see the JSON", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", nargs="?", choices=("serve", "inest", "t3"))
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--only", choices=("core", "ctl"), default=None)
    p.add_argument("--config")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--fixture", default=os.path.join(HERE, "..", "data", "e2e_bite2",
                                                     "sched_d1_ka5_s0_order.json.gz"))
    p.add_argument("--toys", type=int, default=300)
    p.add_argument("--out", default=None)
    p.add_argument("--seconds", type=float, default=40.0)
    a = p.parse_args(argv)
    if a.self_test:
        fx = a.fixture if a.fixture and os.path.exists(a.fixture) else None
        return self_test(fx, a.only)
    if a.cmd == "serve":
        if not a.config:
            print("ERROR: serve needs --config", file=sys.stderr)
            return 2
        return cmd_serve(a.config, a.port)
    if a.cmd == "inest":
        if not os.path.exists(a.fixture):
            print(f"ERROR: no fixture {a.fixture}", file=sys.stderr)
            return 2
        out = inest(a.fixture, None, a.toys)
        print(json.dumps(out, default=str, sort_keys=True))
        return 0 if out["pass"] else 1
    if a.cmd == "t3":
        return t3_run(a.out, a.seconds)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
