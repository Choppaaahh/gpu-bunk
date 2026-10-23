#!/usr/bin/env python3
"""peer_read_during_sleep.py — P1: time an NVLink peer read of a cached 15.8 GiB guest FROM card A while A runs
its own STOCK batch preemption (vLLM 0.30.0 `/sleep?level=1&mode=abort` under batch load = ITEM9's yield, the sims'
stock p 0.842), against the idle-lender read on the same pair. Decides WAIT vs NO-WAIT door branch.

Prereg (frozen before this file existed): (private report). On any
divergence the prereg wins. Reuses gpu_driver shapes (stream_req / Loop / Eng, copied minimal, not imported) and
probe read (`w0.copy_(w1)` timed between two syncs) as R0's PUSH-1.

Output contract:
  --self-test                  exit 0 + "SELF-TEST PASS n/n"; exit 1 + FAIL lines. CPU goldens of the frozen rule + a stub
                               flow smoke (in a temp dir; writes nothing else).
  --stub-smoke DIR             the WHOLE box flow on CPU against an in-process stub engine; WRITES DIR/{rows,logs,DONE,ARM_DONE_*};
                               exit 0 iff DONE starts with OK and --score reads a branch from it, else 1.
  --box --out DIR [--gpu-a 0 --gpu-b 1 --deadline EPOCH --passes 5]
                               ON THE BOX. WRITES DIR/rows/*.jsonl, DIR/logs/*, DIR/ARM_DONE_*, DIR/DONE ("OK ..." | "INFRA-...").
                               exit 0 = DONE OK; 3 = INFRA (DONE names it). It never prints a verdict.
  --box-p3 --out DIR [--gpu-a 0 --deadline EPOCH --n-live 30]
                               ON THE BOX, the conditional P3 (same-model p_same, ITEM9 shape). WRITES DIR/rows/p3_*.jsonl,
                               DIR/logs/*, DIR/DONE_P3; exit 0 / 3 as above.
  --score DIR [--json OUT] [--allow-stub]
                               READ-ONLY on DIR (writes OUT only when given): the frozen §3 rule; one JSON object on stdout;
                               exit 0 (VOID is a value, not an error); exit 2 = DIR unreadable.
  --score-p3 DIR               READ-ONLY: p_med / p_hi per state (gpu_score's formula); one JSON object; exit 0.
  --box --dose --out DIR       addendum A1: R0, warm, R1 (PULL), then Y0 / PRE1-PULL x --passes; WRITES as --box.
  --score-dose DIR [--json OUT] READ-ONLY: addendum A1's frozen rule; one JSON object; exit 0.

Top pitfalls (observed in this program, carried here):
  1. **`ssh host '... &'` holds the session open**: launch with `ssh -n` + `setsid -f`, from a shipped script.
  2. **The kvcached image patches vLLM silently**: every vLLM child runs under `env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH`.
  3. **The box's fd soft limit (1024) breaks the load loop**: every vLLM child runs under `ulimit -n 65536`.
  4. **The first sleep of an engine pins host memory (~25 s)**: two warm sleep/wake cycles run before any scored yield.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

GIB = 1024 ** 3
GUEST_BYTES = 16_965_120_816            # 15.8 GiB rounded down to a multiple of 16 (prereg §1)
K_CHUNKS = 16
R_NB, DELTA = 270.0, 0.25               # R_NB (GB/s) and the deciding δ
FLOOR_GBS = 150.0                       # I-NVLINK floor
WAIT_EARLY_S, WAIT_TOL_S = 0.30, 0.05
INFLATE_TOL_S = 0.05
FREE_RISE_MIN = 15 * GIB
READ_TIMEOUT_S = 10.0
MIN_VALID = 3
MODEL = {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "max_model_len": 16640}
BATCH = {"conc": 8, "L": 512, "out": 512, "settle_s": 10.0}
P3 = {"live_in": 512, "live_out": 32, "live_gap_s": 5.0, "n_idle": 8, "batch_conc": 32, "batch_in": 512,
      "batch_out": 1024, "settle_s": 20.0, "full_kv_bytes": 2684354560, "util": 0.90}
PROV = {"source": "nowait_2937", "trigger": "rental-run", "trust_tier": "autonomous-unsupervised"}
ARMS_PRE = ("Y0", "PRE-PUSH", "PRE-PULL")
ARMS_DOSE = ("Y0", "PRE1-PULL")                       # addendum A1: one PULL read per yield
DOSE_F, DOSE_Y_REF = (0.1, 0.3, 0.5, 0.7, 0.9), 0.7325  # A1: read issued at t_send + f * 0.7325 s (P1's server-side Y0 median)
DOOR = {"NO-WAIT HOLDS": "CW-Q-NW: OPEN-AT-STOCK-p on both maps (worst wake 1.55 / 1.87 s at p 0.842)",
        "FALLS BACK TO WAIT": "CW-Q: OPEN-WITH-p<=0.5 (Y 0.75; SHUF breaks at 0.79)"}


def t_star(nbytes=GUEST_BYTES):
    """Copy part of NO-WAIT charge at δ¼, k_L 1: w_GiB*1.073741824/(270*0.25) s == bytes / 67.5e9."""
    return nbytes / (R_NB * DELTA * 1e9)


def med(xs):
    return statistics.median(xs) if xs else None


def gbs(nbytes, t):
    return (nbytes / t / 1e9) if (t and math.isfinite(t) and t > 0) else 0.0


# ------------------------------------------------------------------------------------------------ the frozen rule (pure)
def _rt(r):
    """A read's time for the rule: a failed read (not ok) = +inf, never dropped (prereg §1)."""
    return r["T"] if (r.get("ok") and r.get("T") is not None and r["T"] <= READ_TIMEOUT_S) else math.inf


def pass_valid(p):
    """prereg §2 valid pass. Returns (bool, reasons)."""
    why = []
    if p.get("sleeping_before") is not False:
        why.append("engine not awake before")
    if (p.get("settle_tokens") or 0) < 1:
        why.append("batch streamed no token in the settle")
    if p.get("post_status") != 200:
        why.append(f"POST status {p.get('post_status')}")
    if (p.get("free_rise") or 0) < FREE_RISE_MIN:
        why.append(f"free rise {(p.get('free_rise') or 0) / GIB:.2f} GiB < 15")
    if p["arm"] != "Y0" and not any(0.0 <= r["s"] < p["y"] for r in p.get("reads", [])):
        why.append("no read issued in W")
    return (not why), why


def during(p):
    return [r for r in p.get("reads", []) if 0.0 <= r["s"] < p["y"]]


def wait_obs(p):
    """Some read issued at s <= y-0.30 whose FIRST chunk ended at >= y-0.05 (moved ~nothing until the yield was over)."""
    for r in during(p):
        ce = r.get("chunk_end") or []
        if r["s"] <= p["y"] - WAIT_EARLY_S and ce and ce[0] is not None and ce[0] >= p["y"] - WAIT_TOL_S:
            return True
    return False


def decide(meta, reads_r, passes, allow_stub=False):
    """meta: {link, guest_bytes, pull_available}; reads_r: {(arm, method): [read rows]}; passes: [pass rows].
    Returns the verdict dict (prereg §3). Pure: no I/O."""
    nb = meta.get("guest_bytes", GUEST_BYTES)
    ts = t_star(nb)
    floor = 0.0 if allow_stub else FLOOR_GBS
    link = str(meta.get("link") or "")
    out = {"t_star": ts, "guest_bytes": nb, "link": link, "methods": {}, "flags": []}
    push1 = [_rt(r) for r in reads_r.get(("R0", "PUSH1"), [])]
    r0_gbs = gbs(nb, med(push1)) if push1 else 0.0
    out["i_nvlink"] = {"link": link, "r0_push1_med_gbs": r0_gbs,
                       "pass": (link.startswith("NV") or (allow_stub and link == "STUB")) and r0_gbs >= floor}
    vp = {a: [] for a in ARMS_PRE}
    out["invalid_passes"] = []
    for p in passes:
        ok, why = pass_valid(p)
        (vp[p["arm"]].append(p) if ok else out["invalid_passes"].append({"arm": p["arm"], "rep": p.get("rep"), "why": why}))
    y0 = [p["y"] for p in vp["Y0"]]
    out["y0"] = {"n": len(y0), "min": min(y0) if y0 else None, "med": med(y0), "max": max(y0) if y0 else None}
    avail = []
    for m in ("PUSH", "PULL"):
        d = {"available": not (m == "PULL" and meta.get("pull_available") is False)}
        r1 = [_rt(r) for r in reads_r.get(("R1", m), [])]
        d["r1_med_s"] = med(r1) if r1 else None
        d["r1_med_gbs"] = gbs(nb, d["r1_med_s"]) if r1 else 0.0
        d["p2p"] = d["r1_med_gbs"] >= floor and bool(r1)
        ps = vp[f"PRE-{m}"]
        d["valid_passes"] = len(ps)
        dr = [r for p in ps for r in during(p)]
        d["n_during"] = len(dr)
        d["pass_max_s"] = [max((_rt(r) for r in during(p)), default=None) for p in ps]
        d["t_max_s"] = max((_rt(r) for r in dr), default=None)
        d["no_wait_ok"] = bool(d["p2p"] and d["t_max_s"] is not None and d["t_max_s"] <= ts)
        d["wait_obs"] = any(wait_obs(p) for p in ps)
        yp = [p["y"] for p in ps]
        d["y_pre_med"] = med(yp)
        d["yield_cost_s"] = (med(yp) - med(y0)) if (yp and len(y0) >= MIN_VALID) else None
        d["inflated"] = None if d["yield_cost_s"] is None else d["yield_cost_s"] > INFLATE_TOL_S
        d["slowdown_vs_r1"] = (d["t_max_s"] / d["r1_med_s"]) if (d["t_max_s"] and d["r1_med_s"]
                                                                  and math.isfinite(d["t_max_s"])) else None
        d["scored"] = d["available"] and d["p2p"] and len(ps) >= MIN_VALID
        if d["scored"]:
            avail.append(m)
        out["methods"][m] = d
    if not out["i_nvlink"]["pass"]:
        out.update(branch="VOID", reason="I-NVLINK failed (topology or R0 rate)")
    elif not avail:
        out.update(branch="VOID", reason="no PRE method has >= 3 valid passes on a p2p-capable, available method")
    else:
        M = out["methods"]
        if any(M[m]["no_wait_ok"] and M[m]["inflated"] is False for m in avail):
            out.update(branch="NO-WAIT HOLDS", reason="a read during the stock preemption ran at >= δ¼ of 270 GB/s, "
                                                      "yield not inflated",
                       by=[m for m in avail if M[m]["no_wait_ok"] and M[m]["inflated"] is False])
        elif all(M[m]["wait_obs"] for m in avail):
            out.update(branch="FALLS BACK TO WAIT", reason="every available method held a read until the yield ended")
        else:
            rs = []
            if any(M[m]["no_wait_ok"] and M[m]["inflated"] is None for m in avail):
                rs.append("YIELD-UNMEASURED")
            if any(M[m]["no_wait_ok"] and M[m]["inflated"] for m in avail):
                rs.append("YIELD-INFLATED")
            if any((not M[m]["no_wait_ok"]) and not M[m]["wait_obs"] for m in avail):
                rs.append("SLOW-NO-WAIT")
            if any(M[m]["wait_obs"] for m in avail) and not all(M[m]["wait_obs"] for m in avail):
                rs.append("MIXED")
            out.update(branch="UNRESOLVED", reason="+".join(rs) or "MIXED")
    out["door"] = DOOR.get(out["branch"], "no door reading (see prereg §3)")
    return out


def decide_dose(meta, reads_r, passes, allow_stub=False):
    """Addendum A1's frozen rule (one read per yield). Pure."""
    nb = meta.get("guest_bytes", GUEST_BYTES)
    ts = t_star(nb)
    floor = 0.0 if allow_stub else FLOOR_GBS
    link = str(meta.get("link") or "")
    push1 = [_rt(r) for r in reads_r.get(("R0", "PUSH1"), [])]
    r0g = gbs(nb, med(push1)) if push1 else 0.0
    out = {"t_star": ts, "link": link, "i_nvlink": (link.startswith("NV") or (allow_stub and link == "STUB")) and r0g >= floor,
           "r0_push1_med_gbs": r0g, "invalid_passes": []}
    vp = {a: [] for a in ARMS_DOSE}
    for p in passes:
        ok, why = pass_valid(p)
        (vp[p["arm"]].append(p) if ok else out["invalid_passes"].append({"arm": p["arm"], "rep": p.get("rep"), "why": why}))
    y0, y1 = [p["y"] for p in vp["Y0"]], [p["y"] for p in vp["PRE1-PULL"]]
    reads = [r for p in vp["PRE1-PULL"] for r in during(p)]
    out.update(n_y0=len(y0), n_pre1=len(y1), y0_med=med(y0), y1_med=med(y1),
               t1max_s=max((_rt(r) for r in reads), default=None),
               per_pass=[{"f": p.get("dose_f"), "y": p["y"], "T": [_rt(r) for r in during(p)],
                          "server_sleep_s": p.get("server_sleep_s")} for p in vp["PRE1-PULL"]])
    out["yield_cost_s"] = (med(y1) - med(y0)) if (y0 and y1) else None
    out["inflated"] = None if out["yield_cost_s"] is None else out["yield_cost_s"] > INFLATE_TOL_S
    held = [wait_obs(dict(p, y=p["y"])) or any((r.get("chunk_end") or [0])[0] >= p["y"] - WAIT_TOL_S for r in during(p))
            for p in vp["PRE1-PULL"] if (p.get("dose_f") or 1) <= 0.5]
    if not out["i_nvlink"]:
        out.update(branch="VOID", reason="I-NVLINK failed")
    elif len(y0) < MIN_VALID or len(y1) < MIN_VALID:
        out.update(branch="VOID", reason=f"valid passes Y0 {len(y0)} / PRE1 {len(y1)} < 3")
    elif out["t1max_s"] is not None and out["t1max_s"] <= ts and out["inflated"] is False:
        out.update(branch="NO-WAIT HOLDS AT ONE READ PER YIELD", reason="T1max <= T* and yield cost <= 0.05 s")
    elif held and all(held):
        out.update(branch="WAIT-1", reason="every early single read was held to the yield's end")
    else:
        rs = (["YIELD-INFLATED"] if out["inflated"] else []) + (["SLOW"] if not (out["t1max_s"] is not None
                                                                              and out["t1max_s"] <= ts) else [])
        out.update(branch="UNRESOLVED-1", reason="+".join(rs) or "MIXED")
    return out


def describe(meta, reads_r, passes):
    """Reported, not deciding: per-arm read stats, the busy-lender rate, bytes moved inside W, drift."""
    nb = meta.get("guest_bytes", GUEST_BYTES)
    arms = {}
    for (a, m), rs in sorted(reads_r.items()):
        ts = [_rt(r) for r in rs]
        fin = [t for t in ts if math.isfinite(t)]
        arms[f"{a}|{m}"] = {"n": len(ts), "failed": len(ts) - len(fin), "min_s": min(fin) if fin else None,
                            "med_s": med(fin), "max_s": max(ts) if ts else None,
                            "med_gbs": gbs(nb, med(fin)) if fin else 0.0, "min_gbs": gbs(nb, max(ts)) if ts else 0.0}
    pre = {}
    for p in passes:
        dr = during(p)
        moved = 0
        for r in dr:
            ce = r.get("chunk_end") or []
            moved += sum(1 for c in ce if c is not None and c < p["y"]) * (nb / max(1, len(ce)))
        pre.setdefault(p["arm"], []).append({"rep": p.get("rep"), "y": p["y"], "n_reads_in_W": len(dr),
                                             "max_T": max((_rt(r) for r in dr), default=None),
                                             "gb_moved_in_W": moved / 1e9, "free_rise_gib": (p.get("free_rise") or 0) / GIB,
                                             "server_sleep_s": p.get("server_sleep_s"), "valid": pass_valid(p)[0]})
    r1 = arms.get("R1|PUSH", {}).get("med_s")
    r1b = arms.get("R1b|PUSH", {}).get("med_s")
    return {"arms": arms, "passes": pre,
            "drift_r1_r1b_push": (r1b / r1 - 1.0) if (r1 and r1b) else None}


def load_run(d):
    d = Path(d)
    meta = json.loads((d / "logs" / "meta.json").read_text())
    reads_r, passes = {}, []
    for f in sorted((d / "rows").glob("reads_*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                reads_r.setdefault((r["arm"], r["method"]), []).append(r)
    pf = d / "rows" / "passes.jsonl"
    if pf.exists():
        passes = [json.loads(ln) for ln in pf.read_text().splitlines() if ln.strip()]
    return meta, reads_r, passes


def score_dir(d, allow_stub=False):
    meta, reads_r, passes = load_run(d)
    v = decide(meta, reads_r, passes, allow_stub=allow_stub)
    v["described"] = describe(meta, reads_r, passes)
    v["meta"] = meta
    return v


def score_p3(d):
    rows = [json.loads(ln) for ln in (Path(d) / "rows" / "p3_live.jsonl").read_text().splitlines() if ln.strip()]
    out = {}
    for var in sorted({r["variant"] for r in rows}):
        idle = [r["ttft"] for r in rows if r["variant"] == var and r["phase"] == "idle" and r.get("ttft") is not None]
        busy = [r["ttft"] for r in rows if r["variant"] == var and r["phase"] == "busy" and r.get("ttft") is not None]
        nbad = sum(1 for r in rows if r["variant"] == var and r.get("ttft") is None)
        if idle and busy:
            out[var] = {"n_idle": len(idle), "n_busy": len(busy), "n_no_ttft": nbad, "ttft_idle_med": med(idle),
                        "ttft_busy_med": med(busy), "p_med": med(busy) - med(idle), "p_hi": max(busy) - med(idle),
                        "busy_max_over_med": max(busy) / med(busy) if med(busy) else None}
        else:
            out[var] = {"status": "INSUFFICIENT", "n_idle": len(idle), "n_busy": len(busy), "n_no_ttft": nbad}
    return out


# ------------------------------------------------------------------------------------------------ box helpers (I/O)
def jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps({**PROV, **r}) + "\n")
        f.flush()


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    with open("logs/drive.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def prompt_ids(rid, n, seed=2937):
    rng = random.Random(f"{seed}:{rid}")
    return [rng.randrange(1000, 30000) for _ in range(n)]


def stream_req(port, name, prompt, max_tokens, extra=None, on_token=None, timeout=900.0):
    """One streaming completion (gpu_driver shape). on_token() is called per streamed text chunk (live batch counter)."""
    body = {"model": name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    t0 = time.time()
    times, usage, status = [], None, "ok"
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if line.startswith(b"data: {") and b'"text"' in line:
                    times.append(time.time())
                    if on_token:
                        on_token()
                if line.startswith(b"data: {") and b'"completion_tokens"' in line:
                    try:
                        usage = (json.loads(line[6:].decode()).get("usage") or {}).get("completion_tokens")
                    except ValueError:
                        pass
    except Exception as e:  # noqa: BLE001
        status = "err:" + repr(e)[:160]
    return {"t_send": t0, "status": status, "tokens": len(times), "usage_tokens": usage,
            "ttft": (times[0] - t0) if times else None, "complete": usage is not None and usage >= max_tokens}


class Loop:
    """Closed-loop streams until stop() (gpu_driver Loop + a live token counter for the settle check)."""

    def __init__(self, port, name, conc, L, out, tag, extra=None):
        self.port, self.name, self.conc, self.L, self.out, self.tag, self.extra = port, name, conc, L, out, tag, extra
        self.ev, self.rows, self.lock, self.tokens = threading.Event(), [], threading.Lock(), 0
        self.th = [threading.Thread(target=self._run, args=(k,), daemon=True) for k in range(conc)]

    def _tok(self):
        with self.lock:
            self.tokens += 1

    def _run(self, k):
        n = 0
        while not self.ev.is_set():
            t0 = time.time()
            r = stream_req(self.port, self.name, prompt_ids(f"{self.tag}:{k}:{n}", self.L), self.out, extra=self.extra,
                           on_token=self._tok)
            with self.lock:
                self.rows.append({**r, "stream": k, "n": n})
            n += 1
            if r["status"] != "ok" and time.time() - t0 < 0.5:
                time.sleep(0.5)          # a fast-failing request (engine asleep) must not spin and starve the GIL

    def start(self):
        [t.start() for t in self.th]
        return self

    def stop(self, join_s=90):
        self.ev.set()
        end = time.time() + join_s
        for t in self.th:
            t.join(timeout=max(0.1, end - time.time()))
        with self.lock:
            return {"rows": len(self.rows), "tokens": self.tokens, "unjoined": sum(t.is_alive() for t in self.th),
                    "errors": sum(1 for r in self.rows if r["status"] != "ok")}


def round_b(port, B, L, out, tag, extra=None):
    """B requests started together, one each (gpu_driver round_B shape): a real warm round."""
    rows, lock = [], threading.Lock()

    def one(i):
        r = stream_req(port, "lender", prompt_ids(f"{tag}:{i}", L), out, extra=extra)
        with lock:
            rows.append(r)
    th = [threading.Thread(target=one, args=(i,)) for i in range(B)]
    [t.start() for t in th]
    [t.join(timeout=600) for t in th]
    return rows


def http(port, path, method="POST", timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=(b"{}" if method == "POST" else None),
                                 headers={"content-type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


class Engine:
    """vLLM 0.30.0 server on ONE card (gpu_driver Eng shape). stub=StubEngine for the CPU smoke."""

    def __init__(self, gpu, port, util, logf, flags=(), stub=None):
        self.gpu, self.port, self.util, self.logf, self.flags, self.stub = gpu, port, util, logf, list(flags), stub
        self.proc, self.t0 = None, None

    def start(self):
        self.t0 = time.time()
        if self.stub:
            self.stub.start(self.port)
            return
        env = {k: v for k, v in os.environ.items() if k not in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH",
                                                                  "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}
        env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), VLLM_SERVER_DEV_MODE="1")
        argv = ["vllm", "serve", MODEL["repo"], "--revision", MODEL["revision"], "--port", str(self.port),
                "--gpu-memory-utilization", f"{self.util}", "--served-model-name", "lender",
                "--max-model-len", str(MODEL["max_model_len"]), "--no-enable-prefix-caching", "--load-format", "dummy",
                *self.flags]
        self.proc = subprocess.Popen(["bash", "-c", "ulimit -n 65536 && exec env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH "
                                      "\"$@\"", "x", *argv], env=env, stdout=open(self.logf, "w"),
                                     stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self):
        return True if self.stub else (self.proc is not None and self.proc.poll() is None)

    def wait_up(self, timeout=1200):
        while time.time() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                http(self.port, "/health", "GET", timeout=5)
                return True, f"up {time.time() - self.t0:.1f}s"
            except Exception:  # noqa: BLE001
                time.sleep(1)
        return False, "timeout"

    def sleeping(self):
        try:
            return bool(json.loads(http(self.port, "/is_sleeping", "GET", timeout=10)[1])["is_sleeping"])
        except Exception:  # noqa: BLE001
            return None

    def stop(self):
        if self.stub:
            self.stub.stop()
            return
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        except ProcessLookupError:
            pass
        time.sleep(5)


def bind_cudart():
    """The CUDA runtime torch already loaded (from /proc/self/maps), for cudaMemcpyPeerAsync on card B's stream.
    Returns (lib | None, path_or_paths)."""
    import ctypes
    paths = []
    try:
        with open("/proc/self/maps") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) >= 6 and "libcudart.so" in parts[-1] and parts[-1] not in paths:
                    paths.append(parts[-1])
    except OSError:
        pass
    for p in paths:
        try:
            lib = ctypes.CDLL(p)
            f = lib.cudaMemcpyPeerAsync
            f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
            f.restype = ctypes.c_int
            g = lib.cudaDeviceEnablePeerAccess
            g.argtypes, g.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
            return lib, p
        except (OSError, AttributeError):
            continue
    return None, paths


class Reader:
    """The guest G on card A (15.8 GiB, random int64) and the target D on card B; one read = all of G into D."""

    def __init__(self, cpu, a, b, nbytes, k=K_CHUNKS, stub=None):
        import torch
        self.t, self.cpu, self.k, self.stub = torch, cpu, k, stub
        n = nbytes // 8
        self.nbytes = n * 8
        if cpu:
            self.dA = self.dB = torch.device("cpu")
            self.ia, self.ib = 0, 0
        else:
            self.dA, self.dB = torch.device(f"cuda:{a}"), torch.device(f"cuda:{b}")
            self.ia, self.ib = a, b
        self.G = torch.empty(n, dtype=torch.int64, device=self.dA)
        self.G.random_(0, 2 ** 62)
        self.gsum = int(self.G.sum().item())
        self.D = torch.zeros(n, dtype=torch.int64, device=self.dB)
        self.bounds = [i * n // k for i in range(k + 1)]
        self.cudart, self.cudart_path, self.peer_rc = None, None, {}
        if not cpu:
            self.sA = torch.cuda.Stream(device=self.dA)
            self.sB = torch.cuda.Stream(device=self.dB)
            self.D[: 1024].copy_(self.G[: 1024])            # torch enables its own p2p mapping (dst -> src)
            self.sync()
            self.cudart, self.cudart_path = bind_cudart()
            if self.cudart is not None:
                for cur, peer in ((self.ib, self.ia), (self.ia, self.ib)):
                    with torch.cuda.device(cur):
                        self.peer_rc[f"{cur}->{peer}"] = int(self.cudart.cudaDeviceEnablePeerAccess(peer, 0))  # 704 = already

    @property
    def pull_available(self):
        return self.cpu or self.cudart is not None

    def sync(self):
        if not self.cpu:
            self.t.cuda.synchronize(self.dA)
            self.t.cuda.synchronize(self.dB)

    def free_a(self):
        if self.stub:
            return self.stub.free_bytes()
        if self.cpu:
            return 0
        return int(self.t.cuda.mem_get_info(self.dA)[0])

    def read(self, method, t_ref=0.0):
        """One timed read. Times are perf_counter seconds minus t_ref. Never raises: a failure is a row with ok False."""
        t, b, k = self.t, self.bounds, self.k
        row = {"method": method, "ok": False, "err": None, "s": None, "e": None, "T": None, "chunk_end": None}
        try:
            self.D.zero_()
            self.sync()
            if method == "PUSH1" or self.cpu:
                s = time.perf_counter()
                if method == "PUSH1":
                    self.D.copy_(self.G, non_blocking=True)
                    self.sync()
                    ce = [time.perf_counter()]
                else:                                   # CPU stub: chunked host copies, host-timed
                    ce = []
                    for i in range(k):
                        self.D[b[i]:b[i + 1]].copy_(self.G[b[i]:b[i + 1]])
                        ce.append(time.perf_counter())
                    if self.stub:
                        self.stub.read_delay()
                        ce[-1] = time.perf_counter()
                e = time.perf_counter()
            elif method == "PUSH":
                evs = [t.cuda.Event(enable_timing=True) for _ in range(k + 1)]
                s = time.perf_counter()
                with t.cuda.stream(self.sA):
                    evs[0].record(self.sA)
                    for i in range(k):
                        self.D[b[i]:b[i + 1]].copy_(self.G[b[i]:b[i + 1]], non_blocking=True)
                        evs[i + 1].record(self.sA)
                self.sync()
                e = time.perf_counter()
                ce = [s + evs[0].elapsed_time(evs[i + 1]) / 1000.0 for i in range(k)]
                row["T_gpu"] = evs[0].elapsed_time(evs[k]) / 1000.0
            elif method == "PULL":
                if self.cudart is None:
                    raise RuntimeError("PULL UNAVAILABLE: libcudart not bound")
                import ctypes
                with t.cuda.device(self.dB):
                    evs = [t.cuda.Event(enable_timing=True) for _ in range(k + 1)]
                    s = time.perf_counter()
                    evs[0].record(self.sB)
                    for i in range(k):
                        off, sz = b[i] * 8, (b[i + 1] - b[i]) * 8
                        rc = self.cudart.cudaMemcpyPeerAsync(ctypes.c_void_p(self.D.data_ptr() + off), self.ib,
                                                             ctypes.c_void_p(self.G.data_ptr() + off), self.ia, sz,
                                                             ctypes.c_void_p(self.sB.cuda_stream))
                        if rc != 0:
                            raise RuntimeError(f"cudaMemcpyPeerAsync rc={rc} chunk {i}")
                        evs[i + 1].record(self.sB)
                self.sync()
                e = time.perf_counter()
                ce = [s + evs[0].elapsed_time(evs[i + 1]) / 1000.0 for i in range(k)]
                row["T_gpu"] = evs[0].elapsed_time(evs[k]) / 1000.0
            else:
                raise ValueError(method)
            got = int(self.D.sum().item())
            row.update(s=s - t_ref, e=e - t_ref, T=e - s, chunk_end=[c - t_ref for c in ce], ok=(got == self.gsum),
                       err=None if got == self.gsum else "checksum mismatch")
        except Exception as ex:  # noqa: BLE001
            row["err"] = repr(ex)[:240]
            try:
                self.sync()
            except Exception:  # noqa: BLE001
                pass
        return row


# ------------------------------------------------------------------------------------------------ CPU stub engine
class StubEngine:
    """In-process HTTP stub of the vLLM endpoints the driver uses (CPU smoke only). Yield = yield_s; free rises 17 GiB."""

    def __init__(self, yield_s=0.4, tok_s=0.002, read_s=0.02):
        self.yield_s, self.tok_s, self.read_s = yield_s, tok_s, read_s
        self.sleeping, self.abort_gen, self.srv, self.lock = False, 0, None, threading.Lock()
        self.awake = threading.Event()
        self.awake.set()

    def free_bytes(self):
        return (40 if self.sleeping else 20) * GIB

    def read_delay(self):
        time.sleep(self.read_s)

    def start(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        st = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    self._json(200, {})
                elif self.path == "/is_sleeping":
                    self._json(200, {"is_sleeping": st.sleeping})
                else:
                    self._json(404, {})

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/sleep"):
                    with st.lock:
                        st.abort_gen += 1
                        st.awake.clear()
                    time.sleep(st.yield_s)
                    st.sleeping = True
                    self._json(200, {})
                elif self.path == "/wake_up":
                    time.sleep(0.1)
                    st.sleeping = False
                    st.awake.set()
                    self._json(200, {})
                elif self.path == "/v1/completions":
                    st.awake.wait(60)
                    gen = st.abort_gen
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    sent = 0
                    for _ in range(int(body.get("max_tokens", 1))):
                        if st.abort_gen != gen:
                            break
                        time.sleep(st.tok_s)
                        self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
                        sent += 1
                    self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":%d}}\n\n' % sent)
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self._json(404, {})

        ThreadingHTTPServer.request_queue_size = 1024
        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()


# ------------------------------------------------------------------------------------------------ the box flow
def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def topo_label(text, a, b):
    """`nvidia-smi topo -m` -> the entry for (GPU<a>, GPU<b>) (e.g. 'NV12', 'NODE'), else None. nvidia-smi underlines the
    header with ANSI escapes even over a non-tty ssh (seen on the box, 2026-10-20): strip them first."""
    import re
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    hdr = None
    for ln in text.splitlines():
        parts = ln.split()
        if not parts:
            continue
        if hdr is None and parts[0].startswith("GPU0"):
            hdr = [p for p in parts if p.startswith("GPU")]
            continue
        if hdr is None and "GPU0" in parts and "GPU1" in parts:
            hdr = [p for p in parts if p.startswith("GPU")]
            continue
        if hdr and parts[0] == f"GPU{a}":
            vals = parts[1:1 + len(hdr)]
            try:
                return vals[hdr.index(f"GPU{b}")]
            except (ValueError, IndexError):
                return None
    return None


def finish(status, code):
    Path("DONE").write_text(status + "\n")
    log(f"DONE {status}")
    return code


def run_window(eng, rd, arm, rep, port, fast, dose_f=None):
    """One preemption cycle: batch -> settle -> POST sleep (+ reads for PRE arms) -> wake -> stop batch. Returns a pass row.
    PRE1-* (addendum A1): exactly one read, issued at t_send + dose_f * DOSE_Y_REF, then wait for the POST to return."""
    p = {"arm": arm, "rep": rep, "sleeping_before": eng.sleeping(), "dose_f": dose_f}
    batch = Loop(port, "lender", BATCH["conc"], BATCH["L"], BATCH["out"], f"b{arm}{rep}").start()
    time.sleep(1.0 if fast else BATCH["settle_s"])
    p["settle_tokens"] = batch.tokens
    free0 = rd.free_a()
    res = {}

    def post():
        res["t_send"] = time.perf_counter()
        res["t_send_epoch"] = time.time()
        try:
            res["status"] = http(port, "/sleep?level=1&mode=abort", timeout=300)[0]
        except Exception as ex:  # noqa: BLE001
            res["status"], res["err"] = None, repr(ex)[:200]
        res["t_ret"] = time.perf_counter()

    th = threading.Thread(target=post, daemon=True)
    th.start()
    while "t_send" not in res:
        time.sleep(0.0002)
    t0 = res["t_send"]
    samples, reads, after = [(0.0, free0)], [], 0
    method = None if arm == "Y0" else arm.split("-", 1)[1]
    single = arm.startswith("PRE1")
    fire_at = t0 + (dose_f or 0.0) * (DOSE_Y_REF if not fast else 0.4)
    while True:
        if "t_ret" in res:
            if method is None or after >= 1 or (single and reads):
                break
            if not single:
                after += 1
        if time.perf_counter() - t0 > 120:
            p["err"] = "window timeout 120 s"
            break
        if method is None or (single and (reads or time.perf_counter() < fire_at)):
            time.sleep(0.002 if single else 0.005)
        else:
            reads.append(rd.read(method, t_ref=t0))
        samples.append((time.perf_counter() - t0, rd.free_a()))
    th.join(timeout=300)
    free1 = rd.free_a()
    p.update(post_status=res.get("status"), post_err=res.get("err"), t_send_epoch=res.get("t_send_epoch"),
             y=(res["t_ret"] - t0) if "t_ret" in res else None, free0=free0, free1=free1, free_rise=free1 - free0,
             free_samples=samples, reads=reads)
    tw = time.perf_counter()
    try:
        p["wake_status"] = http(port, "/wake_up", timeout=300)[0]
    except Exception as ex:  # noqa: BLE001
        p["wake_status"], p["wake_err"] = None, repr(ex)[:200]
    p["wake_s"] = time.perf_counter() - tw
    p["sleeping_after_wake"] = eng.sleeping()
    p["batch"] = batch.stop(join_s=30 if fast else 90)
    if p["y"] is None:
        p["y"] = float("nan")
    return p


def read_series(rd, arm, methods, n, path):
    rows = []
    for rep in range(n):
        for m in (methods if rep % 2 == 0 else list(reversed(methods))):
            r = rd.read(m)
            r.update(arm=arm, rep=rep)
            rows.append(r)
            jsonl(path, [r])
    return rows


def box(args, stub=None):
    out = Path(args.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    if Path("DONE").exists():
        print("REFUSE: DONE exists (a finished run is never relaunched in place)")
        return 2
    fast, cpu = stub is not None, stub is not None
    a, b = args.gpu_a, args.gpu_b
    nbytes = (64 << 20) if cpu else GUEST_BYTES
    port = 8291 if not cpu else _free_port()
    log(f"start box flow cpu={cpu} gpus A={a} B={b} bytes={nbytes}")
    env_txt = []
    for c in ("nvidia-smi", "nvidia-smi topo -m",
              "nvidia-smi --query-gpu=index,name,memory.total,compute_mode,power.limit,pcie.link.gen.current,"
              "pcie.link.width.current --format=csv", "nproc", "free -g",
              "python3 -c 'import torch,vllm;print(torch.__version__,torch.version.cuda,vllm.__version__)'"):
        rc, o = (0, "STUB") if cpu else sh(c, timeout=180)
        env_txt.append(f"$ {c}\n{o}\n[rc={rc}]\n")
    Path("logs/env.txt").write_text("\n".join(env_txt))
    topo = "STUB" if cpu else sh("nvidia-smi topo -m")[1]
    Path("logs/topo.txt").write_text(topo)
    link = "STUB" if cpu else topo_label(topo, a, b)
    meta = {"link": link, "gpu_a": a, "gpu_b": b, "guest_bytes": nbytes, "k_chunks": K_CHUNKS, "cpu": cpu,
            "model": MODEL, "batch": BATCH, "prereg": "9fc7174b7", "t_start_epoch": time.time()}
    Path("logs/meta.json").write_text(json.dumps(meta, indent=1))
    if not cpu and not (link or "").startswith("NV"):
        return finish(f"INFRA-NO-NVLINK link={link}", 3)
    rd = Reader(cpu, 0 if cpu else a, 0 if cpu else b, nbytes, stub=stub)
    meta.update(pull_available=rd.pull_available, cudart=rd.cudart_path, peer_rc=rd.peer_rc, guest_sum=rd.gsum)
    Path("logs/meta.json").write_text(json.dumps(meta, indent=1))
    n_idle = 2 if fast else 5
    r0 = read_series(rd, "R0", ["PUSH1", "PUSH"] + (["PULL"] if rd.pull_available else []), n_idle,
                     "rows/reads_R0.jsonl")
    p1 = [r["T"] for r in r0 if r["method"] == "PUSH1" and r["ok"]]
    r0g = gbs(rd.nbytes, med(p1)) if p1 else 0.0
    meta["r0_push1_med_gbs"] = r0g
    Path("logs/meta.json").write_text(json.dumps(meta, indent=1))
    Path("ARM_DONE_R0").write_text("ok\n")
    log(f"R0 PUSH1 med {r0g:.1f} GB/s link={link} pull_available={rd.pull_available} peer_rc={rd.peer_rc}")
    if not cpu and r0g < FLOOR_GBS:
        return finish(f"INFRA-NVLINK-RATE r0={r0g:.1f}GB/s", 3)
    total = rd.t.cuda.mem_get_info(rd.dA)[1] if not cpu else 80 * GIB
    util = 0.55 if total >= 70 * GIB else 0.50
    if total < 70 * GIB:
        MODEL["max_model_len"] = 4096                      # disclosed deviation for a 40 GB card (KV must hold one seq)
    eng = Engine(a, port, util, "logs/server_lender.log", ["--enable-sleep-mode"], stub=stub)
    meta.update(util=util, max_model_len=MODEL["max_model_len"], card_total_gib=total / GIB)
    Path("logs/meta.json").write_text(json.dumps(meta, indent=1))
    eng.start()
    ok, msg = eng.wait_up(timeout=60 if fast else 1500)
    log(f"engine boot {ok} {msg}")
    if not ok:
        eng.stop()
        return finish(f"INFRA-ENGINE-BOOT {msg}", 3)
    try:
        warm = []
        wr = round_b(port, 2, 64, 8, "warm")
        log(f"warm round {[(r['status'], r['tokens']) for r in wr]}")
        for i in range(2):
            for path in ("/sleep?level=1&mode=wait", "/wake_up"):
                t1 = time.perf_counter()
                try:
                    st = http(port, path, timeout=600)[0]
                except Exception as ex:  # noqa: BLE001
                    st = repr(ex)[:120]
                warm.append({"i": i, "path": path, "s": time.perf_counter() - t1, "status": st})
        jsonl("rows/warm.jsonl", warm)
        Path("ARM_DONE_WARM").write_text("ok\n")
        log(f"warm {[(w['path'], round(w['s'], 3), w['status']) for w in warm]}")
        if eng.sleeping() is not False:
            eng.stop()
            return finish("INFRA-ENGINE-NOT-AWAKE after warm", 3)
        methods = ["PUSH", "PULL"] if rd.pull_available else ["PUSH"]
        arms = ARMS_PRE if rd.pull_available else ARMS_PRE[:2]      # PULL UNAVAILABLE: no PRE-PULL passes (prereg §1)
        if getattr(args, "dose", False):
            raise _DoseDone(_dose_cycles(args, eng, rd, port, fast, n_idle))
        read_series(rd, "R1", methods, n_idle, "rows/reads_R1.jsonl")
        Path("ARM_DONE_R1").write_text("ok\n")
        batch = Loop(port, "lender", BATCH["conc"], BATCH["L"], BATCH["out"], "busyR2").start()
        time.sleep(1.0 if fast else BATCH["settle_s"])
        tok0 = batch.tokens
        read_series(rd, "R2", methods, n_idle, "rows/reads_R2.jsonl")
        bst = batch.stop(join_s=30 if fast else 90)
        jsonl("rows/r2_batch.jsonl", [{"settle_tokens": tok0, **bst}])
        Path("ARM_DONE_R2").write_text("ok\n")
        log(f"R2 busy done (batch tokens {bst['tokens']}, settle {tok0})")
        npass = args.passes
        counts = {arm: 0 for arm in arms}
        valid = {arm: 0 for arm in arms}
        rep = 0
        while any(valid[x] < npass and counts[x] < npass + 3 for x in arms):
            order = list(arms[rep % len(arms):] + arms[:rep % len(arms)])
            for arm in order:
                if valid[arm] >= npass or counts[arm] >= npass + 3:
                    continue
                if args.deadline and time.time() > args.deadline:
                    log("DEADLINE reached: stopping the cycles")
                    raise TimeoutError("deadline")
                if not eng.alive():
                    raise RuntimeError("engine died")
                p = run_window(eng, rd, arm, counts[arm], port, fast)
                counts[arm] += 1
                ok_p, why = pass_valid(p)
                p["valid"], p["why_invalid"] = ok_p, why
                valid[arm] += ok_p
                jsonl("rows/passes.jsonl", [p])
                dr = during(p) if p["y"] == p["y"] else []
                log(f"pass {arm} #{counts[arm]} valid={ok_p} {why} y={p['y']:.3f} reads_in_W={len(dr)} "
                    f"maxT={max((_rt(r) for r in dr), default=float('nan')):.4f} rise={p['free_rise'] / GIB:.2f}GiB")
            rep += 1
        Path("ARM_DONE_CYCLES").write_text(json.dumps({"counts": counts, "valid": valid}) + "\n")
        read_series(rd, "R1b", methods, n_idle, "rows/reads_R1b.jsonl")
        Path("ARM_DONE_R1B").write_text("ok\n")
        status = f"OK valid={valid} counts={counts}"
    except _DoseDone as dd:
        status = dd.status
    except TimeoutError:
        status = "OK-DEADLINE partial"
    except Exception as ex:  # noqa: BLE001
        status = f"INFRA-RUN {repr(ex)[:200]}"
    finally:
        eng.stop()
    return finish(status, 0 if status.startswith("OK") else 3)


class _DoseDone(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def _dose_cycles(args, eng, rd, port, fast, n_idle):
    """Addendum A1: R1 (PULL), then Y0 / PRE1-PULL interleaved, 5 passes each, no extras; finish() in the caller's finally."""
    read_series(rd, "R1", ["PULL"], n_idle, "rows/reads_R1.jsonl")
    Path("ARM_DONE_R1").write_text("ok\n")
    counts = {a: 0 for a in ARMS_DOSE}
    for i in range(args.passes):
        for arm in ARMS_DOSE:
            if args.deadline and time.time() > args.deadline:
                log("DEADLINE reached: stopping the dose cycles")
                raise TimeoutError("deadline")
            f = DOSE_F[i % len(DOSE_F)] if arm != "Y0" else None
            p = run_window(eng, rd, arm, i, port, fast, dose_f=f)
            counts[arm] += 1
            ok_p, why = pass_valid(p)
            p["valid"], p["why_invalid"] = ok_p, why
            jsonl("rows/passes.jsonl", [p])
            dr = during(p) if p["y"] == p["y"] else []
            log(f"dose pass {arm} #{counts[arm]} f={f} valid={ok_p} {why} y={p['y']:.3f} reads_in_W={len(dr)} "
                f"T={[round(r['T'], 4) for r in dr if r.get('T')]}")
    Path("ARM_DONE_DOSE").write_text(json.dumps(counts) + "\n")
    return f"OK dose counts={counts}"


def box_p3(args, stub=None):
    """Conditional P3: ITEM9 same-model arm, n_live per state, dummy weights (prereg §4)."""
    out = Path(args.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    if Path("DONE_P3").exists():
        print("REFUSE: DONE_P3 exists")
        return 2
    fast = stub is not None
    port = 8292 if not fast else _free_port()
    n_live = args.n_live
    status = "OK"
    for var in ("ample", "full"):
        flags = ["--scheduling-policy", "priority"] + (["--kv-cache-memory-bytes", str(P3["full_kv_bytes"])]
                                                       if var == "full" else [])
        eng = Engine(args.gpu_a, port, P3["util"], f"logs/server_p3_{var}.log", flags, stub=stub)
        eng.start()
        ok, msg = eng.wait_up(timeout=60 if fast else 1500)
        log(f"p3 {var} boot {ok} {msg}")
        if not ok:
            eng.stop()
            status = f"INFRA-P3-BOOT {var} {msg}"
            break
        try:
            round_b(port, 2, P3["live_in"], 8, f"pw{var}")
            rows = []
            for k in range(P3["n_idle"]):
                r = stream_req(port, "lender", prompt_ids(f"idle{var}{k}", P3["live_in"]), P3["live_out"],
                               extra={"priority": 0})
                rows.append({**r, "phase": "idle", "k": k, "variant": var})
                time.sleep(0.2 if fast else 1.0)
            batch = Loop(port, "lender", P3["batch_conc"], P3["batch_in"], P3["batch_out"], f"batch{var}",
                         extra={"priority": 10}).start()
            time.sleep(1.0 if fast else P3["settle_s"])
            for k in range(n_live):
                if args.deadline and time.time() > args.deadline:
                    status = "OK-DEADLINE partial"
                    break
                r = stream_req(port, "lender", prompt_ids(f"busy{var}{k}", P3["live_in"]), P3["live_out"],
                               extra={"priority": 0})
                rows.append({**r, "phase": "busy", "k": k, "variant": var})
                time.sleep(0.2 if fast else P3["live_gap_s"])
            bst = batch.stop(join_s=30 if fast else 120)
            jsonl("rows/p3_live.jsonl", rows)
            jsonl("rows/p3_batch.jsonl", [{"variant": var, **bst}])
            Path(f"ARM_DONE_P3_{var}").write_text("ok\n")
        finally:
            eng.stop()
        if status != "OK":
            break
    Path("DONE_P3").write_text(status + "\n")
    log(f"DONE_P3 {status}")
    return 0 if status.startswith("OK") else 3


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ------------------------------------------------------------------------------------------------ self-test (goldens)
def _mk_pass(arm, rep, y, reads=(), rise=17 * GIB, status=200, tok=10, sleeping=False):
    return {"arm": arm, "rep": rep, "y": y, "reads": list(reads), "free_rise": rise, "post_status": status,
            "settle_tokens": tok, "sleeping_before": sleeping}


def _mk_read(s, T, first=None, ok=True, k=K_CHUNKS):
    first = s + T / k if first is None else first
    ce = [first] + [first + (s + T - first) * (i / (k - 1)) for i in range(1, k)]
    return {"s": s, "e": s + T, "T": T, "chunk_end": ce, "ok": ok}


def _series(T, n=5):
    return [{"T": T, "ok": True} for _ in range(n)]


def self_test():
    res = []

    def ck(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"))

    ck("G1 T* = 15.8*1.073741824/(270*0.25) = 0.2513351232 s (the 3.2-byte rounding moves it < 1e-10)",
       abs(t_star() - 0.2513351232474074) < 1e-10, t_star())
    ck("G1b T* equals the bytes / 67.5e9 form", abs(t_star(GUEST_BYTES) - 16_965_120_816 / 67.5e9) < 1e-15)
    meta = {"link": "NV12", "guest_bytes": GUEST_BYTES, "pull_available": True}
    idle = 0.062                                            # scaled to 15.8 GiB: 0.0621 s
    rr = {("R0", "PUSH1"): _series(idle), ("R1", "PUSH"): _series(idle), ("R1", "PULL"): _series(idle)}
    fast_reads = [_mk_read(s, 0.065) for s in (0.001, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80)]
    y0 = [_mk_pass("Y0", i, 0.80) for i in range(5)]
    pre_fast = [_mk_pass(f"PRE-{m}", i, 0.81, fast_reads) for m in ("PUSH", "PULL") for i in range(5)]
    v = decide(meta, rr, y0 + pre_fast)
    ck("G2 fast reads in W + yield +0.01 s -> NO-WAIT HOLDS by both", v["branch"] == "NO-WAIT HOLDS"
       and sorted(v["by"]) == ["PULL", "PUSH"], v)
    ck("G2b the read at s=0.80 (>= y 0.81? no: 0.80 < 0.81) is in W; T_max = 0.065", abs(v["methods"]["PUSH"]["t_max_s"]
                                                                                        - 0.065) < 1e-12)
    # G3 WAIT: reads issued early whose first chunk ends at y-0.02
    wait_reads = [_mk_read(0.001, 0.85, first=0.79), _mk_read(0.851, 0.07)]
    pre_wait = [_mk_pass(f"PRE-{m}", i, 0.81, wait_reads) for m in ("PUSH", "PULL") for i in range(5)]
    v = decide(meta, rr, y0 + pre_wait)
    ck("G3 first chunk held to y-0.02 on both methods -> FALLS BACK TO WAIT", v["branch"] == "FALLS BACK TO WAIT", v)
    ck("G3b the after read (s 0.851 >= y 0.81) is excluded from T_max", abs(v["methods"]["PUSH"]["t_max_s"] - 0.85) < 1e-12)
    # G4 slow but progressing
    slow = [_mk_read(0.001, 0.40), _mk_read(0.41, 0.40)]
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.81, slow) for m in ("PUSH", "PULL") for i in range(5)])
    ck("G4 progress in W at 0.40 s per read -> UNRESOLVED SLOW-NO-WAIT", v["branch"] == "UNRESOLVED"
       and "SLOW-NO-WAIT" in v["reason"], v)
    # G5 adversarial: one failed read in PUSH (fast otherwise), PULL fast -> holds via PULL only
    bad = fast_reads[:3] + [dict(_mk_read(0.25, 0.065), ok=False)] + fast_reads[4:]
    pre5 = [_mk_pass("PRE-PUSH", i, 0.81, bad if i == 2 else fast_reads) for i in range(5)] + \
           [_mk_pass("PRE-PULL", i, 0.81, fast_reads) for i in range(5)]
    v = decide(meta, rr, y0 + pre5)
    ck("G5 a failed read = +inf (never dropped): PUSH not OK; holds by PULL alone", v["branch"] == "NO-WAIT HOLDS"
       and v["by"] == ["PULL"] and v["methods"]["PUSH"]["t_max_s"] == math.inf, v)
    pre5b = [_mk_pass(f"PRE-{m}", i, 0.81, bad if i == 2 else fast_reads) for m in ("PUSH", "PULL") for i in range(5)]
    v = decide(meta, rr, y0 + pre5b)
    ck("G5b failed reads on both methods, no held first chunk -> UNRESOLVED (never WAIT without evidence)",
       v["branch"] == "UNRESOLVED", v)
    # G6 inflated yield
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.90, fast_reads) for m in ("PUSH", "PULL") for i in range(5)])
    ck("G6 fast reads but yield +0.10 s -> UNRESOLVED YIELD-INFLATED", v["branch"] == "UNRESOLVED"
       and "YIELD-INFLATED" in v["reason"], v)
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.85, fast_reads) for m in ("PUSH", "PULL") for i in range(5)])
    ck("G6b yield +0.05 s exactly is NOT inflated (> 0.05 rule)", v["branch"] == "NO-WAIT HOLDS", v)
    # G7 VOID
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.81, fast_reads) for m in ("PUSH", "PULL") for i in range(2)])
    ck("G7 two valid passes per method -> VOID", v["branch"] == "VOID", v)
    v = decide(dict(meta, link="NODE"), rr, y0 + pre_fast)
    ck("G7b topology NODE -> VOID (I-NVLINK)", v["branch"] == "VOID" and "I-NVLINK" in v["reason"], v)
    # G8 boundary
    ts = t_star()
    edge = [_mk_read(0.001, ts)]
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.81, edge) for m in ("PUSH", "PULL") for i in range(5)])
    ck("G8 T_max == T* -> OK (<=)", v["branch"] == "NO-WAIT HOLDS", v)
    edge2 = [_mk_read(0.001, ts + 1e-9)]
    v = decide(meta, rr, y0 + [_mk_pass(f"PRE-{m}", i, 0.81, edge2) for m in ("PUSH", "PULL") for i in range(5)])
    ck("G8b T_max == T* + 1e-9 -> not OK", v["branch"] == "UNRESOLVED", v)
    # G9 NOT-P2P PULL excluded; PUSH WAIT -> WAIT on PUSH alone
    rr9 = dict(rr)
    rr9[("R1", "PULL")] = _series(GUEST_BYTES / 20e9)      # 20 GB/s: host-staged, not the link
    pre9 = [_mk_pass("PRE-PUSH", i, 0.81, wait_reads) for i in range(5)] + \
           [_mk_pass("PRE-PULL", i, 0.81, fast_reads) for i in range(5)]
    v = decide(meta, rr9, y0 + pre9)
    ck("G9 PULL at 20 GB/s idle is NOT-P2P (excluded); PUSH held -> FALLS BACK TO WAIT", v["branch"] == "FALLS BACK TO WAIT"
       and not v["methods"]["PULL"]["p2p"], v)
    # G10 invalid pass excluded (rise < 15 GiB carries the slow read)
    pre10 = [_mk_pass(f"PRE-{m}", i, 0.81, fast_reads) for m in ("PUSH", "PULL") for i in range(5)]
    pre10.append(_mk_pass("PRE-PUSH", 9, 0.81, [_mk_read(0.001, 0.9, first=0.79)], rise=2 * GIB))
    v = decide(meta, rr, y0 + pre10)
    ck("G10 a pass with free rise 2 GiB is INFRA, never scored", v["branch"] == "NO-WAIT HOLDS"
       and len(v["invalid_passes"]) == 1, v)
    # G11 Y0 < 3 valid -> yield unmeasured -> UNRESOLVED
    v = decide(meta, rr, y0[:2] + pre_fast)
    ck("G11 Y0 with 2 valid passes -> UNRESOLVED YIELD-UNMEASURED", v["branch"] == "UNRESOLVED"
       and "YIELD-UNMEASURED" in v["reason"], v)
    # G12 PULL unavailable -> decided on PUSH alone
    v = decide(dict(meta, pull_available=False), rr, y0 + [p for p in pre_fast if p["arm"] == "PRE-PUSH"])
    ck("G12 PULL UNAVAILABLE -> PUSH alone decides", v["branch"] == "NO-WAIT HOLDS" and v["by"] == ["PUSH"], v)
    # G13 mixed: PUSH waits, PULL slow-progress
    pre13 = [_mk_pass("PRE-PUSH", i, 0.81, wait_reads) for i in range(5)] + \
            [_mk_pass("PRE-PULL", i, 0.81, slow) for i in range(5)]
    v = decide(meta, rr, y0 + pre13)
    ck("G13 PUSH held + PULL slow -> UNRESOLVED (MIXED + SLOW-NO-WAIT)", v["branch"] == "UNRESOLVED"
       and "MIXED" in v["reason"], v)
    # G14 topo parser on a real-shaped matrix
    topo = ("\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
            "GPU0\t X \tNV12\t0-63\t0\t\tN/A\nGPU1\tNV12\t X \t64-127\t1\t\tN/A\n")
    ck("G14 topo_label reads NV12 for (0,1)", topo_label(topo, 0, 1) == "NV12", topo_label(topo, 0, 1))
    topo_ansi = ("\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
                 "GPU0\t X \tNV12\t64-127,192-255\t1\t\tN/A\nGPU1\tNV12\t X \t64-127,192-255\t1\t\tN/A\n")
    ck("G14c topo_label strips the ANSI underline in the header (the real box output)",
       topo_label(topo_ansi, 0, 1) == "NV12", topo_label(topo_ansi, 0, 1))
    topo2 = "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNODE\t0-47\nGPU1\tNODE\t X \t0-47\n"
    ck("G14b topo_label reads NODE", topo_label(topo2, 0, 1) == "NODE")
    # G15 wait_obs needs an EARLY read: a held read issued at y-0.2 is not evidence
    late = [_mk_read(0.61, 0.25, first=0.79)]
    ck("G15 a held read issued at y-0.20 is NOT WAIT-OBS", not wait_obs(_mk_pass("PRE-PUSH", 0, 0.81, late)))
    # G19 addendum A1 (one read per yield)
    one = lambda f, y, T=0.065, first=None: _mk_pass("PRE1-PULL", 0, y, [_mk_read(f * DOSE_Y_REF, T, first=first)])
    y0d = [_mk_pass("Y0", i, 0.76) for i in range(5)]
    pd = [dict(one(f, 0.77), dose_f=f) for f in DOSE_F]
    vd = decide_dose(meta, rr, y0d + pd)
    ck("G19 one fast read per yield, +0.01 s -> NO-WAIT HOLDS AT ONE READ PER YIELD",
       vd["branch"] == "NO-WAIT HOLDS AT ONE READ PER YIELD", vd)
    vd = decide_dose(meta, rr, y0d + [dict(one(f, 0.85), dose_f=f) for f in DOSE_F])
    ck("G19b +0.09 s -> UNRESOLVED-1 YIELD-INFLATED", vd["branch"] == "UNRESOLVED-1" and "YIELD-INFLATED" in vd["reason"], vd)
    vd = decide_dose(meta, rr, y0d + [dict(one(f, 0.77, T=0.80 - f * DOSE_Y_REF, first=0.76), dose_f=f) for f in DOSE_F])
    ck("G19c early single reads held to y-0.01 -> WAIT-1", vd["branch"] == "WAIT-1", vd)
    vd = decide_dose(meta, rr, y0d + pd[:2])
    ck("G19d two valid PRE1 passes -> VOID", vd["branch"] == "VOID", vd)
    # G16 p3 scorer formula (gpu_score: p_med = med(busy)-med(idle), p_hi = max(busy)-med(idle))
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "rows").mkdir()
        rows = [{"variant": "ample", "phase": "idle", "ttft": x} for x in (0.040, 0.050, 0.045)] + \
               [{"variant": "ample", "phase": "busy", "ttft": x} for x in (0.10, 0.30, 0.20)]
        (Path(td) / "rows" / "p3_live.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        s3 = score_p3(td)["ample"]
        ck("G16 p3 p_med 0.155 / p_hi 0.255", abs(s3["p_med"] - 0.155) < 1e-12 and abs(s3["p_hi"] - 0.255) < 1e-12, s3)
    # G17 the stub flow end to end (CPU), then the scorer reads a branch from it
    with tempfile.TemporaryDirectory() as td:
        rc, v = stub_smoke(Path(td) / "run", passes=3, quiet=True)
        ck("G17 stub flow: DONE OK + a scored branch + 9 valid passes", rc == 0 and v.get("branch") in
           ("NO-WAIT HOLDS", "FALLS BACK TO WAIT", "UNRESOLVED", "VOID")
           and sum(1 for p in json.loads("[" + ",".join((Path(td) / "run/rows/passes.jsonl").read_text().split("\n")[:-1])
                                          + "]") if p["valid"]) == 9, (rc, v.get("branch"), v.get("reason")))
        rc3 = stub_smoke_p3(Path(td) / "p3", quiet=True)
        ck("G18 stub P3 flow: DONE_P3 OK + both states scored", rc3 == 0, rc3)
    n = len(res)
    print(f"SELF-TEST {'PASS' if all(res) else 'FAIL'} {sum(res)}/{n}")
    return 0 if all(res) else 1


def stub_smoke(d, passes=3, quiet=False):
    cwd = os.getcwd()
    d = Path(d).resolve()
    stub = StubEngine()
    ns = argparse.Namespace(out=str(d), gpu_a=0, gpu_b=1, deadline=None, passes=passes)
    try:
        if quiet:
            with open(os.devnull, "w") as dn:
                old = sys.stdout
                sys.stdout = dn
                try:
                    rc = box(ns, stub=stub)
                finally:
                    sys.stdout = old
        else:
            rc = box(ns, stub=stub)
    finally:
        os.chdir(cwd)
    done = (d / "DONE").read_text() if (d / "DONE").exists() else ""
    if rc != 0 or not done.startswith("OK"):
        return 1, {"done": done}
    v = score_dir(d, allow_stub=True)
    return 0, v


def stub_smoke_p3(d, quiet=False):
    cwd = os.getcwd()
    d = Path(d).resolve()
    ns = argparse.Namespace(out=str(d), gpu_a=0, deadline=None, n_live=4)
    try:
        if quiet:
            with open(os.devnull, "w") as dn:
                old = sys.stdout
                sys.stdout = dn
                try:
                    rc = box_p3(ns, stub=StubEngine())
                finally:
                    sys.stdout = old
        else:
            rc = box_p3(ns, stub=StubEngine())
    finally:
        os.chdir(cwd)
    if rc != 0:
        return rc
    s = score_p3(d)
    return 0 if all("p_hi" in s.get(v, {}) for v in ("ample", "full")) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--stub-smoke", metavar="DIR")
    g.add_argument("--box", action="store_true")
    g.add_argument("--box-p3", action="store_true")
    g.add_argument("--score", metavar="DIR")
    g.add_argument("--score-p3", metavar="DIR")
    g.add_argument("--score-dose", metavar="DIR")
    ap.add_argument("--out")
    ap.add_argument("--gpu-a", type=int, default=0)
    ap.add_argument("--gpu-b", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--n-live", type=int, default=30)
    ap.add_argument("--json")
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--dose", action="store_true", help="with --box: addendum A1's one-read-per-yield arms")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.stub_smoke:
        rc, v = stub_smoke(a.stub_smoke, passes=a.passes)
        print(json.dumps({k: v.get(k) for k in ("branch", "reason", "door")} if rc == 0 else v))
        return rc
    if a.box or a.box_p3:
        if not a.out:
            print("ERROR: --out is required", file=sys.stderr)
            return 2
        return box(a) if a.box else box_p3(a)
    if a.score_dose:
        meta, reads_r, passes = load_run(a.score_dose)
        v = decide_dose(meta, reads_r, passes, allow_stub=a.allow_stub)
        s = json.dumps(v, indent=1, default=str)
        if a.json:
            Path(a.json).write_text(s + "\n")
        print(s)
        return 0
    if a.score or a.score_p3:
        d = a.score or a.score_p3
        if not (Path(d) / "rows").is_dir():
            print(f"ERROR: {d}/rows not found", file=sys.stderr)
            return 2
        v = score_dir(d, allow_stub=a.allow_stub) if a.score else score_p3(d)
        s = json.dumps(v, indent=1, default=str)
        if a.json:
            Path(a.json).write_text(s + "\n")
        print(s)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
