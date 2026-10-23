#!/usr/bin/env python3
"""gpu_driver.py: box driver for the multi-GPU mechanics probe on ONE box with 2 NVLink GPUs (A100 SXM4 class).
Prereg (private report) wins on any divergence. Arms, in plan order (control first):
  CTRL       Qwen3-8B on GPU0: G6 serving control (c=8, in 512, out 128, 30 s) + ANCHOR TPOT at the beta cells
  BETA_IDLE  beta_bench.py on GPU0+GPU1, the neighbour idle
  BETA_BUSY  the same while a Qwen3-8B engine on GPU1 serves its own closed-loop load (the lender), rows per request
  ITEM12     DeepSeek-V2-Lite on GPU0 with --enable-return-routed-experts: routed experts per decode step at B = 1..64
  ITEM10     DeepSeek-V2-Lite on GPU0, UVA expert offload (--cpu-offload-gb, --cpu-offload-params experts) for the
             first L_off MoE layers, TPOT at B = 1, 4, 16 (L_off 0 = the control, first)
  ITEM9      preemption delay p: same model (priority scheduling) and other model (sleep B / wake A, mode abort + wait)
  ITEM8      a KV-starved Qwen3-8B (2 GiB KV) under an open-loop overload: clients that never time out, then 10 s cancel
  DCP        DeepSeek-V2-Lite TP2: DCP 1 vs DCP 2 (TRITON_MLA on sm80), TPOT at c = 1, 8 (reported, never deciding)
Gates before any scored row: env, host gate per GPU (host_gate.py, -i N), NVLink (topo NV# + p2p >= plan floor both ways),
net >= 40 MB/s downloads, one throwaway Qwen3-8B warm boot G6 on box (TPOT p50 <= (1 + tol) x roofline).
Each arm writes rows/*.jsonl, logs/*, ARM_DONE_<arm>; an arm skipped for budget/deadline is listed in logs/skips.json.
Env: MG_DEADLINE (epoch), MG_RENT_EPOCH, MG_DPH, MG_DOWN_COST ($/GB, x2.3 meter), MG_CAP_GUARD ($), MG_ROOT, MG_PLAN,
  MG_STUB (CPU host only: stub vllm, CPU bench, no downloads).
Output contract: WRITES rows/, logs/, ARM_DONE_*, DONE (one status line: OK / INFRA-* / PREMISE-FAIL / CRASH). Exit 0 once
started (the status lives in DONE). `--self-test` = exit 0 + "SELF-TEST PASS n/n", else exit 1 + FAIL lines.
Derived from (private report) (Eng, sse parsing, download, env blocks copied in spirit).
"""
import base64
import io
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.environ.get("MG_ROOT", "/root/mg")
STUB = os.environ.get("MG_STUB") == "1"
GIB = 1024 ** 3
ENV, PY, VLLM = {}, sys.executable, "vllm"
DL = RENT = DPH = DOWN = CAP_GUARD = 0.0
STATE = {"dl_gb": 0.0, "probe": {}}
_RX_BACKEND = re.compile(r"Using (?:AttentionBackendEnum\.)?([A-Z][A-Z0-9_]+)(?: attention)? backend\b")
_RX_OFFLOAD = re.compile(r"Total CPU offloaded parameters: ([0-9.]+) ?GiB")
_RX_MODEL_MEM = re.compile(r"Model loading took ([0-9.]+) GiB")        # gpu_model_runner.py:5417 @ v0.30.0 (I-OFF2)
_RX_TOPO_ROW = re.compile(r"^GPU0\s+(\S+)\s+(\S+)", re.M)
PROV = {"source": "gpu_driver", "trigger": "rental-run", "trust_tier": "autonomous-unsupervised"}


# ----------------------------------------------------------------------------------------------- pure helpers (goldens)
def prompt_ids(rid, n, seed=2746):
    rng = random.Random(f"{seed}:{rid}")
    return [rng.randrange(1000, 30000) for _ in range(n)]


def sse_usage(line):
    if not line.startswith(b"data: {") or b'"completion_tokens"' not in line:
        return None
    try:
        u = json.loads(line[6:].decode()).get("usage") or {}
        v = u.get("completion_tokens")
        return int(v) if v is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def parse_metrics(text):
    """Prometheus text -> {name: summed value} for the series the prereg reads (labels summed; finished_reason kept)."""
    want = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
            "vllm:num_preemptions_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
            "vllm:iteration_tokens_total_sum")
    out = {k: 0.0 for k in want}
    for ln in text.splitlines():
        if not ln or ln[0] == "#":
            continue
        name = ln.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(ln.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        if name in out:
            out[name] += val
        elif name == "vllm:request_success_total":
            m = re.search(r'finished_reason="([a-z_]+)"', ln)
            key = f"success_{m.group(1) if m else 'unknown'}"
            out[key] = out.get(key, 0.0) + val
    return out


def topo_nvlink(text):
    """`nvidia-smi topo -m`: the GPU0 row's GPU1 column must be NV<k>. Returns (ok, cell)."""
    m = _RX_TOPO_ROW.search(text or "")
    if not m:
        return False, None
    cell = m.group(2)
    return bool(re.fullmatch(r"NV\d+", cell)), cell


def offload_gib(l_off, layer_gib):
    """--cpu-offload-gb for the first l_off MoE layers' routed experts. UVA offloads whole parameters in module order until
    the running total >= the budget (uva.py @ v0.30.0), so a budget 1/1024 GiB under l_off whole layers stops exactly there."""
    return 0.0 if l_off <= 0 else round(l_off * layer_gib - 1.0 / 1024, 6)


def g6_ref_s(w_bytes, conc, ctx, kv_bpt, hbm_gbs):
    """Roofline TPOT for the serving gate: (weights + every stream's KV) / measured HBM copy bandwidth (prereg section 3)."""
    return (w_bytes + conc * ctx * kv_bpt) / (hbm_gbs * 1e9)


def budget_ok(est_cost_now, dph, est_min, cap, now, deadline):
    return est_cost_now + dph * est_min / 60.0 <= cap and now + est_min * 60.0 <= deadline


def decode_routed(s):
    """base64 .npy -> nested list (protocol.py @ v0.30.0: shape (num_tokens - 1, num_layers, top_k))."""
    import numpy as np
    a = np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)
    return a.tolist(), list(a.shape)


# ----------------------------------------------------------------------------------------------- box plumbing
def log(msg):
    with open("logs/drive.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}\n")


def finish(status):
    open("DONE", "w").write(status + "\n")
    sh("nvidia-smi > logs/nvidia_smi_end.txt 2>&1", 60)
    log(f"FINISH {status}")
    sys.exit(0)


def mark(arm):
    open(f"ARM_DONE_{arm}", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")


def sh(cmd, timeout=600, env=None):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env or ENV)
    except subprocess.TimeoutExpired:
        return 124, f"timeout {timeout}s: {cmd[:80]}"
    return r.returncode, r.stdout + r.stderr


def est_cost():
    return DPH * (time.time() - RENT) / 3600 + DOWN * STATE["dl_gb"]


def jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps({**PROV, **r}) + "\n")


class Eng:
    def __init__(self, name, model, gpus, port, util, flags=(), env_extra=None, tag=""):
        self.name, self.m, self.gpus, self.port, self.util = name, model, gpus, port, util
        self.flags, self.env_extra = list(flags), env_extra or {}
        self.proc, self.t0 = None, None
        self.log = f"logs/server_{tag}_{name}.log" if tag else f"logs/server_{name}.log"

    def start(self):
        env = dict(ENV)
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)          #: never a thread cap
        env["CUDA_VISIBLE_DEVICES"] = self.gpus
        env.update(self.env_extra)
        argv = [VLLM, "serve", self.m["repo"], "--revision", self.m["revision"], "--port", str(self.port),
                "--gpu-memory-utilization", f"{self.util}", "--served-model-name", self.name,
                "--max-model-len", str(self.m["max_model_len"]), "--no-enable-prefix-caching", *self.flags]
        self.t0 = time.time()
        self.proc = subprocess.Popen(["bash", "-c", "ulimit -n 65536 && exec \"$@\"", "x", *argv], env=env,
                                     stdout=open(self.log, "w"), stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait_up(self, timeout=1800):
        while time.time() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5).read()
                return True, f"up {time.time() - self.t0:.1f}s"
            except Exception:  # noqa: BLE001
                time.sleep(1)
        return False, "timeout"

    def text(self):
        try:
            return open(self.log, errors="replace").read()
        except OSError:
            return ""

    def post(self, path, body=None, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body or {}).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def get(self, path, timeout=30):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=timeout).read()

    def sleeping(self):
        """GET /is_sleeping (dev router @ v0.30.0) -> bool, or None if unreadable."""
        try:
            return bool(json.loads(self.get("/is_sleeping"))["is_sleeping"])
        except Exception:  # noqa: BLE001
            return None

    def metrics(self):
        return parse_metrics(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=5).read().decode())

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)        # our own child's process group: never pkill
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        except ProcessLookupError:
            pass
        time.sleep(6)


def kv_plan(tag, name, kv_bytes, kv_bpt):
    """: every --kv-cache-memory-bytes engine is named in logs/kv_intended_<TAG>.json {"plan": {name: tokens}}
    (engine <name> = logs/server_<TAG>_<name>.log) so the kit's replay checks its 'GPU KV cache size' line against it."""
    p = f"logs/kv_intended_{tag}.json"
    d = {"plan": {}}
    if os.path.exists(p):
        try:                                                       # QA r2: a corrupt earlier write never crashes the run
            with open(p) as fh:
                d = json.load(fh)
        except ValueError:
            d = {"plan": {}}
    d.setdefault("plan", {})[name] = int(kv_bytes) // int(kv_bpt)
    with open(p, "w") as fh:
        json.dump(d, fh, indent=1)


def boot(eng, what):
    eng.start()
    ok, msg = eng.wait_up()
    log(f"boot {what} {eng.name} gpus={eng.gpus} {ok} {msg} backend={(_RX_BACKEND.search(eng.text()) or [None, None])[1]}")
    return ok, msg


def stream_req(port, name, prompt, max_tokens, deadline=None, extra=None, cancel_at=None):
    """One streaming completion. cancel_at (epoch): close the connection if no first token by then (a client timeout)."""
    body = {"model": name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    t0 = time.time()
    row = {"t_send": t0, "max_tokens": max_tokens}
    times, usage, status = [], None, "ok"
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        tmo = 900.0 if cancel_at is None else max(1.0, cancel_at - t0)
        with urllib.request.urlopen(req, timeout=tmo) as resp:
            for line in resp:
                now = time.time()
                if line.startswith(b"data: {") and b'"text"' in line:
                    times.append(now)
                u = sse_usage(line)
                if u is not None:
                    usage = u
                if deadline is not None and now > deadline:
                    status = "cut"
                    break
                if cancel_at is not None and not times and now > cancel_at:
                    status = "cancelled"
                    break
    except Exception as e:  # noqa: BLE001
        status = "cancelled" if (cancel_at is not None and not times and time.time() >= cancel_at - 0.05) \
            else "err:" + repr(e)[:160]
    row.update(status=status, t_cancel=time.time() if status == "cancelled" else None, tokens=len(times),
               usage_tokens=usage, ttft=(times[0] - t0) if times else None, t_first=times[0] if times else None,
               t_last=times[-1] if times else None,
               tpot=((times[-1] - times[0]) / (len(times) - 1)) if len(times) >= 2 else None,
               err=None if status in ("ok", "cut") else status,
               complete=(usage is not None and usage >= max_tokens))
    return row


def round_B(port, name, B, L, out, tag, salt):
    """B requests started together (threads), prompt = L random ids salted per pass (prefix caching is off anyway)."""
    rows, lock = [], threading.Lock()

    def one(i):
        r = stream_req(port, name, prompt_ids(f"{tag}:{salt}:{i}", L), out)
        with lock:
            rows.append({**r, "i": i})
    th = [threading.Thread(target=one, args=(i,)) for i in range(B)]
    [t.start() for t in th]
    [t.join() for t in th]
    return rows


class Loop:
    """Closed-loop streams against one engine until stop(): the lender load, the item-9 batch load, the CTRL replay."""

    def __init__(self, port, name, conc, L, out, tag, extra=None, path=None):
        self.port, self.name, self.conc, self.L, self.out, self.tag = port, name, conc, L, out, tag
        self.extra, self.path, self.ev, self.rows, self.lock = extra, path, threading.Event(), [], threading.Lock()
        self.th = [threading.Thread(target=self._run, args=(k,), daemon=True) for k in range(conc)]

    def _run(self, k):
        n = 0
        while not self.ev.is_set():
            r = stream_req(self.port, self.name, prompt_ids(f"{self.tag}:{k}:{n}", self.L), self.out, extra=self.extra)
            r.update(stream=k, n=n, tag=self.tag)
            with self.lock:
                self.rows.append(r)
            n += 1

    def start(self):
        [t.start() for t in self.th]
        return self

    def stop(self, join_s=120):
        """QA r1: returns a locked snapshot; threads still blocked in a request after join_s are counted in self.unjoined
        (their rows arrive too late for the file, and the count says so instead of a silent gap)."""
        self.ev.set()
        end = time.time() + join_s
        for t in self.th:
            t.join(timeout=max(0.1, end - time.time()))
        self.unjoined = sum(1 for t in self.th if t.is_alive())
        with self.lock:
            snap = list(self.rows)
        if self.unjoined:
            log(f"Loop {self.tag}: {self.unjoined} stream(s) still in a request after {join_s}s (rows not captured)")
        if self.path:
            jsonl(self.path, snap)
        return snap


# ----------------------------------------------------------------------------------------------- phases
def phase_env():
    cmds = ["nvidia-smi", "nvidia-smi topo -m", "nvidia-smi -q | head -60", "free -g", "cat /proc/meminfo | head -8",
            "df -h /root | tail -1", f"{PY} -c 'import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__)'",
            "ulimit -l", "cat /sys/fs/cgroup/memory.max 2>/dev/null", "env | grep -c -i kvcached",
            "grep -o -w -m1 avx512f /proc/cpuinfo; grep -o -w -m1 amx_tile /proc/cpuinfo",
            "lscpu | grep -E 'Model name|^CPU\\(s\\)'", "nproc"]                     # `$ nproc` block
    with open("logs/env.txt", "w") as f:
        for c in cmds:
            rc, out = sh(c, 120)
            f.write(f"$ {c}\n{out}\n")
    rc, out = sh(f"{PY} -c 'import vllm;print(vllm.__version__)'", 300)
    if "0.30.0" not in out:
        finish(f"PREMISE-FAIL vllm={out.strip()[-40:]}")
    rc, out = sh(f"{PY} -c \"from vllm.model_executor.models.registry import _VLLM_MODELS as M; "
                 f"print(all(a in M for a in ('Qwen3ForCausalLM','DeepseekV2ForCausalLM')))\"", 300)
    if "True" not in out:
        finish(f"PREMISE-FAIL registry {out.strip()[-120:]}")


def phase_gates(plan):
    g = plan["gates"]
    for i in (0, 1):
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(i), C15_G4_MIN_TFLOPS=str(g["g4_min_tflops"]))
        rc, out = sh(f"{PY} host_gate.py logs/gpu{i} --gpu-index {i}", 300, env=env)
        log(f"host_gate gpu{i} rc={rc} {out.strip()[-300:]}")
        if rc != 0:
            finish(f"INFRA-GATE gpu{i} rc={rc}")
    rc, topo = sh("nvidia-smi topo -m", 60)
    open("logs/topo.txt", "w").write(topo)
    ok, cell = topo_nvlink(topo)
    rc, out = sh(f"{PY} beta_bench.py --probe --out logs/probe.json" + (" --cpu" if STUB else ""), 900,
                 env=dict(ENV, CUDA_VISIBLE_DEVICES="0,1"))
    log(f"probe rc={rc} {out.strip()[-400:]}")
    if rc != 0:
        finish(f"INFRA-PROBE rc={rc}")
    try:                                                           # QA r1: a malformed probe is INFRA, never a CRASH
        STATE["probe"] = pr = json.load(open("logs/probe.json"))
        p2p_min = min(float(pr["p2p_10_gbs"]), float(pr["p2p_01_gbs"]))
        float(pr["hbm_d2d_gbs_0"])
        float(pr["h2d_gbs_0"])
    except (OSError, ValueError, KeyError, TypeError) as ex:
        finish(f"INFRA-PROBE unreadable logs/probe.json ({type(ex).__name__}: {ex})")
    fast = p2p_min >= g["nvlink_min_gbs"]
    json.dump({"topo_cell": cell, "topo_nv": ok, "p2p_min_gbs": p2p_min,
               "floor": g["nvlink_min_gbs"], "pass": ok and fast}, open("logs/nvlink_gate.json", "w"))
    if not STUB and not (ok and fast):
        finish(f"INFRA-NO-NVLINK topo={cell} p2p_min={p2p_min:.1f} GB/s")


def phase_net(plan):
    url = ("https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/"
           "model-00001-of-00005.safetensors")
    rc, out = sh(f"curl -sL -r 0-2000000000 --max-time 20 -o /dev/null -w '%{{speed_download}}' {url}", 60)
    try:
        bps = float(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        bps = 0.0
    json.dump({"net_Bps": bps}, open("logs/net_gate.json", "w"))
    log(f"net_Bps={bps:.0f}")
    if bps < plan["gates"]["net_min_Bps"]:
        finish(f"INFRA-NET {bps:.0f} B/s < 40 MB/s")


def phase_download(plan):
    res, lock = {}, threading.Lock()

    def dl(key, m):
        t0 = time.time()
        code = (f"from huggingface_hub import snapshot_download as s; print(s('{m['repo']}', revision='{m['revision']}', "
                f"allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model','*.jinja','*.py']))")
        try:
            r = subprocess.run([PY, "-c", code], env=ENV, capture_output=True, text=True, timeout=3000)
        except subprocess.TimeoutExpired:
            with lock:
                res[key] = {"ok": False, "err": "timeout 3000 s"}
            return
        if r.returncode != 0:
            with lock:
                res[key] = {"ok": False, "err": r.stderr[-300:]}
            return
        dirs = [ln.strip() for ln in r.stdout.strip().splitlines() if os.path.isdir(ln.strip())]   # QA r1: the real path
        if not dirs:
            with lock:
                res[key] = {"ok": False, "err": f"no snapshot dir in stdout: {r.stdout[-200:]}"}
            return
        path = dirs[-1]
        st = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs if f.endswith(".safetensors"))
        with lock:            #: byte count of the weights vs the HF API size frozen in the plan
            res[key] = {"ok": st == m["bytes"], "safetensors_bytes": st, "want": m["bytes"], "s": time.time() - t0}
    ths = [threading.Thread(target=dl, args=(k, m)) for k, m in sorted(plan["models"].items())]
    [t.start() for t in ths]
    [t.join() for t in ths]
    STATE["dl_gb"] = sum(v.get("safetensors_bytes", 0) for v in res.values()) / 1e9
    json.dump(res, open("rows/download.json", "w"), indent=1)
    log(f"download {json.dumps(res)[:400]}")
    if len(res) != len(plan["models"]) or not all(v["ok"] for v in res.values()):
        finish("INFRA-DL")


def phase_warm(plan):
    """One throwaway Qwen3-8B start: every later Qwen3-8B engine shares its compile key (same max_model_len,
    util and KV bytes sit outside the hash). DeepSeek-V2-Lite configs differ per arm (offload GiB, TP, DCP)
    and compile cold on first boot, disclosed in the prereg (no DSV2 metric depends on KV size)."""
    e = Eng("warm_q8", plan["models"]["q8"], "0", 8190, 0.90)
    ok, msg = boot(e, "warm")
    if ok:
        round_B(8190, e.name, 4, 512, 32, "warm", 0)
    e.stop()
    json.dump(["warm_q8"], open("logs/warmup_engines.json", "w"))   # marker (kit replay)
    if not ok:
        finish(f"WARM-FAIL {msg}")


def arm_ctrl(plan):
    q8, c, a = plan["models"]["q8"], plan["ctrl"], plan["anchor"]
    e = Eng("q8", q8, "0", 8200, 0.90, tag="CTRL")
    ok, msg = boot(e, "CTRL")
    if not ok:
        e.stop()
        finish(f"CTRL-BOOT-FAIL {msg}")
    for cell in a["cells"]:                                         #: warm requests per shape, discarded
        for w in range(a["warm_rounds"]):
            round_B(8200, "q8", cell["B"], cell["L"], 8, f"aw{cell['name']}", w)
    loop = Loop(8200, "q8", c["conc"], c["in"], c["out"], "ctrl")
    loop.start()
    time.sleep(c["secs"])
    rows = loop.stop()
    with open("rows/ctrl.jsonl", "w") as f:                        # kit host-gate --rows format: {"tpot", "err"}
        for r in rows:
            f.write(json.dumps({**PROV, **r}) + "\n")
    tp = sorted(r["tpot"] for r in rows if r.get("tpot") is not None and not r.get("err"))
    med = tp[len(tp) // 2] if tp else None
    hbm = STATE["probe"].get("hbm_d2d_gbs_0") or 0
    ref = g6_ref_s(q8["bytes"], c["conc"], c["in"] + c["out"] // 2, q8["kv_bpt"], hbm) if hbm else None
    g6 = med is not None and ref and med / ref <= 1 + plan["gates"]["g6_tol"] and len(tp) >= 0.95 * max(1, len(rows))
    json.dump({"tpot_p50": med, "ref_s": ref, "ratio": (med / ref) if (med and ref) else None, "n": len(rows),
               "n_tpot": len(tp), "tol": plan["gates"]["g6_tol"], "pass": bool(g6)}, open("logs/g6_serving.json", "w"))
    log(f"G6 tpot_p50={med} ref={ref} pass={g6}")
    if not g6 and not STUB:
        e.stop()
        finish("INFRA-G6 serving gate failed")
    for p in range(a["passes"]):
        for cell in a["cells"]:
            rows = round_B(8200, "q8", cell["B"], cell["L"], a["out"], f"anchor{cell['name']}", p)
            jsonl("rows/anchor.jsonl", [dict(r, cell=cell["name"], B=cell["B"], L=cell["L"], **{"pass": p}) for r in rows])
            if not e.alive():
                finish("INFRA-ENGINE-DIED CTRL")
    e.stop()
    mark("CTRL")


def run_bench(lender, plan_path, timeout_s):
    cmd = (f"{PY} beta_bench.py --run --plan {plan_path} --lender {lender} --out rows/beta_{lender}.jsonl "
           f"--progress logs/beta_progress_{lender}.json" + (" --cpu" if STUB else ""))
    rc, out = sh(cmd, timeout_s, env=dict(ENV, CUDA_VISIBLE_DEVICES="0,1"))
    open(f"logs/beta_{lender}.log", "w").write(out)
    log(f"beta {lender} rc={rc} {out.strip()[-200:]}")
    return rc


def arm_beta_idle(plan, plan_path):
    rc = run_bench("IDLE", plan_path, int(plan["est_min"]["BETA_IDLE"] * 60 * 2))
    mark("BETA_IDLE")
    return rc == 0


def arm_beta_busy(plan, plan_path):
    q8, ld = plan["models"]["q8"], plan["lender"]
    e = Eng("lender", q8, "1", 8202, ld["util"], tag="BUSY")
    ok, msg = boot(e, "BETA_BUSY lender")
    if not ok:
        e.stop()
        log(f"BETA_BUSY lender boot fail {msg}")
        return False
    round_B(8202, "lender", 4, ld["in"], 16, "lw", 0)
    loop = Loop(8202, "lender", ld["conc"], ld["in"], ld["out"], "lender").start()
    t_settle = time.time()
    time.sleep(ld["settle_s"])
    t_bench = time.time()                                          # the lender's reference window is [t_settle, t_bench)
    json.dump({"t_settle_start": t_settle, "t_bench_start": t_bench}, open("logs/beta_busy_windows.json", "w"))
    rc = run_bench("BUSY", plan_path, int(plan["est_min"]["BETA_BUSY"] * 60 * 2))
    t_end = time.time()
    time.sleep(10)
    rows = loop.stop()
    jsonl("rows/lender_busy.jsonl", rows)
    alive = e.alive()
    e.stop()
    json.dump({"t_settle_start": t_settle, "t_bench_start": t_bench, "t_bench_end": t_end, "lender_alive": alive,
               "n_lender_rows": len(rows)}, open("logs/beta_busy_windows.json", "w"))
    if not alive:
        log("BETA_BUSY lender died during the bench: INFRA")
    mark("BETA_BUSY")
    return rc == 0 and alive


def arm_item12(plan):
    m, it = plan["models"]["dsv2"], plan["item12"]
    e = Eng("dsv2r", m, "0", 8203, 0.92, ["--enable-return-routed-experts", "--max-num-batched-tokens", "8192"], tag="I12")
    ok, msg = boot(e, "ITEM12")
    if not ok:
        e.stop()
        json.dump({"ok": False, "why": msg}, open("rows/routing_meta.json", "w"))
        mark("ITEM12")
        return
    texts = [json.loads(ln)["text"] for ln in open("prompts.jsonl")]
    ids = []
    for t in texts:
        try:
            tok = json.loads(e.post("/tokenize", {"model": "dsv2r", "prompt": t}))["tokens"]
        except Exception as ex:  # noqa: BLE001
            log(f"tokenize fail {ex!r}")
            continue
        if len(tok) >= it["P"]:
            ids.append(tok[: it["P"]])
    meta = {"ok": True, "n_prompts": len(ids), "P": it["P"], "Bs": []}
    for B in it["Bs"]:
        if B > len(ids):
            meta["Bs"].append({"B": B, "skipped": f"only {len(ids)} prompts of >= {it['P']} tokens"})
            continue
        body = {"model": "dsv2r", "prompt": ids[:B], "max_tokens": it["out"], "temperature": it["temperature"],
                "top_p": it["top_p"], "seed": it["seed"], "ignore_eos": True, "routed_experts_prompt_start": it["P"]}
        t0 = time.time()
        try:
            resp = json.loads(e.post("/v1/completions", body, timeout=900))
        except Exception as ex:  # noqa: BLE001
            meta["Bs"].append({"B": B, "err": repr(ex)[:200]})
            continue
        reqs = []
        for ch in resp.get("choices", []):
            s = ch.get("routed_experts")
            if s is None:
                reqs.append({"index": ch.get("index"), "routed": None})
                continue
            arr, shape = decode_routed(s)
            reqs.append({"index": ch.get("index"), "shape": shape, "experts": arr})
        json.dump({**PROV, "B": B, "P": it["P"], "s": time.time() - t0, "reqs": reqs}, open(f"rows/routing_B{B}.json", "w"))
        meta["Bs"].append({"B": B, "n": len(reqs), "s": time.time() - t0,
                           "absent": sum(1 for r in reqs if r.get("routed", 1) is None)})
    json.dump(meta, open("rows/routing_meta.json", "w"), indent=1)
    e.stop()
    mark("ITEM12")


def arm_item10(plan):
    m, it = plan["models"]["dsv2"], plan["item10"]
    meta = []
    for l_off in it["L_off"]:                                      # L_off 0 = the control, first
        gb = offload_gib(l_off, m["layer_expert_gib"])
        flags = ["--cpu-offload-gb", str(gb), "--cpu-offload-params", "experts"] if gb > 0 else []
        e = Eng(f"off{l_off}", m, "0", 8204, it["util"], flags, tag="I10")
        ok, msg = boot(e, f"ITEM10 L_off={l_off}")
        txt = e.text()
        off = _RX_OFFLOAD.search(txt)
        mm = _RX_MODEL_MEM.search(txt)
        rec = {"L_off": l_off, "cpu_offload_gb": gb, "ok": ok, "msg": msg, "boot_s": time.time() - e.t0,
               "logged_offload_gib": float(off.group(1)) if off else None,
               "model_mem_gib": float(mm.group(1)) if mm else None}
        if ok:
            round_B(8204, e.name, 4, it["in"], 8, f"ow{l_off}", 0)
            for p in range(it["passes"]):
                for B in it["Bs"]:
                    rows = round_B(8204, e.name, B, it["in"], it["out"], f"off{l_off}", p)
                    jsonl("rows/offload.jsonl", [dict(r, L_off=l_off, B=B, **{"pass": p}) for r in rows])
                    if not e.alive():
                        rec["died"] = True
                        break
        rec["alive_at_end"] = e.alive()
        e.stop()
        meta.append(rec)
        json.dump(meta, open("rows/offload_meta.json", "w"), indent=1)
        log(f"ITEM10 {rec}")
    mark("ITEM10")


def arm_item9(plan):
    q8, it = plan["models"]["q8"], plan["item9"]
    s = it["same"]
    # (a) same model: priority scheduling; live (priority 0, lower = earlier) vs a low-priority batch (priority 10).
    #     variant 'ample': util-sized KV (no preemption needed); 'full': 2 GiB KV, the live request forces a preemption.
    mets = {}
    for var in s["variants"]:
        flags = ["--scheduling-policy", "priority"] + (["--kv-cache-memory-bytes", str(var["kv_bytes"])]
                                                       if var.get("kv_bytes") else [])
        e = Eng(f"q8p{var['name']}", q8, "0", 8205, 0.90, flags, tag="I9")
        if var.get("kv_bytes"):
            kv_plan("I9", e.name, var["kv_bytes"], q8["kv_bpt"])
        ok, msg = boot(e, f"ITEM9 same {var['name']}")
        if ok:
            round_B(8205, e.name, 2, s["live_in"], 8, "pw", 0)
            rows = []
            for k in range(s["n_live"]):
                rows.append(dict(stream_req(8205, e.name, prompt_ids(f"idle{var['name']}{k}", s["live_in"]),
                                            s["live_out"], extra={"priority": 0}), phase="idle", k=k, variant=var["name"]))
                time.sleep(1.0)
            m0 = e.metrics()
            batch = Loop(8205, e.name, s["batch_conc"], s["batch_in"], s["batch_out"], f"batch{var['name']}",
                         extra={"priority": 10}).start()
            time.sleep(s["batch_settle_s"])
            m_busy0 = e.metrics()
            for k in range(s["n_live"]):
                rows.append(dict(stream_req(8205, e.name, prompt_ids(f"busy{var['name']}{k}", s["live_in"]),
                                            s["live_out"], extra={"priority": 0}), phase="busy", k=k, variant=var["name"]))
                time.sleep(s["live_gap_s"])
            m_busy1 = e.metrics()
            brows = batch.stop()
            m1 = e.metrics()
            jsonl("rows/item9_same.jsonl", rows)
            jsonl("rows/item9_same_batch.jsonl", [dict(r, variant=var["name"]) for r in brows])
            mets[var["name"]] = {"m0": m0, "m_busy0": m_busy0, "m_busy1": m_busy1, "m1": m1, "alive": e.alive()}
        else:
            mets[var["name"]] = {"boot_fail": msg}
        e.stop()
    json.dump(mets, open("rows/item9_same_metrics.json", "w"), indent=1)
    # (b) other model: B (batch) awake, A (live) asleep; live arrival -> B /sleep (mode) -> A /wake_up -> A serves
    o = it["other"]
    dev = {"VLLM_SERVER_DEV_MODE": "1"}
    A = Eng("A", q8, "0", 8211, o["util"], ["--enable-sleep-mode"], dev, tag="I9")
    Bm = Eng("Bm", q8, "0", 8212, o["util"], ["--enable-sleep-mode"], dev, tag="I9")
    okA, okB, pre = boot(A, "ITEM9 A")[0], False, {}
    try:
        if okA:
            round_B(8211, "A", 1, 64, 8, "aw", 0)
            for path in ("/sleep?level=1&mode=wait", "/wake_up", "/sleep?level=1&mode=wait"):   #: pay the first pin
                t1 = time.time()
                A.post(path)
                pre[f"A{path}"] = pre.get(f"A{path}", []) + [time.time() - t1]
            okB = boot(Bm, "ITEM9 B")[0]
        if okB:
            round_B(8212, "Bm", 1, 64, 8, "bw", 0)
            for path in ("/sleep?level=1&mode=wait", "/wake_up"):
                t1 = time.time()
                Bm.post(path)
                pre[f"B{path}"] = [time.time() - t1]
    except Exception as ex:  # noqa: BLE001
        pre["err"] = repr(ex)[:200]
        okB = False
    json.dump(pre, open("rows/item9_presleep.json", "w"), indent=1)
    recs = []
    if okA and okB:
        for mode in o["modes"]:
            for rep in range(o["reps"]):
                state = {"A_sleeping": A.sleeping(), "B_sleeping": Bm.sleeping()}
                if state != {"A_sleeping": True, "B_sleeping": False}:          #: a probe left the wrong state
                    jsonl("rows/item9_other.jsonl", [{"mode": mode, "rep": rep, "err": f"STATE {state}"}])
                    break
                batch = Loop(8212, "Bm", o["batch_conc"], o["batch_in"], o["batch_out"], f"ob{mode}{rep}").start()
                time.sleep(o["batch_settle_s"])
                t_arr = time.time()
                rec = {"mode": mode, "rep": rep, "t_arr": t_arr, "state_before": state}
                try:
                    Bm.post(f"/sleep?level=1&mode={mode}", timeout=900)
                    rec["t_b_slept"] = time.time()
                    A.post("/wake_up")
                    rec["t_a_awake"] = time.time()
                    live = stream_req(8211, "A", prompt_ids(f"live{mode}{rep}", o["live_in"]), o["live_out"])
                    rec.update(live=live, p_s=(live["t_first"] - t_arr) if live.get("t_first") else None)
                    A.post("/sleep?level=1&mode=wait")
                    Bm.post("/wake_up")
                except Exception as ex:  # noqa: BLE001
                    rec["err"] = repr(ex)[:200]
                brows = batch.stop()
                rec["batch"] = [{k: r.get(k) for k in ("status", "tokens", "usage_tokens", "t_send", "t_last", "err",
                                                       "complete")} for r in brows]
                rec["alive"] = A.alive() and Bm.alive()
                recs.append(rec)
                jsonl("rows/item9_other.jsonl", [rec])
                if not rec["alive"]:
                    break
    A.stop()
    Bm.stop()
    mark("ITEM9")


def arm_item8(plan):
    q8, it = plan["models"]["q8"], plan["item8"]
    e = Eng("q8k", q8, "0", 8206, 0.90, ["--kv-cache-memory-bytes", str(it["kv_bytes"])], tag="I8")
    kv_plan("I8", e.name, it["kv_bytes"], q8["kv_bpt"])
    ok, msg = boot(e, "ITEM8")
    if not ok:
        e.stop()
        mark("ITEM8")
        return
    round_B(8206, "q8k", 2, it["in"], 8, "kw", 0)
    for policy in ("Tinf", "T10"):
        stop, samples = threading.Event(), []

        def sampler():
            while not stop.is_set():
                try:
                    samples.append(dict(e.metrics(), t=time.time()))
                except Exception as ex:  # noqa: BLE001
                    samples.append({"t": time.time(), "err": repr(ex)[:80]})
                time.sleep(0.5)
        sth = threading.Thread(target=sampler, daemon=True)
        sth.start()
        rng = random.Random(f"{it['seed']}:{policy}")
        t, arr = 0.0, []
        while t < it["secs"]:
            t += rng.expovariate(it["rate"])
            arr.append(t)
        rows, lock, t0 = [], threading.Lock(), time.time()

        def one(k, at):
            time.sleep(max(0.0, t0 + at - time.time()))
            cancel = (time.time() + it["timeout_s"]) if policy == "T10" else None
            r = stream_req(8206, "q8k", prompt_ids(f"k8{policy}{k}", it["in"]), it["out"], cancel_at=cancel)
            with lock:
                rows.append(dict(r, k=k, at=at))
        th = [threading.Thread(target=one, args=(k, a), daemon=True) for k, a in enumerate(arr)]
        [x.start() for x in th]
        cap_end = time.time() + it["join_cap_s"]
        for x in th:
            x.join(timeout=max(0.1, cap_end - time.time()))
        unfinished = sum(1 for x in th if x.is_alive())             # a client still waiting at the cap has NO row
        t_clients_done = time.time()
        time.sleep(it["tail_s"])                                    # zombie window: tokens generated after the clients
        stop.set()
        sth.join(timeout=10)
        jsonl(f"rows/item8_{policy}.jsonl", rows)
        jsonl(f"rows/item8_metrics_{policy}.jsonl", samples)
        json.dump({"policy": policy, "t0": t0, "t_clients_done": t_clients_done, "n": len(arr), "alive": e.alive(),
                   "unfinished_clients": unfinished}, open(f"rows/item8_{policy}_meta.json", "w"))
        # drain before the next policy: wait for running + waiting == 0 (bounded)
        end = time.time() + it["drain_cap_s"]
        while time.time() < end:
            try:
                mm = e.metrics()
                if mm["vllm:num_requests_running"] + mm["vllm:num_requests_waiting"] == 0:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
    e.stop()
    mark("ITEM8")


def arm_dcp(plan):
    m, it = plan["models"]["dsv2"], plan["dcp"]
    meta = []
    for dcp in (1, 2):
        flags = ["--tensor-parallel-size", "2"] + (["--decode-context-parallel-size", "2"] if dcp > 1 else [])
        e = Eng(f"dcp{dcp}", m, "0,1", 8207, it["util"], flags, tag="DCP")
        ok, msg = boot(e, f"DCP {dcp}")
        rec = {"dcp": dcp, "ok": ok, "msg": msg, "backend": (_RX_BACKEND.search(e.text()) or [None, None])[1]}
        if ok:
            round_B(8207, e.name, 2, 1024, 8, f"dw{dcp}", 0)
            for p in range(it["passes"]):
                for cell in it["cells"]:
                    rows = round_B(8207, e.name, cell["B"], cell["L"], it["out"], f"dcp{dcp}{cell['name']}", p)
                    jsonl("rows/dcp.jsonl", [dict(r, dcp=dcp, cell=cell["name"], B=cell["B"], **{"pass": p}) for r in rows])
        rec["alive_at_end"] = e.alive()
        e.stop()
        meta.append(rec)
        json.dump(meta, open("rows/dcp_meta.json", "w"), indent=1)
    mark("DCP")


# ----------------------------------------------------------------------------------------------- main
def main():
    global ENV, PY, VLLM, DL, RENT, DPH, DOWN, CAP_GUARD
    os.chdir(ROOT)
    for d in ("rows", "logs"):
        os.makedirs(d, exist_ok=True)
    ENV = {k: v for k, v in os.environ.items() if k not in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH")}
    ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"            # CUDA index i == nvidia-smi index i (host_gate -i)
    ENV["HF_XET_HIGH_PERFORMANCE"] = "1"
    PY = sys.executable
    VLLM = os.path.join(os.path.dirname(PY), "vllm") if os.path.exists(os.path.join(os.path.dirname(PY), "vllm")) else "vllm"
    DL = float(os.environ.get("MG_DEADLINE", "0")) or time.time() + 9000
    RENT = float(os.environ.get("MG_RENT_EPOCH", "0")) or time.time()
    DPH = float(os.environ.get("MG_DPH", "1.40"))
    DOWN = float(os.environ.get("MG_DOWN_COST", "0.01"))
    CAP_GUARD = float(os.environ.get("MG_CAP_GUARD", "3.85"))
    plan_path = os.environ.get("MG_PLAN", "plan.json")
    plan = json.load(open(plan_path))
    if plan["order"][0] != "CTRL":
        finish("PREMISE-FAIL the control is not first")
    log(f"START plan={plan['version']} dph={DPH} cap_guard={CAP_GUARD} deadline_in={DL - time.time():.0f}s stub={STUB}")
    phase_env()
    phase_gates(plan)
    if STUB:
        log("STUB: phase_net + phase_download skipped")
    else:
        phase_net(plan)
        phase_download(plan)
    ENV["HF_HUB_OFFLINE"] = "1"                          #: the recorded model id stays the repo id in every process
    phase_warm(plan)
    arms = {"CTRL": lambda: arm_ctrl(plan), "BETA_IDLE": lambda: arm_beta_idle(plan, plan_path),
            "BETA_BUSY": lambda: arm_beta_busy(plan, plan_path), "ITEM12": lambda: arm_item12(plan),
            "ITEM10": lambda: arm_item10(plan), "ITEM9": lambda: arm_item9(plan), "ITEM8": lambda: arm_item8(plan),
            "DCP": lambda: arm_dcp(plan)}
    skips, beta_ok = [], True
    for arm in plan["order"]:
        if arm == "BETA_BUSY" and not beta_ok:
            skips.append({"arm": arm, "why": "BETA_IDLE failed an instrument check (I-MERGE/I-RACE): the bench is invalid"})
            continue
        if not budget_ok(est_cost(), DPH, plan["est_min"][arm], CAP_GUARD, time.time(), DL):
            skips.append({"arm": arm, "why": "budget/deadline", "est_cost": round(est_cost(), 3)})
            log(f"SKIP {arm} budget est={est_cost():.2f} cap_guard={CAP_GUARD}")
            continue
        t0 = time.time()
        r = arms[arm]()
        if arm == "BETA_IDLE":
            beta_ok = bool(r)
        log(f"ARM {arm} done in {time.time() - t0:.0f}s est_cost={est_cost():.2f}")
        json.dump(skips, open("logs/skips.json", "w"), indent=1)
    json.dump(skips, open("logs/skips.json", "w"), indent=1)
    json.dump({"est_cost_end": est_cost(), "dl_gb": STATE["dl_gb"], "t_end": time.time()}, open("logs/cost.json", "w"))
    finish("OK" + (f" skipped={[s['arm'] for s in skips]}" if skips else ""))


# ----------------------------------------------------------------------------------------------- self-test
def self_test():
    fails, n = [], [0]

    def ck(name, cond, detail=""):
        n[0] += 1
        print(f"  {'PASS' if cond else 'FAIL'} {name}{'' if cond else ' :: ' + str(detail)}")
        if not cond:
            fails.append(name)

    # D1 (hand): 13 layers x 1.03125 GiB = 13.40625; minus 1/1024 = 13.405273 (UVA stops exactly after layer 13)
    ck("D1 offload_gib(13) = 13.405273, (0) = 0", offload_gib(13, 1.03125) == 13.405273 and offload_gib(0, 1.03125) == 0.0,
       offload_gib(13, 1.03125))
    # D1b (real data): DeepSeek-V2-Lite routed experts per layer = 64 x 3 x 2048 x 1408 x 2 B = 1,107,296,256 B = 1.03125 GiB
    ck("D1b layer expert bytes = 1.03125 GiB", 64 * 3 * 2048 * 1408 * 2 == 1107296256 and 1107296256 / GIB == 1.03125)
    # D2 (hand): G6 roofline, 8 x 576 x 147,456 = 679,477,248 B; + 16,381,516,776 B weights = 17,060,994,024 B;
    #   / 1,300e9 B/s = 0.0131238416 s (the first hand sum here was off by 25,600 B; the golden caught it, 2026-10-14)
    r = g6_ref_s(16381516776, 8, 576, 147456, 1300.0)
    ck("D2 g6 ref 13.1238 ms", abs(r - 0.0131238416) < 1e-9, r)
    # D3 topo: NV12 passes, PIX / SYS / missing row fail (ADVERSARIAL: an 'X' self column must not be read as the link)
    t_nv = "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNV12\t0-23\nGPU1\tNV12\t X \t0-23\n"
    t_pix = "\tGPU0\tGPU1\nGPU0\t X \tPIX\nGPU1\tPIX\t X \n"
    ck("D3 topo NV12 ok / PIX not / garbage not", topo_nvlink(t_nv) == (True, "NV12") and topo_nvlink(t_pix)[0] is False
       and topo_nvlink("no table")[0] is False, (topo_nvlink(t_nv), topo_nvlink(t_pix)))
    # D4 budget (hand): $1.00 spent + 1.069 x 30/60 = 1.5345 <= 2.0 ok; with cap 1.50 refused; deadline binds too
    ck("D4 budget_ok", budget_ok(1.0, 1.069, 30, 2.0, 0, 1e9) and not budget_ok(1.0, 1.069, 30, 1.50, 0, 1e9)
       and not budget_ok(0.0, 1.0, 30, 9.0, 1000.0, 2000.0))
    # D5 metrics parser on vLLM 0.30 exposition lines (labels summed, finished_reason kept, _total names)
    txt = ('# HELP x\nvllm:num_requests_running{engine="0",model_name="q8k"} 9.0\n'
           'vllm:num_requests_waiting{engine="0",model_name="q8k"} 31.0\n'
           'vllm:num_preemptions_total{engine="0",model_name="q8k"} 4.0\n'
           'vllm:request_success_total{engine="0",finished_reason="abort",model_name="q8k"} 12.0\n'
           'vllm:request_success_total{engine="0",finished_reason="length",model_name="q8k"} 30.0\n')
    pm = parse_metrics(txt)
    ck("D5 metrics", pm["vllm:num_requests_running"] == 9 and pm["vllm:num_requests_waiting"] == 31
       and pm["vllm:num_preemptions_total"] == 4 and pm["success_abort"] == 12 and pm["success_length"] == 30, pm)
    # D6 usage parsing: usage counted, a text chunk ignored, garbage ignored
    ck("D6 sse usage", sse_usage(b'data: {"choices":[],"usage":{"completion_tokens":32}}') == 32
       and sse_usage(b'data: {"choices":[{"text":"a"}]}') is None and sse_usage(b"data: [DONE]") is None)
    # D7 routed experts decode: a real .npy (shape (3, 27, 6), uint8) round-trips through base64
    try:
        import numpy as np
        a = (np.arange(3 * 27 * 6) % 64).astype(np.uint8).reshape(3, 27, 6)
        buf = io.BytesIO()
        np.save(buf, a)
        lst, shape = decode_routed(base64.b64encode(buf.getvalue()).decode())
        ck("D7 routed decode shape (3,27,6) + values", shape == [3, 27, 6] and lst[2][26][5] == a[2, 26, 5], shape)
    except ImportError:
        ck("D7 numpy present", False, "numpy missing")
    # D8 prompt ids deterministic per (rid, seed), in range, different per rid
    p1, p2 = prompt_ids("x", 16), prompt_ids("y", 16)
    ck("D8 prompt ids", p1 == prompt_ids("x", 16) and p1 != p2 and all(1000 <= v < 30000 for v in p1 + p2))
    # D9 (hand): the kit plan for a 2 GiB Qwen3-8B engine: 2,147,483,648 // 147,456 = 14,563 tokens (vLLM logs 910 x 16 =
    #   14,560, inside the kit's max(32, 0.5 %) band); two engines in one tag share one file
    import tempfile
    cwd = os.getcwd()
    got, got2 = None, None
    with tempfile.TemporaryDirectory() as d:
        try:                                                       # QA r2: always return to cwd
            os.chdir(d)
            os.makedirs("logs")
            kv_plan("I9", "q8pfull", 2147483648, 147456)
            kv_plan("I9", "x2", 1073741824, 147456)
            with open("logs/kv_intended_I9.json") as fh:
                got = json.load(fh)["plan"]
            with open("logs/kv_intended_I8.json", "w") as fh:     # ADVERSARIAL: a corrupt earlier file is rewritten
                fh.write("{not json")
            kv_plan("I8", "q8k", 2147483648, 147456)
            with open("logs/kv_intended_I8.json") as fh:
                got2 = json.load(fh)["plan"]
        finally:
            os.chdir(cwd)
    ck("D9 kv_intended plan 14563 + merge + corrupt rewrite", got == {"q8pfull": 14563, "x2": 7281}
       and got2 == {"q8k": 14563}, (got, got2))
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n[0] - len(fails)}/{n[0]}")
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            log(f"CRASH {exc!r}")
            open("DONE", "w").write(f"CRASH {exc!r}"[:300] + "\n")
        finally:
            raise
