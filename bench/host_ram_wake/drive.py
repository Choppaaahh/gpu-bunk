#!/usr/bin/env python3
""" Tiered-residency driver, run ON the rented 1x RTX 5090 box in /root/c13.

Prereg: (private report) (frozen before any rental).
Stock image vllm/vllm-openai:v0.30.0; ENABLE_KVCACHED / KVCACHED_AUTOPATCH unset for every vLLM process.

Phases (prereg section 4, in this order; each phase writes rows/<phase>.json then a ARM_DONE_<phase> marker):
  env     host facts: RAM, cgroup limit, PCIe gen/width under load, CPU threads, gate: P-state/clock/power under a burn
  copy    raw pinned-host -> GPU copy floor (torch), pinned-alloc cost, pageable comparison, pin-capacity probe
  m_<key> per model: download, 3 cold-compile launches, 3 warm-compile launches, then on the live warm server:
          5 awake TTFT baselines, 5 sleep(L1)->wake->first-token cycles, 3 sleep(L2)->reload_weights->first-token
          cycles, an output-identity check (greedy 16 tokens, same single-request shape) before vs after each wake
  coloc   A (small) serves 8 closed-loop streaming clients while B (8B) sleeps/wakes 5x; per-request rows for A
Writes DONE = OK | INFRA-FAIL:<step> | DEADLINE:<phase>. Env C13_DEADLINE (epoch s): no phase starts past it.
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request, random, statistics

ROOT = os.environ.get("BENCH_DIR", "/root/c13")
os.chdir(ROOT)
for d in ("rows", "logs"):
    os.makedirs(d, exist_ok=True)
DL = float(os.environ.get("C13_DEADLINE", "0")) or (time.time() + 3 * 3600)
ENV = {k: v for k, v in os.environ.items() if k not in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH")}
ENV["VLLM_SERVER_DEV_MODE"] = "1"
ENV["HF_XET_HIGH_PERFORMANCE"] = "1"  # amendment 1

MODELS = {  # key: (repo, revision, gpu_memory_utilization)
    "small": ("Qwen/Qwen3-1.7B-FP8", "1641e6c1b620b7ed7e8711b443990429a23b1b99", 0.30),
    "d8b":   ("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3", 0.50),
    "moe":   ("Qwen/Qwen3-30B-A3B-GPTQ-Int4", "9b534e4318b7ebc3c961a839f13eb18b1833f441", 0.85),
}
COMMON = ["--max-model-len", "4096", "--max-num-seqs", "64", "--seed", "0", "--no-enable-prefix-caching",
          "--enable-sleep-mode"]
H = "http://127.0.0.1"
FIXED_PROMPT = "The history of the printing press begins"


def log(msg):
    with open("logs/drive.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}\n")


def finish(status):
    open("DONE", "w").write(status + "\n")
    subprocess.run("nvidia-smi > logs/nvidia_smi_end.txt 2>&1", shell=True)
    sys.exit(0)


def dump(name, obj):
    json.dump(obj, open(f"rows/{name}.json", "w"), indent=1)
    open(f"ARM_DONE_{name}", "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")


def post(port, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"{H}:{port}{path}", data=data, method="POST",
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = r.read()
    return time.perf_counter() - t0, out


def get(port, path, timeout=10):
    with urllib.request.urlopen(f"{H}:{port}{path}", timeout=timeout) as r:
        return r.status, r.read()


def first_token(port, name, prompt="Hello", timeout=600):
    """Streaming 1-token completion; returns seconds from send to the first content chunk."""
    body = {"model": name, "prompt": prompt, "max_tokens": 1, "stream": True, "temperature": 0}
    req = urllib.request.Request(f"{H}:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data: ") and b'"text"' in line:
                dt = time.perf_counter() - t0
                r.read()
                return dt
    raise RuntimeError("no token")


def greedy16(port, name):
    _, out = post(port, "/v1/completions", {"model": name, "prompt": FIXED_PROMPT, "max_tokens": 16,
                                            "temperature": 0, "ignore_eos": True})
    return json.loads(out)["choices"][0]["text"]


def gpu_q(fields):
    r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


class Server:
    def __init__(self, key, port, cache_root, util=None):
        self.key, self.port = key, port
        repo, rev, u = MODELS[key]
        self.name = key
        env = dict(ENV, VLLM_CACHE_ROOT=cache_root)
        cmd = ["vllm", "serve", repo, "--revision", rev, "--port", str(port), "--served-model-name", key,
               "--gpu-memory-utilization", str(util or u)] + COMMON
        self.logpath = f"logs/server_{key}_{int(time.time())}.log"
        self.t_launch = time.perf_counter()
        self.p = subprocess.Popen(cmd, env=env, stdout=open(self.logpath, "w"), stderr=subprocess.STDOUT,
                                  start_new_session=True)

    def wait_ready(self, limit=900):
        while time.perf_counter() - self.t_launch < limit:
            if self.p.poll() is not None:
                raise RuntimeError(f"server {self.key} died rc={self.p.returncode}")
            try:
                if get(self.port, "/health", timeout=2)[0] == 200:
                    return time.perf_counter() - self.t_launch
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"server {self.key} not ready in {limit}s")

    def stop(self):
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.p.pid, signal.SIGKILL)
            self.p.wait(timeout=30)
        for _ in range(30):  # wait until the GPU has no compute apps
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True)
            if not r.stdout.strip():
                break
            time.sleep(1)

    def log_facts(self):
        facts = {}
        keys = ("Loading weights took", "Model loading took", "torch.compile takes", "Compiling a graph",
                "Directly load the compiled graph", "Graph capturing finished", "init engine", "Available KV cache memory",
                "GPU KV cache size", "Sleep mode freed", "CuMemAllocator: sleep freed", "Using sleep-mode backend")
        for line in open(self.logpath, errors="replace"):
            for k in keys:
                if k in line:
                    facts.setdefault(k, []).append(line.strip()[-220:])
        return facts


def drop_caches():
    r = subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, capture_output=True, text=True)
    return r.returncode == 0


def check_deadline(phase):
    if time.time() > DL:
        finish(f"DEADLINE:{phase}")


# ---------------------------------------------------------------- env + gate
def phase_env():
    import torch
    out = {}
    mi = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo")}
    out["MemTotal"], out["MemAvailable"] = mi["MemTotal"], mi["MemAvailable"]
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.exists(p):
            out["cgroup_mem_limit"] = open(p).read().strip()
    out["nproc"] = os.cpu_count()
    out["kvcached_env_lines"] = sum(1 for k in os.environ if "KVCACHED" in k.upper())
    out["ENV_has_kvcached"] = sum(1 for k in ENV if "KVCACHED" in k.upper())
    out["idle_pcie"] = gpu_q("pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")
    # burn (gate): 20 s of bf16 matmuls, sample pstate / sm clock / power / pcie mid-burn
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    samples, t0, n = [], time.time(), 0
    while time.time() - t0 < 20:
        for _ in range(20):
            a @ b
            n += 1
        torch.cuda.synchronize()
        if time.time() - t0 > 5:
            samples.append(gpu_q("pstate,clocks.sm,power.draw,pcie.link.gen.current,pcie.link.width.current"))
    out["burn_tflops"] = round(n * 2 * 8192 ** 3 / (time.time() - t0) / 1e12, 1)
    out["burn_samples"] = samples[-5:]
    out["gpu_name"] = torch.cuda.get_device_name(0)
    out["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
    import vllm
    out["vllm"], out["torch"] = vllm.__version__, torch.__version__
    out["drop_caches_ok"] = drop_caches()
    del a, b
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- copy floor
def phase_copy():
    import torch
    rows = {"h2d_pinned": [], "d2h_pinned": [], "h2d_pageable": [], "pin_alloc": [], "pin_capacity": {}}
    GB = 10 ** 9
    for gb in (1.0, 2.65, 9.44, 16.93):
        n = int(gb * GB)
        t0 = time.perf_counter()
        h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        h.fill_(1)
        rows["pin_alloc"].append({"gb": gb, "s": time.perf_counter() - t0})
        d = torch.empty(n, dtype=torch.uint8, device="cuda")
        d.copy_(h, non_blocking=False)  # warm-up, not scored
        torch.cuda.synchronize()
        for i in range(5):
            t0 = time.perf_counter()
            d.copy_(h, non_blocking=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rows["h2d_pinned"].append({"gb": gb, "rep": i, "s": dt, "GBps": gb / dt})
        if gb in (2.65, 9.44):
            for i in range(3):
                t0 = time.perf_counter()
                h.copy_(d)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                rows["d2h_pinned"].append({"gb": gb, "rep": i, "s": dt, "GBps": gb / dt})
            sample = gpu_q("pcie.link.gen.current,pcie.link.width.current")
            rows.setdefault("pcie_during_copy", []).append(sample)
        del h, d
        torch.cuda.empty_cache()
    n = int(2.65 * GB)
    hp = torch.ones(n, dtype=torch.uint8)
    d = torch.empty(n, dtype=torch.uint8, device="cuda")
    for i in range(3):
        t0 = time.perf_counter()
        d.copy_(hp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows["h2d_pageable"].append({"gb": 2.65, "rep": i, "s": dt, "GBps": 2.65 / dt})
    del hp, d
    # pin-capacity probe: pin 4 GB chunks up to 60 % of min(MemAvailable, cgroup limit); never beyond (OOM-kill risk)
    mi = {l.split(":")[0]: int(l.split(":")[1].split()[0]) * 1024 for l in open("/proc/meminfo")}
    lim = mi["MemAvailable"]
    for p in ("/sys/fs/cgroup/memory.max",):
        if os.path.exists(p):
            v = open(p).read().strip()
            if v.isdigit():
                lim = min(lim, int(v))
    target = int(0.60 * lim)
    held, total, t0, err = [], 0, time.perf_counter(), None
    try:
        while total + 4 * GB <= target:
            x = torch.empty(4 * GB, dtype=torch.uint8, pin_memory=True)
            x[:: 4096] = 1
            held.append(x)
            total += 4 * GB
    except Exception as e:  # noqa
        err = repr(e)[:200]
    rows["pin_capacity"] = {"limit_bytes": lim, "target_bytes": target, "pinned_bytes": total,
                            "s": time.perf_counter() - t0, "err": err}
    del held
    return rows


# ---------------------------------------------------------------- per model
def download(key):
    repo, rev, _ = MODELS[key]
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", f"from huggingface_hub import snapshot_download as s;"
                        f"print(s('{repo}', revision='{rev}', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model']))"],
                       env=ENV, capture_output=True, text=True, timeout=900)  # amendment 1: 15 min cap
    if r.returncode != 0:
        log(f"download {key} failed: {r.stderr[-400:]}")
        return None
    path = r.stdout.strip().splitlines()[-1]
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    return {"path": path, "bytes": size, "s": time.perf_counter() - t0}


def launch_once(key, cache_root, port, keep=False):
    dc = drop_caches()
    s = Server(key, port, cache_root)
    ready = s.wait_ready()
    ft = first_token(port, key)
    row = {"ready_s": ready, "first_req_ttft_s": ft, "launch_to_first_token_s": time.perf_counter() - s.t_launch,
           "drop_caches_ok": dc, "cache_root": cache_root, "facts": s.log_facts()}
    if keep:
        return row, s
    s.stop()
    return row, None


def sleep_cycles(s):
    port, key = s.port, s.key
    out = {"awake_ttft": [], "l1": [], "l2": [], "identity": []}
    ref = greedy16(port, key)
    out["ref_text"] = ref
    for i in range(5):
        out["awake_ttft"].append(first_token(port, key))
        time.sleep(0.5)
    for i in range(5):
        mem_before = gpu_q("memory.used")
        t_sleep, _ = post(port, "/sleep?level=1")
        mem_asleep = gpu_q("memory.used")
        time.sleep(2)
        t0 = time.perf_counter()
        t_wake, _ = post(port, "/wake_up")
        ft = first_token(port, key)
        total = time.perf_counter() - t0
        same = greedy16(port, key) == ref
        out["l1"].append({"rep": i, "sleep_s": t_sleep, "wake_s": t_wake, "ttft_after_wake_s": ft,
                          "pagein_to_first_token_s": total, "mem_before": mem_before, "mem_asleep": mem_asleep,
                          "identical": same})
        time.sleep(1)
    for i in range(3):
        t_sleep, _ = post(port, "/sleep?level=2")
        mem_asleep = gpu_q("memory.used")
        time.sleep(2)
        t0 = time.perf_counter()
        tw, _ = post(port, "/wake_up?tags=weights")
        tr, _ = post(port, "/collective_rpc", {"method": "reload_weights"})
        tk, _ = post(port, "/wake_up?tags=kv_cache")
        ft = first_token(port, key)
        total = time.perf_counter() - t0
        same = greedy16(port, key) == ref
        out["l2"].append({"rep": i, "sleep_s": t_sleep, "wake_weights_s": tw, "reload_s": tr, "wake_kv_s": tk,
                          "ttft_after_wake_s": ft, "reload_to_first_token_s": total, "mem_asleep": mem_asleep,
                          "identical": same})
        time.sleep(1)
    return out


def phase_model(key, port=8001):
    row = {"key": key, "repo": MODELS[key][0], "util": MODELS[key][2]}
    dl = download(key)
    if dl is None:
        return {"key": key, "infra": "download"}
    row["download"] = dl
    warm_root = f"{ROOT}/vcache/{key}/warm"
    row["cold"], row["warm"] = [], []
    # cold #1 primes the warm cache dir; colds #2/#3 use fresh dirs; then 3 warm launches on the primed dir
    r, _ = launch_once(key, warm_root, port)
    row["cold"].append(r)
    for i in (2, 3):
        check_deadline(f"m_{key}")
        r, _ = launch_once(key, f"{ROOT}/vcache/{key}/cold{i}", port)
        row["cold"].append(r)
    srv = None
    for i in range(3):
        check_deadline(f"m_{key}")
        r, srv = launch_once(key, warm_root, port, keep=(i == 2))
        row["warm"].append(r)
    try:
        row["cycles"] = sleep_cycles(srv)
        row["sleep_facts"] = srv.log_facts()
    finally:
        srv.stop()
    return row


# ---------------------------------------------------------------- colocation
def a_client(port, name, stop_at, rows, lock, seed):
    rnd = random.Random(seed)
    while time.time() < stop_at:
        prompt = " ".join(rnd.choice(["alpha", "beta", "gamma", "delta", "river", "stone", "cloud"]) for _ in range(100))
        body = {"model": name, "prompt": prompt, "max_tokens": 64, "stream": True, "temperature": 0, "ignore_eos": True}
        req = urllib.request.Request(f"{H}:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        t0 = time.time()
        times = []
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                for line in r:
                    if line.startswith(b"data: ") and b'"text"' in line:
                        times.append(time.time())
        except Exception as e:  # noqa
            with lock:
                rows.append({"t0": t0, "err": repr(e)[:120]})
            continue
        if len(times) >= 2:
            tpot = (times[-1] - times[0]) / (len(times) - 1)
            with lock:
                rows.append({"t0": t0, "ttft": times[0] - t0, "tpot": tpot, "t_end": times[-1], "n": len(times)})


def phase_coloc():
    root = f"{ROOT}/vcache"
    a = Server("small", 8001, f"{root}/small/warm")
    a.wait_ready()
    b = Server("d8b", 8002, f"{root}/d8b/warm")
    b.wait_ready()
    first_token(8001, "small")
    first_token(8002, "d8b")
    post(8002, "/sleep?level=1")
    time.sleep(3)
    rows, lock = [], threading.Lock()
    events = []
    t_start = time.time()
    stop_at = t_start + 30 + 5 * 14 + 10
    ths = [threading.Thread(target=a_client, args=(8001, "small", stop_at, rows, lock, s)) for s in range(8)]
    for t in ths:
        t.start()
    time.sleep(30)  # baseline window: B asleep
    for i in range(5):
        t0 = time.time()
        tw, _ = post(8002, "/wake_up")
        ft = first_token(8002, "d8b")
        t1 = time.time()
        events.append({"rep": i, "wake_start": t0, "first_token_at": t1, "wake_s": tw, "ttft_s": ft,
                       "pagein_to_first_token_s": t1 - t0})
        time.sleep(4)
        ts, _ = post(8002, "/sleep?level=1")
        events[-1]["sleep_s"] = ts
        events[-1]["sleep_end"] = time.time()
        time.sleep(8)
    for t in ths:
        t.join(timeout=200)
    out = {"t_start": t_start, "baseline_end": t_start + 30, "events": events, "a_rows": rows,
           "a_facts": a.log_facts(), "b_facts": b.log_facts()}
    a.stop()
    b.stop()
    return out


def main():
    # amendment 1: env + copy run in child processes so the driver never holds a CUDA context (host 1 kept 3.3 GB resident)
    if sys.argv[1:2] == ["--phase"]:
        dump(sys.argv[2], {"env": phase_env, "copy": phase_copy}[sys.argv[2]]())
        return
    log("start")
    for ph in ("env", "copy"):
        check_deadline(ph)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", ph], env=ENV)
        if r.returncode != 0 or not os.path.exists(f"ARM_DONE_{ph}"):
            log(f"{ph} fail rc={r.returncode}")
            finish(f"INFRA-FAIL:{ph}")
    for key in ("d8b", "small", "moe"):  # decision model first (prereg section 4 priority)
        check_deadline(f"m_{key}")
        try:
            dump(f"m_{key}", phase_model(key))
        except Exception as e:  # noqa
            log(f"model {key} fail {e!r}")
            dump(f"m_{key}", {"key": key, "infra": repr(e)[:300]})
            subprocess.run("pkill -TERM -f 'vllm serve' ; sleep 5; pkill -KILL -f 'vllm serve'", shell=True)
    check_deadline("coloc")
    try:
        dump("coloc", phase_coloc())
    except Exception as e:  # noqa
        log(f"coloc fail {e!r}")
        dump("coloc", {"infra": repr(e)[:300]})
    finish("OK")


if __name__ == "__main__":
    main()
