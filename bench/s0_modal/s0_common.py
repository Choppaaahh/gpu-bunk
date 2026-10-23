"""s0_common.py -- shared helpers for the S0 harness (Modal 2 x A100-80GB; CPU stub for the $0 rehearsal).

Output contract: a library; every item writes rows/<item>.json (one JSON object, verdict + data) and appends the same
object to rows/all.jsonl. Nothing here decides a verdict except the caller; helpers raise or return None, never fake 0.
Backend: S0_BACKEND=vllm (the box) | stub (bite 1's CPU stub engines + holders on a file fabric, the $0 rehearsal).
Prereg: (private report) (frozen at before any GPU minute).
"""
import base64
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "rows")
LOGS = os.path.join(HERE, "logs")
BACKEND = os.environ.get("S0_BACKEND", "vllm")
DEADLINE = float(os.environ.get("S0_DEADLINE") or 0) or None   # epoch; the runner maps S0_DEADLINE_REL -> S0_DEADLINE
EXT = os.path.join(HERE, "warm_wake_ext.py")
STUB = os.path.join(HERE, "warm_wake_stub.py")
GIB = 1 << 30
MIB = 1 << 20
TOL = 1e-3
PROMPTS = [[(1000 + 37 * k * j + 11 * j) % 100000 + 100 for j in range(48)] for k in (1, 2, 3)]
KVCACHED_ENV = ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH")
_LOCK = threading.Lock()
os.makedirs(ROWS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def left():
    return (DEADLINE - time.time()) if DEADLINE else 1e9


def log(msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    with _LOCK, open(os.path.join(LOGS, "progress.log"), "a") as f:
        f.write(line + "\n")


def put_row(item, verdict, **data):
    row = dict(item=item, verdict=verdict, ts=time.time(), backend=BACKEND, source="s0_harness_2972",
               trigger="S0 on Modal", trust_tier="agent-reasoning", **data)
    with _LOCK:
        with open(os.path.join(ROWS, f"{item}.json"), "w") as f:
            json.dump(row, f, indent=1, default=str)
        with open(os.path.join(ROWS, "all.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    log(f"ROW {item} {verdict}")
    return row


def http_json(url, body=None, method="POST", timeout=60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, (json.loads(raw) if raw else None)
            except json.JSONDecodeError:
                return r.status, {"raw": raw[:300].decode(errors="replace")}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, (json.loads(raw) if raw else None)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:300].decode(errors="replace")}


def child_env(cvd=None, pythonpath=False):
    env = {k: v for k, v in os.environ.items() if k not in KVCACHED_ENV}
    env.update(VLLM_SERVER_DEV_MODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    if cvd is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"          # what fabric_ctl.child_env sets (parity)
        env["CUDA_VISIBLE_DEVICES"] = cvd
    if pythonpath:
        env["PYTHONPATH"] = HERE + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class Proc:
    def __init__(self, name, cmd, env):
        self.name, self.cmd, self.env = name, cmd, env
        self.logpath = os.path.join(LOGS, f"{name}.log")
        self.fh = open(self.logpath, "ab")
        self.p = subprocess.Popen(cmd, env=env, stdout=self.fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True, cwd=HERE)
        log(f"spawn {name} pid={self.p.pid} cmd={' '.join(cmd[:8])} ...")

    @property
    def pid(self):
        return self.p.pid

    def alive(self):
        return self.p.poll() is None

    def stop(self, grace=30):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log(f"UNREAPED {self.name}")
        self.fh.close()
        log(f"stopped {self.name} rc={self.p.returncode}")
        return self.p.returncode

    def text(self):
        try:
            with open(self.logpath, "rb") as f:
                return f.read().decode(errors="replace")
        except OSError:
            return ""


def wait_line(proc, regex, timeout):
    t0 = time.monotonic()
    rx = re.compile(regex)
    while time.monotonic() - t0 < timeout:
        m = rx.search(proc.text())
        if m:
            return m
        if not proc.alive():
            raise RuntimeError(f"{proc.name} exited rc={proc.p.returncode} before /{regex}/")
        time.sleep(0.2)
    raise RuntimeError(f"{proc.name}: /{regex}/ not seen in {timeout}s")


def wait_health(url, timeout, proc=None):
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout:
        if proc is not None and not proc.alive():
            raise RuntimeError(f"{url}: process exited rc={proc.p.returncode} before health")
        try:
            st, _ = http_json(url + "/health", method="GET", timeout=3.0)
            if st == 200:
                return round(time.monotonic() - t0, 3)
            last = st
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:120]
        time.sleep(0.5)
    raise RuntimeError(f"{url}: not healthy in {timeout}s (last {last})")


def smi(args):
    if BACKEND != "vllm":
        return None
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else f"ERR rc={r.returncode} {r.stderr.strip()[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"[:200]


def gpu_used_mib():
    """{index: memory.used MiB} from nvidia-smi --query-gpu, or None (stub / unreadable)."""
    out = smi(["--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if not out or out.startswith("ERR"):
        return None
    res = {}
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():   # QA: '[N/A]' is skipped, not a crash
            res[int(parts[0])] = int(parts[1])
    return res or None


def compute_apps():
    return smi(["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"])


def rpc(port, method, spec, timeout=900):
    st, js = http_json(f"http://127.0.0.1:{port}/collective_rpc",
                       {"method": method, "args": [json.dumps(spec)], "timeout": timeout}, timeout=timeout + 30)
    if st != 200 or not isinstance(js, dict):
        return {"ok": False, "reason": f"RPC_HTTP_{st}:{str(js)[:200]}"}
    res = js.get("results") or [None]
    return res[0] if isinstance(res[0], dict) else {"ok": False, "reason": f"RPC_NO_DICT:{str(res)[:200]}"}


def holder(port, path, body, timeout=900):
    try:
        st, js = http_json(f"http://127.0.0.1:{port}{path}", body, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"CONN:{e!r}"[:200]}
    if st != 200 or not isinstance(js, dict):
        return {"ok": False, "reason": f"HTTP_{st}:{str(js)[:200]}"}
    return js


def handle_len(b64):
    try:
        return len(base64.b64decode(b64 or "", validate=True))
    except Exception:  # noqa: BLE001
        return None


def tf_read(base_url, model, timeout=300):
    """teacher-forced prompt logprobs of the ACTUAL prompt tokens (keyed by id) for P1-P3 -> list of lists, or raises."""
    out = []
    for ids in PROMPTS:
        st, js = http_json(base_url + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                                           "temperature": 0, "prompt_logprobs": 1}, timeout=timeout)
        if st != 200:
            raise RuntimeError(f"tf_read {model} -> {st} {str(js)[:200]}")
        plp = js["choices"][0]["prompt_logprobs"]
        row = []
        for j, (tok, d) in enumerate(zip(ids, plp)):
            if j == 0:
                continue
            v = (d or {}).get(str(tok))
            row.append(None if v is None else float(v["logprob"]))
        out.append(row)
    return out


def gen_read(base_url, model, n=32, timeout=300):
    """greedy decode of P1 (exercises the decode CUDA graphs over the restored weights; reported only, not frozen)."""
    st, js = http_json(base_url + "/v1/completions", {"model": model, "prompt": PROMPTS[0], "max_tokens": n,
                                                       "temperature": 0}, timeout=timeout)
    if st != 200:
        return None
    return js["choices"][0]["text"]


def tf_diff(a, b):
    """max |delta| over every position of P1-P3; inf on any shape or None mismatch."""
    if a is None or b is None or len(a) != len(b):
        return float("inf")
    m = 0.0
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return float("inf")
        for x, y in zip(ra, rb):
            if x is None or y is None:
                return float("inf")
            m = max(m, abs(x - y))
    return m


def stock_sleep(port, mode="wait", timeout=900):
    t0 = time.monotonic()
    st, js = http_json(f"http://127.0.0.1:{port}/sleep?level=1&mode={mode}", timeout=timeout)
    return st, round(time.monotonic() - t0, 6)


def wake(port, tag=None, timeout=900):
    t0 = time.monotonic()
    q = f"?tags={tag}" if tag else ""
    st, js = http_json(f"http://127.0.0.1:{port}/wake_up{q}", timeout=timeout)
    return st, round(time.monotonic() - t0, 6)


def is_sleeping(port):
    st, js = http_json(f"http://127.0.0.1:{port}/is_sleeping", method="GET", timeout=60)
    if st != 200 or not isinstance(js, dict) or "is_sleeping" not in js:
        return None
    return bool(js["is_sleeping"])


def complete(port, model, prompt=(1, 2, 3), n=2, timeout=600):
    st, js = http_json(f"http://127.0.0.1:{port}/v1/completions",
                       {"model": model, "prompt": list(prompt), "max_tokens": n}, timeout=timeout)
    return st


def grep(text, regex):
    return re.findall(regex, text)


class Load:
    """closed-loop streamed batch load (8 x 512/512 ignore_eos, token-id prompts) against one engine; records how each
    stream ended (finish_reason, usage)."""

    def __init__(self, port, model, streams=8, p=512, m=512, seed=0):
        self.port, self.model, self.n, self.p, self.m = port, model, streams, p, m
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.ends = []                                  # (t_end, finish, n_chunks, usage_tokens, err)
        self.seed = seed
        self.ths = []

    def start(self):
        for i in range(self.n):
            th = threading.Thread(target=self._loop, args=(i,), daemon=True)
            th.start()
            self.ths.append(th)

    def _loop(self, i):
        import http.client
        import random
        rng = random.Random(self.seed * 1000 + i)
        while not self.stop_ev.is_set():
            ids = [rng.randint(1000, 100000) for _ in range(self.p)]
            body = {"model": self.model, "prompt": ids, "max_tokens": self.m, "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}}
            fin, got, usage, err = None, 0, None, None
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
                conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    err = f"status {r.status} {r.read()[:200]!r}"
                else:
                    for ln in r:
                        if not ln.startswith(b"data: ") or ln.startswith(b"data: [DONE]"):
                            continue
                        ob = json.loads(ln[6:])
                        if ob.get("usage"):
                            usage = int(ob["usage"].get("completion_tokens") or 0)
                        for ch in ob.get("choices") or []:
                            if ch.get("text") is not None:
                                got += 1
                            fin = ch.get("finish_reason") or fin
                conn.close()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
            with self.lock:
                self.ends.append((time.monotonic(), fin, got, usage, err))
            if err:
                time.sleep(0.5)

    def stop(self):
        self.stop_ev.set()

    def summary(self, since=0.0, until=None):
        with self.lock:
            e = [x for x in self.ends if x[0] >= since and (until is None or x[0] <= until)]
        fins = {}
        for x in e:
            fins[str(x[1])] = fins.get(str(x[1]), 0) + 1
        return dict(n=len(e), finish=fins, usage_tokens=sum(x[3] or 0 for x in e), chunks=sum(x[2] for x in e),
                    errors=sum(1 for x in e if x[4]), first_err=next((x[4] for x in e if x[4]), None))
