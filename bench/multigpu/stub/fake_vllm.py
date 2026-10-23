#!/usr/bin/env python3
"""Prints the log lines the driver parses (backend in the AUTO form, 'Directly load AOT', 'GPU KV cache size', 'Total CPU
offloaded parameters' when --cpu-offload-gb is set), then serves: GET /health /metrics /is_sleeping; POST /v1/completions
(SSE + usage, or non-streaming with a prompt LIST and routed_experts as base64 .npy when --enable-return-routed-experts),
/tokenize, /sleep, /wake_up. A --kv-cache-memory-bytes engine admits floor(bytes / ((in + out) x 147456)) requests at once
(the rest wait: the KV-starved queue of item 8). TPOT grows with --cpu-offload-gb so the scorer sees a non-flat curve."""
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

a = sys.argv[1:]
arg = lambda k, d=None: a[a.index(k) + 1] if k in a else d
port, repo = int(arg("--port")), a[1]
GIB = 1024 ** 3
deep = "DeepSeek" in repo
off_gb = float(arg("--cpu-offload-gb", "0"))
kvb = int(arg("--kv-cache-memory-bytes", "0") or 0)
routed = "--enable-return-routed-experts" in a
dcp = int(arg("--decode-context-parallel-size", "1"))
TPOT = 0.004 + 0.0008 * off_gb + (0.0003 if dcp > 1 else 0.0)
print(f"INFO [cuda.py:538] Using {'TRITON_MLA' if deep else 'FLASH_ATTN'} attention backend out of potential backends: "
      f"['FLASH_ATTN', 'TRITON_ATTN'].", flush=True)
print("INFO [decorators.py:313] Directly load AOT compilation from path /root/.cache/x", flush=True)
if off_gb > 0:
    print(f"INFO [uva.py:70] Total CPU offloaded parameters: {off_gb:.2f} GiB", flush=True)
print(f"INFO [gpu_model_runner.py:5417] Model loading took {(29.26 if deep else 15.26) - off_gb:.4f} GiB memory and 1.000000 seconds", flush=True)
print(f"INFO [kv_cache_utils.py:2395] GPU KV cache size: {(kvb // 147456) if kvb else 120000:,} tokens", flush=True)
ST = {"running": 0, "waiting": 0, "gen": 0, "prompt": 0, "len": 0, "abort": 0, "sleeping": False}
LOCK = threading.Lock()
CAP = None if not kvb else max(1, kvb // ((1024 + 512) * 147456))
SEM = threading.Semaphore(CAP) if CAP else None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *x):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"")
        elif self.path == "/is_sleeping":
            self._send(200, json.dumps({"is_sleeping": ST["sleeping"]}).encode())
        elif self.path == "/metrics":
            with LOCK:
                s = dict(ST)
            body = (f'vllm:num_requests_running{{engine="0",model_name="x"}} {s["running"]}\n'
                    f'vllm:num_requests_waiting{{engine="0",model_name="x"}} {s["waiting"]}\n'
                    f'vllm:kv_cache_usage_perc{{engine="0",model_name="x"}} {min(1.0, s["running"] / (CAP or 64))}\n'
                    f'vllm:num_preemptions_total{{engine="0",model_name="x"}} 0.0\n'
                    f'vllm:generation_tokens_total{{engine="0",model_name="x"}} {s["gen"]}\n'
                    f'vllm:prompt_tokens_total{{engine="0",model_name="x"}} {s["prompt"]}\n'
                    f'vllm:iteration_tokens_total_sum{{engine="0",model_name="x"}} {s["gen"] + s["prompt"]}\n'
                    f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="x"}} {s["len"]}\n'
                    f'vllm:request_success_total{{engine="0",finished_reason="abort",model_name="x"}} {s["abort"]}\n')
            self._send(200, body.encode(), "text/plain")
        else:
            self._send(404, b"")

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        p = self.path.split("?")[0]
        if p == "/sleep":
            time.sleep(0.15)
            ST["sleeping"] = True
            return self._send(200, b"")
        if p == "/wake_up":
            time.sleep(0.25)
            ST["sleeping"] = False
            return self._send(200, b"")
        if p == "/tokenize":
            toks = [1000 + (ord(c) % 500) for c in req.get("prompt", "")][::4]
            return self._send(200, json.dumps({"tokens": toks, "count": len(toks)}).encode())
        if p != "/v1/completions":
            return self._send(404, b"")
        if not req.get("stream"):
            return self._nonstream(req)
        out = int(req["max_tokens"])
        with LOCK:
            ST["waiting"] += 1
        if SEM:
            SEM.acquire()
        with LOCK:
            ST["waiting"] -= 1
            ST["running"] += 1
            ST["prompt"] += len(req["prompt"]) if isinstance(req["prompt"], list) else 100
        done = 0
        try:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for _ in range(out):
                time.sleep(TPOT)
                self.wfile.write(b'data: {"choices":[{"text":"a"}]}\n\n')
                self.wfile.flush()
                done += 1
                with LOCK:
                    ST["gen"] += 1
            u = {"choices": [], "usage": {"completion_tokens": out}}
            self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            with LOCK:
                ST["len"] += 1
        except (BrokenPipeError, ConnectionResetError):
            with LOCK:
                ST["abort"] += 1
        finally:
            with LOCK:
                ST["running"] -= 1
            if SEM:
                SEM.release()

    def _nonstream(self, req):
        import numpy as np
        prompts = req["prompt"] if isinstance(req["prompt"], list) and req["prompt"] and isinstance(req["prompt"][0], list) \
            else [req["prompt"]]
        out = int(req["max_tokens"])
        time.sleep(TPOT * out)
        choices = []
        for i, pr in enumerate(prompts):
            ch = {"index": i, "text": "a" * out, "finish_reason": "length"}
            if routed:
                rng = np.random.default_rng(1000 + i)
                arr = np.stack([np.stack([rng.choice(64, 6, replace=False) for _ in range(27)]) for _ in range(out - 1)])
                buf = io.BytesIO()
                np.save(buf, arr.astype(np.uint8))
                ch["routed_experts"] = base64.b64encode(buf.getvalue()).decode()
            choices.append(ch)
        self._send(200, json.dumps({"choices": choices, "usage": {"completion_tokens": out * len(prompts)}}).encode())


ThreadingHTTPServer.request_queue_size = 1024
ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
