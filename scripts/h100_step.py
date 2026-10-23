#!/usr/bin/env python3
"""h100_step.py:. The frozen Q/S/LC controlled-decode anchor on ONE card (Modal H100 SXM; an A100-80 platform
control), for Qwen3-8B and Qwen3-14B, then the prereg's verdicts against modelled H step.

Prereg: (private report) wins on any divergence. The frozen gpu_driver.py and
gpu_score.py are IMPORTED, never edited (sha256 checked by --run and --score).

Modes:
  --run --plan P     (in the container) WRITES ./rows ./logs ./m_<key>/{rows/anchor.jsonl,meta.json,g6_serving.json,smi.csv}
                     ARM_DONE_<key>, DONE (one status line: OK / STOP-CARD / STOP-D2D / INFRA-*). Exit 0 once started.
  --score --run-dir R --out J [--control-dir C] [--allow-stub]
                     (on the CPU host) prints VERDICT lines and WRITES J (overwritten). exit 0 = verdicts printed; 3 = no verdict
                     (STOP / INFRA on the 8B); 1 = DONE or a required file missing/unreadable; 2 = usage, or a frozen
                     constant / manifest mismatch (the scorer refuses to score).
  --stub-smoke S     (on the CPU host, CPU) in the EMPTY scratch dir S: fake vllm + fake nvidia-smi; an H100-SXM run then --score
                     --allow-stub, and a PCIe-card run that must STOP-CARD before any engine. exit 0 = both behave; else 1.
  --self-test        hand-computed goldens; exit 0 + "SELF-TEST PASS n/n", else exit 1.
Output contract: a STOP or INFRA run is never scored as a number `--score` exit 0 alone is not a pass of any check: read
the VERDICT lines.

Top pitfalls (observed): HF_HUB_OFFLINE changes vLLM's compile key: set only for engines, after downloads;
`hf download --include a b` keeps only the last pattern: this uses snapshot_download + a byte check; a dead engine
scores as slow: every model's cells pass I-ALIVE/I-COMPLETE first; a cold first timed run hits Triton JIT: the
frozen 2 warm rounds per cell run first.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FROZEN_DIR = os.path.join(REPO, "bench", "multigpu")
SHA_DRIVER = "d326720c53a860a60ffebe580137511cdd428c5b3a1b22d75862b9ae44ce7611"
SHA_SCORE = "7c7b7de63d6a37c197b50761affce709fc2ece5080ec24e4f63e55b03cd3dc2d"
MANIFEST_SHA = "d2144cb5271959523ff3c56298ce6fcc0b4a1edf0bcc9bca4f6ac27b10f7b57a"   # prereg §2.1b
CELLS = (("Q", 1, 4096), ("S", 8, 4096), ("LC", 2, 16384))
OUT_TOKENS, PASSES, WARM_ROUNDS = 64, 3, 2
KV_NEED = 8 * (4096 + 64)                                    # I-KV: 33,280 tokens
W8, W14 = 16381516776.0, 29536665640.0
RHO14 = (40 * 128 * 40) / (32 * 128 * 36)
# prereg §3, ms, frozen (recomputed from module at score time; > 1e-6 ms off = refuse)
PRED = {"q8": {"Q": {"A": 11.3567, "HM": 8.5818, "HF": 6.9123}, "S": {"A": 27.0987, "HM": 24.3238, "HF": 16.4938},
               "LC": {"A": 24.9762, "HM": 22.2013, "HF": 15.2019}},
        "q14": {"Q": {"A": 17.7936, "HM": 12.7903, "HF": 10.8302}, "S": {"A": 38.7349, "HM": 33.7316, "HF": 23.5763},
                "LC": {"A": 36.5778, "HM": 31.5745, "HF": 22.2633}}}
PRED_TOL_MS = 5e-5          # the table is printed to 4 dp; the recompute must round to it
A_FIT_MS, F_MS, KB_A_MS, KL_A_S = 9.107822733941803, 2.0170966319484167, 0.338947016095358, 4.6267392100077835e-07
HM_BETA = 3350.0 / 2039.0
SIZE_TERM_HM_MS = 3.4658    # kappa (W14 - W8) / B_H, prereg §5.2
VAST_H2 = {"Q": 11.34188213045635, "S": 26.980297906058173, "LC": 24.946615809486025}   # run_h2 (FIT), ms
PROV = {"source": "h100_step", "trigger": "modal-run", "trust_tier": "autonomous-unsupervised"}
FROZEN_ULIMIT = 'ulimit -n 65536 && exec "$@"'          # gpu_driver.Eng.start's wrapper, verbatim
FALLBACK_ULIMIT = 'ulimit -n "$(ulimit -Hn)"; exec "$@"'   # addendum A1: only when the container refuses 65536


class PopenShim:
    """Addendum A1: stands in for gpu_driver's `subprocess` module; rewrites ONLY Eng.start's frozen ulimit wrapper, every
    other attribute and call passes through unchanged (sh() -> subprocess.run, TimeoutExpired, ...)."""

    def __init__(self, real):
        self.real = real

    def __getattr__(self, k):
        return getattr(self.real, k)

    def Popen(self, args, *a, **k):
        if isinstance(args, list) and len(args) > 3 and args[0] == "bash" and args[1] == "-c" and args[2] == FROZEN_ULIMIT:
            args = [args[0], args[1], FALLBACK_ULIMIT, *args[3:]]
        return self.real.Popen(args, *a, **k)


# ----------------------------------------------------------------------------------------------- pure rules (goldens)
def card_gate(name, mem_mib, pmax_w, rule):
    """prereg §4 I-CARD -> (ok, why)."""
    if rule["name_contains"] not in (name or ""):
        return False, f"name {name!r} lacks {rule['name_contains']!r}"
    if not (rule["mem_min_mib"] <= mem_mib <= rule["mem_max_mib"]):
        return False, f"memory.total {mem_mib} MiB outside [{rule['mem_min_mib']}, {rule['mem_max_mib']}]"
    if pmax_w < rule["pmax_min_w"]:
        return False, f"power.max_limit {pmax_w} W < {rule['pmax_min_w']} W (not the SXM part)"
    return True, "ok"


def parse_card(line):
    """`nvidia-smi --query-gpu=name,memory.total,power.max_limit,... --format=csv,noheader,nounits` first line."""
    parts = [p.strip() for p in (line or "").strip().splitlines()[0].split(",")] if (line or "").strip() else []
    if len(parts) < 3:
        raise ValueError(f"card line unreadable: {line!r}")
    return parts[0], float(parts[1]), float(parts[2]), parts[3:]


def kv_tokens(log_text):
    m = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", log_text or "")
    return int(m[-1].replace(",", "")) if m else None


def backend(log_text):
    m = re.search(r"Using (?:AttentionBackendEnum\.)?([A-Z][A-Z0-9_]+)(?: attention)? backend\b", log_text or "")
    return m.group(1) if m else None


def fit3(q, s, lc):
    """ exact 3-point solve (steptime_crossover.fit3): t = a + kb*b + kl*C at Q (1, 4096), S (8, 32768),
    LC (2, 32768). Inputs and a, kb in the same unit; kl per token."""
    kb = (s - lc) / 6.0
    kl = (lc - q - kb) / 28672.0
    return q - kb - 4096.0 * kl, kb, kl


def band_model(ranges, preds, lo=0.90, hi=1.10):
    """§5.1: every cell's pass range inside [lo, hi] x prediction."""
    return all(lo * preds[c] <= r[0] and r[1] <= hi * preds[c] for c, r in ranges.items())


def model_label(ranges, hm, hf):
    """§5.1 label, the same for 8B and 14B (QA r1: one rule, not two)."""
    hm_ok, hf_ok = band_model(ranges, hm), band_model(ranges, hf)
    return " + ".join(x for x, ok in (("HM-AGREES", hm_ok), ("HF-AGREES", hf_ok)) if ok) or "NEITHER"


def crossover(ranges8, hm8, ranges14=None, hm14=None, edge=1.02):
    """§5.2 -> (band, per-cell detail). ranges: {cell: [min, max]} ms; hm: {cell: P_HM} ms."""
    groups = [("q8", ranges8, hm8)] + ([("q14", ranges14, hm14)] if ranges14 else [])
    detail, all_below, any_above = {}, True, False
    for g, rr, hm in groups:
        for c, (lo, hi) in rr.items():
            lim = edge * hm[c]
            detail[f"{g}.{c}"] = {"min": lo, "max": hi, "edge": lim}
            all_below &= hi <= lim
            any_above |= lo > lim
    if all_below:
        band = "BRACKETED-OR-FASTER"
    elif any_above:
        band = "SLOWER-THAN-HM"
    else:
        band = "AT-EDGE"
    return band + ("" if ranges14 else " (8B only; 13-14 B step still MODELLED)"), detail


def beta_band(a_h_ms):
    """§5.1: implied bandwidth ratio of the weight term vs the 1.643 extrapolation."""
    if a_h_ms <= F_MS:
        return None, "UNDEFINED (a_H <= F)"
    b = (A_FIT_MS - F_MS) / (a_h_ms - F_MS)
    if b < 0.9 * HM_BETA:
        return b, "SLOWER-THAN-EXTRAPOLATED"
    if b > 1.1 * HM_BETA:
        return b, "FASTER-THAN-EXTRAPOLATED"
    return b, "HOLDS"


def term_band(ratio):
    """§5.1: k_b or k_L ratio H / A100 -> HM-like (~1.0) / HF-like (~0.609) / NEITHER."""
    if 0.85 <= ratio <= 1.15:
        return "HM-like"
    if 0.517 <= ratio <= 0.700:
        return "HF-like"
    return "NEITHER"


def a_position(t, p_a, p_hm):
    return (p_a - t) / (p_a - p_hm)


def platform(t_c, vast=VAST_H2, tol=0.03):
    """§5.3 -> (band, delta per cell)."""
    d = {c: t_c[c] / vast[c] - 1.0 for c in vast}
    return ("PLATFORM-CLEAN" if all(abs(v) <= tol for v in d.values()) else "PLATFORM-OFFSET"), d


def prompt_manifest_sha(D):
    man = []
    for p in range(PASSES):
        for n, B, L in CELLS:
            for i in range(B):
                ids = D.prompt_ids(f"anchor{n}:{p}:{i}", L)
                man.append({"cell": n, "pass": p, "i": i, "L": L, "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


def smi_summary(text):
    """D7 sampler CSV (timestamp, clocks.sm, clocks.mem, power.draw, temp, util, event_reasons) -> summary; rows that do not
    parse are counted, never coerced."""
    sm, pw, reasons, bad = [], [], set(), 0
    for ln in (text or "").splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 7:
            bad += 1
            continue
        try:
            sm.append(float(p[1].split()[0]))
            pw.append(float(p[3].split()[0]))
        except (ValueError, IndexError):
            bad += 1
            continue
        if p[6] not in ("0x0000000000000000", "0x0000000000000001", "[N/A]", "N/A"):   # 0x1 = GPU idle
            reasons.add(p[6])
    return {"n": len(sm), "unparsed": bad, "sm_mhz_median": statistics.median(sm) if sm else None,
            "sm_mhz_min": min(sm) if sm else None, "power_w_max": max(pw) if pw else None,
            "event_reasons_seen": sorted(reasons)}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- the container run
class Run:
    def __init__(self, plan, stub):
        self.plan, self.stub = plan, stub
        self.deadline = float(os.environ.get("H2960_DEADLINE") or 0) or time.time() + 3000

    def log(self, msg):
        with open("logs/drive.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}\n")

    def finish(self, status):
        for d in (".", "rows"):
            with open(os.path.join(d, "DONE"), "w") as f:
                f.write(status + "\n")
        self.sh("nvidia-smi > logs/nvidia_smi_end.txt 2>&1", 60)
        self.log(f"FINISH {status}")
        sys.exit(0)

    def sh(self, cmd, timeout=120, env=None):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env or self.D.ENV)
        except subprocess.TimeoutExpired:
            return 124, f"timeout {timeout}s"
        return r.returncode, r.stdout + r.stderr

    def setup(self):
        for d in ("rows", "logs"):
            os.makedirs(d, exist_ok=True)
        for f, want in (("gpu_driver.py", SHA_DRIVER), ("gpu_score.py", SHA_SCORE)):
            if sha256_file(f) != want:
                with open("DONE", "w") as fh:
                    fh.write(f"PREMISE-FAIL {f} sha mismatch\n")
                sys.exit(0)
        sys.path.insert(0, os.getcwd())
        import gpu_driver as D
        self.D = D
        D.ENV = {k: v for k, v in os.environ.items() if k not in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH")}   # as D.main()
        D.ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        D.ENV["HF_XET_HIGH_PERFORMANCE"] = "1"
        D.PY = sys.executable
        cand = os.path.join(os.path.dirname(D.PY), "vllm")
        D.VLLM = cand if (os.path.exists(cand) and not self.stub) else "vllm"
        D.PROV = dict(PROV)
        rc, out = self.sh("bash -c 'ulimit -n 65536 && ulimit -n'", 30)
        self.ulimit = {"rc": rc, "out": out.strip()[-120:], "forced": os.environ.get("H2960_FORCE_ULIMIT_FALLBACK") == "1"}
        if rc != 0 or self.ulimit["forced"]:
            rc2, hard = self.sh("bash -c 'ulimit -Hn'", 30)
            self.ulimit.update(fallback=True, hard=hard.strip())
            D.subprocess = PopenShim(subprocess)
        else:
            self.ulimit["fallback"] = False
        json.dump(self.ulimit, open("logs/ulimit.json", "w"), indent=1)

    def env_dump(self):
        cmds = ["nvidia-smi", "nvidia-smi -q", "free -g", "df -h / | tail -1", "nproc", "lscpu | grep -E 'Model name|^CPU\\(s\\)'",
                f"{self.D.PY} -c 'import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__)'",
                "env | grep -E '^MODAL_(REGION|CLOUD_PROVIDER)=' || true"]
        with open("logs/env.txt", "w") as f:
            for c in cmds:
                rc, out = self.sh(c, 120)
                f.write(f"$ {c}\n{out}\n")
        if not self.stub:                                              # as gpu_driver.phase_env
            rc, out = self.sh(f"{self.D.PY} -c 'import vllm;print(vllm.__version__)'", 300)
            if "0.30.0" not in out:
                self.finish(f"PREMISE-FAIL vllm={out.strip()[-40:]}")

    def card(self):
        q = "name,memory.total,power.max_limit,power.limit,clocks.max.sm,clocks.max.mem,pci.bus_id"
        rc, out = self.sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits", 60)
        doc = {"query": q, "raw": out.strip(), "rc": rc}
        try:
            name, mem, pmax, rest = parse_card(out)
            ok, why = card_gate(name, mem, pmax, self.plan["card"])
            doc.update(name=name, memory_mib=mem, power_max_w=pmax, rest=rest, ok=ok, why=why)
        except ValueError as e:
            doc.update(ok=False, why=str(e))
        rc2, qq = self.sh("nvidia-smi -q", 60)
        for key in ("Product Name", "Product Brand", "Board Part Number", "GPU Part Number", "Module ID", "Max Power Limit"):
            m = re.search(rf"^\s*{key}\s*:\s*(.+)$", qq, re.M)
            doc[f"q_{key}"] = m.group(1).strip() if m else None
        json.dump(doc, open("logs/card_gate.json", "w"), indent=1)
        self.log(f"I-CARD {doc.get('ok')} {doc.get('why')} raw={doc['raw'][:160]}")
        return doc

    def d2d(self):
        """The frozen probe's HBM method on one card (beta_bench.probe): 1 GiB, 10 reps, median, 2*nb / t."""
        import torch
        cpu = self.stub
        dev = torch.device("cpu") if cpu else torch.device("cuda:0")
        nb, reps = (1 << 20, 3) if cpu else (1 << 30, 10)
        b = torch.empty(2 * nb, dtype=torch.uint8, device=dev)
        ts = []
        for _ in range(reps):
            if not cpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            b[nb:].copy_(b[:nb], non_blocking=True)
            if not cpu:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        del b
        res = {"hbm_d2d_gbs_0": 2 * nb / statistics.median(ts) / 1e9, "reps": reps, "nb": nb, "ts": ts, "cpu": cpu}
        json.dump(res, open("logs/probe.json", "w"), indent=1)
        self.log(f"I-D2D {res['hbm_d2d_gbs_0']:.1f} GB/s")
        return res["hbm_d2d_gbs_0"]

    def start_downloads(self):
        self.dl, self.dl_procs = {}, {}
        env = dict(self.D.ENV)                      # captured BEFORE HF_HUB_OFFLINE is set for engines

        def dl(key, m, slot, code=None):
            t0 = time.time()
            code = code or (f"from huggingface_hub import snapshot_download as s; print(s('{m['repo']}', revision='{m['revision']}', "
                            f"allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model','*.jinja','*.py']))")
            pr = subprocess.Popen([self.D.PY, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  start_new_session=True)                     # A2: killable as a group
            self.dl_procs[key] = pr
            try:
                so, se = pr.communicate(timeout=2400)
            except subprocess.TimeoutExpired:
                pr.kill()
                slot.update(ok=False, err="timeout 2400 s", t_end=time.time())
                return
            dirs = [ln.strip() for ln in so.strip().splitlines() if os.path.isdir(ln.strip())]
            if pr.returncode != 0 or not dirs:
                slot.update(ok=False, err=(se or so)[-300:], t_end=time.time())
                return
            st = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(dirs[-1]) for f in fs
                     if f.endswith(".safetensors"))
            slot.update(ok=st == m["bytes"], safetensors_bytes=st, want=m["bytes"], s=time.time() - t0, t_end=time.time())

        for j, (key, m) in enumerate(self.plan["models"]):
            slot = {"t_start": time.time()}
            if self.stub and j > 0 and os.environ.get("H2960_STUB_SLOW_DL") == "1":     # stub: a download that never ends
                th = threading.Thread(target=dl, args=(key, m, slot, "import time; time.sleep(1000)"), daemon=True)
                th.start()
                self.dl[key] = (th, slot)
                continue
            if self.stub:
                slot.update(ok=True, stub=True, t_end=time.time(), s=0.0)
                self.dl[key] = (None, slot)
                continue
            th = threading.Thread(target=dl, args=(key, m, slot), daemon=True)
            th.start()
            self.dl[key] = (th, slot)

    def join_download(self, key, timeout):
        th, slot = self.dl[key]
        if th is not None:
            th.join(timeout=max(1.0, timeout))
        return slot

    def downloads_alive(self):
        return sorted(k for k, (th, _) in self.dl.items() if th is not None and th.is_alive())

    def sampler(self, path):
        q = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active"
        rc, _ = self.sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader", 30)
        if rc != 0:
            q = q.replace("clocks_event_reasons", "clocks_throttle_reasons")
        f = open(path, "w")
        return subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-lms", "500"], stdout=f,
                                stderr=subprocess.STDOUT, env=self.D.ENV), f

    def model(self, key, m, port, probe_gbs):
        D, a = self.D, self.plan
        md = f"m_{key}"
        os.makedirs(f"{md}/rows", exist_ok=True)
        meta = {"key": key, "repo": m["repo"], "revision": m["revision"], "t_begin": time.time()}
        slot = self.join_download(key, self.deadline - time.time() - 60)
        meta["download"] = dict(slot)
        if not slot.get("ok"):
            meta["infra"] = f"I-DL {slot.get('err') or slot}"
            json.dump(meta, open(f"{md}/meta.json", "w"), indent=1)
            self.log(f"{key} INFRA {meta['infra']}")
            return False
        D.ENV["HF_HUB_OFFLINE"] = "1"                                  # engines only
        e = D.Eng(key, m, "0", port, a["util"], tag=key)
        self.cur_eng = e
        e.start()
        ok, msg = e.wait_up(timeout=max(60.0, min(900.0, self.deadline - time.time() - 120)))
        txt = e.text()
        meta.update(boot_ok=ok, boot_msg=msg, backend=backend(txt), kv_tokens=kv_tokens(txt), t_up=time.time())
        self.log(f"boot {key} {ok} {msg} backend={meta['backend']} kv={meta['kv_tokens']}")
        if not ok:
            meta["infra"] = f"BOOT {msg}"
            e.stop()
            json.dump(meta, open(f"{md}/meta.json", "w"), indent=1)
            return False
        meta["i_kv"] = meta["kv_tokens"] is not None and meta["kv_tokens"] >= KV_NEED
        t_w, cap = time.time(), float(self.plan.get("max_dl_wait_s", 300))
        for k in list(self.dl):                                         # D6 / I-NOOVERLAP: no download during warm or timed
            self.join_download(k, min(cap - (time.time() - t_w), self.deadline - time.time() - 60))
        for k in self.downloads_alive():                                # A2: abandon a late download, never overlap it
            self.dl[k][1]["abandoned"] = f"still downloading {cap:.0f}s after {key} was up; killed to keep {key}'s passes clean"
            pr = self.dl_procs.get(k)
            if pr is not None and pr.poll() is None:
                try:
                    os.killpg(pr.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self.join_download(k, 30)
            self.log(f"A2 abandoned download {k}")
        alive_dl = self.downloads_alive()
        meta["downloads_alive_at_warm"] = alive_dl
        meta["i_nooverlap"] = not alive_dl
        meta["t_warm"] = time.time()
        proc, fh = self.sampler(f"{md}/smi.csv")
        try:
            for n, B, L in CELLS:                                       #: warm requests per shape, discarded
                for w in range(WARM_ROUNDS):
                    D.round_B(port, key, B, L, 8, f"aw{n}", w)
            if m.get("g6"):
                c = a["ctrl"]
                loop = D.Loop(port, key, c["conc"], c["in"], c["out"], "ctrl")
                loop.start()
                time.sleep(c["secs"])
                rows = loop.stop()
                D.jsonl(f"{md}/rows/ctrl.jsonl", rows)
                tp = sorted(r["tpot"] for r in rows if r.get("tpot") is not None and not r.get("err"))
                med = tp[len(tp) // 2] if tp else None
                ref = D.g6_ref_s(m["bytes"], c["conc"], c["in"] + c["out"] // 2, m["kv_bpt"], probe_gbs)
                json.dump({"tpot_p50": med, "ref_s": ref, "ratio": (med / ref) if (med and ref) else None, "n": len(rows),
                           "n_tpot": len(tp), "tol": 0.5, "pass": bool(med and med / ref <= 1.5), "gating": False,
                           "a100_ratio_2898_h2": 1.2219866523845206}, open(f"{md}/g6_serving.json", "w"), indent=1)
            met0 = e.metrics()
            meta["t_timed"] = time.time()
            alive = True
            for p in range(PASSES):
                for n, B, L in CELLS:
                    rows = D.round_B(port, key, B, L, OUT_TOKENS, f"anchor{n}", p)
                    D.jsonl(f"{md}/rows/anchor.jsonl", [dict(r, cell=n, B=B, L=L, **{"pass": p}) for r in rows])
                    alive &= e.alive()
            met1 = e.metrics()
            meta["t_timed_end"] = time.time()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            fh.close()
        meta.update(i_alive=alive and e.alive(), metrics_before=met0, metrics_after=met1,
                    preemptions_delta=met1.get("vllm:num_preemptions_total", 0.0) - met0.get("vllm:num_preemptions_total", 0.0))
        meta["i_preempt"] = meta["preemptions_delta"] == 0.0
        e.stop()
        self.cur_eng = None
        rc, used = self.sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", 30)
        meta["mem_used_after_stop"] = used.strip()
        meta["t_end"] = time.time()
        json.dump(meta, open(f"{md}/meta.json", "w"), indent=1)
        for d in (".", "rows"):                                        # rows/ copy: the Modal runner snapshots on rows/ARM_DONE_*
            open(os.path.join(d, f"ARM_DONE_{key}"), "w").write(time.strftime("%FT%TZ", time.gmtime()) + "\n")
        self.log(f"{key} done alive={meta['i_alive']} kv={meta['i_kv']} preempt_delta={meta['preemptions_delta']} "
                 f"nooverlap={meta['i_nooverlap']}")
        return True

    def go(self):
        self.setup()
        self.cur_eng = None
        try:
            self._go()
        except SystemExit:
            raise
        except Exception as ex:  # noqa: BLE001 — QA r1: a crash still writes DONE (never a silent no-status run)
            self._stop_cur()
            self.finish(f"CRASH {type(ex).__name__}: {str(ex)[:200]}")

    def _stop_cur(self):
        e, self.cur_eng = self.cur_eng, None
        if e is not None and e.alive():
            e.stop()

    def _go(self):
        self.log(f"START arm={self.plan['arm']} stub={self.stub} deadline_in={self.deadline - time.time():.0f}s")
        self.env_dump()
        card = self.card()
        if not card.get("ok"):
            self.finish(f"STOP-CARD {card.get('why')}")
        gbs = self.d2d()
        if not self.stub and gbs < self.plan["card"]["d2d_min_gbs"]:
            self.finish(f"STOP-D2D {gbs:.1f} GB/s < {self.plan['card']['d2d_min_gbs']}")
        self.start_downloads()
        done, skipped = [], []
        for i, (key, m) in enumerate(self.plan["models"]):
            left = self.deadline - time.time()
            if i > 0 and left < self.plan["min_s_before_boot"]:
                skipped.append({"key": key, "why": f"deadline: {left:.0f}s left < {self.plan['min_s_before_boot']}"})
                self.log(f"SKIP {key} {skipped[-1]['why']}")
                continue
            try:
                if self.model(key, m, 8200 + i, gbs):
                    done.append(key)
            except Exception as ex:  # noqa: BLE001 — one model's crash never loses the other's rows (QA r1)
                self._stop_cur()
                why = f"CRASH {type(ex).__name__}: {str(ex)[:200]}"
                mp = f"m_{key}/meta.json"
                meta = json.load(open(mp)) if os.path.exists(mp) else {"key": key}
                meta["infra"] = why
                os.makedirs(f"m_{key}", exist_ok=True)
                json.dump(meta, open(mp, "w"), indent=1, default=str)
                self.log(f"{key} {why}")
        json.dump(skipped, open("logs/skips.json", "w"), indent=1)
        self.finish("OK" + (f" done={done}" if done else " done=[]") + (f" skipped={[s['key'] for s in skipped]}" if skipped else ""))


# ----------------------------------------------------------------------------------------------- scoring (CPU host)
def load_frozen():
    sys.path.insert(0, FROZEN_DIR)
    import gpu_driver as D
    import gpu_score as S
    return D, S


def recompute_pred():
    import importlib.util
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location("s2959", os.path.join(HERE, "steptime_crossover.py"))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.MODELS["q3_14b"] = dict(W=W14, Wsw=W14, v=40 * 2 * 8 * 128 * 2, rho=RHO14, P=14.77, name="Qwen3-14B")
    out = {}
    for key, mt in (("q8", "ref8b"), ("q14", "q3_14b")):
        out[key] = {n: {X: M.step(mt, X if X != "A" else "A", B, B * (L + 32)) * 1e3 for X in ("A", "HM", "HF")}
                    for n, B, L in CELLS}
    return out


def score_model(S, run_dir, key, allow_stub):
    md = os.path.join(run_dir, f"m_{key}")
    if not os.path.exists(os.path.join(md, "meta.json")):
        return {"key": key, "status": "ABSENT"}
    meta = json.load(open(os.path.join(md, "meta.json")))
    if meta.get("infra"):
        return {"key": key, "status": "INFRA", "why": meta["infra"]}
    anch = S.anchor_tpot(md, {"anchor": {"cells": [{"name": n, "B": B, "L": L} for n, B, L in CELLS], "passes": PASSES}})
    checks = {"I-KV": bool(meta.get("i_kv")), "I-PREEMPT": bool(meta.get("i_preempt")), "I-ALIVE": bool(meta.get("i_alive")),
              "I-NOOVERLAP": bool(meta.get("i_nooverlap")),
              "I-DL": bool((meta.get("download") or {}).get("ok")) and (allow_stub or not (meta.get("download") or {}).get("stub")),
              "I-COMPLETE": all(anch[n]["tpot_s"] is not None and not anch[n]["infra_passes"] for n, _, _ in CELLS)}
    res = {"key": key, "checks": checks, "anchor": anch, "backend": meta.get("backend"), "kv_tokens": meta.get("kv_tokens"),
           "preemptions_delta": meta.get("preemptions_delta"), "download": meta.get("download")}
    smi = os.path.join(md, "smi.csv")
    res["clock"] = smi_summary(open(smi).read()) if os.path.exists(smi) else None
    g6 = os.path.join(md, "g6_serving.json")
    res["g6"] = json.load(open(g6)) if os.path.exists(g6) else None
    if not all(checks.values()):
        res["status"] = "INFRA"
        res["why"] = [k for k, v in checks.items() if not v]
        return res
    res["status"] = "OK"
    res["t_ms"] = {n: anch[n]["tpot_s"] * 1e3 for n, _, _ in CELLS}
    res["range_ms"] = {n: [anch[n]["spread"][0] * 1e3, anch[n]["spread"][1] * 1e3] for n, _, _ in CELLS}
    return res


def verdicts(r8, r14, ctrl=None):
    """prereg §5 over scored models (status OK); ctrl = the arm-C scored 8B or None. Returns the verdict doc."""
    v = {"V-PLATFORM": "UNMEASURED (arm C did not run)" if ctrl is None else None}
    readings = {"raw": (r8, r14)}
    if ctrl is not None:
        if ctrl.get("status") != "OK":
            v["V-PLATFORM"] = f"UNMEASURED (arm C {ctrl.get('status')})"
        else:
            band, d = platform(ctrl["t_ms"])
            v["V-PLATFORM"], v["platform_delta"] = band, d
            if band == "PLATFORM-OFFSET":
                adj = lambda r: None if r is None else dict(r, t_ms={c: t / (1 + d[c]) for c, t in r["t_ms"].items()},
                                                            range_ms={c: [x / (1 + d[c]) for x in rr]
                                                                      for c, rr in r["range_ms"].items()})
                readings["offset-corrected"] = (adj(r8), adj(r14))
    out = {}
    for name, (a8, a14) in readings.items():
        doc = {}
        hm8 = {c: PRED["q8"][c]["HM"] for c in a8["t_ms"]}
        hf8 = {c: PRED["q8"][c]["HF"] for c in a8["t_ms"]}
        doc["V-MODEL"] = model_label(a8["range_ms"], hm8, hf8)
        a, kb, kl = fit3(a8["t_ms"]["Q"], a8["t_ms"]["S"], a8["t_ms"]["LC"])
        b, bband = beta_band(a)
        doc["fit8"] = {"a_ms": a, "kb_ms": kb, "kl_s": kl / 1e3, "beta_B": b, "beta_band": bband,
                       "kb_ratio": kb / KB_A_MS, "kb_band": term_band(kb / KB_A_MS),
                       "kl_ratio": (kl / 1e3) / KL_A_S, "kl_band": term_band((kl / 1e3) / KL_A_S)}
        doc["cells8"] = {c: {"t_ms": a8["t_ms"][c], "range_ms": a8["range_ms"][c], "t_over_HM": a8["t_ms"][c] / hm8[c],
                             "t_over_HF": a8["t_ms"][c] / hf8[c],
                             "a_position": a_position(a8["t_ms"][c], PRED["q8"][c]["A"], hm8[c])} for c in a8["t_ms"]}
        if a14 is not None:
            hm14 = {c: PRED["q14"][c]["HM"] for c in a14["t_ms"]}
            hf14 = {c: PRED["q14"][c]["HF"] for c in a14["t_ms"]}
            a2, kb2, kl2 = fit3(a14["t_ms"]["Q"], a14["t_ms"]["S"], a14["t_ms"]["LC"])
            doc["V-MODEL-14B"] = model_label(a14["range_ms"], hm14, hf14)
            doc["fit14"] = {"a_ms": a2, "kb_ms": kb2, "kl_s": kl2 / 1e3, "size_term_ms": a2 - a,
                            "size_term_hm_ms": SIZE_TERM_HM_MS, "kl_ratio_14_8": kl2 / kl if kl else None, "rho": RHO14}
            doc["cells14"] = {c: {"t_ms": a14["t_ms"][c], "range_ms": a14["range_ms"][c], "t_over_HM": a14["t_ms"][c] / hm14[c],
                                  "t_over_HF": a14["t_ms"][c] / hf14[c],
                                  "a_position": a_position(a14["t_ms"][c], PRED["q14"][c]["A"], hm14[c])}
                              for c in a14["t_ms"]}
            doc["V-CROSSOVER"], doc["crossover_detail"] = crossover(a8["range_ms"], hm8, a14["range_ms"], hm14)
        else:
            doc["V-CROSSOVER"], doc["crossover_detail"] = crossover(a8["range_ms"], hm8)
        out[name] = doc
    v["readings"] = out
    if "offset-corrected" in out:
        b_raw = out["raw"]["V-CROSSOVER"].split(" (")[0]
        b_adj = out["offset-corrected"]["V-CROSSOVER"].split(" (")[0]
        if b_raw != b_adj:
            v["V-PLATFORM"] = "PLATFORM-CONFOUNDED"
    return v


def score(args):
    D, S = load_frozen()
    for p, want in ((os.path.join(FROZEN_DIR, "gpu_driver.py"), SHA_DRIVER), (os.path.join(FROZEN_DIR, "gpu_score.py"), SHA_SCORE)):
        if sha256_file(p) != want:
            print(f"REFUSED: {p} sha mismatch", file=sys.stderr)
            return 2
    if prompt_manifest_sha(D) != MANIFEST_SHA:
        print("REFUSED: prompt manifest sha mismatch (prereg §2.1b)", file=sys.stderr)
        return 2
    rp = recompute_pred()
    bad = [(k, c, X, rp[k][c][X], PRED[k][c][X]) for k in PRED for c in PRED[k] for X in PRED[k][c]
           if abs(rp[k][c][X] - PRED[k][c][X]) > PRED_TOL_MS]
    if bad:
        print(f"REFUSED: frozen predictions differ from 's module: {bad[:3]}", file=sys.stderr)
        return 2
    done_p = os.path.join(args.run_dir, "DONE")
    if not os.path.exists(done_p):
        print(f"ERROR: {done_p} missing", file=sys.stderr)
        return 1
    done = open(done_p).read().strip()
    doc = {"prereg": "data/runs/hclass_step_prereg_2026-10-22.md", "prereg_commit": "138bb472b", "run_dir": args.run_dir,
           "done": done, "pred_ms": PRED}
    for f in ("logs/card_gate.json", "logs/probe.json"):
        p = os.path.join(args.run_dir, f)
        doc[os.path.basename(f).split(".")[0]] = json.load(open(p)) if os.path.exists(p) else None
    if not done.startswith("OK"):
        doc["verdict"] = f"NO-VERDICT {done}"
        print(f"VERDICT {doc['verdict']}")
        json.dump(doc, open(args.out, "w"), indent=1)
        return 3
    if "STUB" in json.dumps(doc.get("card_gate") or {}).upper() and not args.allow_stub:
        print("REFUSED: a stub run (pass --allow-stub)", file=sys.stderr)
        return 2
    r8 = score_model(S, args.run_dir, "q8", args.allow_stub)
    r14 = score_model(S, args.run_dir, "q14", args.allow_stub)
    ctrl = score_model(S, args.control_dir, "q8", args.allow_stub) if args.control_dir else None
    doc.update(q8=r8, q14=r14, control=ctrl)
    if r8.get("status") != "OK":
        doc["verdict"] = f"NO-VERDICT 8B {r8.get('status')} {r8.get('why')}"
        print(f"VERDICT {doc['verdict']}")
        json.dump(doc, open(args.out, "w"), indent=1, default=str)
        return 3
    v = verdicts(r8, r14 if r14.get("status") == "OK" else None, ctrl)
    doc["verdicts"] = v
    raw = v["readings"]["raw"]
    print(f"VERDICT V-MODEL (8B, ±10 % per cell): {raw['V-MODEL']} | t/HM "
          + " ".join(f"{c} {raw['cells8'][c]['t_over_HM']:.3f}" for c in raw["cells8"])
          + " | t/HF " + " ".join(f"{c} {raw['cells8'][c]['t_over_HF']:.3f}" for c in raw["cells8"]))
    f8 = raw["fit8"]
    bstr = "n/a" if f8["beta_B"] is None else f"{f8['beta_B']:.3f}"            # QA r2: a_H <= F must not crash the scorer
    print(f"VERDICT 1.64x WEIGHT TERM: beta_B {bstr} -> {f8['beta_band']} | k_b {f8['kb_ratio']:.3f} {f8['kb_band']}"
          f" | k_L {f8['kl_ratio']:.3f} {f8['kl_band']}")
    if "V-MODEL-14B" in raw:
        print(f"VERDICT V-MODEL (14B): {raw['V-MODEL-14B']} | t/HM " + " ".join(
            f"{c} {raw['cells14'][c]['t_over_HM']:.3f}" for c in raw["cells14"]) +
            f" | size term {raw['fit14']['size_term_ms']:.3f} ms vs HM {SIZE_TERM_HM_MS}")
    else:
        print(f"VERDICT V-MODEL (14B): {r14.get('status')} {r14.get('why', '')}")
    print(f"VERDICT V-CROSSOVER: {raw['V-CROSSOVER']}")
    print(f"VERDICT V-PLATFORM: {v['V-PLATFORM']}")
    if "offset-corrected" in v["readings"]:
        print(f"VERDICT V-CROSSOVER (offset-corrected): {v['readings']['offset-corrected']['V-CROSSOVER']}")
    json.dump(doc, open(args.out, "w"), indent=1, default=str)
    return 0


# ----------------------------------------------------------------------------------------------- CPU host stub smoke
STUB_SMI = """#!/bin/sh
case "$*" in
  *"--query-gpu=name"*) echo "$H2960_STUB_CARD" ;;
  *"--query-gpu=memory.used"*) echo "3" ;;
  *"--query-gpu=timestamp"*) echo "2026/10/22 00:00:00.000, 1980 MHz, 2619 MHz, 412.50 W, 41, 97 %, 0x0000000000000000" ;;
  *"-q"*) echo "    Product Name                          : STUB $H2960_STUB_CARD" ;;
  *) echo "STUB nvidia-smi" ;;
esac
"""


def stub_plan(arm="H", max_dl_wait_s=300):
    return {"arm": arm, "util": 0.90, "min_s_before_boot": 300, "max_dl_wait_s": max_dl_wait_s,
            "card": {"name_contains": "H100 80GB HBM3", "mem_min_mib": 79000, "mem_max_mib": 82000, "pmax_min_w": 600,
                     "d2d_min_gbs": 2345},
            "ctrl": {"conc": 8, "in": 512, "out": 128, "secs": 3},
            "models": [["q8", {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                               "bytes": 16381516776, "max_model_len": 16640, "kv_bpt": 147456, "g6": True}],
                       ["q14", {"repo": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
                                "bytes": 29536665640, "max_model_len": 16640, "kv_bpt": 163840}]]}


def stub_smoke(scratch):
    os.makedirs(scratch, exist_ok=True)
    if os.listdir(scratch):
        print(f"REFUSE: {scratch} is not empty")
        return 1
    stub = os.path.join(FROZEN_DIR, "stub")
    ok_all = True
    sxm = "NVIDIA H100 80GB HBM3, 81559, 700.00, 700.00, 1980, 2619, 00000000:00:00.0"
    for tag, card, want, xenv in (("sxm", sxm, "OK", {}), ("sxm_fb", sxm, "OK", {"H2960_FORCE_ULIMIT_FALLBACK": "1"}),
                                  ("slowdl", sxm, "OK", {"H2960_STUB_SLOW_DL": "1"}),
                                  ("pcie", "NVIDIA H100 PCIe, 81559, 350.00, 350.00, 1755, 1593, 00000000:00:00.0", "STOP-CARD",
                                   {})):
        R = os.path.join(scratch, tag)
        b = os.path.join(R, "bin")
        os.makedirs(b)
        for f in ("gpu_driver.py", "gpu_score.py"):
            shutil.copy(os.path.join(FROZEN_DIR, f), R)
        shutil.copy(os.path.abspath(__file__), R)
        open(os.path.join(b, "nvidia-smi"), "w").write(STUB_SMI)
        open(os.path.join(b, "vllm"), "w").write(f'#!/bin/sh\nshift\nexec python3 "{stub}/fake_vllm.py" serve "$@"\n')
        for f in ("nvidia-smi", "vllm"):
            os.chmod(os.path.join(b, f), 0o755)
        json.dump(stub_plan(max_dl_wait_s=5 if tag == "slowdl" else 300), open(os.path.join(R, "plan_H.json"), "w"), indent=1)
        env = dict(os.environ, PATH=b + ":" + os.environ["PATH"], H2960_STUB="1", H2960_STUB_CARD=card,
                   H2960_DEADLINE=str(time.time() + 1500), **xenv)
        r = subprocess.run([sys.executable, "h100_step.py", "--run", "--plan", "plan_H.json"], cwd=R, env=env,
                           capture_output=True, text=True, timeout=1200)
        done = open(os.path.join(R, "DONE")).read().strip() if os.path.exists(os.path.join(R, "DONE")) else ""
        good = r.returncode == 0 and done.startswith(want)
        if tag == "slowdl":
            m8 = json.load(open(os.path.join(R, "m_q8", "meta.json")))
            m14 = json.load(open(os.path.join(R, "m_q14", "meta.json")))
            good &= done == "OK done=['q8']" and m8.get("i_nooverlap") is True and "I-DL" in (m14.get("infra") or "") \
                and "abandoned" in json.dumps(m14.get("download") or {})
        if tag == "sxm_fb":
            ul = json.load(open(os.path.join(R, "logs", "ulimit.json")))
            good &= done == "OK done=['q8', 'q14']" and ul.get("fallback") is True and ul.get("forced") is True
        if tag == "sxm":
            good &= done == "OK done=['q8', 'q14']" and json.load(open(os.path.join(R, "logs", "ulimit.json"))).get("fallback") is False
            for f in ("m_q8/rows/anchor.jsonl", "m_q14/rows/anchor.jsonl", "m_q8/g6_serving.json", "m_q8/meta.json",
                      "m_q14/meta.json", "logs/card_gate.json", "logs/probe.json", "ARM_DONE_q8", "ARM_DONE_q14", "m_q8/smi.csv"):
                if not os.path.exists(os.path.join(R, f)):
                    print(f"  MISSING {tag}/{f}")
                    good = False
            if good:
                sr = subprocess.run([sys.executable, os.path.abspath(__file__), "--score", "--run-dir", R, "--out",
                                     os.path.join(R, "score.json"), "--allow-stub"], capture_output=True, text=True, timeout=300)
                print("  " + "\n  ".join(sr.stdout.strip().splitlines()))
                good &= sr.returncode == 0 and "VERDICT V-CROSSOVER" in sr.stdout
                if sr.returncode != 0:
                    print(sr.stderr[-800:])
        elif tag == "pcie":
            good &= not os.path.exists(os.path.join(R, "m_q8")) and not os.path.exists(os.path.join(R, "logs/server_q8_q8.log"))
        print(f"  stub {tag}: rc={r.returncode} DONE={done!r} -> {'PASS' if good else 'FAIL'}")
        if not good:
            print(r.stdout[-600:], r.stderr[-1200:])
            lp = os.path.join(R, "logs", "drive.log")
            if os.path.exists(lp):
                print(open(lp).read()[-1200:])
        ok_all &= good
    print("STUB-SMOKE " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


# ----------------------------------------------------------------------------------------------- self-test
def self_test():
    fails, n = [], [0]

    def ck(name, cond, detail=""):
        n[0] += 1
        print(f"  {'PASS' if cond else 'FAIL'} {name}{'' if cond else ' :: ' + str(detail)}")
        if not cond:
            fails.append(name)

    H = stub_plan()["card"]
    A = {"name_contains": "A100-SXM4-80GB", "mem_min_mib": 79000, "mem_max_mib": 82000, "pmax_min_w": 400, "d2d_min_gbs": 1427}
    ck("G1a SXM H100 passes I-CARD", card_gate("NVIDIA H100 80GB HBM3", 81559, 700.0, H)[0])
    ck("G1b H100 PCIe (350 W) stops", not card_gate("NVIDIA H100 PCIe", 81559, 350.0, H)[0])
    ck("G1c H200 stops (name + memory)", not card_gate("NVIDIA H200", 143771, 700.0, H)[0])
    ck("G1d A100 passes arm C, fails arm H", card_gate("NVIDIA A100-SXM4-80GB", 81920, 400.0, A)[0]
       and not card_gate("NVIDIA A100-SXM4-80GB", 81920, 400.0, H)[0])
    ck("G1e H100 NVL (400 W, 95 GB) stops", not card_gate("NVIDIA H100 NVL", 95830, 400.0, H)[0])
    ck("G1f SXM name at 350 W max stops (power clause alone)", not card_gate("NVIDIA H100 80GB HBM3", 81559, 350.0, H)[0])
    ck("G1g SXM name at 143,771 MiB stops (memory clause alone)", not card_gate("NVIDIA H100 80GB HBM3", 143771, 700.0, H)[0])
    nm, mem, pw, rest = parse_card("NVIDIA H100 80GB HBM3, 81559, 700.00, 700.00, 1980, 2619, 00000000:18:00.0\n")
    ck("G2 parse_card", (nm, mem, pw, rest[0]) == ("NVIDIA H100 80GB HBM3", 81559.0, 700.0, "700.00"), (nm, mem, pw, rest))
    try:
        parse_card("")
        ck("G2b empty card line raises", False)
    except ValueError:
        ck("G2b empty card line raises", True)
    ck("G3 kv_tokens on the real log line",
       kv_tokens("INFO [kv_cache_utils.py:2395] GPU KV cache size: 401,616 tokens, Maximum concurrency") == 401616)
    ck("G3b kv_tokens absent -> None", kv_tokens("nothing here") is None)
    ck("G3c backend on the real line", backend("INFO [cuda.py:538] Using FLASH_ATTN attention backend out of potential")
       == "FLASH_ATTN")
    # fit3 round-trip (hand): a 9, kb 0.3, kl 5e-4 ms/token -> Q = 9+0.3+2.048 = 11.348, S = 9+2.4+16.384 = 27.784,
    # LC = 9+0.6+16.384 = 25.984
    a, kb, kl = fit3(11.348, 27.784, 25.984)
    ck("G4 fit3 round-trip", abs(a - 9) < 1e-9 and abs(kb - 0.3) < 1e-9 and abs(kl - 5e-4) < 1e-12, (a, kb, kl))
    a80 = fit3(11.34188213045635, 26.980297906058173, 24.946615809486025)
    ck("G4b fit3 on h2 = 's FIT_A80", abs(a80[0] - A_FIT_MS) < 1e-9 and abs(a80[1] - KB_A_MS) < 1e-9
       and abs(a80[2] / 1e3 - KL_A_S) < 1e-15, a80)
    hm8 = {c: PRED["q8"][c]["HM"] for c in ("Q", "S", "LC")}
    hf8 = {c: PRED["q8"][c]["HF"] for c in ("Q", "S", "LC")}
    at = lambda preds, x, w=0.002: {c: [preds[c] * x * (1 - w), preds[c] * x * (1 + w)] for c in preds}
    ck("G5a cells at HM -> HM-AGREES, not HF", band_model(at(hm8, 1.0), hm8) and not band_model(at(hm8, 1.0), hf8))
    ck("G5b cells at HF -> HF-AGREES, not HM", band_model(at(hf8, 1.0), hf8) and not band_model(at(hf8, 1.0), hm8))
    ck("G5c edge: 1.10 x HM max is inside, 1.101 is not", band_model({"Q": [8.6, 1.10 * hm8["Q"]]}, {"Q": hm8["Q"]})
       and not band_model({"Q": [8.6, 1.101 * hm8["Q"]]}, {"Q": hm8["Q"]}))
    ck("G6a crossover: 0.95 x HM everywhere -> BRACKETED (8B only suffix)",
       crossover(at(hm8, 0.95), hm8)[0] == "BRACKETED-OR-FASTER (8B only; 13-14 B step still MODELLED)")
    one_slow = dict(at(hm8, 0.95), S=[hm8["S"] * 1.05, hm8["S"] * 1.06])
    ck("G6b one cell min at 1.05 x HM -> SLOWER-THAN-HM", crossover(one_slow, hm8)[0].startswith("SLOWER-THAN-HM"))
    straddle = dict(at(hm8, 0.95), LC=[hm8["LC"] * 1.01, hm8["LC"] * 1.03])
    ck("G6c straddling 1.02 x HM -> AT-EDGE", crossover(straddle, hm8)[0].startswith("AT-EDGE"))
    hm14 = {c: PRED["q14"][c]["HM"] for c in ("Q", "S", "LC")}
    ck("G6d 8B fast + 14B slow -> SLOWER-THAN-HM (no suffix)",
       crossover(at(hm8, 0.9), hm8, at(hm14, 1.2), hm14)[0] == "SLOWER-THAN-HM")
    ck("G6e exactly 1.02 x HM max is BRACKETED", crossover({"Q": [8.0, 1.02 * hm8["Q"]]}, {"Q": hm8["Q"]})[0]
       .startswith("BRACKETED"))
    b, bb = beta_band(6.332914698206481)
    ck("G7a a_H = I_HM -> beta 1.643, HOLDS", abs(b - HM_BETA) < 1e-6 and bb == "HOLDS", (b, bb))
    b, bb = beta_band(A_FIT_MS)
    ck("G7b a_H = A100 intercept -> beta 1.0, SLOWER-THAN-EXTRAPOLATED", abs(b - 1.0) < 1e-12 and bb == "SLOWER-THAN-EXTRAPOLATED")
    ck("G7c a_H <= F -> UNDEFINED", beta_band(2.0)[0] is None)
    ck("G7d a_H 4.0 ms -> beta 3.58, FASTER", beta_band(4.0)[1] == "FASTER-THAN-EXTRAPOLATED")
    ck("G8 term bands", (term_band(1.0), term_band(0.6087), term_band(0.8), term_band(0.517), term_band(1.16))
       == ("HM-like", "HF-like", "NEITHER", "HF-like", "NEITHER"))
    ck("G9 a_position: at HM = 1, at A = 0", abs(a_position(8.5818, 11.3567, 8.5818) - 1.0) < 1e-12
       and a_position(11.3567, 11.3567, 8.5818) == 0.0)
    pb, d = platform({"Q": VAST_H2["Q"] * 1.02, "S": VAST_H2["S"] * 0.99, "LC": VAST_H2["LC"]})
    ck("G10a platform 2 % -> CLEAN", pb == "PLATFORM-CLEAN" and abs(d["Q"] - 0.02) < 1e-12)
    ck("G10b platform 5 % -> OFFSET", platform({"Q": VAST_H2["Q"] * 1.05, "S": VAST_H2["S"], "LC": VAST_H2["LC"]})[0]
       == "PLATFORM-OFFSET")
    s = smi_summary("2026/10/22 08:00:00.000, 1980 MHz, 2619 MHz, 412.50 W, 41, 97 %, 0x0000000000000000\n"
                    "2026/10/22 08:00:00.500, 1755 MHz, 2619 MHz, 650.00 W, 55, 99 %, 0x0000000000000004\n"
                    "garbage\n")
    ck("G11 smi_summary", s["n"] == 2 and s["sm_mhz_min"] == 1755.0 and s["power_w_max"] == 650.0
       and s["event_reasons_seen"] == ["0x0000000000000004"] and s["unparsed"] == 1, s)
    seen = []

    class _Rec:
        TimeoutExpired = subprocess.TimeoutExpired

        def Popen(self, args, *a, **k):
            seen.append(args)
    sh_ = PopenShim(_Rec())
    sh_.Popen(["bash", "-c", FROZEN_ULIMIT, "x", "vllm", "serve", "Qwen/Qwen3-8B", "--port", "8200"])
    sh_.Popen(["nvidia-smi", "-lms", "500"])
    ck("G19 A1 shim rewrites only the frozen wrapper; argv tail unchanged; other attrs pass through",
       seen[0] == ["bash", "-c", FALLBACK_ULIMIT, "x", "vllm", "serve", "Qwen/Qwen3-8B", "--port", "8200"]
       and seen[1] == ["nvidia-smi", "-lms", "500"] and sh_.TimeoutExpired is subprocess.TimeoutExpired, seen)
    D, S = load_frozen()
    ck("G12 prompt manifest sha = prereg §2.1b", prompt_manifest_sha(D) == MANIFEST_SHA)
    src = open(os.path.join(FROZEN_DIR, "gpu_driver.py")).read()
    ck("G20 FROZEN_ULIMIT is verbatim in gpu_driver.Eng.start", '"ulimit -n 65536 && exec \\"$@\\""' in src)
    rp = recompute_pred()
    worst = max(abs(rp[k][c][X] - PRED[k][c][X]) for k in PRED for c in PRED[k] for X in PRED[k][c])
    ck("G13 frozen §3 table = 's step() (4 dp)", worst <= PRED_TOL_MS, worst)
    # Q-cell gap 14B - 8B under HM = kappa (W14 - W8) / B_H + k_L * 4128 * (rho14 - 1) = 3.4658 + 0.7427 = 4.2085 ms (hand)
    ck("G13b size term HM = kappa (W14 - W8) / B_H", abs((rp["q14"]["Q"]["HM"] - rp["q8"]["Q"]["HM"])
                                                          - (SIZE_TERM_HM_MS + KL_A_S * 1e3 * 4128 * (RHO14 - 1))) < 1e-3
       and abs(rp["q14"]["Q"]["HM"] - rp["q8"]["Q"]["HM"] - 4.2085) < 1e-3, rp["q14"]["Q"]["HM"] - rp["q8"]["Q"]["HM"])
    # G14: end-to-end over a synthetic run dir scored by the REAL anchor_tpot (rows at exactly HM x 0.97 per pass)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for key, fac in (("q8", 0.97), ("q14", 0.97)):
            md = os.path.join(td, f"m_{key}", "rows")
            os.makedirs(md)
            with open(os.path.join(md, "anchor.jsonl"), "w") as f:
                for p in range(PASSES):
                    for cn, B, L in CELLS:
                        for i in range(B):
                            t = PRED[key][cn]["HM"] * fac * (1 + 0.001 * p) / 1e3
                            f.write(json.dumps({"cell": cn, "pass": p, "i": i, "B": B, "L": L, "tpot": t, "complete": True,
                                                "err": None}) + "\n")
            json.dump({"i_kv": True, "i_preempt": True, "i_alive": True, "i_nooverlap": True, "download": {"ok": True}},
                      open(os.path.join(td, f"m_{key}", "meta.json"), "w"))
        r8, r14 = score_model(S, td, "q8", False), score_model(S, td, "q14", False)
        v = verdicts(r8, r14)
        ck("G14 synthetic run at 0.97 x HM -> BRACKETED + HM-AGREES", v["readings"]["raw"]["V-CROSSOVER"] == "BRACKETED-OR-FASTER"
           and v["readings"]["raw"]["V-MODEL"] == "HM-AGREES" and v["V-PLATFORM"].startswith("UNMEASURED"),
           (v["readings"]["raw"]["V-CROSSOVER"], v["readings"]["raw"]["V-MODEL"]))
        # G15: a pass with fewer than B complete rows -> INFRA, never slow
        lines = open(os.path.join(td, "m_q8", "rows", "anchor.jsonl")).read().splitlines()
        lines = [ln for ln in lines if not ('"cell": "S", "pass": 1, "i": 7' in ln)]
        open(os.path.join(td, "m_q8", "rows", "anchor.jsonl"), "w").write("\n".join(lines) + "\n")
        r8b = score_model(S, td, "q8", False)
        ck("G15 a missing S row in pass 1 -> INFRA (I-COMPLETE)", r8b["status"] == "INFRA" and "I-COMPLETE" in r8b["why"], r8b.get("why"))
        json.dump({"i_kv": True, "i_preempt": False, "i_alive": True, "i_nooverlap": True, "download": {"ok": True}},
                  open(os.path.join(td, "m_q14", "meta.json"), "w"))
        ck("G16 a preemption -> INFRA (I-PREEMPT)", score_model(S, td, "q14", False).get("why") == ["I-PREEMPT"])
        json.dump({"i_kv": True, "i_preempt": True, "i_alive": True, "i_nooverlap": True, "download": {"ok": True, "stub": True}},
                  open(os.path.join(td, "m_q14", "meta.json"), "w"))
        ck("G17 a stub download is not I-DL without --allow-stub", score_model(S, td, "q14", False).get("why") == ["I-DL"]
           and score_model(S, td, "q14", True)["status"] == "OK")
    # G18: platform offset that flips the band -> PLATFORM-CONFOUNDED
    r8 = {"status": "OK", "t_ms": {c: hm8[c] * 1.01 for c in hm8}, "range_ms": at(hm8, 1.01, 0.001)}
    ctrl = {"status": "OK", "t_ms": {c: VAST_H2[c] * 0.95 for c in VAST_H2}}
    v = verdicts(r8, None, ctrl)
    ck("G18 raw AT/BRACKETED vs offset-corrected SLOWER -> PLATFORM-CONFOUNDED", v["V-PLATFORM"] == "PLATFORM-CONFOUNDED",
       (v["V-PLATFORM"], v["readings"]["raw"]["V-CROSSOVER"], v["readings"].get("offset-corrected", {}).get("V-CROSSOVER")))
    print(f"SELF-TEST {'PASS' if not fails else 'FAIL'} {n[0] - len(fails)}/{n[0]}")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--stub-smoke", metavar="SCRATCH")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--plan")
    ap.add_argument("--run-dir")
    ap.add_argument("--control-dir")
    ap.add_argument("--out")
    ap.add_argument("--allow-stub", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.stub_smoke:
        return stub_smoke(a.stub_smoke)
    if a.run:
        if not a.plan:
            ap.error("--run needs --plan")
        Run(json.load(open(a.plan)), os.environ.get("H2960_STUB") == "1").go()
        return 0
    if not (a.run_dir and a.out):
        ap.error("--score needs --run-dir and --out")
    return score(a)


if __name__ == "__main__":
    sys.exit(main())
