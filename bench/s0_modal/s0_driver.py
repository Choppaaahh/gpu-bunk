"""s0_driver.py -- S0 direct phase: S0-0 .. S0-11 against stock vLLM 0.30.0 engines + bite 1's holders, no controller.

Prereg: (private report) (frozen). Each item writes rows/<item>.json with the
verdict the prereg's criterion gives; an item whose prerequisite failed writes NOT-RUN with the reason. Continue-on-fail.
Output contract: exit 0 after the last item whatever the verdicts (they live in rows/); exit 1 only on a harness crash
(traceback in logs/modal_cmd.out). S0_BACKEND=stub runs the same sequence on CPU stub engines (the $0 rehearsal).
"""
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C  # noqa: E402

V = C.BACKEND == "vllm"
REV14 = "971185c6cf17f6c89168ad95b3c316ef27d0b459"
REV8 = "b968826d9c46dd6066d109eabc6255188de91218"
REPO14 = "AnastasiaFantasy/affine-ana11-3-5CJXygeziPM2F8C1bhupwAKpKmx28cw1zD15Eoa5QbFPSXXE"
REPO8 = "Qwen/Qwen3-8B"
OWNER_FLAGS = ["--quantization", "fp8", "--linear-backend", "marlin", "--no-enable-prefix-caching",
               "--gpu-memory-utilization", "0.45"]
# Amendment A1 (after R1): vLLM 0.30.0 picks CutlassFP8ScaledMMLinearKernel on sm80 (is_supported has no capability
# check, cutlass.py L164-169) and torch.compile then asserts 'auto_functionalized was not removed' (R1, all 3 owners);
# --linear-backend marlin (arg_utils L1678-1680) forces MarlinFP8ScaledMMLinearKernel, the A100 W8A16 path.
ENG = {   # kind -> (repo, rev, extra flags, stub weights, stub kv)
    "e14": (REPO14, REV14, OWNER_FLAGS + ["--kv-cache-memory-bytes", "17179869184", "--max-model-len", "40960",
                                          "--tokenizer-revision", REV14], [1 * C.MIB, 3 * C.MIB, 2 * C.MIB], [2 * C.MIB]),
    "e8": (REPO8, REV8, OWNER_FLAGS + ["--kv-cache-memory-bytes", "24159191040", "--max-model-len", "32768",
                                       "--tokenizer-revision", REV8], [2 * C.MIB, 2 * C.MIB], [2 * C.MIB]),
    "b": (REPO8, REV8, ["--no-enable-prefix-caching", "--gpu-memory-utilization", "0.45", "--kv-cache-memory-bytes",
                        "17179869184", "--max-model-len", "16640", "--tokenizer-revision", REV8], [16 * C.MIB], [8 * C.MIB]),
}
PORT = dict(h0=8300, h1=8301, e14=8400, e8=8401, e8p=8402, b0=8410, b1=8411)
S02_BYTES = 16_970_000_000 if V else 16 * C.MIB
PROBE_BYTES = 4 * C.GIB if V else 4 * C.MIB
FILLER = 35 * C.GIB if V else 36 * C.MIB
HOLDER_CAP = ["--cap-gib", "35.62"] if V else ["--cap-mib", "40"]
BOOT_S = 1200 if V else 60
FAB = os.path.join(C.HERE, "stubfab")
KV_RE = r"GPU KV cache size: ([\d,]+) tokens"
INJ_RE = (r"Injected <class 'warm_wake_ext\.WarmWakeExt'> into <class 'vllm\.v1\.worker\.gpu_worker\.Worker'>"
          if V else r"Injected <class '[\w.]*WarmWakeExt'> into <class '[\w.]+'>")
LOAD_RE = r"Model loading took ([\d.]+) GiB memory"
PROCS = {}
ST = {}          # shared state between steps


def wait_dl(key, timeout=1500):
    if not V:
        return True
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if os.path.exists(os.path.join(C.LOGS, f"dl_{key}.done")):
            return True
        if os.path.exists(os.path.join(C.LOGS, f"dl_{key}.fail")):
            return False
        time.sleep(1)
    return False


def start_holder(name, cvd, port):
    if V:
        cmd = [sys.executable, C.EXT, "--serve-holder", "--port", str(port), "--allow-corrupt"] + HOLDER_CAP
        env = C.child_env(cvd)
    else:
        cmd = [sys.executable, C.EXT, "--serve-holder", "--stub-fabric", FAB, "--stub-cvd", cvd, "--port", str(port),
               "--allow-corrupt"] + HOLDER_CAP
        env = C.child_env(None)
    p = C.Proc(name, cmd, env)
    PROCS[name] = p
    m = C.wait_line(p, r"BB-HOLDER-READY port=(\d+)", 120)
    return p, int(m.group(1))


def start_engine(name, kind, cvd, port, ext=True):
    repo, rev, flags, sw, skv = ENG[kind]
    if V:
        cmd = ["vllm", "serve", repo, "--revision", rev, "--host", "127.0.0.1", "--port", str(port),
               "--served-model-name", name, "--enable-sleep-mode"] + list(flags)
        if ext:
            cmd += ["--worker-extension-cls", "warm_wake_ext.WarmWakeExt"]
        env = C.child_env(cvd, pythonpath=ext)
    else:
        cmd = [sys.executable, C.STUB, "--serve-engine", "--fabric", FAB, "--cvd", cvd, "--port", str(port),
               "--weights-bytes", ",".join(map(str, sw)), "--kv-bytes", ",".join(map(str, skv)),
               "--seed", str(sum(map(ord, name))), "--ext-file", C.EXT]
        env = C.child_env(None)
    p = C.Proc(name, cmd, env)
    PROCS[name] = p
    hs = C.wait_health(f"http://127.0.0.1:{port}", BOOT_S, p)
    return p, hs


def boot(name, kind, cvd, port, ext=True):
    """start_engine, and on a boot failure ONE retry of that engine kind with --enforce-eager (A1: recorded as a
    deviation; ENG[kind] keeps the flags that booted, so every later engine of that kind uses them)."""
    try:
        return start_engine(name, kind, cvd, port, ext)
    except Exception as e:  # noqa: BLE001
        p = PROCS.get(name)
        tail = ""
        if p is not None:
            tail = "\n".join(x for x in p.text().splitlines() if "Error" in x or "error" in x)[-1500:]
            p.stop()
            os.replace(p.logpath, p.logpath.replace(".log", ".try1.log"))
        ST.setdefault("boot_failures", []).append(dict(name=name, kind=kind, err=str(e)[:300], tail=tail))
        if "--enforce-eager" in ENG[kind][2]:
            raise
        r, v, fl, w, k = ENG[kind]
        ENG[kind] = (r, v, fl + ["--enforce-eager"], w, k)
        C.log(f"boot {name} failed -> retry kind {kind} with --enforce-eager")
        time.sleep(3)
        return start_engine(name, kind, cvd, port, ext)


def model_of(name):
    return name if V else "stub"


def step(item, fn, needs=()):
    miss = [n for n in needs if not ST.get(n)]
    if miss:
        return C.put_row(item, "NOT-RUN", reason=f"prerequisite failed: {miss}")
    if C.left() < 120:
        return C.put_row(item, "NOT-RUN", reason=f"deadline: {C.left():.0f}s left")
    try:
        for nm in ("e14", "e8"):                    # rehearsal 1: a step that raised mid-way left E14 asleep
            if ST.get(nm) and nm in PROCS and PROCS[nm].alive():
                ensure_awake(PORT[nm])
        return fn()
    except Exception as e:  # noqa: BLE001
        C.log(f"STEP {item} raised: {e!r}")
        return C.put_row(item, "FAIL", reason=f"harness/step exception: {type(e).__name__}: {e}"[:500],
                         tb=traceback.format_exc()[-1500:])


def push(port, gid, al, owner, nbytes):
    spec = {"guest": {"guest_id": gid, "ipc_handle": al["ipc_handle"], "nbytes": nbytes}, "owner_id": owner}
    r = C.rpc(port, "bb_push_to_peer", spec)
    tries = [dict(mode="direct", res={k: v for k, v in r.items() if k != "manifest"})]
    if not r.get("ok") and str(r.get("reason", "")).startswith("PUSH_FAILED"):
        spec["copy_mode"] = "staged"
        r = C.rpc(port, "bb_push_to_peer", spec)
        tries.append(dict(mode="staged", res={k: v for k, v in r.items() if k != "manifest"}))
    return r, tries


def restore(port, gid, handle, nbytes, mid, mode="direct"):
    spec = {"guest": {"guest_id": gid, "ipc_handle": handle, "nbytes": nbytes, "manifest_id": mid}, "copy_mode": mode}
    r = C.rpc(port, "bb_restore_from_peer", spec)
    tries = [dict(mode=mode, res=r)]
    if mode == "direct" and r.get("served") == "HOST_FALLBACK" and str(r.get("reason", "")).startswith("COPY:"):
        spec["copy_mode"] = "staged"
        r = C.rpc(port, "bb_restore_from_peer", spec)
        tries.append(dict(mode="staged", res=r))
    return r, tries


def ensure_awake(port):
    """precondition for any step that pushes or reads: a step that raised mid-way may have left the engine asleep."""
    if C.is_sleeping(port):
        C.log(f"ensure_awake: engine on {port} was asleep -> stock /wake_up")
        C.wake(port)
    return C.is_sleeping(port) is False


def both_wakes(port):
    a = C.wake(port, "weights")
    b = C.wake(port, "kv_cache")
    return dict(wake_weights=a, wake_kv=b, is_sleeping=C.is_sleeping(port))


# ------------------------------------------------------------------------------------------------ items
def s0_1():
    if not V:
        return C.put_row("S0-1", "N/A-STUB", reason="no CUDA in the stub rehearsal")
    import subprocess
    env = C.child_env("0,1")
    r = subprocess.run([sys.executable, C.EXT, "--s0-digest", "--s0-mib", "1024"], capture_output=True, text=True,
                       env=env, timeout=300, cwd=C.HERE)
    out = r.stdout + r.stderr
    ok = r.returncode == 0 and "S0-DIGEST-PASS" in out
    rate = C.grep(out, r"rate ([\d.]+) GB/s")
    t16 = C.grep(out, r"16\.46 GB at this rate = ([\d.]+) s")
    return C.put_row("S0-1", "PASS" if ok else "FAIL", cmd="CUDA_VISIBLE_DEVICES=0,1 python3 warm_wake_ext.py "
                     "--s0-digest --s0-mib 1024", rc=r.returncode, rate_gbs=rate, t_16_46GB_s=t16, out=out[-2500:])


def holders():
    if not V:
        sys.path.insert(0, C.HERE)
        import warm_wake_ext as ext
        stub = ext.load_stub()
        stub.StubFabric.create(FAB, ngpu=2, cap_bytes=64 * C.MIB)
    ST["mem_t0"] = C.gpu_used_mib()
    _, p0 = start_holder("holder0", "0,1", PORT["h0"])
    _, p1 = start_holder("holder1", "1,0", PORT["h1"])
    ST["holders"] = True
    res = {}
    ok = True
    for nm, port in (("holder0", p0), ("holder1", p1)):
        al = C.holder(port, "/holder/alloc", {"guest_id": "s02", "nbytes": S02_BYTES, "owner_id": "s0-2"})
        hl = C.handle_len(al.get("ipc_handle"))
        fr = C.holder(port, "/holder/free", {"guest_id": "s02"}) if al.get("ok") else None
        res[nm] = dict(port=port, alloc_ok=al.get("ok"), reason=al.get("reason"), handle_bytes=hl,
                       free_ok=(fr or {}).get("ok"))
        ok &= bool(al.get("ok")) and hl == 64 and bool((fr or {}).get("ok"))
    ST["mem_holders"] = C.gpu_used_mib()
    return C.put_row("S0-2", "PASS" if ok else "FAIL", cmd="CUDA_VISIBLE_DEVICES=<k>,<other> python3 "
                     "warm_wake_ext.py --serve-holder --port 830<k> --cap-gib 35.62 --allow-corrupt; "
                     f"POST /holder/alloc nbytes={S02_BYTES}", holders=res, mem_before=ST["mem_t0"],
                     mem_after_holders=ST["mem_holders"])


def s0_0():
    import subprocess
    al = C.holder(PORT["h1"], "/holder/alloc", {"guest_id": "probe", "nbytes": PROBE_BYTES, "owner_id": "s0-0"})
    if not al.get("ok"):
        ST["ipc"] = False
        return C.put_row("S0-0", "FAIL", reason=f"holder1 alloc: {al.get('reason')}")
    cmd = [sys.executable, os.path.join(C.HERE, "s0_probe.py"), "--handle", al["ipc_handle"], "--nbytes",
           str(PROBE_BYTES)]
    if V:
        env = C.child_env("0,1")
    else:
        cmd += ["--stub-fabric", FAB, "--stub-cvd", "0,1"]
        env = C.child_env(None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300, cwd=C.HERE)
    try:
        js = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        js = {"ok": False, "raw": (r.stdout + r.stderr)[-1500:]}
    C.holder(PORT["h1"], "/holder/free", {"guest_id": "probe", "force": True})
    ST["ipc"] = bool(js.get("ok")) and r.returncode == 0
    return C.put_row("S0-0", "PASS" if ST["ipc"] else "FAIL", cmd="CUDA_VISIBLE_DEVICES=0,1 python3 s0_probe.py "
                     f"--handle <holder1 alloc> --nbytes {PROBE_BYTES}", rc=r.returncode, probe=js,
                     stderr=r.stderr[-1500:])


def boots():
    """E14 on GPU0 (reordered, ext) in parallel with S0-4a: E8's flags PINNED on GPU1; then E8 reordered on GPU1."""
    res = {}

    def boot14():
        if not wait_dl("e14"):
            res["e14"] = "download failed"
            return
        try:
            _, hs = boot("e14", "e14", "0,1", PORT["e14"])
            res["e14"] = dict(health_s=hs)
            ST["e14"] = True
        except Exception as e:  # noqa: BLE001
            res["e14"] = f"boot failed: {e}"[:400]

    def pin8():
        if not wait_dl("e8"):
            res["e8p"] = "download failed"
            return
        try:
            p, hs = boot("e8p", "e8", "1", PORT["e8p"], ext=False)
            txt = p.text()
            res["e8p"] = dict(health_s=hs, kv=C.grep(txt, KV_RE), load=C.grep(txt, LOAD_RE),
                              mem=C.gpu_used_mib())
            p.stop()
            time.sleep(3)
        except Exception as e:  # noqa: BLE001
            res["e8p"] = f"boot failed: {e}"[:400]
            if "e8p" in PROCS:
                PROCS["e8p"].stop()
    t14 = threading.Thread(target=boot14)
    t8 = threading.Thread(target=pin8)
    t14.start()
    t8.start()
    t14.join()
    t8.join()
    mem_before = C.gpu_used_mib()
    apps_before = C.compute_apps()
    try:
        p, hs = boot("e8", "e8", "1,0", PORT["e8"])
        res["e8"] = dict(health_s=hs)
        ST["e8"] = True
    except Exception as e:  # noqa: BLE001
        res["e8"] = f"boot failed: {e}"[:400]
    ST["boots"] = res
    ST["mem_e8_before"], ST["mem_e8_after"] = mem_before, C.gpu_used_mib()
    ST["apps_e8"] = (apps_before, C.compute_apps())
    C.log(f"boots: {json.dumps(res, default=str)[:600]}")


def s0_4():
    res = ST["boots"]
    pin = res.get("e8p") if isinstance(res.get("e8p"), dict) else None
    txt = PROCS["e8"].text()
    kv_re = C.grep(txt, KV_RE)
    cyc = {}
    st, cyc["sleep_s"] = C.stock_sleep(PORT["e8"])
    cyc["sleep_http"] = st
    cyc["asleep"] = C.is_sleeping(PORT["e8"])
    st, cyc["wake_s"] = C.wake(PORT["e8"])
    cyc["wake_http"] = st
    cyc["is_sleeping_after"] = C.is_sleeping(PORT["e8"])
    cyc["completion_http"] = C.complete(PORT["e8"], model_of("e8"))
    cyc_ok = (cyc["sleep_http"] == 200 and cyc["asleep"] is True and cyc["wake_http"] == 200
              and cyc["is_sleeping_after"] is False and cyc["completion_http"] == 200)
    if V:
        kv_ok = bool(pin) and bool(pin["kv"]) and pin["kv"] == kv_re
    else:
        kv_ok = True                                     # the stub prints no KV line
    m0, m1 = ST["mem_e8_before"], ST["mem_e8_after"]
    d0 = (m1[0] - m0[0]) if (m0 and m1 and 0 in m0 and 0 in m1) else None
    return C.put_row("S0-4", "PASS" if (kv_ok and cyc_ok) else "FAIL",
                     cmd="vllm serve Qwen/Qwen3-8B <E8 flags> with CUDA_VISIBLE_DEVICES=1 (pinned, no ext) then =1,0 "
                         "(reordered, ext); stock sleep/wake on the reordered engine", kv_pinned=(pin or {}).get("kv"),
                     kv_reordered=kv_re, load_pinned=(pin or {}).get("load"), load_reordered=C.grep(txt, LOAD_RE),
                     stock_cycle=cyc, gpu0_used_delta_mib_on_reordered_launch=d0, mem_before=m0, mem_after=m1,
                     compute_apps=ST.get("apps_e8"), boots=res)


def s0_3():
    out, ok = {}, True
    for nm in ("e14", "e8"):
        if not ST.get(nm):
            out[nm] = "not booted"
            ok = False
            continue
        txt = PROCS[nm].text()
        inj = C.grep(txt, INJ_RE)
        inv = C.rpc(PORT[nm], "bb_inventory", {})
        w = ((inv or {}).get("tags") or {}).get("weights") or {}
        out[nm] = dict(injected_lines=len(inj), shape_problem=inv.get("shape_problem", "MISSING"), cvd=inv.get("cvd"),
                       device=inv.get("device"), allocator=inv.get("allocator"), tags=inv.get("tags"),
                       load_line=C.grep(txt, LOAD_RE), kv_line=C.grep(txt, KV_RE), inv_ok=inv.get("ok"))
        ST[f"w_{nm}"] = int(w.get("bytes") or 0)
        ST[f"n_{nm}"] = int(w.get("n") or 0)
        ok &= bool(inj) and inv.get("ok") is True and inv.get("shape_problem") is None and ST[f"w_{nm}"] > 0
    ST["inv"] = ok
    return C.put_row("S0-3", "PASS" if ok else "FAIL", cmd="VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=<own>,<other> "
                     "PYTHONPATH=<harness> vllm serve ... --worker-extension-cls warm_wake_ext.WarmWakeExt"
                     "; POST /collective_rpc bb_inventory", engines=out)


def warm_and_reads():
    """one stock sleep/wake on E14 (+ a COOL wake time), then the pre-sleep teacher-forced reads of both."""
    st, s = C.stock_sleep(PORT["e14"])
    st2, w = C.wake(PORT["e14"])
    ST["cool14"] = dict(sleep_http=st, sleep_s=s, wake_http=st2, wake_s=w, is_sleeping=C.is_sleeping(PORT["e14"]))
    ST["tf14"] = C.tf_read("http://127.0.0.1:%d" % PORT["e14"], model_of("e14"))
    ST["tf8"] = C.tf_read("http://127.0.0.1:%d" % PORT["e8"], model_of("e8"))
    ST["gen14"] = C.gen_read("http://127.0.0.1:%d" % PORT["e14"], model_of("e14"))
    ST["gen8"] = C.gen_read("http://127.0.0.1:%d" % PORT["e8"], model_of("e8"))
    ST["reads"] = True
    C.log(f"cool14 {ST['cool14']}")


def push_item(kind, lender_port, lender_gpu, gid):
    port = PORT[kind]
    nbytes = ST[f"w_{kind}"]
    al = C.holder(lender_port, "/holder/alloc", {"guest_id": gid, "nbytes": nbytes, "owner_id": kind})
    if not al.get("ok"):
        return None, dict(alloc=al)
    m0, a0 = C.gpu_used_mib(), C.compute_apps()
    r, tries = push(port, gid, al, kind, nbytes)
    m1, a1 = C.gpu_used_mib(), C.compute_apps()
    at = C.holder(lender_port, "/holder/attach", {"guest_id": gid, "manifest": r.get("manifest") or {}, "verify": True}) \
        if r.get("ok") else {"ok": False, "reason": "no push"}
    d = (m1[lender_gpu] - m0[lender_gpu]) if (m0 and m1) else None
    info = dict(gid=gid, nbytes=nbytes, push=tries, attach=at, lender_used_before=m0, lender_used_after=m1,
                lender_delta_mib=d, apps_before=a0, apps_after=a1)
    g = dict(gid=gid, handle=al["ipc_handle"], nbytes=nbytes, mid=r.get("manifest_id"), offsets=r.get("offsets"),
             n_alloc=r.get("n_alloc"), port=lender_port) if (r.get("ok") and at.get("ok")) else None
    return g, info


def s0_5_7():
    res, ok = {}, {}
    for kind, lp, lg, gid in (("e14", PORT["h1"], 1, "e14-g1"), ("e8", PORT["h0"], 0, "e8-g0")):
        g, info = push_item(kind, lp, lg, gid)
        ST[f"g_{kind}"] = g
        res[kind] = info
        modes = [t["mode"] for t in (info.get("push") or []) if t["res"].get("ok")]
        ok[kind] = "PASS" if (g and modes == ["direct"]) else ("PASS-STAGED" if (g and "staged" in modes) else "FAIL")
    v = "PASS" if all(x == "PASS" for x in ok.values()) else (
        "PASS-STAGED" if all(x.startswith("PASS") for x in ok.values()) else "FAIL")
    C.put_row("S0-5", v, cmd="POST /holder/alloc on the lender -> POST /collective_rpc bb_push_to_peer (direct, staged "
              "on failure) -> POST /holder/attach verify", per_kind=ok,
              push={k: dict(push=r_.get("push"), attach=r_.get("attach"), alloc=r_.get("alloc"))   # QA c7: an alloc
                    for k, r_ in res.items()})                                                           # failure has no push
    C.put_row("S0-7", ("PASS" if all(r_.get("lender_delta_mib") is not None for r_ in res.values()) else "FAIL")
              if V else "N/A-STUB",
              cmd="nvidia-smi --query-gpu=index,memory.used right before and after the engine's FIRST IPC open of the "
                  "lender (its first push; the guest buffer was allocated before the 'before' read)",
              pairs={f"{k}->gpu{1 if k == 'e14' else 0}": dict(delta_mib=r_.get("lender_delta_mib"),
                                                                 before=r_.get("lender_used_before"),
                                                                 after=r_.get("lender_used_after"),
                                                                 apps_before=r_.get("apps_before"),
                                                                 apps_after=r_.get("apps_after")) for k, r_ in res.items()})


def cycle(kind, g, tf0, seal=True):
    port = PORT[kind]
    c = {}
    st, c["sleep_s"] = C.stock_sleep(port)
    c["sleep_http"] = st
    c["asleep"] = C.is_sleeping(port)
    if seal:
        t0 = time.monotonic()
        c["seal"] = C.rpc(port, "bb_seal_after_sleep", {"guest_id": g["gid"]})
        c["seal_wall_s"] = round(time.monotonic() - t0, 6)
    r, tries = restore(port, g["gid"], g["handle"], g["nbytes"], g["mid"])
    c["restore"] = r
    c["restore_tries"] = [t["mode"] for t in tries]
    c.update(both_wakes(port))
    tf = C.tf_read("http://127.0.0.1:%d" % port, model_of(kind))
    c["tf_maxdiff"] = C.tf_diff(tf0, tf)
    g0 = ST.get(f"gen{kind[1:]}")
    c["greedy32_equal"] = (g0 is not None and C.gen_read("http://127.0.0.1:%d" % port, model_of(kind)) == g0)  # reported
    c["pass"] = (c["asleep"] is True and (not seal or (c["seal"].get("ok") and c["seal"].get("stale") is False))
                 and r.get("served") == "PEER" and c["is_sleeping"] is False and c["tf_maxdiff"] <= C.TOL)
    return c


def s0_6_8():
    allc, okk = {}, {}
    for kind in ("e14", "e8"):
        g = ST.get(f"g_{kind}")
        if not g:
            allc[kind] = "no guest (S0-5 failed)"
            okk[kind] = False
            continue
        cs = [cycle(kind, g, ST[f"tf{kind[1:]}"]) for _ in range(3)]
        allc[kind] = cs
        okk[kind] = all(c["pass"] for c in cs)
    ST["cycles_ok"] = all(okk.values())
    C.put_row("S0-6", "PASS" if ST["cycles_ok"] else "FAIL",
              cmd="/sleep?level=1&mode=wait -> bb_seal_after_sleep -> bb_restore_from_peer (direct) -> "
                  "/wake_up?tags=weights -> /wake_up?tags=kv_cache -> /is_sleeping; teacher-forced P1-P3; n=3 per kind",
              per_kind=okk, cycles=allc,
              summary={k: [dict(open_s=c["restore"].get("open_s"), copy_s=c["restore"].get("copy_s"),
                                verify_s=c["restore"].get("verify_s"), total_s=c["restore"].get("total_s"),
                                rate_gbs=c["restore"].get("rate_gbs"), served=c["restore"].get("served"),
                                seal_s=(c.get("seal") or {}).get("seconds"), sleep_s=c["sleep_s"],
                                wake_kv_s=c["wake_kv"][1], tf=c["tf_maxdiff"], greedy32_equal=c["greedy32_equal"])
                           for c in v]
                       for k, v in allc.items() if isinstance(v, list)})
    opens = {k: [c["restore"].get("open_s") for c in v] for k, v in allc.items() if isinstance(v, list)}
    C.put_row("S0-8", "PASS" if opens and all(all(x is not None for x in v) for v in opens.values()) else "FAIL",
              cmd="open_s of restores 1..3 from the same lender (S0-6's cycles)", open_s=opens)


def s0_9():
    g = ST["g_e14"]
    port = PORT["e14"]
    off = int(g["offsets"][1]) + 12345
    cr = C.holder(g["port"], "/holder/corrupt", {"guest_id": g["gid"], "offset": off})
    st, s = C.stock_sleep(port)
    r, tries = restore(port, g["gid"], g["handle"], g["nbytes"], g["mid"])
    w = both_wakes(port)
    tf = C.tf_read("http://127.0.0.1:%d" % port, model_of("e14"))
    d = C.tf_diff(ST["tf14"], tf)
    ok = (cr.get("ok") and r.get("served") == "HOST_FALLBACK" and r.get("reason") == "DIGEST_MISMATCH"
          and r.get("rolled_back") == g["n_alloc"] and 1 in (r.get("bad_allocs") or []) and w["is_sleeping"] is False
          and d <= C.TOL)
    fr = C.holder(g["port"], "/holder/free", {"guest_id": g["gid"], "force": True})
    v = "FAIL-STOP-THE-RUN" if r.get("served") == "PEER" else ("PASS" if ok else "FAIL")
    return C.put_row("S0-9", v, cmd=f"POST /holder/corrupt {{guest_id: {g['gid']}, offset: offsets[1]+12345 = {off}}} "
                     "-> sleep -> bb_restore_from_peer -> stock wakes", corrupt=cr, restore=r, wakes=w, tf_maxdiff=d,
                     n_alloc=g["n_alloc"], freed=fr)


def s0_10():
    port = PORT["e14"]
    C.holder(PORT["h0"], "/holder/free", {"guest_id": "e8-g0", "force": True})     # E8 stays awake; its guest goes
    g, info = push_item("e14", PORT["h1"], 1, "e14-g4")
    if not g:
        return C.put_row("S0-10", "FAIL", reason="re-push for S0-10 failed", push=info)
    st, s = C.stock_sleep(port)
    fl = C.holder(PORT["h0"], "/holder/alloc", {"guest_id": "filler", "nbytes": FILLER, "owner_id": "cap-stand-in"})
    b0 = None
    try:
        b0, hs = boot("b0", "b", "0", PORT["b0"], ext=False)
        C.complete(PORT["b0"], model_of("b0"))
        m_pre = C.gpu_used_mib()
        r, tries = restore(port, g["gid"], g["handle"], g["nbytes"], g["mid"])
        health = C.http_json(f"http://127.0.0.1:{port}/health", method="GET", timeout=10)[0]
        yst, ys = C.stock_sleep(PORT["b0"], "abort")
        m_post = C.gpu_used_mib()
        w = both_wakes(port)
        tf = C.tf_read("http://127.0.0.1:%d" % port, model_of("e14"))
        d = C.tf_diff(ST["tf14"], tf)
    finally:                                        # rehearsal 1: never leave the filler or B0 behind on a raise
        C.holder(PORT["h0"], "/holder/free", {"guest_id": "filler", "force": True})
        if b0 is not None:
            b0.stop()
        elif "b0" in PROCS:
            PROCS["b0"].stop()
    ok = (fl.get("ok") and r.get("served") == "HOST_FALLBACK" and str(r.get("reason", "")).startswith("CREATE_AND_MAP")
          and not r.get("poisoned") and r.get("probe_rc") == 0 and health == 200 and yst == 200
          and w["is_sleeping"] is False and d <= C.TOL)
    return C.put_row("S0-10", "PASS" if ok else "FAIL", cmd="holder0 /holder/alloc filler (CAP stand-in) -> B0 boot "
                     "(CUDA_VISIBLE_DEVICES=0) awake -> E14 bb_restore_from_peer BEFORE B0's /sleep?mode=abort -> B0 "
                     "yield -> stock wakes", filler=fl, b0_health_s=hs, mem_before_restore=m_pre, restore=r,
                     e14_health=health, b0_yield=dict(http=yst, s=ys), mem_after_yield=m_post, wakes=w, tf_maxdiff=d)


def s0_11a():
    port = PORT["e14"]
    C.holder(PORT["h0"], "/holder/free", {"guest_id": "e8-g0", "force": True})     # E8 stays awake; its guest goes
    g, info = push_item("e14", PORT["h1"], 1, "e14-g2")
    if not g:
        return C.put_row("S0-11a", "FAIL", reason="push for S0-11a failed", push=info)
    m0 = C.gpu_used_mib()
    da = C.holder(PORT["h0"], "/holder/alloc", {"guest_id": g["gid"], "nbytes": g["nbytes"], "owner_id": "rehome"})
    rh = C.holder(PORT["h1"], "/holder/rehome_to", {"guest_id": g["gid"], "dest_handle": da.get("ipc_handle"),
                                                    "dest_nbytes": g["nbytes"]}) if da.get("ok") else {"ok": False}
    m1 = C.gpu_used_mib()
    at = C.holder(PORT["h0"], "/holder/attach", {"guest_id": g["gid"], "manifest": rh.get("manifest") or {},
                                                 "verify": True}) if rh.get("ok") else {"ok": False}
    fr = C.holder(PORT["h1"], "/holder/free", {"guest_id": g["gid"]}) if at.get("ok") else None
    st, s = C.stock_sleep(port)
    r, tries = restore(port, g["gid"], da.get("ipc_handle"), g["nbytes"], g["mid"]) if at.get("ok") else ({}, [])
    w = both_wakes(port)
    tf = C.tf_read("http://127.0.0.1:%d" % port, model_of("e14"))
    d = C.tf_diff(ST["tf14"], tf)
    C.holder(PORT["h0"], "/holder/free", {"guest_id": g["gid"], "force": True})
    ok = (at.get("ok") and at.get("state") == "FILLED" and at.get("bad") == [] and r.get("served") == "PEER"
          and w["is_sleeping"] is False and d <= C.TOL)
    gbs = round(g["nbytes"] / rh["seconds"] / 1e9, 3) if rh.get("ok") and rh.get("seconds") else None
    return C.put_row("S0-11a", "PASS" if ok else "FAIL", cmd="holder0 /holder/alloc -> holder1 /holder/rehome_to -> "
                     "holder0 /holder/attach verify -> holder1 free -> E14 sleep -> restore from holder0 -> wakes",
                     dest_alloc={k: v for k, v in da.items() if k != "ipc_handle"},
                     rehome={k: v for k, v in rh.items() if k != "manifest"}, rehome_gbs=gbs, attach=at, source_free=fr,
                     gpu0_delta_mib_on_holder1_first_open=((m1[0] - m0[0]) if (m0 and m1) else None),
                     restore=r, wakes=w, tf_maxdiff=d, note="holder0 is E14's own card: the read is local")


def s0_11b():
    port = PORT["e14"]
    g, info = push_item("e14", PORT["h1"], 1, "e14-g3")
    if not g:
        return C.put_row("S0-11b", "FAIL", reason="push for C6 failed", push=info)
    st, s = C.stock_sleep(port)
    C.stock_sleep(PORT["e8"])                                   # GPU1 is parked: its owner sleeps
    b1, load = None, None
    try:
        return _s0_11b_body(port, g)
    finally:                                                    # rehearsal 1: cleanup on every path
        if ST.get("load11") is not None:
            ST["load11"].stop()
        if "b1" in PROCS:
            PROCS["b1"].stop()
        C.holder(PORT["h1"], "/holder/free", {"guest_id": g["gid"], "force": True})


def _s0_11b_body(port, g):
    b1, hs = boot("b1", "b", "1", PORT["b1"], ext=False)
    load = C.Load(PORT["b1"], model_of("b1"), streams=8, p=512 if V else 8, m=512 if V else 4)
    ST["load11"] = load
    load.start()
    time.sleep(12 if V else 2)
    t_y0 = time.monotonic()
    y0st, y0 = C.stock_sleep(PORT["b1"], "abort")
    time.sleep(1.0)                                  # let the aborted streams' last chunks land
    base_ends = load.summary(since=t_y0, until=time.monotonic())
    C.wake(PORT["b1"])
    time.sleep(10 if V else 2)
    res = {}

    def yld():
        res["t_y"] = time.monotonic()
        res["yield"] = C.stock_sleep(PORT["b1"], "abort")
        res["t_y_end"] = time.monotonic()

    def rst():
        res["t_r"] = time.monotonic()
        res["restore"] = restore(port, g["gid"], g["handle"], g["nbytes"], g["mid"])[0]
        res["t_r_end"] = time.monotonic()
    ty, tr = threading.Thread(target=yld), threading.Thread(target=rst)
    t_c = time.monotonic()
    ty.start()
    tr.start()
    ty.join()
    tr.join()
    time.sleep(1.0)
    conc_ends = load.summary(since=t_c, until=time.monotonic())
    load.stop()
    w = both_wakes(port)
    tf = C.tf_read("http://127.0.0.1:%d" % port, model_of("e14"))
    d = C.tf_diff(ST["tf14"], tf)
    r = res.get("restore") or {}
    ok = r.get("served") == "PEER" and w["is_sleeping"] is False and d <= C.TOL and res.get("yield", (0,))[0] == 200
    ST["load_finish"] = dict(baseline=base_ends, concurrent=conc_ends)
    return C.put_row("S0-11b", "PASS" if ok else "FAIL", cmd="B1 (CUDA_VISIBLE_DEVICES=1) 8x512/512 ignore_eos load; "
                     "baseline /sleep?mode=abort; wake; then B1 /sleep?mode=abort || E14 bb_restore_from_peer from holder1",
                     b1_health_s=hs, baseline_yield=dict(http=y0st, s=y0), baseline_stream_ends=base_ends,
                     concurrent=dict(yield_http=res.get("yield", (None, None))[0], yield_s=res.get("yield", (None, None))[1],
                                     restore_start_minus_yield_start_s=round(res["t_r"] - res["t_y"], 6),
                                     restore_wall_s=round(res["t_r_end"] - res["t_r"], 6),
                                     yield_wall_s=round(res["t_y_end"] - res["t_y"], 6)),
                     restore=r, concurrent_stream_ends=conc_ends, wakes=w, tf_maxdiff=d)


def main():
    C.log(f"s0_driver start backend={C.BACKEND} left={C.left():.0f}s")
    step("S0-1", s0_1)
    step("S0-2", holders)
    step("S0-0", s0_0, needs=("holders",))
    step("BOOTS", boots, needs=("holders",))
    step("S0-4", s0_4, needs=("e8",))
    step("S0-3", s0_3)
    step("READS", warm_and_reads, needs=("e14", "e8"))
    step("S0-5", s0_5_7, needs=("inv", "reads"))
    step("S0-6", s0_6_8, needs=("inv", "reads"))
    step("S0-9", s0_9, needs=("g_e14",))
    step("S0-11a", s0_11a, needs=("inv", "reads"))
    step("S0-11b", s0_11b, needs=("inv", "reads"))
    step("S0-10", s0_10, needs=("inv", "reads"))         # last: the step most likely to break E14 (an OOM path)
    for nm, p in list(PROCS.items()):
        p.stop()
    with open(os.path.join(C.LOGS, "working_flags.json"), "w") as f:      # A1: the controller phase inherits them
        json.dump({k: v[2] for k, v in ENG.items()}, f, indent=1)
    C.put_row("DRIVER-STATE", "INFO", cool14=ST.get("cool14"), load_finish=ST.get("load_finish"),
              flags={k: v[2] for k, v in ENG.items()}, boot_failures=ST.get("boot_failures"),
              w={k: ST.get(f"w_{k}") for k in ("e14", "e8")}, n={k: ST.get(f"n_{k}") for k in ("e14", "e8")},
              mem_end=C.gpu_used_mib())
    C.log("s0_driver done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        for p in list(PROCS.values()):
            try:
                p.stop()
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
