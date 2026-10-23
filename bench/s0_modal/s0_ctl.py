"""s0_ctl.py -- S0 controller phase: S0-12 .. S0-15 through `fabric_ctl.py serve` (backend vllm on the box,
backend stub for the $0 rehearsal). S0-16 is scored afterwards on the CPU host from ctl_logs/ (e2e_score.log_replay +
sim_order). Prereg: (private report) (frozen).

Output contract: exit 0 after the last item whatever the verdicts (rows/S0-12..S0-15.json); exit 1 on a harness crash.
WRITES ctl_cfg.json, ctl_logs/ (the controller's own event/request/sampler/boot logs + one log per child), rows/.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C  # noqa: E402
import s0_driver as D  # noqa: E402

V = C.BACKEND == "vllm"
CTL = os.path.join(C.HERE, "fabric_ctl.py")
LOGDIR = os.path.join(C.HERE, "ctl_logs")
CFG = os.path.join(C.HERE, "ctl_cfg.json")
NAMES = {0: "e14", 1: "e8"}


def build_cfg():
    wf = os.path.join(C.LOGS, "working_flags.json")         # A1: use the flags that booted in the driver phase
    if os.path.exists(wf):
        for k, fl in json.load(open(wf)).items():
            if k in D.ENG:
                r, v, _f, w, kv = D.ENG[k]
                D.ENG[k] = (r, v, list(fl), w, kv)
    if V:
        def vs(kind):
            repo, rev, flags, _w, _k = D.ENG[kind]
            return {"repo": repo, "revision": rev, "flags": list(flags)}
        slots = [dict(slot=0, gpu=0, name="e14", holder_port=8300, owner=dict(port=8400, vllm=vs("e14")),
                      batch=dict(port=8410, vllm=vs("b"))),
                 dict(slot=1, gpu=1, name="e8", holder_port=8301, owner=dict(port=8401, vllm=vs("e8")),
                      batch=dict(port=8411, vllm=vs("b")))]
        cfg = dict(mode="fabric", backend="vllm", gpus=[0, 1], slots=slots, log_dir=LOGDIR, host="127.0.0.1",
                   cap_bytes=int(35.62 * C.GIB), keepalive_s=3600.0, tick_s=0.25, sample_s=1.0, time_scale=1.0,
                   request_timeout_s=600.0, boot_timeout_s=1500,
                   batch_load=dict(streams=8, prompt_tokens=512, max_tokens=512, seed=0),
                   ext_file=C.EXT, port=8500)
    else:
        sys.path.insert(0, C.HERE)
        import fabric_ctl as F
        cfg = F.stub_cfg(os.path.join(C.HERE, "ctlstub"), 2, w=(1 * C.MIB, 3 * C.MIB), cap_bytes=8 * C.MIB,
                         keepalive_s=3600.0, tick_s=0.05)
        cfg["log_dir"] = LOGDIR
        cfg["ext_file"] = C.EXT
        cfg["stub"]["stub_file"] = C.STUB
        for sl in cfg["slots"]:
            sl["name"] = NAMES[sl["slot"]]
    with open(CFG, "w") as f:
        json.dump(cfg, f, indent=1)
    return cfg


def events(kind=None):
    p = os.path.join(LOGDIR, "events.jsonl")
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if kind is None or r.get("ev") == kind:
                out.append(r)
    return out


def jl(name):
    """QA (3 arms): the controller appends concurrently, so a torn last line is skipped, never a crash."""
    p = os.path.join(LOGDIR, f"{name}.jsonl")
    out = []
    if os.path.exists(p):
        for x in open(p):
            if not x.strip():
                continue
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                continue
    return out


def wait_ev(kind, pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        hit = [e for e in events(kind) if pred(e)]
        if hit:
            return hit[-1]
        time.sleep(0.25)
    return None


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except OSError as e:
        return f"unreadable: {e}"


def main():
    C.log(f"s0_ctl start backend={C.BACKEND} left={C.left():.0f}s")
    cfg = build_cfg()
    url = None
    out_f = open(os.path.join(C.LOGS, "ctl_serve.out"), "wb")
    err_f = open(os.path.join(C.LOGS, "ctl_serve.err"), "wb")
    proc = subprocess.Popen([sys.executable, CTL, "serve", "--config", CFG], stdout=out_f, stderr=err_f,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd=C.HERE)
    C.log(f"ctl serve pid={proc.pid}")
    t0 = time.monotonic()
    port = None
    while time.monotonic() - t0 < (1700 if V else 120):
        try:
            txt = open(os.path.join(C.LOGS, "ctl_serve.out")).read()
        except OSError:
            txt = ""
        if "BB-FABRIC-READY port=" in txt:
            port = int(txt.split("BB-FABRIC-READY port=")[1].split()[0])
            break
        if proc.poll() is not None or C.left() < 300:
            break
        time.sleep(1)
    boot_s = round(time.monotonic() - t0, 1)
    if port is None:
        err = open(os.path.join(C.LOGS, "ctl_serve.err")).read()[-3000:]
        C.put_row("S0-12", "FAIL", reason="no BB-FABRIC-READY", boot_s=boot_s, rc=proc.poll(), stderr=err)
        for it in ("S0-13", "S0-14", "S0-15"):
            C.put_row(it, "NOT-RUN", reason="controller did not boot (S0-12)")
        stop(proc, None)
        return 0
    url = f"http://127.0.0.1:{port}"
    crashed = False
    try:
        run_items(cfg, url, boot_s)
    except Exception as e:  # noqa: BLE001
        crashed = True
        C.log(f"s0_ctl items raised {e!r}")
        C.put_row("CTL-CRASH", "FAIL", reason=repr(e)[:400], tb=traceback.format_exc()[-2000:])
    stop(proc, url)
    return 1 if crashed else 0                      # QA: the contract says exit 1 on a harness crash


def stop(proc, url):
    if url:
        try:
            C.http_json(url + "/fabric/stop", {}, timeout=30)
        except Exception:  # noqa: BLE001
            pass
    try:
        rc = proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    C.put_row("CTL-EXIT", "INFO", rc=rc, stderr_tail=open(os.path.join(C.LOGS, "ctl_serve.err")).read()[-2000:])


def run_items(cfg, url, boot_s):
    # ---------------------------------------------------------------- S0-12
    boot = [r for r in jl("boot") if r.get("ev") == "boot_done"]
    bd = boot[-1] if boot else {}
    pids = bd.get("pids") or {}
    lines = {k: cmdline(v) for k, v in pids.items()}
    inv = {}
    for sl in cfg["slots"]:
        inv[sl["name"]] = C.rpc(sl["owner"]["port"], "bb_inventory", {})
    wb = bd.get("w_bytes") or {}
    chk = {}
    for sl in cfg["slots"]:
        s, nm = sl["slot"], sl["name"]
        own = lines.get(f"{s}:owner")
        bat = lines.get(f"{s}:batch")
        own_s = " ".join(own) if isinstance(own, list) else str(own)
        bat_s = " ".join(bat) if isinstance(bat, list) else str(bat)
        want_cvd = ",".join(str(g) for g in [sl["gpu"]] + [g for g in cfg["gpus"] if g != sl["gpu"]])
        chk[nm] = dict(
            w_bytes=wb.get(nm), shape_problem=inv[nm].get("shape_problem", "MISSING"), cvd=inv[nm].get("cvd"),
            want_cvd=want_cvd,
            owner_has_ext=("--worker-extension-cls warm_wake_ext.WarmWakeExt" in own_s) if V else None,
            owner_host=("--host 127.0.0.1" in own_s) if V else None,
            owner_served=(f"--served-model-name {nm}" in own_s) if V else None,
            batch_no_ext=("worker-extension-cls" not in bat_s) if V else None,
            owner_cmd=own, batch_cmd=bat)
    def ok12(c):
        base = bool(c["w_bytes"]) and c["shape_problem"] is None and (c["cvd"] == c["want_cvd"] or not V)
        return base and (not V or (c["owner_has_ext"] and c["owner_host"] and c["owner_served"] and c["batch_no_ext"]))
    C.put_row("S0-12", "PASS" if all(ok12(c) for c in chk.values()) else "FAIL",
              cmd="python3 fabric_ctl.py serve --config ctl_cfg.json (backend vllm, fabric, 2 owners + 2 batch "
                  "+ 2 holders); /proc/<pid>/cmdline; bb_inventory", boot_s=boot_s, boot_done=bd, slots=chk)
    # ---------------------------------------------------------------- S0-13 / S0-14
    tf0 = {nm: C.tf_read(url, nm) for nm in ("e14", "e8")}
    st, pk = C.http_json(url + "/fabric/park", {"slots": [0, 1]}, timeout=900)
    pdone = {s: wait_ev("park_done", lambda e, s=s: e.get("slot") == s, 300) for s in (0, 1)}
    t_park = time.monotonic()
    snap0 = C.http_json(url + "/fabric/state", method="GET")[1]
    time.sleep(35 if V else 3)
    snap1 = C.http_json(url + "/fabric/state", method="GET")[1]
    tf13 = C.tf_read(url, "e14")
    rd0 = wait_ev("ready", lambda e: e.get("slot") == 0, 300)
    time.sleep(2)
    snap2 = C.http_json(url + "/fabric/state", method="GET")[1]
    d13 = C.tf_diff(tf0["e14"], tf13)
    parts = {k: (rd0 or {}).get(k) for k in ("yield_s", "t_yield_done", "restore_end", "wake_weights_s", "wake_kv_s",
                                              "D", "source", "restore", "restore_card", "cowake")}
    ok13 = (st == 200 and rd0 is not None and rd0.get("source") == "HOLDER" and (rd0.get("restore") or {}).get("served")
            == "PEER" and d13 <= C.TOL and all(parts[k] is not None for k in ("yield_s", "t_yield_done", "restore_end",
                                                                               "wake_weights_s", "wake_kv_s")))
    tf1 = C.tf_read(url, "e8")
    rd1 = wait_ev("ready", lambda e: e.get("slot") == 1, 300)
    d13b = C.tf_diff(tf0["e8"], tf1)
    C.put_row("S0-13", "PASS" if ok13 else "FAIL",
              cmd="router teacher-forced P1-P3 for e14/e8; POST /fabric/park {slots:[0,1]}; 35 s parked; one router "
                  "request for e14 (then e8)", park_http=st, placements=pk, park_done=pdone, ready_slot0=rd0,
              parts=parts, tf_maxdiff_e14=d13, ready_slot1=rd1, tf_maxdiff_e8=d13b,
              slot1_reported_only="prereg S0-13: slot 1's HOST wake is reported, not deciding",
              note="2-GPU form: both owners parked in one tick (a lone parked owner has no lender)")
    b_a = ((snap0 or {}).get("batch") or {}).get("0") or {}
    b_b = ((snap1 or {}).get("batch") or {}).get("0") or {}
    b_c = ((snap2 or {}).get("batch") or {}).get("0") or {}
    samp = [r for r in jl("sampler") if r.get("ev") == "sample"]
    ut = [((r.get("slots") or {}).get("0") or {}).get("batch", {}).get("usage_tokens") for r in samp]
    ut = [x for x in ut if x is not None]
    a_ok = b_b.get("usage_tokens", 0) > b_a.get("usage_tokens", 0) and b_b.get("errors", 1) == 0
    b_ok = b_c.get("aborted", 0) >= 1
    c_ok = (rd0 is not None and rd0.get("yield_s") is not None and rd0.get("t_yield_done") is not None
            and rd0.get("restore_end") is not None and rd0["t_yield_done"] <= rd0["restore_end"])
    C.put_row("S0-14", "PASS" if (a_ok and b_ok and c_ok) else "FAIL",
              cmd="the controller's 8-stream batch load (token-id prompts, ignore_eos) on parked slot 0's batch engine; "
                  "slot 0's reclaim (/sleep?mode=abort)", a_usage_grew=a_ok, b_aborted=b_ok, c_order=c_ok,
              batch0_at_park=b_a, batch0_after_35s=b_b, batch0_after_reclaim=b_c, sampler_usage_tokens_series=ut[-60:],
              batch1_after=((snap2 or {}).get("batch") or {}).get("1"))
    # ---------------------------------------------------------------- S0-15 (partial, reported)
    st2, pk2 = C.http_json(url + "/fabric/park", {"slots": [0, 1]}, timeout=900)
    for s in (0, 1):
        wait_ev("park_done", lambda e, s=s: e.get("slot") == s and e.get("mono", 0) > t_park + 1, 300)
    time.sleep(15 if V else 2)
    n_ready = len(events("ready"))
    res = {}

    def go(nm):
        try:
            res[nm] = C.tf_diff(tf0[nm], C.tf_read(url, nm))
        except Exception as e:  # noqa: BLE001
            res[nm] = repr(e)[:200]
    ths = [threading.Thread(target=go, args=(nm,)) for nm in ("e14", "e8")]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    time.sleep(3)
    rds = events("ready")[n_ready:]
    drops = [e for e in events() if e.get("ev") in ("rehome_drop", "rehome_done", "rehome_failed", "free_timeout",
                                                    "handoff_read", "reader_refused")]
    C.put_row("S0-15", "N/A-ON-MODAL", reason="RECLAIMED-CARD needs a re-home target that is neither the reclaimed card "
              "nor the guest's owner: >= 3 GPUs (fabric_ctl._drain_items / FabricCore); this box has 2",
              cmd="partial in-situ read: POST /fabric/park {slots:[0,1]}; 15 s; router requests for e14 and e8 together",
              park2=dict(http=st2, placements=pk2), cowake_ready=rds, drain_events=drops[-12:], tf_maxdiff=res,
              final_state=C.http_json(url + "/fabric/state", method="GET")[1])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
