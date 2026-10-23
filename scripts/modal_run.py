"""modal_run.py — run one gpu-bunk GPU harness in ONE Modal container and bring every result home.

Usage (foreground; blocks until the container returns or Modal's timeout ends it):
    ~/.local/bin/modal run scripts/modal_run.py --spec <spec.json>

Output contract: success = exit 0 + <out>/<tag>.tar.gz (sha256 re-checked against the in-container digest) extracted to
<out>/<tag>/ + <out>/<tag>.meta.json (identity, rc, wall seconds, app id). Any failure raises (non-zero exit); the
Volume `gpu-bunk-results` holds the last snapshot under <tag>/ either way (fetch-then-destroy: a watcher thread
tars the declared items every 120 s and at every new rows/ARM_DONE_* or rows/DONE, BEFORE the function returns).
WRITES: <out>/ only, and the Modal Volume.

spec.json keys: tag, out (local dir), harness_dir, extra_files [paths placed at the harness root], exclude [names],
workdir (in-container), mkdirs [dirs to create, default rows logs], prep [shell cmds, each must exit 0 or the run stops], cmd (shell), env {K: V}, items [names
under workdir to snapshot, default rows logs], gpu, cpu (physical cores; request = limit), mem_mib, mem_limit_mib,
timeout_s (Modal's hard cap), cmd_timeout_s (the harness's own bound, < timeout_s), region (optional; Modal
bills a pinned region at 1.15x broad / 1.75x narrow).
No secret goes into the image or the container: the spec's env is checked for secret-shaped NAMES and refused.

Top pitfalls (observed): `nvidia-smi topo -m` fails and bus ids read 'Unknown Error' inside Modal's sandbox
a result written only inside the container is gone when the function returns (fetch-then-destroy).
"""
import hashlib, io, json, os, re, subprocess, sys, tarfile, threading, time

import modal

APP = modal.App("gpu-bunk-rows")
# Modal needs `python` and `pip` on PATH (modal.com/docs/guide/existing-images); the vLLM image ships `python3` only
# (first smoke: "unable to determine the version of Python installed in the Image", 2026-10-21).
IMAGE = modal.Image.from_registry(
    "vllm/vllm-openai:v0.30.0",
    setup_dockerfile_commands=[
        'RUN ln -sf "$(command -v python3)" /usr/local/bin/python && '
        '(python -m pip --version || (curl -sS https://bootstrap.pypa.io/get-pip.py | python)) && '
        'ln -sf /usr/local/bin/python /usr/local/bin/python3-modal && python --version && python -m pip --version',
        'RUN command -v pip || printf "#!/bin/sh\\nexec python -m pip \\"\\$@\\"\\n" > /usr/local/bin/pip && chmod +x /usr/local/bin/pip',
    ]).entrypoint([])
VOL = modal.Volume.from_name("gpu-bunk-results", create_if_missing=True)
SECRETISH = re.compile(r"TOKEN|SECRET|KEY|PASS|CRED|AUTH", re.I)
SECRET_VALUE = re.compile(r"^(sk-|hf_|ghp_|gho_|github_pat_|xox[bp]-|AKIA|ak-|as-|eyJ)")   # review: values, not only names
_SNAP_LOCK = threading.Lock()
ID_KEEP = ("MODAL_REGION", "MODAL_CLOUD_PROVIDER", "MODAL_TASK_ID", "MODAL_ENVIRONMENT", "MODAL_IMAGE_ID")


def _identity():
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"ERR {e!r}"[:200]
    env = {}
    for k in sorted(os.environ):
        if k.startswith("MODAL_"):
            env[k] = "<redacted>" if re.search(r"TOKEN|SECRET|KEY|PASS|IDENTITY|CRED|AUTH", k, re.I) else (
                os.environ[k] if k in ID_KEEP else "<present>")
    q = ("name,uuid,serial,memory.total,driver_version,power.limit,clocks.max.sm,pcie.link.gen.current,"
         "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")
    smi = run(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    uuids = re.findall(r"GPU-[0-9a-f-]{36}", smi)
    return {"gpu_query": q, "gpu": smi, "gpu_uuids": uuids, "env": env, "hostname": run("hostname"),
            "cpu_model": run("grep -m1 'model name' /proc/cpuinfo"), "nproc": run("nproc"),
            "memtotal": run("grep -m1 MemTotal /proc/meminfo"), "kernel": run("uname -r")}


def _snap(workdir, items, dest_dir):
    with _SNAP_LOCK:                 # review: the watcher and the final snapshot never share the temp file
        return _snap_locked(workdir, items, dest_dir)


def _snap_locked(workdir, items, dest_dir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n in items:
            p = os.path.join(workdir, n)
            if os.path.exists(p):
                tf.add(p, arcname=n)
    b = buf.getvalue()
    sha = hashlib.sha256(b).hexdigest()
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "snap.tar.gz.tmp"), "wb") as f:
        f.write(b)
    os.replace(os.path.join(dest_dir, "snap.tar.gz.tmp"), os.path.join(dest_dir, "snap.tar.gz"))
    with open(os.path.join(dest_dir, "snap.sha256"), "w") as f:
        f.write(sha + "\n")
    VOL.commit()
    return b, sha


@APP.function(image=IMAGE, gpu="RTX-PRO-6000", timeout=600, volumes={"/results": VOL}, retries=0)
def run_harness(harness_tar: bytes, spec: dict) -> dict:
    t_start = time.time()
    wd = spec["workdir"]
    items = spec.get("items") or ["rows", "logs"]
    dest = os.path.join("/results", spec["tag"])
    VOL.reload()
    prev = os.path.join(dest, "snap.tar.gz")
    if os.path.exists(prev):     # a preempted/retried input must never re-run a paid harness: hand back what exists
        b = open(prev, "rb").read()
        return {"rc": None, "err": "RERUN-REFUSED: a snapshot for this tag already exists (preemption or reuse)",
                "tar": b, "sha256": hashlib.sha256(b).hexdigest(), "identity": _identity(), "t_start": t_start,
                "t_end": time.time()}
    os.makedirs(wd, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(harness_tar), mode="r:gz") as tf:
        tf.extractall(wd)
    for d in spec.get("mkdirs") or ["rows", "logs"]:   # dirs only: an item may be a FILE (e.g. DONE); h1 launch 1
        os.makedirs(os.path.join(wd, d), exist_ok=True)  # made a DONE directory that would have blocked `echo > DONE`
    os.makedirs(os.path.join(wd, "logs"), exist_ok=True)  # review: logs/ always exists
    ident = _identity()
    with open(os.path.join(wd, "logs", "modal_identity.json"), "w") as f:
        json.dump(ident, f, indent=1)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    env["T99_RENT_EPOCH"] = env.get("T99_RENT_EPOCH") or str(t_start)
    for k in [k for k in env if k.endswith("_DEADLINE_REL")]:    # <X>_DEADLINE_REL (s) -> <X>_DEADLINE (epoch)
        env[k[:-4]] = str(t_start + float(env[k]))
    prep_log = []
    for c in spec.get("prep") or []:
        try:
            r = subprocess.run(c, shell=True, cwd=wd, env=env, capture_output=True, text=True, timeout=600)
            prc, pout = r.returncode, (r.stdout + r.stderr)[-3000:]
        except subprocess.TimeoutExpired:   # review: a hung prep is a prep failure with a snapshot, not a traceback
            prc, pout = "timeout", "prep timed out after 600 s"
        prep_log.append({"cmd": c, "rc": prc, "out": pout})
        if prc != 0:
            json.dump(prep_log, open(os.path.join(wd, "logs", "modal_prep.json"), "w"), indent=1)
            b, sha = _snap(wd, items, dest)
            return {"rc": None, "prep_failed": c, "prep": prep_log, "tar": b, "sha256": sha, "identity": ident,
                    "t_start": t_start, "t_end": time.time()}
    json.dump(prep_log, open(os.path.join(wd, "logs", "modal_prep.json"), "w"), indent=1)

    stop = threading.Event()

    def watcher():
        seen = set()
        last = 0.0
        while not stop.is_set():
            rows = os.path.join(wd, "rows")
            marks = {f for f in os.listdir(rows) if f.startswith("ARM_DONE_") or f == "DONE"} if os.path.isdir(rows) else set()
            if marks - seen or time.time() - last > 120:
                try:
                    _snap(wd, items, dest)
                except Exception as e:  # noqa: BLE001 — a failed snapshot must not kill the run; it is logged
                    with open(os.path.join(wd, "logs", "modal_snap_errors.txt"), "a") as f:
                        f.write(f"{time.time():.0f} {e!r}\n")
                seen |= marks
                last = time.time()
            stop.wait(5)

    th = threading.Thread(target=watcher, daemon=True)
    th.start()
    rc, err = None, None
    try:
        with open(os.path.join(wd, "logs", "modal_cmd.out"), "w") as out:
            r = subprocess.run(spec["cmd"], shell=True, cwd=wd, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=float(spec["cmd_timeout_s"]))
        rc = r.returncode
    except subprocess.TimeoutExpired:
        err = f"cmd timeout after {spec['cmd_timeout_s']} s"
    finally:
        stop.set()
        th.join(timeout=30)
    subprocess.run("nvidia-smi > logs/nvidia_smi_final.txt 2>&1", shell=True, cwd=wd)
    b, sha = _snap(wd, items, dest)
    return {"rc": rc, "err": err, "tar": b, "sha256": sha, "identity": ident, "t_start": t_start, "t_end": time.time()}


def _harness_tar(spec):
    excl = set(spec.get("exclude") or []) | {"__pycache__"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        hd = spec["harness_dir"]
        for n in sorted(os.listdir(hd)):
            if n in excl or (n.startswith("run_") and os.path.isdir(os.path.join(hd, n))):   # run_* result DIRS only:
                continue                                                                     # h1 lost run_ch19.sh
            tf.add(os.path.join(hd, n), arcname=n, filter=lambda ti: None if "__pycache__" in ti.name else ti)
        for p in spec.get("extra_files") or []:
            tf.add(p, arcname=os.path.basename(p))
    return buf.getvalue()


@APP.local_entrypoint()
def main(spec: str):
    s = json.load(open(spec))
    bad = [k for k, v in (s.get("env") or {}).items() if SECRETISH.search(k) or SECRET_VALUE.search(str(v))]
    if bad:
        raise SystemExit(f"REFUSED: secret-shaped env names in spec: {bad}")
    if float(s["cmd_timeout_s"]) > int(s["timeout_s"]) - 60:      # QA: compare against the int Modal enforces
        raise SystemExit("REFUSED: cmd_timeout_s must be <= int(timeout_s) - 60 (the snapshot and return need time)")
    tar = _harness_tar(s)
    os.makedirs(s["out"], exist_ok=True)
    meta = {"tag": s["tag"], "spec": spec, "app_id": APP.app_id, "t_submit": time.time(),
            "harness_tar_sha256": hashlib.sha256(tar).hexdigest(), "gpu": s["gpu"], "cpu": s["cpu"],
            "mem_mib": s["mem_mib"], "mem_limit_mib": s["mem_limit_mib"], "timeout_s": s["timeout_s"]}
    json.dump(meta, open(os.path.join(s["out"], s["tag"] + ".meta.json"), "w"), indent=1)
    print(f"[modal_run] app {APP.app_id} tag {s['tag']} gpu {s['gpu']} timeout {s['timeout_s']} s", flush=True)
    opts = dict(gpu=s["gpu"], cpu=(float(s["cpu"]), float(s["cpu"])),
                memory=(int(s["mem_mib"]), int(s["mem_limit_mib"])), timeout=int(s["timeout_s"]))
    if s.get("region"):              # a pinned region costs 1.15x (broad) / 1.75x (narrow): the spec's cap must say so
        opts["region"] = s["region"]
    fn = run_harness.with_options(**opts)
    res = fn.remote(tar, s)
    b = res.pop("tar")
    got = hashlib.sha256(b).hexdigest()
    path = os.path.join(s["out"], s["tag"] + ".tar.gz")
    with open(path, "wb") as f:
        f.write(b)
    meta.update({k: v for k, v in res.items()}, t_return=time.time(), local_sha256=got, sha_ok=(got == res["sha256"]))
    json.dump(meta, open(os.path.join(s["out"], s["tag"] + ".meta.json"), "w"), indent=1)
    if got != res["sha256"]:
        raise SystemExit(f"SHA MISMATCH: local {got} vs container {res['sha256']}")
    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        unsafe = [m.name for m in members if m.name.startswith("/") or ".." in m.name.split("/") or m.issym() or m.islnk()]
        if unsafe:
            raise SystemExit(f"REFUSED to extract unsafe members: {unsafe[:5]}")
        tf.extractall(os.path.join(s["out"], s["tag"]), members=members)
    print(json.dumps({k: meta[k] for k in ("tag", "app_id", "rc", "sha_ok") if k in meta} |
                     {"err": res.get("err"), "prep_failed": res.get("prep_failed"),
                      "wall_s": round(res["t_end"] - res["t_start"], 1)}), flush=True)
    if res.get("rc") != 0 or res.get("err") or res.get("prep_failed"):   # QA deepseek FATAL: the contract says raise
        raise SystemExit(f"HARNESS-FAILED rc={res.get('rc')} err={res.get('err')} prep_failed={res.get('prep_failed')} "
                         f"(results are saved and extracted above)")
