"""s0_probe.py -- S0-0, the platform gate (prereg §2): can a SECOND process open a holder's legacy CUDA IPC buffer on the
other GPU and copy through it (no vLLM)? Opens the handle with the ext's own CudaRtBackend (lazy peer access), copies a
local buffer -> peer -> a second local buffer, and compares the pws1 digests of the two local buffers.

Output contract: exactly one JSON object on stdout; exit 0 = PASS, 1 = FAIL (the JSON says where), 2 = bad usage.
Usage: CUDA_VISIBLE_DEVICES=0,1 python3 s0_probe.py --handle <b64> --nbytes N [--stub-fabric DIR --stub-cvd 0,1]
"""
import argparse
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warm_wake_ext as ext  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--handle", required=True)
    p.add_argument("--nbytes", type=int, required=True)
    p.add_argument("--stub-fabric", default=None)
    p.add_argument("--stub-cvd", default="0,1")
    a = p.parse_args()
    out = {"ok": False, "nbytes": a.nbytes}
    if a.nbytes <= 0 or a.nbytes % 4:                   # QA pair: a ragged size would overrun the int32 buffers
        print(json.dumps({"ok": False, "stage": "usage", "err": "nbytes must be a positive multiple of 4"}))
        return 2
    try:
        h = base64.b64decode(a.handle, validate=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "stage": "decode", "err": repr(e)}))
        return 2
    try:
        be, src, dst, keep = _setup(a)
    except Exception as e:  # noqa: BLE001  review: a setup failure still prints the one JSON object
        print(json.dumps({"ok": False, "stage": "setup", "err": f"{type(e).__name__}: {e}"[:300]}))
        return 1
    return _run(a, out, h, be, src, dst, keep)


def _setup(a):
    if a.stub_fabric:
        stub = ext.load_stub()
        be = stub.StubBackend(stub.StubFabric.open(a.stub_fabric), [int(x) for x in a.stub_cvd.split(",")],
                              f"probe-pid{os.getpid()}", ext.chunk_sums_numpy)
        src = be.malloc(0, a.nbytes)
        be.poke_bytes(src, bytes((7 * j + 3) % 256 for j in range(min(a.nbytes, 4096))))
        dst = be.malloc(0, a.nbytes)
        keep = None
    else:
        import torch
        be = ext.CudaRtBackend()
        t = torch.randint(-2 ** 31, 2 ** 31 - 1, (a.nbytes // 4,), dtype=torch.int32, device="cuda:0")
        d = torch.empty_like(t)
        torch.cuda.synchronize()
        src, dst, keep = t.data_ptr(), d.data_ptr(), (t, d)
    return be, src, dst, keep


def _run(a, out, h, be, src, dst, keep):
    stage = "ipc_open"
    try:
        t0 = time.perf_counter()
        g = be.ipc_open(0, h)
        out["open_s"] = round(time.perf_counter() - t0, 6)
        stage = "copy_to_peer"
        t1 = time.perf_counter()
        be.memcpy(0, g, src, a.nbytes)
        rc = be.synchronize_rc(0)
        out["to_peer_s"] = round(time.perf_counter() - t1, 6)
        if rc:
            raise ext.CudaError("sync", rc)
        stage = "copy_from_peer"
        t2 = time.perf_counter()
        be.memcpy(0, dst, g, a.nbytes)
        rc = be.synchronize_rc(0)
        out["from_peer_s"] = round(time.perf_counter() - t2, 6)
        if rc:
            raise ext.CudaError("sync", rc)
        stage = "digest"
        ds = ext.digest_from_sums(a.nbytes, be.chunk_sums(0, src, a.nbytes, "device"))
        dd = ext.digest_from_sums(a.nbytes, be.chunk_sums(0, dst, a.nbytes, "device"))
        out.update(digest_src=ds, digest_back=dd, match=(ds == dd),
                   to_peer_gbs=round(a.nbytes / out["to_peer_s"] / 1e9, 3) if out["to_peer_s"] > 0 else None,
                   from_peer_gbs=round(a.nbytes / out["from_peer_s"] / 1e9, 3) if out["from_peer_s"] > 0 else None)
        stage = "close"
        be.ipc_close(0, g)
        out["ok"] = bool(out["match"])
    except Exception as e:  # noqa: BLE001
        out.update(stage=stage, err=f"{type(e).__name__}: {e}"[:300])
        try:
            out["last_error_after"] = be.clear_last_error()
        except Exception:  # noqa: BLE001
            pass
    if keep is None:                    # stub fabric: a dead process's allocations are never reclaimed (real CUDA
        for ptr in (src, dst):          # frees them at exit), so the probe frees its own (rehearsal 1 leaked 8 MiB)
            try:
                be.free(0, ptr)
            except Exception:  # noqa: BLE001
                pass
    del keep
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
