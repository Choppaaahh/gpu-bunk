#!/usr/bin/env python3
"""warm_wake_ext.py -- the WARM engine-level wake (bite 1: B1 worker extension + B2 guest holder).

WHAT. A sleeping vLLM 0.30.0 engine restores its weights from a copy parked in a NEIGHBOUR GPU's memory (NVLink peer copy)
instead of paging them in from its pinned host backup. Stock vLLM, no core patch:
  * `--worker-extension-cls warm_wake_ext.WarmWakeExt` mixes `WarmWakeExt` into the worker
    (v1/worker/worker_base.py L285-311 @ v0.30.0; L292-299 assert that NO non-dunder name collides, so the class carries
    only `bb_*` methods and every helper lives at module level);
  * the controller calls them over `POST /collective_rpc` (serve/dev/rpc/api_router.py L23; same VLLM_SERVER_DEV_MODE gate
    as /sleep, serve/__init__.py L43-69). Args arrive as STRINGS only (api_router L38-41), so every method takes one JSON
    string; a dict return passes through as JSON (L50-53);
  * then the stock partial wake `POST /wake_up?tags=weights` (now a no-op: nothing of that tag is asleep) and
    `POST /wake_up?tags=kv_cache` (v1/executor/abstract.py L350-375; the scheduler resumes only when both tags are resident,
    v1/engine/core.py L917-933).

THE RESTORE (bb_restore_from_peer). For each weights-tag allocation, in the allocator's own order: `create_and_map(handle)`
(the stock C call, same VA) -> one copy from the guest buffer (CUDA IPC handle exported by the guest holder on another GPU)
-> device sync -> verify every allocation's `pws1` digest against the digests THIS engine computed at push time -> COMMIT
(`is_asleep=False`, host backup dropped, exactly what stock wake_up does, cumem.py L341-356) or ROLL BACK (unmap every mapped
allocation; the allocator is then byte-for-byte stock-asleep with every host backup intact, so the next stock
`/wake_up?tags=weights` pages from host = COOL). Every failure path clears the CUDA last-error first an error that
survives a device sync (sticky) is reported as `served: NONE` (restart the engine), never as a host fallback.

WHY THE ENGINE DOES EVERY COPY. On A100 the cumem allocator requests no exportable handle type (csrc/cumem_allocator.cpp
L143-149: FABRIC only where supported; POSIX-FD only as the fabric fallback) and grants access to the owning device only
(L206-211). Nobody but the engine's own process can touch its weights, so it PUSHes at park and PULLs at wake; the holder
only owns the guest buffers (plain cudaMalloc, legacy IPC-exportable).

Output contract:
  --self-test     success = exit 0 + one `ok  <golden>` line per check + `SELFTEST-OK <n>/<n> (<s> skipped)`;
                  failure = exit 1 + `FAIL <golden> - <detail>` lines + `SELFTEST-FAIL <k>/<n>`. A SKIP is printed and
                  counted as not run, never as a pass. Writes only under a tempdir (/dev/shm when present); starts
                  loopback HTTP servers and Python children; never imports vllm or touches a GPU.
  --t3            exit 0 + one JSON object on stdout (and in --out when given, which it WRITES); exit 1 + `T3-FAIL ...` on
                  stderr. Stub-only: a full-size layout/CAP pass with no bytes, and a byte pass at 1/--scale.
  --serve-holder  runs a guest holder HTTP server until killed; prints `BB-HOLDER-READY port=<n>` on stdout when up.
                  Real mode needs torch + a GPU; `--stub-fabric DIR` runs it on the CPU stub instead.
  bb_* methods    never raise: each returns a dict with `ok`, and the restore also returns `served` in
                  {PEER, HOST_FALLBACK, NONE} + `reason`.

Top pitfalls (observed):
  1. **A ctypes CUDA call that fails leaves the runtime's last-error set, and torch raises it on its next kernel launch**
     (: a 704 from cudaDeviceEnablePeerAccess failed the next read in both runs) -- every failure path here calls
     cudaGetLastError before returning.
  2. **A guard that raises out of sleep/wake desyncs the engine** (: abstract.py sets is_sleeping only after the worker
     returns) -- nothing here raises out of a bb_* method, and nothing here runs inside stock sleep or wake_up.
  3. **A stub that prints the parser's expected format proves nothing** -- the stub engine copies its log lines
     from real v0.30.0 logs ((private report) (private report)).

Usage:
  python3 scripts/warm_wake_ext.py --self-test
  python3 scripts/warm_wake_ext.py --t3 --scale 16 --out (private report)
  CUDA_VISIBLE_DEVICES=5,0,1,2,3,4,6,7 python3 scripts/warm_wake_ext.py --serve-holder --port 8305 --cap-gib 35.62
  VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=2,0,1,3,4,5,6,7 PYTHONPATH=scripts vllm serve ... --enable-sleep-mode \\
      --worker-extension-cls warm_wake_ext.WarmWakeExt
"""

import argparse
import base64
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EXT_VERSION = "bb-warm-1"
DEFAULT_CHUNK_WORDS = 262144          # 1 MiB of int32 words per digest chunk
MAX_CHUNK_WORDS = 1 << 24             # keeps every chunk sum inside int64 (2^24 words x 2^31 < 2^63)
IPC_HANDLE_SIZE = 64                  # CUDA_IPC_HANDLE_SIZE (driver_types.h); vLLM's own wrapper declares 128
MASK64 = (1 << 64) - 1
SERVED = ("PEER", "HOST_FALLBACK", "NONE")
MAX_STAGE_ROUNDS = 1 << 20            # bite-2 H3: a staged copy needing more rounds is refused before any copy
GIB = 1 << 30


# ------------------------------------------------------------------------------------------------ the pws1 digest
def fold_chunk_sums(sums):
    """s1 = sum c_k mod 2^64; s2 = sum (k+1)*c_k mod 2^64 (exact Python ints, negatives wrap)."""
    s1 = 0
    s2 = 0
    for k, c in enumerate(sums):
        s1 += int(c)
        s2 += (k + 1) * int(c)
    return s1 & MASK64, s2 & MASK64


def digest_from_sums(nbytes, sums):
    s1, s2 = fold_chunk_sums(sums)
    return f"pws1:{int(nbytes)}:{s1:016x}:{s2:016x}"


def _check_chunk_words(chunk_words):
    cw = int(chunk_words)
    if cw < 1 or cw > MAX_CHUNK_WORDS:
        raise ValueError(f"chunk_words {cw} outside 1..{MAX_CHUNK_WORDS}")
    return cw


def chunk_sums_numpy(buf, chunk_words=DEFAULT_CHUNK_WORDS):
    """Per-chunk sums MOD 2^32 (as signed int32) of the little-endian int32 words of `buf`, zero-padded to a multiple of
    4 bytes. Mod 2^32, not exact (amendment A1, 2026-10-21): an exact sum on the GPU needs `sum(dtype=int64)`, which
    casts the whole input first (a 3.1 GB transient for one 1.56 GB embedding); an int32 sum is one pass, no copy."""
    import numpy as np
    cw = _check_chunk_words(chunk_words)
    a = np.frombuffer(buf, dtype=np.uint8)
    pad = (-a.size) % 4
    if pad:
        b = np.zeros(a.size + pad, dtype=np.uint8)
        b[: a.size] = a
        a = b
    words = a.view("<i4")
    nw = words.size
    full = nw // cw
    out = []
    if full:
        out = words[: full * cw].reshape(full, cw).sum(axis=1, dtype=np.int32).tolist()
    if nw % cw:
        out.append(int(words[full * cw:].sum(dtype=np.int32)))
    return [int(x) for x in out]


def chunk_sums_torch(t, chunk_words=DEFAULT_CHUNK_WORDS):
    """The same function on a 1-D int32 torch tensor (GPU in production). dtype=int32 keeps the reduction in place: the
    result wraps mod 2^32 like numpy's. Agreement with numpy is G1e/G1f on CPU and `--s0-digest` on CUDA (S0 item)."""
    import torch
    cw = _check_chunk_words(chunk_words)
    t = t.reshape(-1)
    if t.dtype != torch.int32:
        raise ValueError(f"expected int32, got {t.dtype}")
    nw = t.numel()
    full = nw // cw
    out = []
    if full:
        out = t[: full * cw].view(full, cw).sum(dim=1, dtype=torch.int32).cpu().tolist()
    if nw % cw:
        out.append(int(t[full * cw:].sum(dtype=torch.int32).item()))
    return [int(x) for x in out]


def digest_bytes(buf, chunk_words=DEFAULT_CHUNK_WORDS):
    return digest_from_sums(len(memoryview(buf).cast("B")), chunk_sums_numpy(buf, chunk_words))


# ------------------------------------------------------------------------------------------------ device backends
class CudaError(RuntimeError):
    def __init__(self, op, code, detail=""):
        super().__init__(f"{op}: rc={int(code)} {detail}".strip())
        self.op, self.code = op, int(code)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * IPC_HANDLE_SIZE)]


class _CAI:
    """A __cuda_array_interface__ view of raw device memory (int32 words), for the torch digest. UNPROVEN on a cumem VMM
    pointer until S0 runs it (torch reads the pointer's device through cudaPointerGetAttributes)."""

    def __init__(self, ptr, nwords):
        self.__cuda_array_interface__ = {"shape": (int(nwords),), "typestr": "<i4", "data": (int(ptr), False),
                                         "version": 3, "strides": None}


class CudaRtBackend:
    """The CUDA runtime that torch already loaded (found in /proc/self/maps, as bind_cudart). Device ids are
    LOGICAL (CUDA_VISIBLE_DEVICES order). vLLM's CudaRTLibrary has no cudaGetLastError and no cudaIpcCloseMemHandle, so
    this is its own small binding to the same library."""

    kind = "cuda"

    def __init__(self, lib_path=None):
        self.lib_path = lib_path or self._find()
        lib = ctypes.CDLL(self.lib_path)
        c_int, c_void_p, c_size_t = ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t

        def fn(name, res, args):
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
            return f

        self._get = fn("cudaGetDevice", c_int, [ctypes.POINTER(c_int)])
        self._set = fn("cudaSetDevice", c_int, [c_int])
        self._malloc = fn("cudaMalloc", c_int, [ctypes.POINTER(c_void_p), c_size_t])
        self._free = fn("cudaFree", c_int, [c_void_p])
        self._memcpy = fn("cudaMemcpy", c_int, [c_void_p, c_void_p, c_size_t, c_int])
        self._ipc_get = fn("cudaIpcGetMemHandle", c_int, [ctypes.POINTER(_IpcHandle), c_void_p])
        self._ipc_open = fn("cudaIpcOpenMemHandle", c_int, [ctypes.POINTER(c_void_p), _IpcHandle, ctypes.c_uint])
        self._ipc_close = fn("cudaIpcCloseMemHandle", c_int, [c_void_p])
        self._sync = fn("cudaDeviceSynchronize", c_int, [])
        self._last = fn("cudaGetLastError", c_int, [])
        self._peek = fn("cudaPeekAtLastError", c_int, [])
        self._err = fn("cudaGetErrorString", ctypes.c_char_p, [c_int])

    @staticmethod
    def _scan():
        out = []
        try:
            with open("/proc/self/maps") as fh:
                for ln in fh:
                    parts = ln.split()
                    if len(parts) >= 6 and "libcudart.so" in parts[-1] and parts[-1] not in out:
                        out.append(parts[-1])
        except OSError:
            pass
        return out

    @classmethod
    def _find(cls):
        paths = cls._scan()
        if not paths:
            import torch  # noqa: (loads the runtime torch was built against)
            torch.cuda.init()
            paths = cls._scan()
        if not paths:
            raise RuntimeError("no libcudart.so loaded in this process")
        return paths[0]

    def _chk(self, op, rc):
        if rc != 0:
            msg = self._err(rc)
            raise CudaError(op, rc, msg.decode(errors="replace") if msg else "")

    @contextmanager
    def _on(self, dev):
        cur = ctypes.c_int()
        self._chk("cudaGetDevice", self._get(ctypes.byref(cur)))
        if cur.value != dev:
            self._chk("cudaSetDevice", self._set(int(dev)))
        try:
            yield
        finally:
            if cur.value != dev:
                self._set(cur.value)

    def malloc(self, dev, n):
        p = ctypes.c_void_p()
        with self._on(dev):
            self._chk("cudaMalloc", self._malloc(ctypes.byref(p), int(n)))
        return int(p.value)

    def free(self, dev, ptr):
        with self._on(dev):
            self._chk("cudaFree", self._free(ctypes.c_void_p(int(ptr))))

    def ipc_export(self, dev, ptr):
        h = _IpcHandle()
        with self._on(dev):
            self._chk("cudaIpcGetMemHandle", self._ipc_get(ctypes.byref(h), ctypes.c_void_p(int(ptr))))
        return ctypes.string_at(ctypes.addressof(h), IPC_HANDLE_SIZE)

    def ipc_open(self, dev, handle):
        if len(handle) != IPC_HANDLE_SIZE:
            raise CudaError("cudaIpcOpenMemHandle", 1, f"handle is {len(handle)} bytes")
        h = _IpcHandle()
        ctypes.memmove(ctypes.addressof(h), bytes(handle), IPC_HANDLE_SIZE)
        p = ctypes.c_void_p()
        with self._on(dev):   # cudaIpcMemLazyEnablePeerAccess = 1: no separate EnablePeerAccess call ('s 704 path)
            self._chk("cudaIpcOpenMemHandle", self._ipc_open(ctypes.byref(p), h, 1))
        return int(p.value)

    def ipc_close(self, dev, ptr):
        with self._on(dev):
            self._chk("cudaIpcCloseMemHandle", self._ipc_close(ctypes.c_void_p(int(ptr))))

    def memcpy(self, dev, dst, src, n):
        with self._on(dev):   # cudaMemcpyDefault (UVA); the engine's own device does the copy
            self._chk("cudaMemcpy", self._memcpy(ctypes.c_void_p(int(dst)), ctypes.c_void_p(int(src)), int(n), 4))

    def flip_byte(self, dev, ptr, xor=0xFF):
        """Test-only (holder --allow-corrupt): XOR one device byte through a 1-byte host buffer (pageable host memory
        under cudaMemcpyDefault: UNPROVEN until S0 runs it)."""
        b = (ctypes.c_ubyte * 1)()
        self.memcpy(dev, ctypes.addressof(b), ptr, 1)
        old = int(b[0])
        b[0] = old ^ (int(xor) & 0xFF)
        self.memcpy(dev, ptr, ctypes.addressof(b), 1)
        return old, int(b[0])

    def synchronize_rc(self, dev):
        try:
            with self._on(dev):
                return int(self._sync())
        except CudaError as e:
            return e.code

    def clear_last_error(self):
        return int(self._last())

    def peek_last_error(self):
        return int(self._peek())

    def chunk_sums(self, dev, ptr, nbytes, where, chunk_words=DEFAULT_CHUNK_WORDS):
        nbytes = int(nbytes)
        if where == "host":
            import numpy as np
            arr = np.frombuffer((ctypes.c_char * nbytes).from_address(int(ptr)), dtype=np.uint8)
            return chunk_sums_numpy(arr, chunk_words)
        if nbytes % 4:                  # cumem sizes are 2 MiB multiples; a ragged device size is a caller bug
            raise CudaError("digest", 1, "device digest needs nbytes % 4 == 0")
        import torch
        with self._on(dev):
            t = torch.as_tensor(_CAI(ptr, nbytes // 4), device=torch.device("cuda", int(dev)))
            sums = chunk_sums_torch(t, chunk_words)
            torch.cuda.synchronize(int(dev))
        return sums


# ------------------------------------------------------------------------------------------------ the allocator view
def h_dev(d):
    return int(d.handle[0])


def h_size(d):
    return int(d.handle[1])


def h_ptr(d):
    return int(d.handle[2])


def backup_of(d):
    """(host ptr, nbytes) of an allocation's host backup, or None. Same fields stock cumem.py reads (L348-355)."""
    t = getattr(d, "cpu_backup_tensor", None)
    if t is None:
        return None
    return int(t.data_ptr()), int(t.numel()) * int(t.element_size())


class AllocatorView:
    """Duck-typed view of vLLM's CuMemAllocator (or the stub's transliteration): `pointer_to_data` {ptr: AllocationData
    (handle=(dev, size, ptr, phandle), tag, cpu_backup_tensor, is_asleep)} plus the module-level create_and_map /
    unmap_and_release the stock wake/sleep call."""

    def __init__(self, allocator, create_and_map, unmap_and_release, label="vllm"):
        self.allocator, self.label = allocator, label
        self.create_and_map, self.unmap_and_release = create_and_map, unmap_and_release

    def shape_problem(self):
        p2d = getattr(self.allocator, "pointer_to_data", None)
        if not isinstance(p2d, dict):
            return "no pointer_to_data dict"
        for ptr, d in list(p2d.items()):          # every allocation (bite-2 H1; was the first 4 only)
            for f in ("handle", "tag", "is_asleep", "cpu_backup_tensor"):
                if not hasattr(d, f):
                    return f"AllocationData has no {f}"
            if not (isinstance(d.handle, tuple) and len(d.handle) == 4 and int(d.handle[2]) == int(ptr)):
                return "handle is not (dev, size, ptr, phandle) keyed by ptr"
        return None

    def entries(self, tags=None):
        return [(int(p), d) for p, d in list(self.allocator.pointer_to_data.items()) if tags is None or d.tag in tags]

    def get(self, ptr):
        return self.allocator.pointer_to_data.get(ptr)


def vllm_ctx(worker):
    """The production context: vLLM's CuMemAllocator singleton + the loaded libcudart. Imported lazily so this file
    loads, and self-tests, with no vLLM present."""
    from vllm.device_allocator import cumem
    try:
        import vllm
        ver = getattr(vllm, "__version__", "?")
    except Exception:  # noqa: BLE001
        ver = "?"
    view = AllocatorView(cumem.CuMemAllocator.get_instance(), lambda h: cumem.create_and_map(h),
                         lambda h: cumem.unmap_and_release(h), label=f"vllm-{ver}")
    return WarmCtx(view, CudaRtBackend(), cvd=os.environ.get("CUDA_VISIBLE_DEVICES", ""), log=_stderr_log)


def _stderr_log(ev):
    print("BB-WARM " + json.dumps(ev, sort_keys=True, default=str), file=sys.stderr, flush=True)


_CTX_FACTORY = None


def set_ctx_factory(fn):
    """Dependency injection for stubs: fn(worker) -> WarmCtx. None = production (vllm_ctx)."""
    global _CTX_FACTORY
    _CTX_FACTORY = fn


class WarmCtx:
    def __init__(self, view, backend, cvd="", log=None):
        self.view, self.backend, self.cvd = view, backend, cvd
        self.log = log or (lambda ev: None)
        self.boot_nonce = uuid.uuid4().hex[:16]
        self.manifests = {}
        self.lock = threading.Lock()
        self.counts = {"push": 0, "restore_peer": 0, "restore_fallback": 0, "restore_none": 0, "seal": 0}


def _ctx_for(worker):
    ctx = worker.__dict__.get("_bb_ctx")
    if ctx is None:
        ctx = (_CTX_FACTORY or vllm_ctx)(worker)
        worker.__dict__["_bb_ctx"] = ctx
    return ctx


def _parse(spec):
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return spec
    return json.loads(spec)


def _call(worker, op, spec):
    t0 = time.perf_counter()
    try:
        s = _parse(spec)
        ctx = _ctx_for(worker)
        prob = ctx.view.shape_problem()
        if prob and op in ("push", "restore", "seal"):
            out = {"ok": False, "reason": f"ALLOCATOR_SHAPE:{prob}"}
            if op == "restore":
                out["served"] = "HOST_FALLBACK"
        else:
            with ctx.lock:
                out = OPS[op](ctx, s)
        out.setdefault("op", op)
        out["ext"] = EXT_VERSION
        out["wall_s"] = round(time.perf_counter() - t0, 6)
        try:
            ctx.log({k: v for k, v in out.items() if k not in ("manifest",)})
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception as e:  # noqa: BLE001 -- never raise out of collective_rpc ('s spirit)
        out = {"ok": False, "op": op, "reason": f"INTERNAL:{type(e).__name__}:{str(e)[:200]}", "ext": EXT_VERSION}
        if op == "restore":
            out["served"] = "HOST_FALLBACK"   # only reached before any mapping: ctx_restore traps its own errors
        return out


# ------------------------------------------------------------------------------------------------ engine-side ops
def _layout(entries):
    layout, offsets, off = [], [], 0
    for ptr, d in entries:
        layout.append([ptr, h_size(d), d.tag])
        offsets.append(off)
        off += h_size(d)
    return layout, offsets, off


def _own_dev(entries):
    devs = sorted({h_dev(d) for _, d in entries})
    return devs[0] if len(devs) == 1 else None


def ctx_inventory(ctx, spec):
    ents = ctx.view.entries()
    by = {}
    for ptr, d in ents:
        t = by.setdefault(d.tag, {"n": 0, "bytes": 0, "asleep": 0, "host_backups": 0, "sizes": []})
        t["n"] += 1
        t["bytes"] += h_size(d)
        t["asleep"] += int(bool(d.is_asleep))
        t["host_backups"] += int(backup_of(d) is not None)
        t["sizes"].append(h_size(d))
    for t in by.values():
        t["layout_sha"] = hashlib.sha256(json.dumps(t.pop("sizes")).encode()).hexdigest()[:16]
    return {"ok": True, "boot_nonce": ctx.boot_nonce, "cvd": ctx.cvd, "device": _own_dev(ents) if ents else None,
            "tags": by, "allocator": ctx.view.label, "shape_problem": ctx.view.shape_problem(),
            "guests": sorted(ctx.manifests)}


def ctx_status(ctx, spec):
    return {"ok": True, "boot_nonce": ctx.boot_nonce, "counts": dict(ctx.counts),
            "manifests": {g: {"manifest_id": m["manifest_id"], "total": m["total"], "stale": m["stale"],
                              "n_alloc": len(m["layout"])} for g, m in ctx.manifests.items()}}


def ctx_drop(ctx, spec):
    gid = spec.get("guest_id")
    return {"ok": True, "guest_id": gid, "dropped": ctx.manifests.pop(gid, None) is not None}


def _guest_fields(spec):
    g = spec.get("guest") or {}
    gid = g.get("guest_id")
    nbytes = g.get("nbytes")
    try:
        handle = base64.b64decode(g.get("ipc_handle") or "", validate=True)
    except Exception:  # noqa: BLE001
        handle = b""
    return g, gid, nbytes, handle


def ctx_push(ctx, spec):
    """Copy every allocation of `tags` (default weights) into the guest buffer, packed in allocator order; record the
    per-allocation pws1 digests of the SOURCE as this engine's manifest. Requires the engine awake. Changes no engine
    state. A failed push leaves no manifest (the engine is simply COOL)."""
    g, gid, nbytes, handle = _guest_fields(spec)
    tags = list(spec.get("tags") or ["weights"])
    cw = int(spec.get("chunk_words") or DEFAULT_CHUNK_WORDS)
    base = {"op": "push", "guest_id": gid}
    if not gid or len(handle) != IPC_HANDLE_SIZE or not isinstance(nbytes, int) or isinstance(nbytes, bool):
        return {**base, "ok": False, "reason": "BAD_SPEC:guest"}
    copy_mode = spec.get("copy_mode", "direct")
    stage, serr = _parse_stage(spec)
    if copy_mode not in ("direct", "staged") or serr:
        return {**base, "ok": False, "reason": serr or "BAD_SPEC:mode"}
    _check_chunk_words(cw)
    ents = ctx.view.entries(tags)
    if not ents:
        return {**base, "ok": False, "reason": "NO_ALLOCATIONS"}
    if any(d.is_asleep for _, d in ents):
        return {**base, "ok": False, "reason": "ENGINE_ASLEEP"}
    dev = _own_dev(ents)
    if dev is None:
        return {**base, "ok": False, "reason": "MULTI_DEVICE"}
    layout, offsets, total = _layout(ents)
    if nbytes < total:
        return {**base, "ok": False, "reason": "GUEST_TOO_SMALL", "total": total, "nbytes": nbytes}
    if copy_mode == "staged" and _stage_rounds([row[1] for row in layout], stage) > MAX_STAGE_ROUNDS:
        return {**base, "ok": False, "reason": "BAD_SPEC:stage_rounds"}
    be = ctx.backend
    t0 = time.perf_counter()
    try:
        gptr = be.ipc_open(dev, handle)
    except Exception as e:  # noqa: BLE001
        be.clear_last_error()
        return {**base, "ok": False, "reason": f"IPC_OPEN:{e}"[:240]}
    t_open = time.perf_counter()
    digests = []
    err = None
    scratch = None
    try:
        if copy_mode == "staged":
            scratch = be.malloc(dev, stage)
        for (ptr, size, _tag), off in zip(layout, offsets):
            _copy(be, dev, gptr + off, ptr, size, scratch, stage)
        rc = be.synchronize_rc(dev)
        if rc:
            raise CudaError("sync", rc)
        t_copy = time.perf_counter()
        for ptr, size, _tag in layout:
            digests.append(digest_from_sums(size, be.chunk_sums(dev, ptr, size, "device", cw)))
        t_dig = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        err = e
    finally:
        try:
            be.ipc_close(dev, gptr)
        except Exception:  # noqa: BLE001
            be.clear_last_error()
        if scratch is not None:
            try:
                be.free(dev, scratch)
            except Exception:  # noqa: BLE001
                be.clear_last_error()
    if err is not None:
        be.clear_last_error()
        probe = be.synchronize_rc(dev)
        be.clear_last_error()
        return {**base, "ok": False, "reason": f"PUSH_FAILED:{err}"[:240], "poisoned": bool(probe), "probe_rc": probe}
    mid = hashlib.sha256(json.dumps([ctx.boot_nonce, gid, layout, digests]).encode()).hexdigest()[:16]
    replaced = gid in ctx.manifests
    m = {"manifest_id": mid, "guest_id": gid, "owner_id": spec.get("owner_id", ""), "boot_nonce": ctx.boot_nonce,
         "tags": tags, "chunk_words": cw, "layout": layout, "offsets": offsets, "total": total, "digests": digests,
         "pushed_unix": time.time(), "stale": False, "stale_why": None}
    ctx.manifests[gid] = m
    ctx.counts["push"] += 1
    copy_s = t_copy - t_open
    return {**base, "ok": True, "manifest_id": mid, "total": total, "n_alloc": len(layout), "offsets": offsets,
            "replaced": replaced, "copy_mode": copy_mode, "open_s": round(t_open - t0, 6), "copy_s": round(copy_s, 6),
            "digest_s": round(t_dig - t_copy, 6), "rate_gbs": round(total / copy_s / 1e9, 3) if copy_s > 0 else None,
            "manifest": m}


def ctx_seal(ctx, spec):
    """After the stock /sleep: compare the host backups' digests with the push digests. A byte that changed between the
    push and the sleep makes the guest STALE (its restore will refuse; the host copy, which is newer, serves)."""
    gid = spec.get("guest_id")
    m = ctx.manifests.get(gid)
    if m is None:
        return {"ok": False, "guest_id": gid, "reason": "NO_MANIFEST"}
    ents = ctx.view.entries(m["tags"])
    have = [[p, h_size(d), d.tag] for p, d in ents]
    if have != m["layout"]:
        m["stale"], m["stale_why"] = True, "LAYOUT_MISMATCH"
        return {"ok": True, "guest_id": gid, "stale": True, "reason": "LAYOUT_MISMATCH", "mismatched": []}
    if not all(d.is_asleep for _, d in ents):
        return {"ok": False, "guest_id": gid, "reason": "NOT_ASLEEP"}
    t0 = time.perf_counter()
    bad, missing = [], []
    dev = _own_dev(ents)
    for i, (ptr, d) in enumerate(ents):
        b = backup_of(d)
        if b is None or b[1] != h_size(d):
            missing.append(i)
            continue
        if digest_from_sums(h_size(d), ctx.backend.chunk_sums(dev, b[0], b[1], "host", m["chunk_words"])) != m["digests"][i]:
            bad.append(i)
    stale = bool(bad or missing)
    if stale:
        m["stale"], m["stale_why"] = True, ("HOST_DIGEST_MISMATCH" if bad else "NO_HOST_BACKUP")
    ctx.counts["seal"] += 1
    return {"ok": True, "guest_id": gid, "stale": stale, "mismatched": bad, "missing_backup": missing,
            "seconds": round(time.perf_counter() - t0, 6)}


def _parse_stage(spec):
    """(stage_bytes, error). Missing -> 256 MiB; an explicit value below 4096 (or not an int) is refused, never
    replaced by the default (QA 2026-10-21: a 0 stage made the staged loop spin forever)."""
    try:
        stage = spec.get("stage_bytes")
        stage = (256 << 20) if stage is None else int(stage)
    except (TypeError, ValueError):
        return 0, "BAD_SPEC:stage_bytes"
    if stage < 4096:
        return 0, "BAD_SPEC:stage_bytes"
    return stage, None


def _stage_rounds(sizes, stage):
    return sum(-(-int(n) // int(stage)) for n in sizes)


def _copy(be, dev, dst, src, size, scratch=None, stage=0):
    """direct: one cudaMemcpy. staged: through a plain cudaMalloc scratch on the engine's own device -- the fallback if
    S0 shows a cumem VMM allocation (access granted to its own device only) cannot be copied to/from a peer mapping."""
    if scratch is None:
        be.memcpy(dev, dst, src, size)
        return
    if size < 0 or stage < 1 or -(-size // stage) > MAX_STAGE_ROUNDS:
        raise ValueError(f"staged copy refused: size {size} stage {stage}")
    o = 0
    while o < size:
        c = min(stage, size - o)
        be.memcpy(dev, scratch, src + o, c)
        be.memcpy(dev, dst + o, scratch, c)
        o += c


class _Fail(Exception):
    def __init__(self, reason, **kw):
        super().__init__(reason)
        self.reason, self.kw = reason, kw


def ctx_restore(ctx, spec):
    """WARM restore of a sleeping engine from its guest. Precondition checks have no side effects. See the module
    docstring for the commit / roll-back contract."""
    g, gid, nbytes, handle = _guest_fields(spec)
    base = {"op": "restore", "guest_id": gid, "bytes": 0}

    def fb(reason, **kw):
        ctx.counts["restore_fallback"] += 1
        return {**base, "ok": False, "served": "HOST_FALLBACK", "reason": reason, **kw}

    m = ctx.manifests.get(gid)
    if m is None:
        return fb("NO_MANIFEST")
    if m["stale"]:
        return fb("MANIFEST_STALE", stale_why=m["stale_why"])
    if m["boot_nonce"] != ctx.boot_nonce:
        return fb("BOOT_NONCE_MISMATCH")
    if g.get("manifest_id") not in (None, m["manifest_id"]):
        return fb("MANIFEST_ID_MISMATCH")
    tags = list(spec.get("tags") or m["tags"])
    if not set(tags) <= set(m["tags"]):
        return fb("TAG_NOT_IN_MANIFEST")
    if len(handle) != IPC_HANDLE_SIZE:
        return fb("BAD_SPEC:ipc_handle")
    if isinstance(nbytes, bool):                        # bite-2 H2: True is an int to Python, never a size
        return fb("BAD_SPEC:nbytes")
    if not isinstance(nbytes, int) or nbytes < m["total"]:
        return fb("GUEST_TOO_SMALL", total=m["total"], nbytes=nbytes)
    verify = spec.get("verify", "digest")
    copy_mode = spec.get("copy_mode", "direct")
    if verify not in ("digest", "layout") or copy_mode not in ("direct", "staged"):
        return fb("BAD_SPEC:mode")
    sel = [i for i, row in enumerate(m["layout"]) if row[2] in tags]
    if not sel:
        return fb("NOTHING_TO_RESTORE")
    cur = ctx.view.entries(tags)
    if [[p, h_size(d), d.tag] for p, d in cur] != [m["layout"][i] for i in sel]:
        return fb("LAYOUT_MISMATCH")
    for _p, d in cur:
        if not d.is_asleep:
            return fb("NOT_ASLEEP")
        b = backup_of(d)
        if b is None or b[1] != h_size(d):
            return fb("NO_HOST_BACKUP")
    dev = _own_dev(cur)
    if dev is None:
        return fb("MULTI_DEVICE")
    stage, serr = _parse_stage(spec)
    if serr:
        return fb(serr)
    be = ctx.backend
    datas = [d for _p, d in cur]
    mapped, gptr, scratch = [], None, None
    committed = 0
    tm = {"t0": time.perf_counter()}
    moved = 0
    try:
        try:
            gptr = be.ipc_open(dev, handle)
        except Exception as e:  # noqa: BLE001
            raise _Fail(f"IPC_OPEN:{e}"[:200])
        tm["open"] = time.perf_counter()
        if copy_mode == "staged":
            try:
                scratch = be.malloc(dev, stage)
            except Exception as e:  # noqa: BLE001
                raise _Fail(f"STAGE_ALLOC:{e}"[:200])
        for k, i in enumerate(sel):
            ptr, size, _tag = m["layout"][i]
            off = m["offsets"][i]
            d = datas[k]
            try:
                ctx.view.create_and_map(d.handle)
            except Exception as e:  # noqa: BLE001
                raise _Fail(f"CREATE_AND_MAP:{e}"[:200], at=k)
            mapped.append(d)
            try:
                _copy(be, dev, ptr, gptr + off, size, scratch, stage)
            except Exception as e:  # noqa: BLE001
                raise _Fail(f"COPY:{e}"[:200], at=k)
            moved += size
        rc = be.synchronize_rc(dev)
        if rc:
            raise _Fail(f"SYNC:rc={rc}")
        tm["copy"] = time.perf_counter()
        if verify == "digest":
            bad = []
            try:
                for i in sel:
                    ptr, size, _tag = m["layout"][i]
                    if digest_from_sums(size, be.chunk_sums(dev, ptr, size, "device", m["chunk_words"])) != m["digests"][i]:
                        bad.append(i)
            except Exception as e:  # noqa: BLE001
                raise _Fail(f"VERIFY_ERROR:{e}"[:200])   # fail closed: unverifiable = not served
            if bad:
                raise _Fail("DIGEST_MISMATCH", bad_allocs=bad)
        tm["verify"] = time.perf_counter()
        for d in mapped:                     # COMMIT: exactly what stock wake_up leaves (cumem.py L347, L356)
            d.is_asleep = False
            d.cpu_backup_tensor = None
            committed += 1
        ctx.counts["restore_peer"] += 1
        T = tm["verify"] - tm["t0"]
        cs = tm["copy"] - tm["open"]
        return {**base, "ok": True, "served": "PEER", "reason": "OK", "bytes": moved, "n_alloc": len(mapped),
                "copy_mode": copy_mode, "verify": verify, "open_s": round(tm["open"] - tm["t0"], 6),
                "copy_s": round(cs, 6), "verify_s": round(tm["verify"] - tm["copy"], 6), "total_s": round(T, 6),
                "rate_gbs": round(moved / cs / 1e9, 3) if cs > 0 else None}
    except Exception as e:  # noqa: BLE001 -- _Fail or anything unexpected: roll back to stock-asleep
        f = e if isinstance(e, _Fail) else _Fail(f"INTERNAL:{type(e).__name__}:{str(e)[:160]}")
        seen = be.peek_last_error()                        # diagnostic only; the clear below is the load-bearing one
        rb_err = []
        for d in reversed(mapped[committed:]):  # review: never unmap a committed (backup-dropped) entry
            try:
                ctx.view.unmap_and_release(d.handle)
            except Exception as ue:  # noqa: BLE001
                rb_err.append(f"{h_ptr(d)}:{ue}"[:120])
        probe = be.synchronize_rc(dev)
        be.clear_last_error()                              #: never leave a stale error for torch to raise
        out = {**base, "ok": False, "reason": f.reason, "rolled_back": len(mapped) - committed - len(rb_err),
               "committed": committed,
               "bytes_copied": moved, "last_error_seen": seen, "probe_rc": probe, **f.kw}
        if probe or rb_err:
            ctx.counts["restore_none"] += 1
            out.update(served="NONE", fatal=f"STICKY:{probe}" if probe else "ROLLBACK_FAILED", poisoned=bool(probe),
                       rollback_errors=rb_err)
            return out
        ctx.counts["restore_fallback"] += 1
        out["served"] = "HOST_FALLBACK"
        return out
    finally:
        if gptr is not None:
            try:
                be.ipc_close(dev, gptr)
            except Exception:  # noqa: BLE001
                be.clear_last_error()
        if scratch is not None:
            try:
                be.free(dev, scratch)
            except Exception:  # noqa: BLE001
                be.clear_last_error()


OPS = {"inventory": ctx_inventory, "status": ctx_status, "drop": ctx_drop, "push": ctx_push, "seal": ctx_seal,
       "restore": ctx_restore}


# ------------------------------------------------------------------------------------------------ the worker extension
class WarmWakeExt:
    """`--worker-extension-cls warm_wake_ext.WarmWakeExt`. Every name here is checked against the worker
    class by worker_base.py L292-299 -- keep it to bb_* methods (G10 pins the list). Each takes ONE JSON string."""

    def bb_inventory(self, spec="{}"):
        return _call(self, "inventory", spec)

    def bb_status(self, spec="{}"):
        return _call(self, "status", spec)

    def bb_push_to_peer(self, spec):
        return _call(self, "push", spec)

    def bb_seal_after_sleep(self, spec):
        return _call(self, "seal", spec)

    def bb_restore_from_peer(self, spec):
        return _call(self, "restore", spec)

    def bb_drop_manifest(self, spec):
        return _call(self, "drop", spec)


# ------------------------------------------------------------------------------------------------ B2: the guest holder
class GuestHolder:
    """Owns the guest buffers on ONE GPU (logical `dev`, normally 0 with CUDA_VISIBLE_DEVICES = own GPU first, then the
    rest so a re-home can open a neighbour's handle). Budget CAP bytes. A buffer is kept while any reader that began on it
    has not ended (the NO-WAIT handoff reads from a card that is being reclaimed)."""

    def __init__(self, backend, dev=0, cap_bytes=int(35.62 * GIB), label="", log=None, allow_corrupt=False):
        self.be, self.dev, self.cap, self.label = backend, int(dev), int(cap_bytes), label
        self.allow_corrupt = bool(allow_corrupt)
        self.log = log or (lambda ev: None)
        self.lock = threading.Lock()
        self.guests = {}
        self.used = 0
        self.history = []

    def _g(self, gid):
        return self.guests.get(gid)

    def alloc(self, guest_id, nbytes, owner_id=""):
        if isinstance(nbytes, bool):                    # bite-2 H2
            return {"ok": False, "reason": "BAD_SIZE", "guest_id": guest_id}
        nbytes = int(nbytes)
        with self.lock:
            if guest_id in self.guests:
                return {"ok": False, "reason": "DUPLICATE", "guest_id": guest_id}
            if nbytes <= 0:
                return {"ok": False, "reason": "BAD_SIZE", "guest_id": guest_id}
            if self.used + nbytes > self.cap:
                return {"ok": False, "reason": "CAP", "guest_id": guest_id, "used": self.used, "need": nbytes,
                        "cap": self.cap}
            self.used += nbytes
            self.guests[guest_id] = {"state": "ALLOCATING", "nbytes": nbytes, "owner_id": owner_id, "ptr": None,
                                     "readers": set(), "manifest": None, "bad": []}
        ptr = None
        try:
            ptr = self.be.malloc(self.dev, nbytes)
            h = self.be.ipc_export(self.dev, ptr)
        except Exception as e:  # noqa: BLE001
            self.be.clear_last_error()
            if ptr is not None:                 # malloc worked, export failed: never leak the buffer
                try:
                    self.be.free(self.dev, ptr)
                except Exception:  # noqa: BLE001
                    self.be.clear_last_error()
            with self.lock:
                self.guests.pop(guest_id, None)
                self.used -= nbytes
            return {"ok": False, "reason": f"MALLOC:{e}"[:200], "guest_id": guest_id}
        with self.lock:
            g = self.guests[guest_id]
            g.update(ptr=ptr, state="ALLOCATED", handle=base64.b64encode(h).decode())
            return {"ok": True, "guest_id": guest_id, "ipc_handle": g["handle"], "nbytes": nbytes, "used": self.used,
                    "holder": self.label}

    def attach(self, guest_id, manifest, verify=True):
        """Record the owner's push manifest (offsets, sizes, digests); verify=True checks this buffer against it."""
        with self.lock:
            g = self._g(guest_id)
            if g is None or g["ptr"] is None:
                return {"ok": False, "reason": "NO_GUEST", "guest_id": guest_id}
            if manifest.get("guest_id") != guest_id or int(manifest.get("total", 1 << 62)) > g["nbytes"]:
                return {"ok": False, "reason": "MANIFEST_DOES_NOT_FIT", "guest_id": guest_id}
            g["manifest"] = {k: manifest[k] for k in ("manifest_id", "guest_id", "owner_id", "chunk_words", "layout",
                                                       "offsets", "total", "digests")}
            g["state"] = "ATTACHED"
        if verify:
            return self.verify(guest_id)
        with self.lock:
            g["state"] = "FILLED"
        return {"ok": True, "guest_id": guest_id, "state": "FILLED", "verified": False}

    def verify(self, guest_id):
        with self.lock:
            g = self._g(guest_id)
            if g is None or g["manifest"] is None:
                return {"ok": False, "reason": "NO_MANIFEST", "guest_id": guest_id}
            m, ptr = g["manifest"], g["ptr"]
        t0 = time.perf_counter()
        bad = []
        try:
            for i, ((_p, size, _t), off) in enumerate(zip(m["layout"], m["offsets"])):
                if digest_from_sums(size, self.be.chunk_sums(self.dev, ptr + off, size, "device",
                                                             m["chunk_words"])) != m["digests"][i]:
                    bad.append(i)
        except Exception as e:  # noqa: BLE001
            self.be.clear_last_error()
            with self.lock:
                g["state"], g["bad"] = "BAD", ["VERIFY_ERROR"]
            return {"ok": False, "reason": f"VERIFY_ERROR:{e}"[:200], "guest_id": guest_id, "state": "BAD"}
        with self.lock:
            g["state"], g["bad"] = ("BAD" if bad else "FILLED"), bad
        return {"ok": not bad, "guest_id": guest_id, "state": g["state"], "bad": bad, "verified": True,
                "seconds": round(time.perf_counter() - t0, 6), **({"reason": "DIGEST_MISMATCH"} if bad else {})}

    def reader_begin(self, guest_id, reader):
        with self.lock:
            g = self._g(guest_id)
            if g is None or g["state"] != "FILLED":
                return {"ok": False, "reason": "NOT_READABLE", "guest_id": guest_id,
                        "state": None if g is None else g["state"]}
            g["readers"].add(str(reader))
            return {"ok": True, "guest_id": guest_id, "readers": sorted(g["readers"])}

    def reader_end(self, guest_id, reader):
        with self.lock:
            g = self._g(guest_id)
            if g is None or str(reader) not in g["readers"]:
                return {"ok": False, "reason": "NOT_A_READER", "guest_id": guest_id}
            g["readers"].discard(str(reader))
            return {"ok": True, "guest_id": guest_id, "readers": sorted(g["readers"])}

    def free(self, guest_id, force=False):
        with self.lock:
            g = self._g(guest_id)
            if g is None:
                return {"ok": False, "reason": "NO_GUEST", "guest_id": guest_id}
            if g["state"] in ("ALLOCATING", "FREEING"):   # review: a free racing alloc leaked the buffer
                return {"ok": False, "reason": "BUSY_" + g["state"], "guest_id": guest_id}
            if g["readers"] and not force:
                return {"ok": False, "reason": "BUSY", "guest_id": guest_id, "readers": sorted(g["readers"])}
            g["state"] = "FREEING"
        try:
            if g["ptr"] is not None:
                self.be.free(self.dev, g["ptr"])
        except Exception as e:  # noqa: BLE001
            self.be.clear_last_error()
            with self.lock:
                g["state"] = "FREE_FAILED"
            return {"ok": False, "reason": f"FREE:{e}"[:200], "guest_id": guest_id}
        with self.lock:
            self.guests.pop(guest_id, None)
            self.used -= g["nbytes"]
            self.history.append({"guest_id": guest_id, "freed_unix": time.time(), "forced": bool(force)})
            return {"ok": True, "guest_id": guest_id, "used": self.used}

    def rehome_to(self, guest_id, dest_handle_b64, dest_nbytes):
        """Copy this guest into another holder's buffer (this card's copy engine writes over NVLink). The destination
        verifies its own bytes with `attach(manifest, verify=True)`; the source is freed only after that."""
        with self.lock:
            g = self._g(guest_id)
            if g is None or g["state"] != "FILLED":
                return {"ok": False, "reason": "NOT_READABLE", "guest_id": guest_id}
            total = int(g["manifest"]["total"])
            if int(dest_nbytes) < total:
                return {"ok": False, "reason": "DEST_TOO_SMALL", "guest_id": guest_id}
            g["readers"].add("_rehome")
            ptr = g["ptr"]
        t0 = time.perf_counter()
        dptr = None
        try:
            dptr = self.be.ipc_open(self.dev, base64.b64decode(dest_handle_b64, validate=True))
            self.be.memcpy(self.dev, dptr, ptr, total)
            rc = self.be.synchronize_rc(self.dev)
            if rc:
                raise CudaError("sync", rc)
            out = {"ok": True, "guest_id": guest_id, "bytes": total, "seconds": round(time.perf_counter() - t0, 6),
                   "manifest": g["manifest"]}
        except Exception as e:  # noqa: BLE001
            self.be.clear_last_error()
            out = {"ok": False, "reason": f"REHOME:{e}"[:200], "guest_id": guest_id}
        finally:
            if dptr is not None:
                try:
                    self.be.ipc_close(self.dev, dptr)
                except Exception:  # noqa: BLE001
                    self.be.clear_last_error()
            with self.lock:
                g["readers"].discard("_rehome")
        return out

    def corrupt(self, guest_id, offset, xor=0xFF):
        """S0's adversarial step on real hardware: flip one byte of a guest so the owner's restore must refuse it.
        Disabled unless the holder was started with --allow-corrupt."""
        if not self.allow_corrupt:
            return {"ok": False, "reason": "CORRUPT_DISABLED", "guest_id": guest_id}
        with self.lock:
            g = self._g(guest_id)
            if g is None or g["ptr"] is None or not 0 <= int(offset) < g["nbytes"]:
                return {"ok": False, "reason": "NO_GUEST_OR_OFFSET", "guest_id": guest_id}
            ptr = g["ptr"]
        try:
            old, new = self.be.flip_byte(self.dev, ptr + int(offset), xor)
        except Exception as e:  # noqa: BLE001
            self.be.clear_last_error()
            return {"ok": False, "reason": f"FLIP:{e}"[:200], "guest_id": guest_id}
        return {"ok": True, "guest_id": guest_id, "offset": int(offset), "old": old, "new": new}

    def status(self):
        with self.lock:
            return {"ok": True, "holder": self.label, "cap": self.cap, "used": self.used,
                    "guests": {k: {"state": g["state"], "nbytes": g["nbytes"], "owner_id": g["owner_id"],
                                   "readers": sorted(g["readers"]),
                                   "manifest_id": (g["manifest"] or {}).get("manifest_id")}
                               for k, g in self.guests.items()}}


HOLDER_ROUTES = {
    "/holder/alloc": lambda h, b: h.alloc(b["guest_id"], b["nbytes"], b.get("owner_id", "")),
    "/holder/attach": lambda h, b: h.attach(b["guest_id"], b["manifest"], bool(b.get("verify", True))),
    "/holder/verify": lambda h, b: h.verify(b["guest_id"]),
    "/holder/reader_begin": lambda h, b: h.reader_begin(b["guest_id"], b["reader"]),
    "/holder/reader_end": lambda h, b: h.reader_end(b["guest_id"], b["reader"]),
    "/holder/free": lambda h, b: h.free(b["guest_id"], bool(b.get("force", False))),
    "/holder/rehome_to": lambda h, b: h.rehome_to(b["guest_id"], b["dest_handle"], b["dest_nbytes"]),
    "/holder/corrupt": lambda h, b: h.corrupt(b["guest_id"], b["offset"], b.get("xor", 0xFF)),
}


def make_holder_server(holder, port=0, host="127.0.0.1"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/holder/status":
                self._send(200, holder.status())
            elif self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"ok": False, "reason": "NOT_FOUND"})

        def do_POST(self):
            fn = HOLDER_ROUTES.get(self.path)
            if fn is None:
                return self._send(404, {"ok": False, "reason": "NOT_FOUND"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"ok": False, "reason": f"BAD_JSON:{e}"})
            try:
                out = fn(holder, body)
            except KeyError as e:
                return self._send(400, {"ok": False, "reason": f"MISSING:{e}"})
            except Exception as e:  # noqa: BLE001
                out = {"ok": False, "reason": f"INTERNAL:{type(e).__name__}:{e}"[:200]}
            holder.log({"path": self.path, **{k: v for k, v in out.items() if k != "manifest"}})
            self._send(200, out)

    return ThreadingHTTPServer((host, int(port)), H)


# ------------------------------------------------------------------------------------------------ stub loading (tests)
def _here():
    return os.path.dirname(os.path.abspath(__file__))


def load_stub():
    """The sibling stub module, by path (a mutation-test copy sits beside it and must test ITSELF, so the stub
    never imports this module by name: everything it needs is handed to it)."""
    import importlib.util
    p = os.path.join(_here(), "warm_wake_stub.py")
    spec = importlib.util.spec_from_file_location("warm_wake_stub", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("warm_wake_stub", mod)
    spec.loader.exec_module(mod)
    return mod


def _tmp_root():
    import tempfile
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    return tempfile.mkdtemp(prefix="bbwarm_", dir=base)


# ------------------------------------------------------------------------------------------------ self-test (goldens)
MIB = 1 << 20
W_MIB = [2, 4, 2]
KV_MIB = [4, 4]
EXT_CALLS = ["bb_drop_manifest", "bb_inventory", "bb_push_to_peer", "bb_restore_from_peer", "bb_seal_after_sleep",
             "bb_status"]


def pattern_byte(i, j, seed=0):
    return (31 * i + 7 * j + 5 + seed) % 256


def self_test(vllm_src=None):
    import shutil
    import struct
    sys.dont_write_bytecode = True
    me = sys.modules[__name__]
    stub = load_stub()
    res = []
    skipped = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond)))
        print(("ok   " if cond else "FAIL ") + name + ("" if cond else f" - {detail}"), flush=True)

    # G1: the digest, hand-computed
    check("G1a digest [1..5] CH2", digest_bytes(struct.pack("<5i", 1, 2, 3, 4, 5), 2)
          == "pws1:20:000000000000000f:0000000000000020")
    check("G1b ragged 5 bytes", digest_bytes(bytes([1, 0, 0, 0, 2]), 2) == "pws1:5:0000000000000003:0000000000000003")
    check("G1c negative wraps", digest_bytes(struct.pack("<2i", -1, 0), 2) == "pws1:8:ffffffffffffffff:ffffffffffffffff")
    g1d = digest_bytes(struct.pack("<5i", 3, 4, 1, 2, 5), 2)
    check("G1d chunk swap caught by s2 only", g1d == "pws1:20:000000000000000f:000000000000001c", g1d)
    g1f = digest_bytes(struct.pack("<2i", 2 ** 31 - 1, 1), 2)
    check("G1f (added, amendment A1) chunk sum wraps mod 2^32", g1f == "pws1:8:ffffffff80000000:ffffffff80000000", g1f)
    try:
        import numpy as np
        import torch
        rng = np.random.default_rng(2967)
        raw = rng.integers(0, 256, size=(1 << 20) + 4 * 1001, dtype=np.uint8)   # full-range words: chunk sums overflow
        t = torch.from_numpy(raw.copy()).view(torch.int32)
        wide = sum(1 for c in raw.view("<i4")[:262144].reshape(64, 4096).sum(axis=1, dtype=np.int64)
                   if not -2 ** 31 <= int(c) < 2 ** 31)
        check("G1e torch formula == numpy formula (CPU), incl. wrapped chunks",
              chunk_sums_torch(t, 4096) == chunk_sums_numpy(raw, 4096) and wide > 0, wide)
    except ImportError as e:
        skipped.append(f"G1e ({e})")
        print(f"SKIP G1e torch not importable: {e}")

    # G11 (adversarial): SIGTERM must run `finally` (exit 143), not die silently with children orphaned.
    import signal
    import subprocess
    probe = subprocess.Popen(
        [sys.executable, "-c",
         # load THIS file (a mutant copy included), not the module by name from scripts/
         "import sys, time, importlib.util; s = importlib.util.spec_from_file_location('X', %r)\n"
         "X = importlib.util.module_from_spec(s); s.loader.exec_module(X)\n"
         "X._install_term_handlers()\n"
         "try:\n    print('UP', flush=True); time.sleep(30)\n"
         "finally:\n    print('FINALLY-RAN', flush=True)\n" % os.path.abspath(__file__)],
        stdout=subprocess.PIPE, text=True)
    first = probe.stdout.readline().strip()
    probe.send_signal(signal.SIGTERM)
    rest, _ = probe.communicate(timeout=20)
    check("G11 SIGTERM runs finally and exits 143", first == "UP" and "FINALLY-RAN" in rest
          and probe.returncode == 143, (first, rest, probe.returncode))

    root = _tmp_root()
    try:
        _goldens_engine(stub, me, root, check)
        _golden_holder(stub, me, root, check)
        _golden_inject(stub, me, check, vllm_src, skipped)
        _golden_http(stub, root, check)
        _goldens_hardening(stub, me, root, check)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("self-test ran to completion", False, repr(e))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n = len(res)
    k = sum(1 for _n, ok in res if not ok)
    if k:
        print(f"SELFTEST-FAIL {k}/{n}")
        return 1
    print(f"SELFTEST-OK {n}/{n} ({len(skipped)} skipped{': ' + '; '.join(skipped) if skipped else ''})")
    return 0


def _new_fabric(stub, root, name, cap_mib=64):
    d = os.path.join(root, name)
    return stub.StubFabric.create(d, ngpu=3, cap_bytes=cap_mib * MIB)


def _rig(stub, me, fab, cvd=(2, 0, 1), seed=0, proc="eng", holder_phys=0, holder_cap_mib=32, w_mib=W_MIB,
         kv_mib=KV_MIB):
    eng = stub.make_engine(fab, list(cvd), me, [m * MIB for m in w_mib], [m * MIB for m in kv_mib], seed=seed, proc=proc)
    hcvd = [holder_phys] + [p for p in range(fab.ngpu) if p != holder_phys]
    hold = GuestHolder(stub.StubBackend(fab, hcvd, f"holder{holder_phys}-{proc}", me.chunk_sums_numpy), 0,
                       holder_cap_mib * MIB, label=f"gpu{holder_phys}")
    return eng, hold


def _push(eng, hold, gid="g1", owner="slotA"):
    total = sum(eng.weight_sizes)
    a = hold.alloc(gid, total, owner)
    r = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": gid, "ipc_handle": a["ipc_handle"], "nbytes": total},
                                    "tags": ["weights"], "owner_id": owner})
    if r.get("ok"):
        hold.attach(gid, r["manifest"], True)
    return a, r


def _rspec(a, r, **kw):
    s = {"guest": {"guest_id": a["guest_id"], "ipc_handle": a["ipc_handle"], "nbytes": a["nbytes"],
                   "manifest_id": r.get("manifest_id")}}
    s.update(kw)
    return s


def _weights_match(eng, seed=0):
    for i, (ptr, size) in enumerate(zip(eng.weight_ptrs, eng.weight_sizes)):
        if eng.read(ptr, 0, size) != stub_pattern(i, size, seed):
            return False
    return True


def stub_pattern(i, size, seed=0):
    import numpy as np
    j = np.arange(size, dtype=np.int64)
    return ((31 * i + 7 * j + 5 + seed) % 256).astype(np.uint8).tobytes()


def _goldens_engine(stub, me, root, check):
    TOTAL = 8388608
    # ---------------- G2: the WARM path end to end (in-process engine, stock-transliterated allocator/executor/core)
    fab = _new_fabric(stub, root, "g2")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    check("G2 push total/offsets", r.get("ok") and r["total"] == TOTAL and r["offsets"] == [0, 2097152, 6291456], r)
    check("G2 holder attach verified FILLED", hold.status()["guests"]["g1"]["state"] == "FILLED")
    lp0 = eng.logprobs([1, 2, 3, 4, 5])
    eng.sleep(1, "wait")
    tr = eng.truth()
    check("G2 after sleep: 5 asleep, 3 host backups 8388608 B",
          tr["asleep"] == 5 and tr["backups"] == 3 and tr["backup_bytes"] == TOTAL, tr)
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    c = eng.backend.ctr
    check("G2 restore served PEER 8388608 B", rr.get("served") == "PEER" and rr["bytes"] == TOTAL, rr)
    check("G2 bytes peer->local 8388608, 0 host->local copies",
          c["bytes:peer->local"] == TOTAL and c["copies:host->local"] == 0, dict(c))
    tr = eng.truth()
    check("G2 3 weights awake, 2 kv asleep, 0 backups", tr["weights_awake"] == 3 and tr["kv_asleep"] == 2
          and tr["backups"] == 0, tr)
    eng.backend.ctr.clear()
    eng.wake_up(["weights"])
    check("G2 wake_up(weights) copies 0, sleeping_tags {kv_cache}, scheduler paused",
          eng.backend.ctr["copies:host->local"] == 0 and eng.executor.sleeping_tags == {"kv_cache"}
          and eng.core.paused, (dict(eng.backend.ctr), eng.executor.sleeping_tags, eng.core.paused))
    eng.wake_up(["kv_cache"])
    check("G2 wake_up(kv_cache): not sleeping, scheduler resumed", not eng.core.is_sleeping() and not eng.core.paused)
    spots = (eng.read(eng.weight_ptrs[0], 0, 1)[0], eng.read(eng.weight_ptrs[1], 1000, 1)[0],
             eng.read(eng.weight_ptrs[2], 2097151, 1)[0])
    check("G2 spot bytes 5/124/60", spots == (5, 124, 60), spots)
    check("G2 every weights byte matches the formula", _weights_match(eng))
    check("G2 logprobs equal pre-sleep", eng.logprobs([1, 2, 3, 4, 5]) == lp0)

    # ---------------- G2b staged copy
    fab = _new_fabric(stub, root, "g2b")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r, copy_mode="staged", stage_bytes=MIB))
    check("G2b staged PEER, 8 scratch round trips", rr.get("served") == "PEER"
          and eng.backend.ctr["copies:peer->local"] == 8 and eng.backend.ctr["copies:local->local"] == 8,
          (rr, dict(eng.backend.ctr)))
    eng.wake_up(None)
    check("G2b bytes match", _weights_match(eng))
    eng.sleep(1, "wait")
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r, copy_mode="staged", stage_bytes=0))
    check("G2c (added, QA fix) staged with stage_bytes 0 -> BAD_SPEC, no hang, 0 B moved",
          rr.get("reason") == "BAD_SPEC:stage_bytes" and sum(eng.backend.ctr.values()) == 0, rr)

    # G2d (added): a STAGED push (the S0 fallback if VMM -> peer direct copies fail), then a direct restore
    fab = _new_fabric(stub, root, "g2d")
    eng, hold = _rig(stub, me, fab)
    total = sum(eng.weight_sizes)
    a = hold.alloc("g1", total, "slotA")
    eng.backend.ctr.clear()
    r = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "g1", "ipc_handle": a["ipc_handle"], "nbytes": total},
                                    "copy_mode": "staged", "stage_bytes": MIB})
    pc = dict(eng.backend.ctr)
    at = hold.attach("g1", r.get("manifest") or {}, True)
    eng.sleep(1, "wait")
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    eng.wake_up(None)
    check("G2d (added) staged push: 8 local->local + 8 local->peer, holder verifies, restore PEER, bytes match",
          r.get("ok") and pc.get("copies:local->local") == 8 and pc.get("copies:local->peer") == 8 and at.get("ok")
          and rr.get("served") == "PEER" and _weights_match(eng), (r.get("reason"), pc, at, rr.get("reason")))

    # ---------------- G3a the holder freed the guest
    fab = _new_fabric(stub, root, "g3a")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    hold.free("g1")
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    tr = eng.truth()
    check("G3a HOST_FALLBACK reason IPC_OPEN", rr.get("served") == "HOST_FALLBACK"
          and rr["reason"].startswith("IPC_OPEN"), rr)
    check("G3a 3 weights asleep with 3 backups", tr["weights_asleep"] == 3 and tr["backups"] == 3, tr)
    eng.backend.ctr.clear()
    eng.wake_up(["weights"])
    check("G3a stock wake: 3 host copies 8388608 B", eng.backend.ctr["copies:host->local"] == 3
          and eng.backend.ctr["bytes:host->local"] == TOTAL, dict(eng.backend.ctr))
    eng.wake_up(["kv_cache"])
    check("G3a bytes match", _weights_match(eng))

    # ---------------- G3b holder GPU invisible (engine CUDA_VISIBLE_DEVICES=2,1; push via GPU 1, re-home to GPU 0)
    fab = _new_fabric(stub, root, "g3b")
    eng, hold1 = _rig(stub, me, fab, cvd=(2, 1), holder_phys=1)
    hold0 = GuestHolder(stub.StubBackend(fab, [0, 1, 2], "holder0-g3b", me.chunk_sums_numpy), 0, 32 * MIB, "gpu0")
    a1, r = _push(eng, hold1)
    eng.sleep(1, "wait")
    a0 = hold0.alloc("g1", a1["nbytes"], "slotA")
    rh = hold1.rehome_to("g1", a0["ipc_handle"], a0["nbytes"])
    at = hold0.attach("g1", rh.get("manifest") or {}, True)
    check("G3b re-home 1->0 verified (added)", rh.get("ok") and at.get("ok"), (rh.get("reason"), at))
    rr = eng.rpc("bb_restore_from_peer", _rspec(a0, r))
    check("G3b GPU 0 invisible -> HOST_FALLBACK IPC_OPEN", rr.get("served") == "HOST_FALLBACK"
          and rr["reason"].startswith("IPC_OPEN"), rr)
    rr2 = eng.rpc("bb_restore_from_peer", _rspec(a1, r))
    check("G3b restore from the visible home serves PEER (added)", rr2.get("served") == "PEER", rr2)
    eng.wake_up(None)
    check("G3b bytes match", _weights_match(eng))

    # ---------------- G3c no manifest
    fab = _new_fabric(stub, root, "g3c")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.backend.ctr.clear()
    spec = _rspec(a, r)
    spec["guest"]["guest_id"] = "never-pushed"
    before = eng.truth()
    rr = eng.rpc("bb_restore_from_peer", spec)
    check("G3c NO_MANIFEST, 0 B moved, state untouched", rr.get("served") == "HOST_FALLBACK"
          and rr["reason"] == "NO_MANIFEST" and sum(eng.backend.ctr.values()) == 0 and eng.truth() == before, rr)

    # ---------------- G3d owner GPU full mid-restore
    fab = _new_fabric(stub, root, "g3d")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    batch = stub.StubBackend(fab, [2, 0, 1], "batch2", me.chunk_sums_numpy)
    bptr = batch.malloc(0, 60 * MIB)
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    tr = eng.truth()
    check("G3d CREATE_AND_MAP fallback, 1 rolled back, 0 weights mapped",
          rr.get("served") == "HOST_FALLBACK" and rr["reason"].startswith("CREATE_AND_MAP")
          and rr.get("rolled_back") == 1 and tr["weights_mapped"] == 0, (rr, tr))
    batch.free(0, bptr)
    eng.wake_up(None)
    check("G3d after the batch frees, the stock wake serves", _weights_match(eng))

    # ---------------- G4 partial wake by tag
    fab = _new_fabric(stub, root, "g4")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    tr = eng.truth()
    check("G4a restore touched 0 kv allocations", rr.get("served") == "PEER" and tr["kv_mapped"] == 0
          and eng.backend.kv_touched == 0, (rr.get("served"), tr))
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r, tags=["kv_cache"]))
    check("G4c tags kv_cache -> TAG_NOT_IN_MANIFEST", rr.get("reason") == "TAG_NOT_IN_MANIFEST"
          and rr["bytes"] == 0, rr)
    fab = _new_fabric(stub, root, "g4b")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.wake_up(["kv_cache"])
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    eng.wake_up(["weights"])
    check("G4b kv first, then restore, then weights: PEER, awake, scheduler resumed",
          rr.get("served") == "PEER" and not eng.core.is_sleeping() and not eng.core.paused and _weights_match(eng), rr)

    # ---------------- G5 adversarial: sha / shape mismatches
    fab = _new_fabric(stub, root, "g5a")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    gp = hold.guests["g1"]["ptr"]
    old = hold.be.peek(gp + 2097152 + 12345)
    hold.be.poke(gp + 2097152 + 12345, old ^ 0xFF)
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    tr = eng.truth()
    check("G5a flipped byte -> DIGEST_MISMATCH [1], HOST_FALLBACK, rolled back",
          rr.get("reason") == "DIGEST_MISMATCH" and rr.get("bad_allocs") == [1] and rr["served"] == "HOST_FALLBACK"
          and tr["weights_mapped"] == 0 and tr["backups"] == 3, (rr, tr))
    eng.wake_up(None)
    b = eng.read(eng.weight_ptrs[1], 12345, 1)[0]
    check("G5a stock wake serves 179, never 76", old == 179 and b == 179, (old, b))

    fab = _new_fabric(stub, root, "g5b")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    other = stub.make_engine(fab, [1, 0, 2], me, [m * MIB for m in W_MIB], [m * MIB for m in KV_MIB], seed=1,
                             proc="eng-other")
    aB = hold.alloc("gB", a["nbytes"], "slotB")
    rB = other.rpc("bb_push_to_peer", {"guest": {"guest_id": "gB", "ipc_handle": aB["ipc_handle"],
                                                 "nbytes": aB["nbytes"]}})
    eng.sleep(1, "wait")
    spec = _rspec(a, r)
    spec["guest"]["ipc_handle"] = aB["ipc_handle"]       # right metadata, another engine's bytes
    rr = eng.rpc("bb_restore_from_peer", spec)
    check("G5b wrong-content guest -> DIGEST_MISMATCH [0,1,2]", rB.get("ok") and rr.get("reason") == "DIGEST_MISMATCH"
          and rr.get("bad_allocs") == [0, 1, 2] and rr["served"] == "HOST_FALLBACK", rr)
    eng.wake_up(None)
    check("G5b stock wake serves this engine's weights", _weights_match(eng))

    fab = _new_fabric(stub, root, "g5c")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    extra = eng.worker.allocator.allocate(2 * MIB, "weights")     # e.g. an adapter loaded after the push
    eng.sleep(1, "wait")
    eng.backend.ctr.clear()
    small = _rspec(a, r)
    small["guest"]["nbytes"] = 6291456
    r1 = eng.rpc("bb_restore_from_peer", small)
    r2 = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G5c GUEST_TOO_SMALL and LAYOUT_MISMATCH, 0 B moved", r1.get("reason") == "GUEST_TOO_SMALL"
          and r2.get("reason") == "LAYOUT_MISMATCH" and sum(eng.backend.ctr.values()) == 0
          and extra is not None, (r1.get("reason"), r2.get("reason"), dict(eng.backend.ctr)))

    # ---------------- G6 sticky error mid-restore
    fab = _new_fabric(stub, root, "g6")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.backend.install_fault("memcpy", 2, 700, sticky=True)
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G6 sticky 700 -> served NONE, fatal STICKY:700, poisoned", rr.get("served") == "NONE"
          and rr.get("fatal") == "STICKY:700" and rr.get("poisoned") is True, rr)
    fab = _new_fabric(stub, root, "g6b")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(1, "wait")
    eng.backend.install_fault("ipc_open", 1, 700, sticky=True)
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G6b (added, a mutant survived) sticky BEFORE any mapping -> NONE / STICKY:700, never HOST_FALLBACK",
          rr.get("served") == "NONE" and rr.get("fatal") == "STICKY:700" and rr.get("rolled_back") == 0, rr)

    # ---------------- G7: a non-sticky error is cleared
    fab = _new_fabric(stub, root, "g7")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    lp0 = eng.logprobs([7, 8, 9])
    eng.sleep(1, "wait")
    eng.backend.install_fault("ipc_open", 1, 1, sticky=False)
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G7 HOST_FALLBACK and last-error 0 afterwards", rr.get("served") == "HOST_FALLBACK"
          and eng.backend.peek_last_error() == 0, (rr, eng.backend.peek_last_error()))
    eng.wake_up(None)
    try:
        ok7 = eng.logprobs([7, 8, 9]) == lp0
    except RuntimeError as e:
        ok7 = False
        print(f"     G7 forward raised: {e}")
    check("G7 forward after the stock wake succeeds", ok7)

    # ---------------- added after the freeze (mutation probe found these guards unexercised)
    fab = _new_fabric(stub, root, "g7b")
    eng, hold = _rig(stub, me, fab)
    lp0 = eng.logprobs([4, 5, 6])
    total = sum(eng.weight_sizes)
    a = hold.alloc("g1", total, "slotA")
    eng.backend.install_fault("ipc_open", 1, 1, sticky=False)
    r = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "g1", "ipc_handle": a["ipc_handle"], "nbytes": total}})
    try:
        ok7b = eng.logprobs([4, 5, 6]) == lp0
    except RuntimeError as e:
        ok7b = False
        print(f"     G7b forward raised: {e}")
    check("G7b (added) a failed push on an AWAKE engine clears the error; it keeps serving",
          r.get("reason", "").startswith("IPC_OPEN") and eng.backend.peek_last_error() == 0 and ok7b, r)
    a, r = _push(eng, hold, gid="g2")
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G4d (added) restore while awake -> NOT_ASLEEP, 0 B moved", rr.get("reason") == "NOT_ASLEEP"
          and sum(eng.backend.ctr.values()) == 0, rr)
    bad_id = _rspec(a, r)
    bad_id["guest"]["manifest_id"] = "0" * 16
    eng.sleep(1, "wait")
    rr = eng.rpc("bb_restore_from_peer", bad_id)
    check("G5d (added) wrong manifest_id -> MANIFEST_ID_MISMATCH", rr.get("reason") == "MANIFEST_ID_MISMATCH", rr)
    fab = _new_fabric(stub, root, "g4e")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    eng.sleep(2, "wait")
    eng.backend.ctr.clear()
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    check("G4e (added) level-2 sleep (no host backup) -> NO_HOST_BACKUP, 0 B moved",
          rr.get("reason") == "NO_HOST_BACKUP" and sum(eng.backend.ctr.values()) == 0, rr)

    # ---------------- G11 seal: a byte changed after the push
    fab = _new_fabric(stub, root, "g11")
    eng, hold = _rig(stub, me, fab)
    a, r = _push(eng, hold)
    p2 = eng.weight_ptrs[2]
    v = eng.read(p2, 0, 1)[0]
    eng.backend.poke(p2, v ^ 0xFF)
    eng.sleep(1, "wait")
    s = eng.rpc("bb_seal_after_sleep", {"guest_id": "g1"})
    rr = eng.rpc("bb_restore_from_peer", _rspec(a, r))
    eng.wake_up(None)
    check("G11 seal stale [2], restore MANIFEST_STALE, stock wake serves the newer byte",
          s.get("stale") is True and s.get("mismatched") == [2] and rr.get("reason") == "MANIFEST_STALE"
          and v == 67 and eng.read(p2, 0, 1)[0] == (67 ^ 0xFF), (s, rr.get("reason"), v))


def _golden_holder(stub, me, root, check):
    fab = _new_fabric(stub, root, "g9")
    be0 = stub.StubBackend(fab, [0, 1, 2], "holder0-g9", me.chunk_sums_numpy)
    be1 = stub.StubBackend(fab, [1, 0, 2], "holder1-g9", me.chunk_sums_numpy)
    h0 = GuestHolder(be0, 0, 16 * MIB, "gpu0")
    h1 = GuestHolder(be1, 0, 16 * MIB, "gpu1")
    x = h0.alloc("A", 8 * MIB, "o")
    y = h0.alloc("B", 10 * MIB, "o")
    check("G9 alloc 8 MiB used 8388608; 10 MiB refused CAP", x.get("ok") and x["used"] == 8388608
          and y.get("reason") == "CAP" and y["used"] + y["need"] == 18874368, (x, y))
    eng = stub.make_engine(fab, [2, 0, 1], me, [m * MIB for m in W_MIB], [m * MIB for m in KV_MIB], seed=0,
                           proc="eng-g9")
    r = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "A", "ipc_handle": x["ipc_handle"], "nbytes": x["nbytes"]}})
    at = h0.attach("A", r["manifest"], True)
    h0.reader_begin("A", "e2")
    f1 = h0.free("A")
    check("G9 free with reader e2 -> BUSY ['e2']", at.get("ok") and f1.get("reason") == "BUSY"
          and f1.get("readers") == ["e2"], (at, f1))
    d = h1.alloc("A", 8 * MIB, "o")
    rh = h0.rehome_to("A", d["ipc_handle"], d["nbytes"])
    vd = h1.attach("A", rh["manifest"], True)
    h0.reader_end("A", "e2")
    f2 = h0.free("A")
    check("G9 re-home 0->1 verifies, then the source frees (used 0)", rh.get("ok") and vd.get("ok")
          and f2.get("ok") and f2["used"] == 0, (rh.get("reason"), vd, f2))
    e = h0.alloc("C", 8 * MIB, "o")
    r2 = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "C", "ipc_handle": e["ipc_handle"], "nbytes": e["nbytes"]}})
    h0.attach("C", r2["manifest"], True)
    d2 = h1.alloc("C", 8 * MIB, "o")
    rh2 = h0.rehome_to("C", d2["ipc_handle"], d2["nbytes"])
    dp = h1.guests["C"]["ptr"]
    be1.poke(dp + 2097152 + 7, be1.peek(dp + 2097152 + 7) ^ 0x01)
    vd2 = h1.attach("C", rh2["manifest"], True)
    check("G9 corrupted re-home: dest BAD [1], source still FILLED", rh2.get("ok") and vd2.get("ok") is False
          and vd2.get("bad") == [1] and h1.guests["C"]["state"] == "BAD" and h0.guests["C"]["state"] == "FILLED", vd2)
    # G9c (added): /holder/corrupt is off by default; on, a flipped guest byte makes the owner's restore refuse
    fab9c = _new_fabric(stub, root, "g9c")
    eng9, h9 = _rig(stub, me, fab9c)
    a9, r9 = _push(eng9, h9)
    off_ = h9.corrupt("g1", 2097152 + 5)
    h9.allow_corrupt = True
    on_ = h9.corrupt("g1", 2097152 + 5)
    eng9.sleep(1, "wait")
    rr9 = eng9.rpc("bb_restore_from_peer", _rspec(a9, r9))
    eng9.wake_up(None)
    check("G9c (added) corrupt disabled by default; enabled -> restore DIGEST_MISMATCH [1] -> host serves",
          off_.get("reason") == "CORRUPT_DISABLED" and on_.get("ok") and rr9.get("bad_allocs") == [1]
          and rr9.get("served") == "HOST_FALLBACK" and _weights_match(eng9), (off_, on_, rr9))

    # G9b (added, QA fix): a free racing an in-flight alloc is refused; the alloc completes; nothing leaks
    be2 = stub.StubBackend(fab, [2, 0, 1], "holder2-g9b", me.chunk_sums_numpy)
    h2 = GuestHolder(be2, 0, 16 * MIB, "gpu2")
    gate, entered = threading.Event(), threading.Event()
    real_malloc = be2.malloc

    def slow_malloc(dev, n):
        entered.set()
        gate.wait(10)
        return real_malloc(dev, n)

    be2.malloc = slow_malloc
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("a", h2.alloc("R", 2 * MIB, "o")))
    th.start()
    entered.wait(10)
    fr = h2.free("R")
    gate.set()
    th.join(10)
    check("G9b free during alloc -> BUSY_ALLOCATING; alloc completes; then free ok, used 0",
          fr.get("reason") == "BUSY_ALLOCATING" and box.get("a", {}).get("ok") and h2.free("R").get("ok")
          and h2.used == 0, (fr, box.get("a")))


def _golden_inject(stub, me, check, vllm_src, skipped):
    W = type("Worker", (stub.StubWorkerBase,), {})
    calls = stub.inject_extension(W, me.WarmWakeExt, log=lambda *a: None)
    check("G10 inject: extended calls exactly the 6 bb_* names", calls == EXT_CALLS, calls)
    if not vllm_src:
        skipped.append("G10b (no --vllm-src)")
        print("SKIP G10b no vLLM source given")
        return
    coll = vllm_name_collisions(vllm_src)
    if coll is None:
        skipped.append("G10b (v0.30.0 not readable)")
        print(f"SKIP G10b v0.30.0 not readable under {vllm_src}")
        return
    check("G10b 0 collisions with v0.30.0 Worker/WorkerBase names", coll == [], coll)


def vllm_name_collisions(src):
    """AST over the stock v0.30.0 Worker + WorkerBase: class attributes, methods and self.<attr> targets. None if the
    source is not readable. Read-only (`git show v0.30.0:<path>`)."""
    import ast
    import subprocess
    names = set()
    for path, cls in (("vllm/v1/worker/gpu_worker.py", "Worker"), ("vllm/v1/worker/worker_base.py", "WorkerBase")):
        try:
            txt = subprocess.run(["git", "-C", src, "show", f"v0.30.0:{path}"], capture_output=True, text=True,
                                 timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if txt.returncode != 0:
            return None
        tree = ast.parse(txt.stdout)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                found = True
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(sub.name)
                    elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                        names.add(sub.attr)
                    elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
        if not found:
            return None
    ours = {a for a in dir(WarmWakeExt) if not a.startswith("__")} | {"_bb_ctx"}
    return sorted(ours & names)


def _http(url, body=None, method="POST", timeout=60):
    import urllib.request
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


def _ready_port(proc, tag, timeout=60):
    """bite-2 H4: os.read on the raw fd + line splitting, so a child that writes a partial line and stalls cannot block
    past the deadline (a select-then-readline did: readline waits for the newline)."""
    import select
    fd = proc.stdout.fileno()
    t0 = time.monotonic()
    buf = seen = b""
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{tag} exited rc={proc.returncode}")
        r, _, _ = select.select([fd], [], [], min(0.5, max(0.01, timeout - (time.monotonic() - t0))))
        if r:
            chunk = os.read(fd, 4096)
            if not chunk:
                time.sleep(0.05)                        # QA 2026-10-21: stdout closed, child alive -- no hot spin
                continue
            buf += chunk
            for line in buf.split(b"\n")[:-1]:
                if line.startswith(tag.encode()):
                    return int(line.decode().split("port=")[1].split()[0])
            seen = (seen + buf[: buf.rfind(b"\n") + 1])[-300:]
            buf = buf[buf.rfind(b"\n") + 1:]           # QA 2026-10-21: keep only the unterminated tail
    raise RuntimeError(f"{tag} not ready in {timeout}s; got {(seen + buf)[-300:]!r}")


def spawn_pair(root, fab_dir, w_mib, kv_mib, holder_cap_mib, eng_cvd="2,0,1", tag=""):
    """Holder process (GPU 0) + stub engine process (GPU 2), both on the stub fabric. Returns (hp, ep, hport, eport,
    log path)."""
    import subprocess
    log = os.path.join(root, f"engine{tag}.log")
    hf = open(os.path.join(root, f"holder{tag}.log"), "wb")       # bite-2 H6: was DEVNULL
    hp = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve-holder", "--stub-fabric", fab_dir,
                           "--stub-cvd", "0,1,2", "--port", "0", "--cap-mib", str(holder_cap_mib)],
                          stdout=subprocess.PIPE, stderr=hf)
    hf.close()
    try:                                                #: a failing engine Popen must not orphan the holder
        lf = open(log, "wb")
        try:
            ep = subprocess.Popen([sys.executable, os.path.join(_here(), "warm_wake_stub.py"), "--serve-engine",
                                   "--fabric", fab_dir, "--cvd", eng_cvd, "--port", "0", "--weights-bytes",
                                   ",".join(str(x) for x in w_mib), "--kv-bytes", ",".join(str(x) for x in kv_mib),
                                   "--ext-file", os.path.abspath(__file__), "--ready-fd", "1"],
                                  stdout=subprocess.PIPE, stderr=lf)
        finally:
            lf.close()
    except BaseException:
        _stop(hp)
        raise
    try:
        hport = _ready_port(hp, "BB-HOLDER-READY")
        eport = _ready_port(ep, "BB-STUB-READY")
    except Exception:
        for p in (hp, ep):
            p.kill()
            p.wait()                                    # no zombies until exit
        raise
    return hp, ep, hport, eport, log


def _install_term_handlers():
    """: SIGTERM/SIGHUP end the process through SystemExit(143) so the `finally` blocks that stop the
    holder/engine children and rmtree the /dev/shm root run (Python's default exits without them)."""
    import signal

    def _raise(signum, _frame):
        raise SystemExit(128 + signum if signum != signal.SIGTERM else 143)
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _raise)


def _stop(*procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            p.kill()
        if p.stdout:
            p.stdout.close()


def http_cycle(hport, eport, total, gid="g1", corrupt_at=None, timeout=600):
    """push -> attach -> logprobs -> sleep(wait) -> seal -> [corrupt] -> restore -> wake weights -> wake kv -> logprobs.
    Returns a dict of every step's result and wall time (the controller's call sequence, minus scheduling)."""
    H = f"http://127.0.0.1:{hport}"
    E = f"http://127.0.0.1:{eport}"
    out = {}

    def rpc(method, spec):
        st, js = _http(E + "/collective_rpc", {"method": method, "args": [json.dumps(spec)], "timeout": timeout},
                       timeout=timeout)
        return st, (js or {}).get("results", [None])[0]

    def lps():
        st, js = _http(E + "/v1/completions", {"model": "stub", "prompt": [1, 2, 3, 4, 5], "max_tokens": 1,
                                                "prompt_logprobs": 1}, timeout=timeout)
        return [None if x is None else list(x.values())[0]["logprob"] for x in js["choices"][0]["prompt_logprobs"]]

    t = time.perf_counter
    _, al = _http(H + "/holder/alloc", {"guest_id": gid, "nbytes": total, "owner_id": "slot"})
    out["alloc"] = al
    t0 = t()
    st, push = rpc("bb_push_to_peer", {"guest": {"guest_id": gid, "ipc_handle": al["ipc_handle"], "nbytes": total},
                                       "owner_id": "slot"})
    out["push_http"], out["push_s"] = st, t() - t0
    out["push"] = {k: v for k, v in (push or {}).items() if k != "manifest"}
    _, out["attach"] = _http(H + "/holder/attach", {"guest_id": gid, "manifest": (push or {}).get("manifest") or {}})
    out["lp0"] = lps()
    t0 = t()
    out["sleep_http"] = _http(E + "/sleep?level=1&mode=wait")[0]
    out["sleep_s"] = t() - t0
    out["is_sleeping_1"] = _http(E + "/is_sleeping", method="GET")[1]
    _, out["seal"] = rpc("bb_seal_after_sleep", {"guest_id": gid})
    if corrupt_at is not None:
        out["poke"] = _http(E + "/stub/poke_guest", {"guest_handle": al["ipc_handle"], "offset": int(corrupt_at)})[1]
    t0 = t()
    _, out["restore"] = rpc("bb_restore_from_peer", {"guest": {"guest_id": gid, "ipc_handle": al["ipc_handle"],
                                                               "nbytes": total,
                                                               "manifest_id": (push or {}).get("manifest_id")}})
    out["restore_s"] = t() - t0
    t0 = t()
    out["wake_w_http"] = _http(E + "/wake_up?tags=weights")[0]
    out["wake_w_s"] = t() - t0
    t0 = t()
    out["wake_kv_http"] = _http(E + "/wake_up?tags=kv_cache")[0]
    out["wake_kv_s"] = t() - t0
    out["is_sleeping_2"] = _http(E + "/is_sleeping", method="GET")[1]
    out["lp1"] = lps()
    out["truth"] = _http(E + "/stub/truth", method="GET")[1]
    return out


def _golden_http(stub, root, check):
    fab_dir = os.path.join(root, "g8")
    stub.StubFabric.create(fab_dir, ngpu=3, cap_bytes=64 * MIB)
    hp, ep, hport, eport, log = spawn_pair(root, fab_dir, [m * MIB for m in W_MIB], [m * MIB for m in KV_MIB], 32)
    try:
        o = http_cycle(hport, eport, 8388608)
    finally:
        _stop(hp, ep)
    rs = o["restore"] or {}
    check("G8 HTTP push ok via /collective_rpc string arg", o["push_http"] == 200 and o["push"].get("ok"), o["push"])
    check("G8 HTTP holder attach verified", (o["attach"] or {}).get("state") == "FILLED", o["attach"])
    check("G8 HTTP restore results[0].served == PEER", rs.get("served") == "PEER", rs)
    check("G8 HTTP sleeping then awake", o["is_sleeping_1"] == {"is_sleeping": True}
          and o["is_sleeping_2"] == {"is_sleeping": False}, (o["is_sleeping_1"], o["is_sleeping_2"]))
    check("G8 HTTP prompt logprobs equal pre-sleep", o["lp0"] == o["lp1"] and o["lp0"][0] is None and len(o["lp0"]) == 5,
          (o["lp0"], o["lp1"]))
    with open(log, errors="replace") as fh:
        txt = fh.read()
    import re
    pats = {"fall asleep": r"\(EngineCore pid=\d+\) INFO \d\d-\d\d \d\d:\d\d:\d\d \[abstract\.py:346\] It took "
                           r"\d+\.\d{6} seconds to fall asleep\.",
            "wake api": r"\(APIServer pid=\d+\) INFO \d\d-\d\d \d\d:\d\d:\d\d \[api_router\.py:36\] wake up the engine "
                        r"with tags: \['weights'\]",
            "wake tags": r"\[abstract\.py:364\] It took \d+\.\d{6} seconds to wake up tags \['weights'\]\.",
            "injected": r"\[worker_base\.py:306\] Injected <class 'warm_wake_ext\.WarmWakeExt'> into "
                        r"<class '[\w.]+\.Worker'> for extended collective_rpc calls \['bb_drop_manifest', "}
    miss = [k for k, p in pats.items() if not re.search(p, txt)]
    check("G8 engine log carries the real-format lines", not miss, miss)


def _copied(eng):
    """bytes this engine process copied (the stub's `bytes:<src>-><dst>` counters)."""
    return sum(v for k, v in eng.backend.ctr.items() if k.startswith("bytes:"))


def _goldens_hardening(stub, me, root, check):
    """bite 2: the six pre-S0 review items, H1-H6."""
    import subprocess
    global MAX_STAGE_ROUNDS
    # H1: a malformed 5th allocation is reported (the old loop read only the first 4)
    fab = _new_fabric(stub, root, "h1")
    eng, _hold = _rig(stub, me, fab, w_mib=(1, 1, 1, 1, 1, 1), kv_mib=(1,))
    view = AllocatorView(eng.worker.allocator, None, None, "stub")
    p5 = list(eng.worker.allocator.pointer_to_data)[4]
    d5 = eng.worker.allocator.pointer_to_data[p5]
    good = d5.handle
    d5.handle = (good[0], good[1], int(good[2]) + 4096, good[3])
    check("H1 shape_problem reads every allocation (a bad 5th is reported)", view.shape_problem() is not None,
          view.shape_problem())
    d5.handle = good
    check("H1 ... and the repaired table reads clean", view.shape_problem() is None, view.shape_problem())
    # H2: bool is not a size (push, restore, holder alloc)
    fab = _new_fabric(stub, root, "h2")
    eng, hold = _rig(stub, me, fab)
    total = sum(eng.weight_sizes)
    a = hold.alloc("g1", total, "slotA")
    pr = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "g1", "ipc_handle": a["ipc_handle"], "nbytes": True}})
    check("H2 push nbytes=True refused BAD_SPEC:guest, 0 B moved", pr.get("reason") == "BAD_SPEC:guest"
          and _copied(eng) == 0, (pr, dict(eng.backend.ctr)))
    a2, r2 = _push(eng, hold, gid="g2")
    eng.sleep(1, "wait")
    rr = eng.rpc("bb_restore_from_peer", {"guest": {"guest_id": "g2", "ipc_handle": a2["ipc_handle"], "nbytes": True,
                                                    "manifest_id": r2.get("manifest_id")}})
    check("H2 restore nbytes=True refused BAD_SPEC:nbytes (host fallback, nothing mapped)",
          rr.get("reason") == "BAD_SPEC:nbytes" and rr.get("served") == "HOST_FALLBACK"
          and eng.truth()["weights_asleep"] == 3, (rr, eng.truth()))
    check("H2 holder alloc nbytes=True refused BAD_SIZE", hold.alloc("g3", True).get("reason") == "BAD_SIZE")
    # H3: staged rounds cap, refused before any copy
    fab = _new_fabric(stub, root, "h3")
    eng, hold = _rig(stub, me, fab)
    a = hold.alloc("g1", total, "slotA")
    old = MAX_STAGE_ROUNDS
    MAX_STAGE_ROUNDS = 1000                         # 8 MiB at 4096 B = 2048 rounds > 1000
    try:
        before = _copied(eng)
        pr = eng.rpc("bb_push_to_peer", {"guest": {"guest_id": "g1", "ipc_handle": a["ipc_handle"], "nbytes": total},
                                         "copy_mode": "staged", "stage_bytes": 4096})
        check("H3 staged push over the rounds cap refused before any copy",
              pr.get("reason") == "BAD_SPEC:stage_rounds" and _copied(eng) == before == 0, pr)
        try:
            _copy(eng.backend, 0, 0, 0, 8 * MIB, scratch=1, stage=4096)
            check("H3 _copy itself refuses past the cap", False, "accepted")
        except ValueError:
            check("H3 _copy itself refuses past the cap", True)
    finally:
        MAX_STAGE_ROUNDS = old
    # H4: a child that writes a partial line and stalls cannot hold _ready_port past its deadline
    cp = subprocess.Popen([sys.executable, "-c", "import sys,time; sys.stdout.write('BB-HOLDER-READY por'); "
                           "sys.stdout.flush(); time.sleep(30)"], stdout=subprocess.PIPE)
    t0 = time.monotonic()
    try:
        _ready_port(cp, "BB-HOLDER-READY", timeout=1.0)
        check("H4 _ready_port honours its deadline on a partial line", False, "returned")
    except RuntimeError:
        check("H4 _ready_port honours its deadline on a partial line", time.monotonic() - t0 < 2.0,
              time.monotonic() - t0)
    finally:
        cp.kill()
        cp.wait()
        cp.stdout.close()
    # H5: --scale 0 exits 2 with one ERROR line
    pr5 = subprocess.run([sys.executable, os.path.abspath(__file__), "--t3", "--scale", "0"], capture_output=True,
                         text=True, timeout=60)
    check("H5 --scale 0 -> exit 2 + one ERROR line", pr5.returncode == 2 and pr5.stderr.strip().startswith("ERROR:")
          and len(pr5.stderr.strip().splitlines()) == 1, (pr5.returncode, pr5.stderr[-200:]))
    # H6: holder stderr lands in a file
    fab_dir = os.path.join(root, "h6")
    stub.StubFabric.create(fab_dir, ngpu=3, cap_bytes=64 * MIB)
    hp, ep, hport, eport, log = spawn_pair(root, fab_dir, [m * MIB for m in W_MIB], [m * MIB for m in KV_MIB], 32,
                                           tag="h6")
    try:
        _http(f"http://127.0.0.1:{hport}/holder/alloc", {"guest_id": "x", "nbytes": 4096})
    finally:
        _stop(hp, ep)
    hl = os.path.join(root, "holderh6.log")
    check("H6 holder stderr captured to holder<tag>.log", os.path.exists(hl) and os.path.getsize(hl) > 0,
          os.path.exists(hl))


# ------------------------------------------------------------------------------------------------ T3 (realistic table)
def t3_run(scale, out_path=None):
    stub = load_stub()
    me = sys.modules[__name__]
    res = {"scale": scale, "started_unix": time.time(), "host": os.uname().nodename, "notes": []}
    # full-size layout + CAP pass (no bytes)
    tabs = {name: stub.table_sizes(name, 1) for name in ("qwen3-14b-fp8", "qwen3-8b-fp8")}
    full = {}
    for name, sizes in tabs.items():
        offs, o = [], 0
        for s in sizes:
            offs.append(o)
            o += s
        full[name] = {"n_alloc": len(sizes), "total": o, "total_gib": round(o / GIB, 3),
                      "all_2mib_multiples": all(s % (2 * MIB) == 0 for s in sizes),
                      "max_offset": offs[-1], "max_alloc": max(sizes)}
    res["full_size"] = full
    meta = stub.MetaBackend()
    cap = int(35.62 * GIB)
    combos = {"14b+8b+8b": ["qwen3-14b-fp8", "qwen3-8b-fp8", "qwen3-8b-fp8"],
              "14b+8b+8b+8b": ["qwen3-14b-fp8", "qwen3-8b-fp8", "qwen3-8b-fp8", "qwen3-8b-fp8"]}
    cap_res = {}
    for cname, models in combos.items():
        h = GuestHolder(meta, 0, cap, "meta")
        rows = [h.alloc(f"{cname}-{i}", full[m]["total"], m) for i, m in enumerate(models)]
        cap_res[cname] = {"accepted": [r.get("ok") for r in rows], "used_gib": round(h.used / GIB, 3),
                          "refused": [r.get("reason") for r in rows if not r.get("ok")]}
    res["cap_35.62GiB"] = cap_res
    # byte pass at 1/scale, cross-process over HTTP
    sizes = stub.table_sizes("qwen3-14b-fp8", scale)
    total = sum(sizes)
    kv = [max(4096, (1 << 30) // scale)] * 2
    root = _tmp_root()
    try:
        fab_dir = os.path.join(root, "t3")
        stub.StubFabric.create(fab_dir, ngpu=3, cap_bytes=int(total * 2.2 + sum(kv) * 2 + 64 * MIB))
        hp, ep, hport, eport, log = spawn_pair(root, fab_dir, sizes, kv, int(total * 2.2 / MIB) + 8, tag="-t3")
        try:
            ok_run = http_cycle(hport, eport, total, gid="t3a")
            bad_run = http_cycle(hport, eport, total, gid="t3b", corrupt_at=sizes[0] + sizes[1] // 2)
        finally:
            _stop(hp, ep)
        with open(log, errors="replace") as fh:
            res["engine_log_tail"] = fh.read().splitlines()[-12:]
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    for k, run in (("clean", ok_run), ("corrupted_guest", bad_run)):
        r = run["restore"] or {}
        res[k] = {"n_alloc": len(sizes), "total_bytes": total, "push": run["push"], "seal": run["seal"],
                  "restore": {kk: r.get(kk) for kk in ("served", "reason", "bytes", "bytes_copied", "n_alloc", "copy_s",
                                                         "verify_s", "total_s", "bad_allocs", "rolled_back")},
                  "restore_http_s": round(run["restore_s"], 4), "wake_weights_s": round(run["wake_w_s"], 4),
                  "wake_kv_s": round(run["wake_kv_s"], 4), "sleep_s": round(run["sleep_s"], 4),
                  "logprobs_equal": run["lp0"] == run["lp1"], "awake_after": run["is_sleeping_2"],
                  "truth": run["truth"]}
    res["pass"] = bool(res["clean"]["restore"]["served"] == "PEER" and res["clean"]["logprobs_equal"]
                       and res["corrupted_guest"]["restore"]["served"] == "HOST_FALLBACK"
                       and res["corrupted_guest"]["restore"]["reason"] == "DIGEST_MISMATCH"
                       and res["corrupted_guest"]["logprobs_equal"]
                       and cap_res["14b+8b+8b"]["accepted"] == [True, True, True]
                       and cap_res["14b+8b+8b+8b"]["accepted"] == [True, True, True, False])
    res["finished_unix"] = time.time()
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(txt + "\n")
    print(txt)
    if not res["pass"]:
        print("T3-FAIL see JSON", file=sys.stderr)
        return 1
    return 0


def s0_digest(mib=1024):
    """S0 item (run ON THE BOX): the GPU digest equals the numpy digest on hand fixtures and on random full-range data
    (so the int32 wrap is the same on CUDA), the __cuda_array_interface__ path reads a raw device pointer, and the
    digest rate in GB/s. Exit 0 PASS, 1 FAIL, 3 UNCHECKED (no CUDA here: not a pass)."""
    try:
        import numpy as np
        import torch
    except ImportError as e:
        print(f"UNCHECKED: {e}")
        return 3
    if not torch.cuda.is_available():
        print("UNCHECKED: no CUDA device")
        return 3
    ok = True
    for words, cw, want in (((1, 2, 3, 4, 5), 2, "pws1:20:000000000000000f:0000000000000020"),
                            ((2 ** 31 - 1, 1), 2, "pws1:8:ffffffff80000000:ffffffff80000000"),
                            ((3, 4, 1, 2, 5), 2, "pws1:20:000000000000000f:000000000000001c")):
        t = torch.tensor(words, dtype=torch.int32, device="cuda:0")
        got = digest_from_sums(4 * len(words), chunk_sums_torch(t, cw))
        print(("ok   " if got == want else "FAIL ") + f"fixture {words} -> {got}")
        ok &= got == want
    n = (int(mib) << 20) // 4
    t = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g = chunk_sums_torch(t, DEFAULT_CHUNK_WORDS)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    c = chunk_sums_numpy(t.cpu().numpy().view(np.uint8), DEFAULT_CHUNK_WORDS)
    v = torch.as_tensor(_CAI(t.data_ptr(), n), device=torch.device("cuda", 0))
    cai = chunk_sums_torch(v, DEFAULT_CHUNK_WORDS)
    print(("ok   " if g == c else "FAIL ") + f"random {mib} MiB: cuda == numpy ({len(g)} chunks)")
    print(("ok   " if cai == g else "FAIL ") + "__cuda_array_interface__ view == the tensor")
    dt = max(dt, 1e-9)
    print(f"rate {n * 4 / dt / 1e9:.1f} GB/s over {mib} MiB (first call includes warm-up); "
          f"16.46 GB at this rate = {16.456e9 / (n * 4 / dt):.3f} s")
    ok &= g == c and cai == g
    print("S0-DIGEST-" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ CLI
def serve_holder(a):
    if a.stub_fabric:
        stub = load_stub()
        fab = stub.StubFabric.open(a.stub_fabric)
        cvd = [int(x) for x in a.stub_cvd.split(",")]
        be = stub.StubBackend(fab, cvd, f"holder-pid{os.getpid()}", chunk_sums_numpy)
        label = f"stub-gpu{cvd[0]}"
    else:
        be = CudaRtBackend()
        label = f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?').split(',')[0]}"
    cap = int(a.cap_mib * MIB) if a.cap_mib else int(a.cap_gib * GIB)
    holder = GuestHolder(be, 0, cap, label, log=(lambda ev: None) if a.quiet else _stderr_log,
                         allow_corrupt=a.allow_corrupt)
    srv = make_holder_server(holder, a.port)
    print(f"BB-HOLDER-READY port={srv.server_address[1]} cap={cap} label={label}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--vllm-src", default=os.environ.get("VLLM_SRC", ""),
                   help="a vLLM git checkout carrying tag v0.30.0, for G10b (read-only); '' to skip")
    p.add_argument("--t3", action="store_true")
    p.add_argument("--scale", type=int, default=16)
    p.add_argument("--out", default=None, help="T3 JSON path (WRITTEN)")
    p.add_argument("--serve-holder", action="store_true")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--cap-gib", type=float, default=35.62)
    p.add_argument("--cap-mib", type=float, default=0.0)
    p.add_argument("--stub-fabric", default=None)
    p.add_argument("--stub-cvd", default="0,1,2")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--allow-corrupt", action="store_true", help="enable /holder/corrupt (S0 adversarial step only)")
    p.add_argument("--s0-digest", action="store_true", help="GPU digest check for the box (exit 3 = no CUDA here)")
    p.add_argument("--s0-mib", type=int, default=1024)
    a = p.parse_args(argv)
    if a.s0_digest:
        return s0_digest(a.s0_mib)
    if a.self_test:
        src = a.vllm_src if a.vllm_src and os.path.isdir(a.vllm_src) else None
        _install_term_handlers()
        return self_test(src)
    if a.t3:
        if a.scale < 1:                                 # bite-2 H5: --scale 0 divided by zero
            print(f"ERROR: --scale must be >= 1 (got {a.scale})", file=sys.stderr)
            return 2
        _install_term_handlers()
        return t3_run(a.scale, a.out)
    if a.serve_holder:
        return serve_holder(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
