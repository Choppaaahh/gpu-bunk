#!/usr/bin/env python3
"""warm_wake_stub.py -- B7: CPU stub engines for the WARM wake (bite 1).

WHAT. Everything the WARM path touches, on a CPU, with no vLLM and no torch:
  * StubFabric / StubBackend -- N fake GPUs as files (one per allocation) under a directory, shared ACROSS PROCESSES
    (fcntl locks). It enforces the CUDA rules the design depends on: a process may touch only memory it allocated or
    memory it opened by IPC; an IPC handle opens only if the owning GPU is in the process's CUDA_VISIBLE_DEVICES; a
    process cannot open its own export; per-GPU capacity (OOM = rc 2); an unmapped VA cannot be copied to. Every failed
    call sets the per-thread last error and a sticky code poisons every later call, as on a real context.
    Error codes are the CUDA ones where the cause is certain (2 OOM, 101 invalid device, 400 invalid handle, 700 illegal
    address); WHICH code real CUDA returns for "IPC open of a GPU not in CUDA_VISIBLE_DEVICES" is not verified (101 here).
  * StubCuMemAllocator -- vllm/device_allocator/cumem.py @ v0.30.0 sleep (L229-294) and wake_up (L327-356),
    transliterated; StubExecutor -- v1/executor/abstract.py L337-375; StubEngineCore -- v1/engine/core.py L879-937
    (pause modes abort/wait/keep, the scheduler resumes only when no tag is asleep); inject_extension --
    v1/worker/worker_base.py L285-311 (the conflict assert included); run_method -- v1/serial_utils.py L486-510.
  * an HTTP engine (`--serve-engine`): GET /health /is_sleeping /v1/models, POST /sleep /wake_up /collective_rpc
    /v1/completions (prompt token ids, prompt_logprobs, stream), plus stub-only /stub/truth /stub/fault /stub/poke_guest.
    Log lines are copied from real v0.30.0 logs: (private report) and
    (private report)
  The "model" is a deterministic function of the weights' bytes: prompt logprobs change if any weight byte changes, so
  a wrong restore is visible as an identity failure (the stub analogue of S0's teacher-forced logprob check).

WHAT IT CANNOT SHOW: anything about CUDA itself -- cross-process IPC on real GPUs, VMM access rules, peer contexts,
NVLink rates, CUDA_VISIBLE_DEVICES reordering inside vLLM. That list is the S0 checklist in
(private report)

Output contract:
  --self-test     exit 0 + `ok ...` lines + `STUB-SELFTEST-OK <n>/<n>`; exit 1 + `FAIL ...` + `STUB-SELFTEST-FAIL k/n`.
                  Writes only a tempdir; loopback HTTP; one Python child.
  --serve-engine  serves until killed; prints exactly one stdout line `BB-STUB-READY port=<n>`; logs go to stderr.
                  Needs --ext-file (the extension is loaded from that PATH, never by module name).

Top pitfalls (observed):
  1. **`/sleep` defaults to mode=abort and drops in-flight requests** (stock api_router L25) -- this stub keeps the
     same default; owners sleep with mode=wait.
  2. **A stub that prints the parser's expected format proves nothing** -- the log formats here were copied from
     real logs, and the ext self-test (G8) matches them by regex built from those lines.
"""

import argparse
import base64
import collections
import dataclasses
import fcntl
import hashlib
import json
import mmap
import os
import struct
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MIB = 1 << 20
GIB = 1 << 30
DEV_SHIFT = 52
ALIAS_BIT = 1 << 51
AID_SHIFT = 36
AID_MASK = (1 << 15) - 1
OFF_MASK = (1 << 36) - 1
HOST_CODE = 2047
MAGIC = b"BBSTUB01"
COPY_CHUNK = 8 * MIB


class StubCudaError(RuntimeError):
    def __init__(self, op, code, detail=""):
        super().__init__(f"{op}: rc={int(code)} {detail}".strip())
        self.op, self.code = op, int(code)


def mkptr(code, aid, off=0, alias=False):
    return (int(code) << DEV_SHIFT) | (ALIAS_BIT if alias else 0) | (int(aid) << AID_SHIFT) | int(off)


def split_ptr(ptr):
    ptr = int(ptr)
    return ptr >> DEV_SHIFT, bool(ptr & ALIAS_BIT), (ptr >> AID_SHIFT) & AID_MASK, ptr & OFF_MASK


# ------------------------------------------------------------------------------------------------ the fabric (files)
class StubFabric:
    """root/fabric.json {ngpu, cap}; root/lock; root/seq; root/d<code>/<aid>.meta (json) + <aid>.bin (mapped bytes)."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "fabric.json")) as fh:
            cfg = json.load(fh)
        self.ngpu, self.cap = int(cfg["ngpu"]), int(cfg["cap"])

    @classmethod
    def create(cls, root, ngpu=3, cap_bytes=64 * MIB):
        os.makedirs(root, exist_ok=True)
        for c in list(range(1, ngpu + 1)) + [HOST_CODE]:
            os.makedirs(os.path.join(root, f"d{c}"), exist_ok=True)
        with open(os.path.join(root, "fabric.json"), "w") as fh:
            json.dump({"ngpu": int(ngpu), "cap": int(cap_bytes), "created_unix": time.time()}, fh)
        with open(os.path.join(root, "seq"), "w") as fh:
            fh.write("0")
        open(os.path.join(root, "lock"), "a").close()
        return cls(root)

    @classmethod
    def open(cls, root):
        return cls(root)

    @contextmanager
    def locked(self):
        with open(os.path.join(self.root, "lock"), "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def next_aid(self):
        p = os.path.join(self.root, "seq")
        with open(p) as fh:
            n = int(fh.read().strip() or 0) + 1
        if n > AID_MASK:
            raise StubCudaError("malloc", 2, "stub address space exhausted")
        with open(p, "w") as fh:
            fh.write(str(n))
        return n

    def mpath(self, code, aid):
        return os.path.join(self.root, f"d{code}", f"{aid}.meta")

    def bpath(self, code, aid):
        return os.path.join(self.root, f"d{code}", f"{aid}.bin")

    def read_meta(self, code, aid):
        try:
            with open(self.mpath(code, aid)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def write_meta(self, code, aid, meta):
        p = self.mpath(code, aid)
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp, p)

    def used(self, code):
        d = os.path.join(self.root, f"d{code}")
        return sum(e.stat().st_size for e in os.scandir(d) if e.name.endswith(".bin"))

    def create_bin(self, code, aid, size, fill=0xEE):
        with open(self.bpath(code, aid), "wb") as fh:
            if fill == 0:
                fh.truncate(size)
            else:
                blk = bytes([fill]) * min(size, COPY_CHUNK)
                left = size
                while left > 0:
                    fh.write(blk[: min(left, len(blk))])
                    left -= min(left, len(blk))

    def poke_handle(self, handle, offset, xor=0xFF):
        code, aid, _secret = parse_handle(handle)
        with open(self.bpath(code, aid), "r+b") as fh:
            fh.seek(int(offset))
            b = fh.read(1)[0]
            fh.seek(int(offset))
            fh.write(bytes([b ^ xor]))
        return {"ok": True, "old": b, "new": b ^ xor}


def make_handle(code, aid, secret_hex):
    raw = MAGIC + struct.pack("<HI", code, aid) + bytes.fromhex(secret_hex)
    return raw + b"\0" * (64 - len(raw))


def parse_handle(handle):
    handle = bytes(handle)
    if len(handle) != 64 or handle[:8] != MAGIC:
        raise StubCudaError("cudaIpcOpenMemHandle", 400, "not a stub IPC handle")
    code, aid = struct.unpack("<HI", handle[8:14])
    return code, aid, handle[14:30].hex()


class HostBuf:
    """Stands in for the pinned torch CPU tensor stock sleep allocates (cumem.py L264-269): data_ptr/numel/element_size.
    Dropping the last reference frees it, as `data.cpu_backup_tensor = None` does in stock."""

    def __init__(self, be, ptr, n):
        self.be, self.ptr, self.n = be, ptr, n

    def data_ptr(self):
        return self.ptr

    def numel(self):
        return self.n

    def element_size(self):
        return 1

    def __del__(self):
        try:
            self.be.host_free(self.ptr)
        except Exception:  # noqa: BLE001
            pass


class StubBackend:
    """One process's view of the fabric. `cvd` = CUDA_VISIBLE_DEVICES as a list of PHYSICAL ids (logical i -> cvd[i]).
    `proc` identifies the process for ownership (two backends with different procs in one Python process act as two
    processes). Implements the ext's backend interface."""

    kind = "stub"

    def __init__(self, fab, cvd, proc, sums_fn):
        self.fab, self.cvd, self.proc, self.sums_fn = fab, [int(x) for x in cvd], str(proc), sums_fn
        self.last = 0
        self.sticky = 0
        self.faults = []
        self.ctr = collections.Counter()
        self.watch = set()        # aids whose every touch is counted (kv allocations, for G4a)
        self.kv_touched = 0

    # ---- error plumbing (every failure sets the last error; sticky poisons everything after)
    def _enter(self, op):
        if self.sticky:
            self.last = self.sticky
            raise StubCudaError(op, self.sticky, "sticky (context poisoned)")
        for f in list(self.faults):
            if f["op"] == op:
                f["left"] -= 1
                if f["left"] <= 0:
                    self.faults.remove(f)
                    self.last = f["code"]
                    if f["sticky"]:
                        self.sticky = f["code"]
                    raise StubCudaError(op, f["code"], "injected")

    def _fail(self, op, code, detail):
        self.last = int(code)
        raise StubCudaError(op, code, detail)

    def install_fault(self, op, nth, code, sticky=False):
        self.faults.append({"op": op, "left": int(nth), "code": int(code), "sticky": bool(sticky)})

    def _phys(self, op, dev):
        if not 0 <= int(dev) < len(self.cvd):
            self._fail(op, 101, f"invalid device {dev}")
        return self.cvd[int(dev)]

    def _resolve(self, op, ptr, n):
        code, alias, aid, off = split_ptr(ptr)
        meta = self.fab.read_meta(code, aid)
        if meta is None:
            self._fail(op, 1, f"invalid pointer {ptr:#x} (freed or never allocated)")
        if alias:
            if int(meta.get("opens", {}).get(self.proc, 0)) <= 0:
                self._fail(op, 1, "IPC alias not opened by this process")
            kind = "peer"
        else:
            if meta["owner"] != self.proc:
                self._fail(op, 1, f"pointer owned by {meta['owner']}, not {self.proc}")
            kind = "host" if code == HOST_CODE else "local"
        if not os.path.exists(self.fab.bpath(code, aid)):
            self._fail(op, 1, "unmapped address")
        if off + int(n) > int(meta["size"]):
            self._fail(op, 1, "out of bounds")
        if aid in self.watch:
            self.kv_touched += 1
        return self.fab.bpath(code, aid), off, kind

    # ---- the ext backend interface
    def malloc(self, dev, n):
        self._enter("malloc")
        phys = self._phys("cudaMalloc", dev)
        code = phys + 1
        with self.fab.locked():
            if self.fab.used(code) + int(n) > self.fab.cap:
                self._fail("cudaMalloc", 2, "out of memory")
            aid = self.fab.next_aid()
            self.fab.write_meta(code, aid, {"owner": self.proc, "size": int(n), "kind": "malloc",
                                            "secret": uuid.uuid4().hex, "opens": {}})
            self.fab.create_bin(code, aid, int(n), fill=0)
        return mkptr(code, aid)

    def free(self, dev, ptr):
        self._enter("free")
        code, alias, aid, off = split_ptr(ptr)
        with self.fab.locked():
            meta = self.fab.read_meta(code, aid)
            if alias or off or meta is None or meta["owner"] != self.proc or meta["kind"] != "malloc":
                self._fail("cudaFree", 1, "not a base pointer this process allocated")
            others = {p: c for p, c in meta.get("opens", {}).items() if int(c) > 0}
            if others:
                self._fail("cudaFree", 1, f"stub-strict: freed while IPC-opened by {sorted(others)}")
            for p in (self.fab.bpath(code, aid), self.fab.mpath(code, aid)):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    def ipc_export(self, dev, ptr):
        self._enter("ipc_export")
        code, alias, aid, off = split_ptr(ptr)
        meta = self.fab.read_meta(code, aid)
        if alias or off or meta is None or meta["owner"] != self.proc or meta["kind"] != "malloc":
            self._fail("cudaIpcGetMemHandle", 1, "not an exportable base pointer")
        return make_handle(code, aid, meta["secret"])

    def ipc_open(self, dev, handle):
        self._enter("ipc_open")
        try:
            code, aid, secret = parse_handle(handle)
        except StubCudaError as e:
            self._fail("cudaIpcOpenMemHandle", e.code, str(e))
        self._phys("cudaIpcOpenMemHandle", dev)
        if code - 1 not in self.cvd:
            self._fail("cudaIpcOpenMemHandle", 101, f"GPU {code - 1} not in CUDA_VISIBLE_DEVICES {self.cvd}")
        with self.fab.locked():
            meta = self.fab.read_meta(code, aid)
            if meta is None or meta.get("secret") != secret:
                self._fail("cudaIpcOpenMemHandle", 400, "invalid resource handle (freed or forged)")
            if meta["owner"] == self.proc:
                self._fail("cudaIpcOpenMemHandle", 400, "a process cannot open its own export")
            ops = meta.setdefault("opens", {})
            ops[self.proc] = int(ops.get(self.proc, 0)) + 1
            self.fab.write_meta(code, aid, meta)
        return mkptr(code, aid, alias=True)

    def ipc_close(self, dev, ptr):
        self._enter("ipc_close")
        code, alias, aid, _off = split_ptr(ptr)
        with self.fab.locked():
            meta = self.fab.read_meta(code, aid)
            if not alias or meta is None or int(meta.get("opens", {}).get(self.proc, 0)) <= 0:
                self._fail("cudaIpcCloseMemHandle", 400, "not an open IPC mapping")
            meta["opens"][self.proc] = int(meta["opens"][self.proc]) - 1
            self.fab.write_meta(code, aid, meta)

    def memcpy(self, dev, dst, src, n):
        self._enter("memcpy")
        self._phys("cudaMemcpy", dev)
        n = int(n)
        sp, so, sk = self._resolve("cudaMemcpy", src, n)
        dp, do, dk = self._resolve("cudaMemcpy", dst, n)
        with open(sp, "rb") as fs, open(dp, "r+b") as fd:
            left, a, b = n, so, do
            while left > 0:
                c = min(left, COPY_CHUNK)
                buf = os.pread(fs.fileno(), c, a)
                os.pwrite(fd.fileno(), buf, b)
                a, b, left = a + c, b + c, left - c
        self.ctr[f"copies:{sk}->{dk}"] += 1
        self.ctr[f"bytes:{sk}->{dk}"] += n

    def flip_byte(self, dev, ptr, xor=0xFF):
        self._enter("flip_byte")
        path, off, kind = self._resolve("flip_byte", ptr, 1)
        if kind == "peer":
            self._fail("flip_byte", 1, "flip only memory this process owns")
        with open(path, "r+b") as fh:
            old = os.pread(fh.fileno(), 1, off)[0]
            os.pwrite(fh.fileno(), bytes([old ^ (int(xor) & 0xFF)]), off)
        return old, old ^ (int(xor) & 0xFF)

    def synchronize_rc(self, dev):
        if self.sticky:
            self.last = self.sticky
            return self.sticky
        try:
            self._enter("sync")
        except StubCudaError as e:
            return e.code
        return 0

    def clear_last_error(self):
        if self.sticky:
            return self.sticky
        rc, self.last = self.last, 0
        return rc

    def peek_last_error(self):
        return self.sticky or self.last

    def chunk_sums(self, dev, ptr, nbytes, where, chunk_words=262144):
        self._enter("digest")
        import numpy as np
        path, off, kind = self._resolve("digest", ptr, nbytes)
        if (where == "host") != (kind == "host"):
            self._fail("digest", 1, f"asked for {where} memory, pointer is {kind}")
        nbytes = int(nbytes)
        if nbytes == 0:
            return []
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                arr = np.frombuffer(mm, dtype=np.uint8, count=nbytes, offset=off)
                sums = self.sums_fn(arr, chunk_words)
                del arr
            finally:
                mm.close()
        self.ctr["digest_bytes"] += nbytes
        return sums

    # ---- VMM (what cumem's C extension does) + host memory
    def vmm_reserve(self, dev, size):
        phys = self._phys("cuMemAddressReserve", dev)
        code = phys + 1
        with self.fab.locked():
            aid = self.fab.next_aid()
            self.fab.write_meta(code, aid, {"owner": self.proc, "size": int(size), "kind": "vmm", "secret": "",
                                            "opens": {}})
        return mkptr(code, aid)

    def vmm_map(self, ptr):
        self._enter("create_and_map")
        code, _alias, aid, _off = split_ptr(ptr)
        meta = self.fab.read_meta(code, aid)
        if meta is None or meta["owner"] != self.proc or meta["kind"] != "vmm":
            self._fail("create_and_map", 1, "not this process's VMM reservation")
        if aid in self.watch:
            self.kv_touched += 1
        with self.fab.locked():
            if os.path.exists(self.fab.bpath(code, aid)):
                self._fail("create_and_map", 1, "range already mapped")
            if self.fab.used(code) + int(meta["size"]) > self.fab.cap:
                self._fail("create_and_map", 2, "CUDA_ERROR_OUT_OF_MEMORY")
            self.fab.create_bin(code, aid, int(meta["size"]), fill=0xEE)   # fresh physical memory = garbage

    def vmm_unmap(self, ptr):
        self._enter("unmap_and_release")
        code, _alias, aid, _off = split_ptr(ptr)
        try:
            os.remove(self.fab.bpath(code, aid))
        except FileNotFoundError:
            self._fail("unmap_and_release", 1, "range not mapped")

    def host_alloc(self, n):
        self._enter("host_alloc")
        with self.fab.locked():
            aid = self.fab.next_aid()
            self.fab.write_meta(HOST_CODE, aid, {"owner": self.proc, "size": int(n), "kind": "host", "secret": "",
                                                 "opens": {}})
            self.fab.create_bin(HOST_CODE, aid, int(n), fill=0)
        return HostBuf(self, mkptr(HOST_CODE, aid), int(n))

    def host_free(self, ptr):
        code, _a, aid, _o = split_ptr(ptr)
        for p in (self.fab.bpath(code, aid), self.fab.mpath(code, aid)):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    def device_free(self, dev):
        return self.fab.cap - self.fab.used(self.cvd[int(dev)] + 1)

    # ---- test helpers (bypass ownership; never used by the ext)
    def _raw_path(self, ptr):
        code, _alias, aid, off = split_ptr(ptr)
        return self.fab.bpath(code, aid), off

    def peek(self, ptr):
        return self.peek_bytes(ptr, 1)[0]

    def peek_bytes(self, ptr, n):
        p, off = self._raw_path(ptr)
        with open(p, "rb") as fh:
            return os.pread(fh.fileno(), int(n), off)

    def poke(self, ptr, value):
        self.poke_bytes(ptr, bytes([int(value) & 0xFF]))

    def poke_bytes(self, ptr, data):
        p, off = self._raw_path(ptr)
        with open(p, "r+b") as fh:
            os.pwrite(fh.fileno(), bytes(data), off)


class MetaBackend:
    """Size-only backend for the full-size CAP pass (no bytes anywhere)."""

    kind = "meta"

    def __init__(self):
        self.n = 0

    def malloc(self, dev, n):
        self.n += 1
        return mkptr(1, self.n % AID_MASK)

    def ipc_export(self, dev, ptr):
        return b"\0" * 64

    def free(self, dev, ptr):
        return None

    def clear_last_error(self):
        return 0


# ------------------------------------------------------------------------------------------------ vLLM transliterations
def _ts():
    return time.strftime("%m-%d %H:%M:%S")


def vlog(proc, where, msg, level="INFO", stream=None):
    """The real v0.30.0 line shape, e.g.
    (EngineCore pid=27433) INFO 10-13 23:54:46 [abstract.py:346] It took 5.589358 seconds to fall asleep."""
    print(f"({proc} pid={os.getpid()}) {level} {_ts()} [{where}] {msg}", file=stream or sys.stderr, flush=True)


@dataclasses.dataclass
class StubAllocationData:
    handle: tuple
    tag: str
    cpu_backup_tensor: object = None
    is_asleep: bool = False


class StubCuMemAllocator:
    """cumem.py @ v0.30.0, sleep L229-294 and wake_up L327-356, with libcudart.cudaMemcpy -> backend.memcpy and the C
    create_and_map/unmap_and_release -> backend.vmm_map/vmm_unmap. One allocate() = one cumem handle."""

    default_tag = "default"

    def __init__(self, be, dev, log):
        self.be, self.dev, self.log = be, int(dev), log
        self.pointer_to_data = {}
        self.current_tag = self.default_tag

    def create_and_map(self, handle):
        self.be.vmm_map(handle[2])

    def unmap_and_release(self, handle):
        self.be.vmm_unmap(handle[2])

    def allocate(self, size, tag):
        ptr = self.be.vmm_reserve(self.dev, int(size))
        handle = (self.dev, int(size), ptr, 0)
        self.create_and_map(handle)
        self.pointer_to_data[ptr] = StubAllocationData(handle, tag)
        return ptr

    def sleep(self, offload_tags=None):
        if offload_tags is None:
            offload_tags = (self.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        assert isinstance(offload_tags, tuple)
        total_bytes = 0
        backup_bytes = 0
        has_policy_conflict = False
        for ptr, data in self.pointer_to_data.items():
            if data.is_asleep:
                requests_offload = data.tag in offload_tags
                was_offloaded = data.cpu_backup_tensor is not None
                if requests_offload != was_offloaded:
                    has_policy_conflict = True
                continue
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                size_in_bytes = handle[1]
                cpu_backup_tensor = self.be.host_alloc(size_in_bytes)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                self.be.memcpy(self.dev, cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            try:
                self.unmap_and_release(handle)
            finally:
                data.is_asleep = True
        self.log("EngineCore", "cumem.py:278",
                 "CuMemAllocator: sleep freed %.2f GiB memory in total, of which %.2f GiB is backed up in CPU and the "
                 "rest %.2f GiB is discarded directly." % (total_bytes / GIB, backup_bytes / GIB,
                                                           (total_bytes - backup_bytes) / GIB))
        if has_policy_conflict:
            self.log("EngineCore", "cumem.py:288", "CuMemAllocator: sleep cannot change the policy of already-asleep "
                     "allocations; the existing policy was kept.", "WARNING")

    def wake_up(self, tags=None):
        for ptr, data in self.pointer_to_data.items():
            if not data.is_asleep:
                continue
            if tags is None or data.tag in tags:
                handle = data.handle
                self.create_and_map(handle)
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        self.be.memcpy(self.dev, ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    def get_current_usage(self):
        return sum(d.handle[1] for d in self.pointer_to_data.values())


def fill_pattern(be, ptr, size, i, seed):
    import numpy as np
    j0 = 0
    while j0 < size:
        j = np.arange(j0, min(size, j0 + COPY_CHUNK), dtype=np.int64)
        be.poke_bytes(ptr + j0, ((31 * i + 7 * j + 5 + seed) % 256).astype(np.uint8).tobytes())
        j0 += COPY_CHUNK


class StubWorkerBase:
    """The worker the extension is mixed into. sleep/wake_up follow gpu_worker.py L245-304 through the default cumem
    SleepModeBackend (sleep_mode_backend.py L120-133): level 1 offloads `weights`, discards the rest."""

    def __init__(self, be, dev, weight_sizes, kv_sizes, seed, log):
        self.backend, self.dev, self.log = be, int(dev), log
        self.allocator = StubCuMemAllocator(be, dev, log)
        self.weight_ptrs, self.kv_ptrs = [], []
        for i, s in enumerate(weight_sizes):
            p = self.allocator.allocate(s, "weights")
            fill_pattern(be, p, int(s), i, seed)
            self.weight_ptrs.append(p)
        for s in kv_sizes:
            p = self.allocator.allocate(s, "kv_cache")
            be.poke_bytes(p, b"\xaa" * min(int(s), 4096))
            self.kv_ptrs.append(p)
        be.watch = {split_ptr(p)[2] for p in self.kv_ptrs}
        self._backend_logged = False

    def sleep(self, level=1):
        free_before = self.backend.device_free(self.dev)
        if not self._backend_logged:
            self.log("EngineCore", "sleep_mode_backend.py:183", "Using sleep-mode backend: cumem")
            self._backend_logged = True
        self.allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_after = self.backend.device_free(self.dev)
        self.log("EngineCore", "gpu_worker.py:276", "Sleep mode freed %s GiB memory, %s GiB memory is still in use."
                 % (round((free_after - free_before) / GIB, 2), round((self.backend.fab.cap - free_after) / GIB, 2)))

    def wake_up(self, tags=None):
        self.allocator.wake_up(tags)

    def stub_logprobs(self, token_ids):
        """A deterministic 'model': logprob(token) = f(digest of every weights byte, token, position). Raises on a stale
        CUDA error the way torch raises on its next launch check consuming it."""
        stale = self.backend.peek_last_error()
        if stale:
            self.backend.clear_last_error()
            raise RuntimeError(f"CUDA error: rc={stale} (a stale last-error; torch raises it at its next launch check)")
        for d in self.allocator.pointer_to_data.values():
            if d.is_asleep:
                raise RuntimeError(f"forward with an asleep {d.tag} allocation")
        h = hashlib.sha256()
        for p in self.weight_ptrs:
            d = self.allocator.pointer_to_data[p]
            h.update(json.dumps(self.backend.chunk_sums(self.dev, p, d.handle[1], "device", 262144)).encode())
        dig = h.hexdigest()
        out = [None]
        for pos, t in enumerate(token_ids[1:], start=1):
            v = int(hashlib.sha256(f"{dig}:{t}:{pos}".encode()).hexdigest()[:8], 16)
            out.append(-(v % 100000) / 10000.0)
        return out


def inject_extension(worker_class, worker_extension_cls, log):
    """v1/worker/worker_base.py L285-311 @ v0.30.0, verbatim logic."""
    extended_calls = []
    if worker_extension_cls not in worker_class.__bases__:
        for attr in dir(worker_extension_cls):
            if attr.startswith("__"):
                continue
            assert not hasattr(worker_class, attr), (
                f"Worker class {worker_class} already has an attribute"
                f" {attr}, which conflicts with the worker"
                f" extension class {worker_extension_cls}."
            )
            if callable(getattr(worker_extension_cls, attr)):
                extended_calls.append(attr)
        worker_class.__bases__ = worker_class.__bases__ + (worker_extension_cls,)
        log("EngineCore", "worker_base.py:306", "Injected %s into %s for extended collective_rpc calls %s"
            % (worker_extension_cls, worker_class, extended_calls))
    return extended_calls


def run_method(obj, method, args, kwargs):
    """v1/serial_utils.py L486-510 (the str branch; bytes/callable are not reachable over HTTP)."""
    if isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(f"Method {method!r} is not implemented.") from None
    else:
        raise TypeError("stub supports string methods only")
    return func(*args, **kwargs)


class StubExecutor:
    """v1/executor/abstract.py L337-375 @ v0.30.0."""

    def __init__(self, worker, log):
        self.worker, self.log = worker, log
        self.is_sleeping = False
        self.sleeping_tags = set()

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [run_method(self.worker, method, tuple(args), kwargs or {})]

    def sleep(self, level=1):
        if self.is_sleeping:
            self.log("EngineCore", "abstract.py:339", "Executor is already sleeping.", "WARNING")
            return
        t0 = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        t1 = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        self.log("EngineCore", "abstract.py:346", "It took %.6f seconds to fall asleep." % (t1 - t0))

    def wake_up(self, tags=None):
        if not self.is_sleeping:
            self.log("EngineCore", "abstract.py:352", "Executor is not sleeping.", "WARNING")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    self.log("EngineCore", "abstract.py:357", "Tag %s is not in sleeping tags %s"
                             % (tag, self.sleeping_tags), "WARNING")
                    return
        t0 = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        t1 = time.perf_counter()
        self.log("EngineCore", "abstract.py:364", "It took %.6f seconds to wake up tags %s."
                 % (t1 - t0, tags if tags is not None else self.sleeping_tags))
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False


class StubEngineCore:
    """v1/engine/core.py L841-937 @ v0.30.0, reduced: pause modes, sleep, wake_up (the scheduler resumes only once no tag
    is asleep), collective_rpc. Utility calls are serialized like the engine core's busy loop."""

    def __init__(self, executor, log):
        self.executor, self.log = executor, log
        self.cv = threading.Condition()
        self.util = threading.Lock()
        self.paused = False
        self.inflight = 0
        self.queued = 0
        self.abort_gen = 0

    def pause_scheduler(self, mode="abort"):
        if mode not in ("abort", "wait", "keep"):
            raise ValueError(f"Invalid pause mode: {mode}")
        with self.cv:
            self.paused = True
            if mode == "abort":
                self.abort_gen += 1
            if mode == "wait":
                if self.inflight:
                    self.log("EngineCore", "core.py:913", "Waiting for in-flight requests to complete before "
                             "sleeping...")
                while self.inflight:
                    self.cv.wait(0.1)

    def resume_scheduler(self):
        with self.cv:
            self.paused = False
            self.cv.notify_all()

    def sleep(self, level=1, mode="abort"):
        with self.util:
            self.pause_scheduler(mode)
            if level < 1:
                return
            self.executor.sleep(level)

    def wake_up(self, tags=None):
        with self.util:
            if tags is not None and "scheduling" in tags:
                tags = [t for t in tags if t != "scheduling"]
            if tags is None or tags:
                self.executor.wake_up(tags)
            if not self.executor.is_sleeping:
                self.resume_scheduler()

    def is_sleeping(self):
        return self.paused or self.executor.is_sleeping

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        with self.util:
            return self.executor.collective_rpc(method, timeout, args, kwargs)

    def run_request(self, fn, timeout=120.0):
        t0 = time.time()
        with self.cv:
            self.queued += 1
            try:
                while self.paused:        # new adds are queued while paused (core.py L846)
                    if time.time() - t0 > timeout:
                        raise TimeoutError("engine paused")
                    self.cv.wait(0.05)
            finally:
                self.queued -= 1
            self.inflight += 1
            gen = self.abort_gen
        try:
            out = fn()
        finally:
            with self.cv:
                self.inflight -= 1
                self.cv.notify_all()
        return out, gen != self.abort_gen


def stub_ctx_factory(ext, log=None):
    def f(worker):
        a = worker.allocator
        view = ext.AllocatorView(a, a.create_and_map, a.unmap_and_release, label="stub-cumem-v0.30.0")
        return ext.WarmCtx(view, worker.backend, cvd=",".join(str(x) for x in worker.backend.cvd),
                           log=log or (lambda ev: None))
    return f


class InProcEngine:
    def __init__(self, fab, cvd, ext, weight_sizes, kv_sizes, seed=0, proc="eng", log=None, ctx_log=None):
        self.log = log or (lambda *a, **k: None)
        self.backend = StubBackend(fab, cvd, proc, ext.chunk_sums_numpy)
        W = type("Worker", (StubWorkerBase,), {})
        self.calls = inject_extension(W, ext.WarmWakeExt, self.log)
        ext.set_ctx_factory(stub_ctx_factory(ext, ctx_log))
        self.worker = W(self.backend, 0, weight_sizes, kv_sizes, seed, self.log)
        self.executor = StubExecutor(self.worker, self.log)
        self.core = StubEngineCore(self.executor, self.log)
        self.weight_ptrs = list(self.worker.weight_ptrs)
        self.weight_sizes = [int(s) for s in weight_sizes]

    def rpc(self, method, spec):
        return self.core.collective_rpc(method, args=(json.dumps(spec),))[0]

    def sleep(self, level=1, mode="wait"):
        self.core.sleep(level, mode)

    def wake_up(self, tags=None):
        self.core.wake_up(tags)

    def logprobs(self, ids):
        out, _aborted = self.core.run_request(lambda: self.worker.stub_logprobs(list(ids)))
        return out

    def read(self, ptr, off, n):
        return self.backend.peek_bytes(ptr + off, n)

    def truth(self):
        fab = self.backend.fab
        rows = []
        for p, d in self.worker.allocator.pointer_to_data.items():
            code, _a, aid, _o = split_ptr(p)
            rows.append((d.tag, bool(d.is_asleep), d.cpu_backup_tensor is not None,
                         os.path.exists(fab.bpath(code, aid)), int(d.handle[1])))
        return {"asleep": sum(r[1] for r in rows), "backups": sum(r[2] for r in rows),
                "backup_bytes": sum(r[4] for r in rows if r[2]),
                "weights_awake": sum(1 for r in rows if r[0] == "weights" and not r[1]),
                "weights_asleep": sum(1 for r in rows if r[0] == "weights" and r[1]),
                "weights_mapped": sum(1 for r in rows if r[0] == "weights" and r[3]),
                "kv_asleep": sum(1 for r in rows if r[0] == "kv_cache" and r[1]),
                "kv_mapped": sum(1 for r in rows if r[0] == "kv_cache" and r[3]),
                "ctr": dict(self.backend.ctr), "last_error": self.backend.peek_last_error(),
                "sticky": self.backend.sticky, "sleeping_tags": sorted(self.executor.sleeping_tags),
                "paused": self.core.paused, "queued": self.core.queued}


def make_engine(fab, cvd, ext, weight_sizes, kv_sizes, seed=0, proc="eng", log=None, ctx_log=None):
    return InProcEngine(fab, cvd, ext, weight_sizes, kv_sizes, seed, proc, log, ctx_log)


# ------------------------------------------------------------------------------------------------ realistic tables
CONFIGS = {   # public HF config.json values (re-read 2026-10-21 for the T3; see the bite-1 report)
    "qwen3-14b-fp8": dict(h=5120, inter=17408, L=40, heads=40, kvh=8, hd=128, vocab=151936, tie=False),
    "qwen3-8b-fp8": dict(h=4096, inter=12288, L=36, heads=32, kvh=8, hd=128, vocab=151936, tie=False),
}


def table_sizes(name, scale=1):
    """cumem allocation sizes of an fp8 (Marlin, weight-only) engine: fused qkv / o / gate_up / down per layer, each its
    own 2 MiB-rounded segment; per-channel bf16 scales + norms packed into 2 MiB small segments; bf16 embed + lm_head.
    An ESTIMATE of the allocator's table (count and sizes), not a read of a real engine: S0's bb_inventory reads that."""
    c = CONFIGS[name]
    seg = 2 * MIB

    def r(n):
        return -(-int(n) // seg) * seg

    q = c["heads"] * c["hd"]
    kv = 2 * c["kvh"] * c["hd"]
    sizes = [r(c["vocab"] * c["h"] * 2)]
    small = 0
    for _ in range(c["L"]):
        sizes += [r((q + kv) * c["h"]), r(c["h"] * q), r(2 * c["inter"] * c["h"]), r(c["h"] * c["inter"])]
        small += ((q + kv) + c["h"] + 2 * c["inter"] + c["h"]) * 2 + 2 * c["h"] * 2 + 2 * c["hd"] * 2
    sizes += [seg] * (-(-small // seg))
    if not c["tie"]:
        sizes.append(r(c["vocab"] * c["h"] * 2))
    if scale > 1:
        sizes = [max(4096, -(-(s // scale) // 4096) * 4096) for s in sizes]
    return sizes


# ------------------------------------------------------------------------------------------------ HTTP engine
def load_ext(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("warm_wake_ext", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["warm_wake_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_engine_server(eng, port=0, host="127.0.0.1"):
    def api_log(where, msg):
        vlog("APIServer", where, msg)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, ctype="application/json"):
            body = b"" if obj is None else (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                return self._send(200)
            if u.path == "/is_sleeping":
                return self._send(200, {"is_sleeping": eng.core.is_sleeping()})
            if u.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": "stub", "object": "model"}]})
            if u.path == "/stub/truth":
                return self._send(200, eng.truth())
            return self._send(404, {"detail": "Not Found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            raw = self._body()
            try:
                if u.path == "/sleep":                  # dev/sleep/api_router.py L21-27: default mode = abort
                    eng.core.sleep(int(q.get("level", ["1"])[0]), q.get("mode", ["abort"])[0])
                    return self._send(200)
                if u.path == "/wake_up":                # L30-38
                    tags = q.get("tags", [])
                    if tags == []:
                        tags = None
                    api_log("api_router.py:36", "wake up the engine with tags: %s" % (tags,))
                    eng.core.wake_up(tags)
                    return self._send(200)
                if u.path == "/collective_rpc":         # dev/rpc/api_router.py L23-54
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError as e:
                        return self._send(400, {"detail": f"JSON decode error: {e}"})
                    method = body.get("method")
                    if method is None:
                        return self._send(400, {"detail": "Missing 'method' in request body"})
                    results = eng.core.collective_rpc(method, body.get("timeout"), tuple(body.get("args", [])),
                                                      body.get("kwargs", {}))
                    if results is None:
                        return self._send(200)
                    return self._send(200, {"results": [r if (r is None or isinstance(r, (dict, list))) else str(r)
                                                        for r in results]})
                if u.path == "/v1/completions":
                    return self._completions(json.loads(raw or b"{}"))
                if u.path == "/stub/fault":
                    b = json.loads(raw)
                    eng.backend.install_fault(b["op"], b.get("nth", 1), b["code"], b.get("sticky", False))
                    return self._send(200, {"ok": True})
                if u.path == "/stub/poke_guest":
                    b = json.loads(raw)
                    return self._send(200, eng.backend.fab.poke_handle(base64.b64decode(b["guest_handle"]),
                                                                       int(b["offset"])))
                return self._send(404, {"detail": "Not Found"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"detail": f"{type(e).__name__}: {e}"})

        def _completions(self, b):
            prompt = b.get("prompt")
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
                prompt = prompt[0]
            if not isinstance(prompt, list):
                return self._send(400, {"detail": "stub takes token-id prompts only"})
            n_out = int(b.get("max_tokens") or 16)
            t0 = time.time()
            lps, aborted = eng.core.run_request(lambda: eng.worker.stub_logprobs(prompt))
            usage = {"prompt_tokens": len(prompt), "completion_tokens": n_out, "total_tokens": len(prompt) + n_out}
            rid = f"cmpl-{uuid.uuid4().hex[:24]}"
            fin = "abort" if aborted else "length"
            plp = None
            if b.get("prompt_logprobs") is not None:
                plp = [None if x is None else {str(t): {"logprob": x, "rank": 1, "decoded_token": f"t{t}"}}
                       for t, x in zip(prompt, lps)]
            if not b.get("stream"):
                return self._send(200, {"id": rid, "object": "text_completion", "created": int(t0), "model": "stub",
                                        "choices": [{"index": 0, "text": "t" * n_out, "logprobs": None,
                                                     "finish_reason": fin, "stop_reason": None,
                                                     "prompt_logprobs": plp}],
                                        "usage": usage})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(obj):
                data = (b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            for k in range(n_out):
                chunk({"id": rid, "object": "text_completion", "created": int(t0), "model": "stub",
                       "choices": [{"index": 0, "text": "t", "logprobs": None,
                                    "finish_reason": fin if k == n_out - 1 else None, "stop_reason": None}],
                       "usage": None})
            if (b.get("stream_options") or {}).get("include_usage"):
                chunk({"id": rid, "object": "text_completion", "created": int(t0), "model": "stub", "choices": [],
                       "usage": usage})
            chunk(b"[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return ThreadingHTTPServer((host, int(port)), H)


def serve_engine(a):
    ext = load_ext(a.ext_file)
    fab = StubFabric.open(a.fabric)
    cvd_s = a.cvd or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd = [int(x) for x in cvd_s.split(",") if x.strip() != ""]
    if not cvd:
        print("BB-STUB-ERROR no CUDA_VISIBLE_DEVICES / --cvd", file=sys.stderr, flush=True)
        return 2
    wb = [int(x) for x in a.weights_bytes.split(",")]
    kb = [int(x) for x in a.kv_bytes.split(",")] if a.kv_bytes else []

    def log(proc, where, msg, level="INFO"):
        vlog(proc, where, msg, level)

    eng = InProcEngine(fab, cvd, ext, wb, kb, a.seed, f"engine-pid{os.getpid()}", log=log,
                       ctx_log=lambda ev: print("BB-WARM " + json.dumps(ev, sort_keys=True, default=str),
                                                file=sys.stderr, flush=True))
    vlog("EngineCore", "gpu_model_runner.py:5417", "Model loading took %.4f GiB memory and 1.000000 seconds"
         % (sum(wb) / GIB))
    srv = make_engine_server(eng, a.port)
    vlog("APIServer", "launcher.py:46", "Route: /collective_rpc, Methods: POST")
    print(f"BB-STUB-READY port={srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# ------------------------------------------------------------------------------------------------ stub self-test
def self_test():
    import re
    import shutil
    import subprocess
    import tempfile
    sys.dont_write_bytecode = True
    here = os.path.dirname(os.path.abspath(__file__))
    ext = load_ext(os.path.join(here, "warm_wake_ext.py"))
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        print(("ok   " if cond else "FAIL ") + name + ("" if cond else f" - {detail}"), flush=True)

    root = tempfile.mkdtemp(prefix="bbstub_", dir="/dev/shm" if os.access("/dev/shm", os.W_OK) else None)
    try:
        fab = StubFabric.create(os.path.join(root, "s"), ngpu=3, cap_bytes=16 * MIB)
        a = StubBackend(fab, [0, 1, 2], "A", ext.chunk_sums_numpy)
        b = StubBackend(fab, [1, 0], "B", ext.chunk_sums_numpy)
        c = StubBackend(fab, [2], "C", ext.chunk_sums_numpy)
        pa = a.malloc(0, MIB)
        pb = b.malloc(0, MIB)
        # S1 ownership: B cannot touch A's memory without IPC
        try:
            b.memcpy(0, pb, pa, 16)
            s1 = False
        except StubCudaError as e:
            s1 = e.code == 1 and b.peek_last_error() == 1
        check("S1 touching another process's memory fails rc 1 and sets the last error", s1)
        b.clear_last_error()
        h = a.ipc_export(0, pa)
        al = b.ipc_open(0, h)
        a.poke_bytes(pa, b"hello")
        b.memcpy(0, pb, al, 5)
        check("S2 IPC open + copy works; counts peer->local", b.peek_bytes(pb, 5) == b"hello"
              and b.ctr["bytes:peer->local"] == 5)
        try:
            a.free(0, pa)
            s3 = False
        except StubCudaError:
            s3 = True
        b.ipc_close(0, al)
        a.clear_last_error()
        a.free(0, pa)
        check("S3 free while IPC-opened elsewhere refused (stub-strict); after close, freed", s3
              and fab.read_meta(*split_ptr(pa)[0:1], split_ptr(pa)[2]) is None)
        try:
            c.ipc_open(0, b.ipc_export(0, pb))
            s4 = False
        except StubCudaError as e:
            s4 = e.code == 101
        check("S4 IPC open of a GPU outside CUDA_VISIBLE_DEVICES fails (101, stub-chosen code)", s4)
        c.clear_last_error()
        try:
            b.ipc_open(0, b.ipc_export(0, pb))
            s5 = False
        except StubCudaError as e:
            s5 = e.code == 400
        check("S5 a process cannot open its own export (400)", s5)
        b.clear_last_error()
        try:
            a.malloc(0, 17 * MIB)
            s6 = False
        except StubCudaError as e:
            s6 = e.code == 2
        check("S6 capacity: OOM rc 2", s6)
        a.clear_last_error()
        a.install_fault("memcpy", 1, 700, sticky=True)
        pa2 = a.malloc(1, MIB)
        try:
            a.memcpy(0, pa2, pa2, 1)
        except StubCudaError:
            pass
        st = (a.clear_last_error(), a.synchronize_rc(0))
        try:
            a.malloc(0, 4096)
            s7 = False
        except StubCudaError as e:
            s7 = e.code == 700
        check("S7 sticky 700: not cleared, sync returns it, every later call fails", st == (700, 700) and s7, st)

        # S8 allocator transliteration: sleep backs up weights only; wake(tags) copies only that tag
        fab2 = StubFabric.create(os.path.join(root, "s8"), ngpu=3, cap_bytes=64 * MIB)
        eng = make_engine(fab2, [2, 0, 1], ext, [2 * MIB, 4 * MIB, 2 * MIB], [4 * MIB, 4 * MIB], seed=0, proc="e8")
        lp = eng.logprobs([1, 2, 3])
        eng.sleep(1, "wait")
        t1 = eng.truth()
        eng.backend.ctr.clear()
        eng.wake_up(["kv_cache"])
        t2 = eng.truth()
        check("S8 sleep: 3 weights backed up, kv discarded; wake(kv) maps kv with 0 host copies, still paused",
              t1["backups"] == 3 and t1["asleep"] == 5 and t2["kv_mapped"] == 2
              and eng.backend.ctr["copies:host->local"] == 0 and t2["paused"], (t1, t2))
        eng.wake_up(["kv_cache"])
        check("S9 wake of a tag that is not asleep warns and returns (abstract.py L355-360)",
              eng.executor.sleeping_tags == {"weights"})
        eng.wake_up(["weights"])
        check("S10 wake(weights): 3 host copies, resumed, identity holds",
              eng.backend.ctr["copies:host->local"] == 3 and not eng.core.paused and eng.logprobs([1, 2, 3]) == lp)
        Bad = type("Bad", (), {"stub_logprobs": lambda self: None})
        W = type("Worker", (StubWorkerBase,), {})
        try:
            inject_extension(W, Bad, lambda *x: None)
            s11 = False
        except AssertionError as e:
            s11 = "conflicts with the worker extension class" in str(e)
        check("S11 an extension name that collides with the worker fails the L295 assert", s11)
        check("S12 table: Qwen3-14B fp8 shape 165 allocations, 16456351744 B (15.33 GiB)",
              len(table_sizes("qwen3-14b-fp8")) == 165 and sum(table_sizes("qwen3-14b-fp8")) == 16456351744,
              (len(table_sizes("qwen3-14b-fp8")), sum(table_sizes("qwen3-14b-fp8"))))

        # S13-S15 the HTTP engine (one child process)
        fab3 = os.path.join(root, "http")
        StubFabric.create(fab3, ngpu=3, cap_bytes=64 * MIB)
        logp = os.path.join(root, "e.log")
        with open(logp, "wb") as lf:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve-engine", "--fabric", fab3,
                                  "--cvd", "2,0,1", "--port", "0", "--weights-bytes", f"{MIB},{MIB}", "--kv-bytes",
                                  f"{MIB}", "--ext-file", os.path.join(here, "warm_wake_ext.py")],
                                 stdout=subprocess.PIPE, stderr=lf)
        try:
            port = ext._ready_port(p, "BB-STUB-READY")      # review: a bare readline could hang forever
            E = f"http://127.0.0.1:{port}"
            st, js = ext._http(E + "/collective_rpc", {"method": "bb_inventory", "args": ["{}"]})
            inv = js["results"][0]
            check("S13 /collective_rpc reaches the injected extension (string arg)", st == 200 and inv["ok"]
                  and inv["tags"]["weights"]["n"] == 2 and inv["cvd"] == "2,0,1", js)
            import urllib.error
            try:
                ext._http(E + "/collective_rpc", {"args": []})
                s14 = False
            except urllib.error.HTTPError as e:              # review: match the status, not a substring
                s14 = e.code == 400
            check("S14 /collective_rpc without method -> 400", s14)
            ext._http(E + "/sleep")                       # default mode = abort, like stock
            asleep = ext._http(E + "/is_sleeping", method="GET")[1]
            import urllib.request
            got = {}

            def req():
                got["r"] = ext._http(E + "/v1/completions", {"prompt": [5, 6, 7], "max_tokens": 2,
                                                             "prompt_logprobs": 1})

            th = threading.Thread(target=req)
            th.start()
            t_q = time.time()
            queued = False
            while time.time() - t_q < 10 and not queued:   # review: prove it is WAITING, not unsent
                queued = ext._http(E + "/stub/truth", method="GET")[1]["queued"] == 1 and "r" not in got
                time.sleep(0.05)
            ext._http(E + "/wake_up?tags=weights")
            ext._http(E + "/wake_up?tags=kv_cache")
            th.join(30)
            check("S15 a request sent while asleep is queued, then served after both wakes",
                  asleep == {"is_sleeping": True} and queued and got.get("r", (0,))[0] == 200
                  and got["r"][1]["usage"]["prompt_tokens"] == 3, (asleep, queued, got.get("r")))
            with urllib.request.urlopen(urllib.request.Request(
                    E + "/v1/completions", data=json.dumps({"prompt": [1, 2], "max_tokens": 3, "stream": True,
                                                            "stream_options": {"include_usage": True}}).encode(),
                    headers={"Content-Type": "application/json"}), timeout=30) as r:
                sse = r.read().decode()
            ev = [x[6:] for x in sse.split("\n") if x.startswith("data: ")]
            check("S16 streamed completion: 3 token events + a usage event + [DONE]", len(ev) == 5
                  and ev[-1] == "[DONE]" and json.loads(ev[-2])["usage"]["completion_tokens"] == 3, ev)
        finally:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:                 # QA 2026-10-21 (glm)
                p.kill()
                p.wait(10)
            p.stdout.close()
        with open(logp) as fh:
            txt = fh.read()
        real = ["(EngineCore pid=27433) INFO 10-13 23:54:46 [cumem.py:278] CuMemAllocator: sleep freed 11.93 GiB memory "
                "in total, of which 8.55 GiB is backed up in CPU and the rest 3.38 GiB is discarded directly.",
                "(EngineCore pid=27433) INFO 10-13 23:54:46 [gpu_worker.py:276] Sleep mode freed 11.91 GiB memory, "
                "4.63 GiB memory is still in use.",
                "(EngineCore pid=27433) INFO 10-13 23:54:46 [abstract.py:346] It took 5.589358 seconds to fall asleep.",
                "(APIServer pid=12542) INFO 10-13 15:11:11 [api_router.py:36] wake up the engine with tags: ['weights']",
                "(EngineCore pid=21561) INFO 10-13 15:38:59 [abstract.py:364] It took 0.087705 seconds to wake up tags "
                "['weights']."]

        def shape(s):     # digits -> \d, so a real line and a stub line compare by FORMAT
            return re.sub(r"\d+", "#", s)

        mine = {shape(x) for x in txt.splitlines()}
        miss = [x for x in real if shape(x) not in mine]
        check("S17 the stub's sleep/wake lines have the same format as 5 real v0.30.0 log lines", not miss, miss)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    n, k = len(res), res.count(False)
    print(f"STUB-SELFTEST-{'FAIL' if k else 'OK'} {n - k if not k else k}/{n}")
    return 1 if k else 0


def main(argv=None):
    p = argparse.ArgumentParser(description="CPU stub engines for the WARM wake (B7)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--serve-engine", action="store_true")
    p.add_argument("--fabric")
    p.add_argument("--cvd", default=None, help="CUDA_VISIBLE_DEVICES as physical ids; default: the env var")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--weights-bytes", default=f"{2 * MIB},{4 * MIB},{2 * MIB}")
    p.add_argument("--kv-bytes", default=f"{4 * MIB},{4 * MIB}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ext-file", default=None)
    p.add_argument("--ready-fd", default="1", help="accepted for clarity; the ready line goes to stdout")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.serve_engine:
        if not a.fabric or not a.ext_file:
            print("BB-STUB-ERROR --serve-engine needs --fabric and --ext-file", file=sys.stderr)
            return 2
        return serve_engine(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
