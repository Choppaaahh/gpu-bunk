"""s0_r3.py -- run R3 (Amendment A2): S0-6 / S0-8 / S0-9 with the controller's per-park sequence (a fresh push into
the SAME guest buffer before every sleep, as fabric_ctl._do_park does), plus D-STALE (reported only): a guest
pushed BEFORE a serving period, restored WITHOUT the seal, then teacher-forced + greedy identity -- does restoring a
stale copy of the 150 MiB weights-pool allocation that serving mutates (R2: E14 idx 167, E8 idx 150) change outputs?

Output contract: rows/S0-6.json, S0-8.json, S0-9.json, D-STALE.json (+ the driver's S0-1/S0-2/S0-0 rows); exit 0 whatever
the verdicts, exit 1 only on a harness crash. Prereg: (private report) (§2 frozen; A1, A2).
"""
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C  # noqa: E402
import s0_driver as D  # noqa: E402


def alloc(lport, gid, nbytes, owner):
    al = C.holder(lport, "/holder/alloc", {"guest_id": gid, "nbytes": nbytes, "owner_id": owner})
    if not al.get("ok"):
        raise RuntimeError(f"alloc {gid}: {al.get('reason')}")
    return al


def push_into(kind, lport, gid, al):
    port = D.PORT[kind]
    nbytes = D.ST[f"w_{kind}"]
    r, tries = D.push(port, gid, al, kind, nbytes)
    at = C.holder(lport, "/holder/attach", {"guest_id": gid, "manifest": r.get("manifest") or {}, "verify": True}) \
        if r.get("ok") else {"ok": False}
    g = dict(gid=gid, handle=al["ipc_handle"], nbytes=nbytes, mid=r.get("manifest_id"), offsets=r.get("offsets"),
             n_alloc=r.get("n_alloc"), port=lport) if (r.get("ok") and at.get("ok")) else None
    return g, dict(push=[dict(mode=t["mode"], ok=t["res"].get("ok"), reason=t["res"].get("reason"),
                              rate_gbs=t["res"].get("rate_gbs"), copy_s=t["res"].get("copy_s"),
                              digest_s=t["res"].get("digest_s"), replaced=t["res"].get("replaced")) for t in tries],
                   attach={k: at.get(k) for k in ("ok", "state", "bad", "seconds", "reason")})


def s0_6_8():
    allc, okk = {}, {}
    for kind, lport in (("e14", D.PORT["h1"]), ("e8", D.PORT["h0"])):
        gid = f"{kind}-r3"
        al = alloc(lport, gid, D.ST[f"w_{kind}"], kind)
        cs = []
        for i in range(3):
            g, pinfo = push_into(kind, lport, gid, al)            # the per-park push (bite 2 _do_park)
            if not g:
                cs.append(dict(pass_=False, push=pinfo))
                continue
            c = D.cycle(kind, g, D.ST[f"tf{kind[1:]}"])
            c["push"] = pinfo
            cs.append(c)
        D.ST[f"g3_{kind}"] = (al, gid, lport)
        allc[kind] = cs
        okk[kind] = all(c.get("pass") for c in cs)
    v = "PASS" if all(okk.values()) else "FAIL"
    C.put_row("S0-6", v, cmd="per cycle: bb_push_to_peer into the SAME guest + /holder/attach verify (the per-park push) -> "
              "/sleep?level=1&mode=wait -> bb_seal_after_sleep -> bb_restore_from_peer (direct) -> /wake_up?tags=weights "
              "-> /wake_up?tags=kv_cache -> /is_sleeping; teacher-forced P1-P3; n=3 per kind (A2)", per_kind=okk,
              summary={k: [dict(push_rate=(c.get("push") or {}).get("push", [{}])[0].get("rate_gbs"),
                                seal_stale=(c.get("seal") or {}).get("stale"), seal_s=(c.get("seal") or {}).get("seconds"),
                                served=(c.get("restore") or {}).get("served"), reason=(c.get("restore") or {}).get("reason"),
                                open_s=(c.get("restore") or {}).get("open_s"), copy_s=(c.get("restore") or {}).get("copy_s"),
                                verify_s=(c.get("restore") or {}).get("verify_s"), total_s=(c.get("restore") or {}).get("total_s"),
                                rate_gbs=(c.get("restore") or {}).get("rate_gbs"), sleep_s=c.get("sleep_s"),
                                wake_kv_s=(c.get("wake_kv") or [None, None])[1], tf=c.get("tf_maxdiff"),
                                greedy32_equal=c.get("greedy32_equal")) for c in v] for k, v in allc.items()},
              cycles=allc)
    opens = {k: [(c.get("restore") or {}).get("open_s") for c in v] for k, v in allc.items()}
    C.put_row("S0-8", "PASS" if all(all(x is not None for x in v) for v in opens.values()) else "FAIL",
              cmd="open_s of restores 1..3 from the same lender + the same guest buffer (S0-6 A2 cycles)", open_s=opens)


def s0_9():
    al, gid, lport = D.ST["g3_e14"]
    port = D.PORT["e14"]
    g, pinfo = push_into("e14", lport, gid, al)
    if not g:
        return C.put_row("S0-9", "FAIL", reason="push before corrupt failed", push=pinfo)
    off = int(g["offsets"][1]) + 12345
    cr = C.holder(lport, "/holder/corrupt", {"guest_id": gid, "offset": off})
    st, s = C.stock_sleep(port)
    r, tries = D.restore(port, gid, g["handle"], g["nbytes"], g["mid"])
    w = D.both_wakes(port)
    d = C.tf_diff(D.ST["tf14"], C.tf_read("http://127.0.0.1:%d" % port, D.model_of("e14")))
    ok = (cr.get("ok") and r.get("served") == "HOST_FALLBACK" and r.get("reason") == "DIGEST_MISMATCH"
          and r.get("rolled_back") == g["n_alloc"] and 1 in (r.get("bad_allocs") or []) and w["is_sleeping"] is False
          and d <= C.TOL)
    v = "FAIL-STOP-THE-RUN" if r.get("served") == "PEER" else ("PASS" if ok else "FAIL")
    return C.put_row("S0-9", v, cmd=f"push (per-park) -> POST /holder/corrupt {{guest_id: {gid}, offset: offsets[1]+12345 = "
                     f"{off}}} -> sleep -> bb_restore_from_peer -> stock wakes", push=pinfo, corrupt=cr, restore=r,
                     wakes=w, tf_maxdiff=d, n_alloc=g["n_alloc"])


def d_stale_unsealed():
    """push -> serve -> sleep -> restore WITHOUT a seal (the controller's default: seal off) -> identity."""
    out = {}
    for kind in ("e14", "e8"):
        al, gid, lport = D.ST[f"g3_{kind}"]
        port = D.PORT[kind]
        g, pinfo = push_into(kind, lport, gid, al)
        if not g:
            out[kind] = dict(err="push failed", push=pinfo)
            continue
        base = f"http://127.0.0.1:{port}"
        C.tf_read(base, D.model_of(kind))
        C.gen_read(base, D.model_of(kind))
        st, s = C.stock_sleep(port)
        r, tries = D.restore(port, gid, g["handle"], g["nbytes"], g["mid"])
        w = D.both_wakes(port)
        tf = C.tf_diff(D.ST[f"tf{kind[1:]}"], C.tf_read(base, D.model_of(kind)))
        gen = C.gen_read(base, D.model_of(kind)) == D.ST.get(f"gen{kind[1:]}")
        tf2 = C.tf_diff(D.ST[f"tf{kind[1:]}"], C.tf_read(base, D.model_of(kind)))     # a second read after more serving
        out[kind] = dict(push=pinfo, restore={k: r.get(k) for k in ("served", "reason", "total_s", "rate_gbs")},
                         wakes=w, tf_maxdiff=tf, tf_maxdiff_second_read=tf2, greedy32_equal=gen)
    ok = all(isinstance(v, dict) and v.get("restore", {}).get("served") == "PEER" for v in out.values())
    return C.put_row("D-STALE", "INFO" if ok else "INFO-INCOMPLETE",
                     cmd="push -> serve (P1-P3 + greedy) -> sleep -> bb_restore_from_peer with NO seal -> wakes -> "
                         "teacher-forced + greedy identity vs the pre-sleep reads (reported only, A2)",
                     per_kind=out)


def main():
    C.log(f"s0_r3 start backend={C.BACKEND} left={C.left():.0f}s")
    D.step("S0-1", D.s0_1)
    D.step("S0-2", D.holders)
    D.step("S0-0", D.s0_0, needs=("holders",))

    def boots():
        res = {}

        def b(name, kind, cvd):
            if not D.wait_dl(kind):
                res[name] = "download failed"
                return
            try:
                _, hs = D.boot(name, kind, cvd, D.PORT[name])
                res[name] = dict(health_s=hs)
                D.ST[name] = True
            except Exception as e:  # noqa: BLE001
                res[name] = f"boot failed: {e}"[:400]
        ts = [threading.Thread(target=b, args=a) for a in (("e14", "e14", "0,1"), ("e8", "e8", "1,0"))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        D.ST["boots"] = res
        C.log(f"boots: {json.dumps(res)[:400]}")
    D.step("BOOTS", boots, needs=("holders",))
    D.step("S0-3", D.s0_3)
    D.step("READS", D.warm_and_reads, needs=("e14", "e8"))
    D.step("S0-6", s0_6_8, needs=("inv", "reads"))
    D.step("S0-9", s0_9, needs=("g3_e14",))
    D.step("D-STALE", d_stale_unsealed, needs=("g3_e14", "g3_e8"))
    for p in list(D.PROCS.values()):
        p.stop()
    C.put_row("R3-STATE", "INFO", cool14=D.ST.get("cool14"), flags={k: v[2] for k, v in D.ENG.items()},
              boot_failures=D.ST.get("boot_failures"), boots=D.ST.get("boots"), mem_end=C.gpu_used_mib())
    C.log("s0_r3 done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        for p in list(D.PROCS.values()):
            try:
                p.stop()
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
