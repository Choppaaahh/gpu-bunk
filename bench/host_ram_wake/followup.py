#!/usr/bin/env python3
""" Follow-on (plan amendment 2), run ON the box in /root/c13 after drive.py is stopped mid-moe.
Order: coloc (D2) first, then moe: 3 warm-compile-cache launches on the cache dir primed by moe cold #1, sleep/wake cycles.
moe cold n=1 only (cold #1 row reconstructed from its server log by the caller; labelled log-derived). Writes rows/coloc.json,
rows/m_moe.json, ARM_DONE_* markers, then DONE2 = OK | DEADLINE:<phase>."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive as d  # module import does not run main()


def main():
    d.check_deadline = lambda ph: (open("DONE2", "w").write(f"DEADLINE:{ph}\n"), sys.exit(0)) if time.time() > d.DL else None
    try:
        d.dump("coloc", d.phase_coloc())
    except Exception as e:  # noqa
        d.log(f"coloc fail {e!r}")
        d.dump("coloc", {"infra": repr(e)[:300]})
    key, port = "moe", 8001
    row = {"key": key, "repo": d.MODELS[key][0], "util": d.MODELS[key][2], "cold": [], "warm": [],
           "note": "amendment 2: cold n=1 (log-derived, see results); 3 warm launches on the cold-#1-primed cache dir"}
    warm_root = f"{d.ROOT}/vcache/{key}/warm"
    srv = None
    try:
        row["download"] = {"bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in
                                        os.walk("/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/blobs") for f in fs)}
        for i in range(3):
            d.check_deadline("m_moe")
            r, srv = d.launch_once(key, warm_root, port, keep=(i == 2))
            row["warm"].append(r)
        row["cycles"] = d.sleep_cycles(srv)
        row["sleep_facts"] = srv.log_facts()
    except SystemExit:
        raise
    except Exception as e:  # noqa
        row["infra"] = repr(e)[:300]
    finally:
        if srv:
            srv.stop()
        d.dump("m_moe", row)
    open("DONE2", "w").write("OK\n")


if __name__ == "__main__":
    main()
