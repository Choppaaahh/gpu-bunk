#!/usr/bin/env python3
""" multi-GPU edit (2026-10-14): the original sampled `nvidia-smi --query-gpu` WITHOUT -i and parsed line 0, so on a
2-GPU box a gate run for GPU 1 (CUDA_VISIBLE_DEVICES=1) burns GPU 1 but reads GPU 0's pstate/clock/power (nvidia-smi ignores
CUDA_VISIBLE_DEVICES). Here `--gpu-index N` passes `-i N` to every nvidia-smi call and a sample whose query returns != 1 line
is an error sample (never silently line 0). The caller sets CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=N so the burn
runs on the same card nvidia-smi index N names.
copy of the Stage D2 HOST GATE at H100 thresholds. Runs ON the box before the model download (so a bad host costs no model pull).
Phase A: bf16 matmul burn for 25 s (compute load). Phase B: pinned host->device copies for 8 s (forces the PCIe link up).
A sampler thread reads nvidia-smi every 1 s; a full `nvidia-smi -q` is captured mid-burn as the receipt.
Frozen criteria (prereg (private report) §2), ALL required:
  G1 pstate in {P0,P1,P2} on >= 80% of phase-A samples with utilization >= 50%  (and >= 5 such samples)
  G2 max SM clock in phase A >= 0.70 x clocks.max.sm, OR power-limited (draw >= 0.95 x limit AND clock >= 0.50 x max; amendment 3)
  G3 max power draw in phase A >= max(250 W, 0.80 x power.limit) (amendment 1: SXM 700 / NVL 400 / PCIe 350 W)
  G4 matmul throughput >= C15_G4_MIN_TFLOPS (bf16, 8192^3; 400 H100-class, 150 GB202-class, amendment 2)
  G5 max pcie.link.gen.current in phase B >= 3  AND  H2D bandwidth >= 5 GB/s
Writes logs/host_gate.json + logs/host_gate_smi_q.txt. Exit 0 = PASS, 1 = FAIL, 2 = could not measure (also INFRA).
Usage: host_gate.py <logs dir> --gpu-index N      host_gate.py --self-test"""
import json, os, subprocess, sys, threading, time

Q = "pstate,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,pcie.link.gen.current,pcie.link.gen.max,power.limit"


def smi_argv(idx, q=Q):
    """: always name the card."""
    return ["nvidia-smi", "-i", str(idx), f"--query-gpu={q}", "--format=csv,noheader,nounits"]


def one_line(out):
    """: exactly one CSV line per query, else ValueError (a multi-line answer means -i was ignored or missing)."""
    lines = [x for x in out.splitlines() if x.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected 1 nvidia-smi line, got {len(lines)}")
    return lines[0]


def parse_line(line):
    p = [x.strip() for x in line.split(",")]
    f = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
    return {"pstate": p[0], "sm": f(p[1]), "sm_max": f(p[2]), "power": f(p[3]), "util": f(p[4]),
            "gen": f(p[5]), "gen_max": f(p[6]), "power_limit": f(p[7])}


G4_MIN = float(os.environ.get("C15_G4_MIN_TFLOPS", "400"))   # amendment 2: 400 H100-class, 150 GB202-class


def judge(a, b, tflops, h2d_gbs, g4_min=None):
    """a, b: lists of sample dicts for phase A / phase B. Returns (verdict, checks)."""
    loaded = [s for s in a if (s["util"] or 0) >= 50]
    pfrac = (sum(s["pstate"] in ("P0", "P1", "P2") for s in loaded) / len(loaded)) if loaded else 0.0
    sm_max = max((s["sm_max"] or 0) for s in a) if a else 0
    sm_peak = max((s["sm"] or 0) for s in a) if a else 0
    pw = max((s["power"] or 0) for s in a) if a else 0
    plim = max((s["power_limit"] or 0) for s in a) if a else 0
    gen = max((s["gen"] or 0) for s in b) if b else 0
    c = {"G1_pstate_frac": round(pfrac, 3), "G1_n_loaded": len(loaded), "G1": len(loaded) >= 5 and pfrac >= 0.80,
         "G2_sm_peak": sm_peak, "G2_sm_max": sm_max,
         # amendment 3: a POWER-LIMITED clock (draw >= 0.95 x limit, clock >= 0.50 x max) is not the idle clock
         "G2": sm_max > 0 and (sm_peak >= 0.70 * sm_max or (pw >= 0.95 * plim and sm_peak >= 0.50 * sm_max)),
         "G3_power_peak_w": pw, "G3": pw >= max(250.0, 0.80 * max((s["power_limit"] or 0) for s in a)) if a else False,
         "G4_tflops": round(tflops, 1), "G4": tflops >= (G4_MIN if g4_min is None else g4_min),
         "G4_min": G4_MIN if g4_min is None else g4_min,                  # QA: record the active floor
         "G5_gen_peak": gen, "G5_h2d_gbs": round(h2d_gbs, 2), "G5": gen >= 3 and h2d_gbs >= 5}
    return ("PASS" if all(c[k] for k in ("G1", "G2", "G3", "G4", "G5")) else "FAIL"), c


def self_test():
    good = [parse_line("P0, 2800, 3090, 500.1, 100, 4, 5, 575.00")] * 10   # 500 >= 0.8 x 575 = 460
    p8 = [parse_line("P8, 210, 3090, 54.9, 100, 1, 5, 400.00")] * 10
    fails = []
    v, _ = judge(good, good, 650.0, 20.0)
    if v != "PASS": fails.append("healthy host should PASS")
    v, c = judge(p8, p8, 14.0, 1.5)   # the JP host readings (nvidia_smi_q_at_stop.txt)
    if v != "FAIL" or c["G1"] or c["G2"] or c["G3"] or c["G4"] or c["G5"]: fails.append(f"P8 host should fail all five: {c}")
    v, c = judge([parse_line("P0, 1980, 1980, 300.0, 100, 5, 5, 350.00")] * 10, good, 500.0, 20.0)   # PCIe 350 W: 300 >= 280
    if v != "PASS": fails.append(f"PCIe H100 at 300 W of a 350 W limit should PASS: {c}")
    v, c = judge([parse_line("P0, 1980, 1980, 450.0, 100, 5, 5, 700.00")] * 10, good, 650.0, 20.0)   # SXM: 450 < 560
    if v != "FAIL" or c["G3"]: fails.append("SXM at 450 W of 700 should fail G3")
    pro = [parse_line("P1, 2152, 3090, 600.03, 100, 5, 5, 600.00")] * 24    # host 55808982 reading, 2026-10-13
    v, c = judge(pro, pro, 391.0, 42.68, g4_min=150.0)
    if v != "PASS" or not c["G2"]: fails.append(f"power-limited GB202 (0.696 clock at 600/600 W) should PASS: {c}")
    v, c = judge([parse_line("P1, 1700, 3090, 300.0, 100, 5, 5, 600.00")] * 10, good, 391.0, 20.0, g4_min=150.0)
    if v != "FAIL" or c["G2"]: fails.append("0.55 clock at half power must fail G2")
    v, c = judge(good, good, 216.0, 20.0, g4_min=150.0)   # GB202 class at the 5090's measured 216 TFLOPS
    if v != "PASS": fails.append(f"GB202 at 216 TFLOPS with floor 150 should PASS: {c}")
    v, c = judge(good, good, 216.0, 20.0, g4_min=400.0)
    if v != "FAIL" or c["G4"]: fails.append("216 TFLOPS must fail the H100 floor")
    v, c = judge(good, [parse_line("P0, 2800, 3090, 450.1, 100, 1, 5, 575.00")] * 3, 650.0, 20.0)
    if v != "FAIL" or c["G5"]: fails.append("gen-1 link should fail G5")
    v, c = judge([parse_line("P0, 2800, 3090, 450.1, 10, 4, 5, 575.00")] * 10, good, 650.0, 20.0)
    if v != "FAIL" or c["G1"]: fails.append("no loaded samples should fail G1")
    # goldens: the query names the card; a 2-line answer (the old bug's input) is refused, never read as line 0
    if smi_argv(1)[:3] != ["nvidia-smi", "-i", "1"]:
        fails.append("smi_argv must pass -i <index>")
    two = "P8, 210, 1410, 55.0, 0, 1, 4, 400.00\nP0, 1410, 1410, 390.0, 100, 4, 4, 400.00\n"
    try:
        one_line(two)
        fails.append("ADVERSARIAL: a 2-line nvidia-smi answer must be refused")
    except ValueError:
        pass
    if parse_line(one_line("P0, 1410, 1410, 390.0, 100, 4, 4, 400.00\n"))["pstate"] != "P0":
        fails.append("one-line answer should parse")
    # QA r1: the burned card must be -i N. Same UUID with/without the GPU- prefix and case matches; ADVERSARIAL: another
    # card's UUID must not
    if not same_card("58d3a1b2-0c4e-4b1f-9e2a-1f2e3d4c5b6a", "GPU-58D3A1B2-0C4E-4B1F-9E2A-1F2E3D4C5B6A\n"):
        fails.append("same UUID must match across prefix/case")
    if same_card("58d3a1b2-0c4e-4b1f-9e2a-1f2e3d4c5b6a", "GPU-77d3a1b2-0c4e-4b1f-9e2a-1f2e3d4c5b6a"):
        fails.append("ADVERSARIAL: a different card's UUID must not match")
    # A100 SXM: 400 W limit, G3 floor 320 W; 260 TFLOPS passes the 200 floor
    a100 = [parse_line("P0, 1410, 1410, 390.0, 100, 4, 4, 400.00")] * 10
    v, c = judge(a100, a100, 260.0, 20.0, g4_min=200.0)
    if v != "PASS":
        fails.append(f"healthy A100 SXM should PASS at the 200 floor: {c}")
    print(json.dumps({"self_test": "PASS" if not fails else "FAIL", "fails": fails}))
    return 0 if not fails else 1


def same_card(torch_uuid, smi_uuid):
    """ QA: the burned card (CUDA device 0 of the visible set) must be nvidia-smi's -i N. UUIDs compared without the
    'GPU-' prefix, case-folded."""
    norm = lambda u: str(u).strip().lower().replace("gpu-", "", 1)
    return norm(torch_uuid) == norm(smi_uuid)


def unmeasured(logs, why):
    json.dump({"verdict": "UNMEASURED", "error": str(why)[:500]}, open(os.path.join(logs, "host_gate.json"), "w"))
    print("UNMEASURED", why)
    return 2


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    if len(sys.argv) != 4 or sys.argv[2] != "--gpu-index":       # QA: check argc before reading argv
        print("USAGE host_gate.py <logs dir> --gpu-index N", file=sys.stderr)
        return 2
    logs = sys.argv[1]
    try:
        idx = int(sys.argv[3])
        if idx < 0:
            raise ValueError("negative index")
    except ValueError as e:
        print(f"USAGE --gpu-index must be an integer >= 0 ({e})", file=sys.stderr)
        return 2
    os.makedirs(logs, exist_ok=True)
    try:
        import torch
    except Exception as e:  # noqa: BLE001  (QA: a broken torch is UNMEASURED, exit 2, never a FAIL)
        return unmeasured(logs, f"import torch: {e}")
    card = {"checked": False}
    try:                                                          # QA: prove the burn runs on card -i N
        tu = torch.cuda.get_device_properties(0).uuid
        su = one_line(subprocess.check_output(smi_argv(idx, "uuid"), timeout=10).decode())
        card = {"checked": True, "torch_uuid": str(tu), "smi_uuid": su.strip(), "same": same_card(tu, su)}
    except Exception as e:  # noqa: BLE001
        card = {"checked": False, "why": str(e)[:200]}
    if card.get("checked") and not card["same"]:
        return unmeasured(logs, f"card mismatch: CUDA device 0 is not nvidia-smi -i {idx} ({card})")
    samples, phase, stop = [], {"p": "idle"}, threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.check_output(smi_argv(idx), timeout=10).decode()
                s = parse_line(one_line(out)); s["phase"] = phase["p"]; s["t"] = time.time(); s["gpu_index"] = idx
                samples.append(s)
            except Exception as e:  # noqa: BLE001
                samples.append({"phase": phase["p"], "err": str(e)[:200]})
            time.sleep(1)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    try:
        x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); y = torch.randn_like(x)
        for _ in range(5): x @ y
        torch.cuda.synchronize()
        phase["p"] = "A"; n = 0; t0 = time.time(); q_done = False
        while time.time() - t0 < 25:
            for _ in range(20): x @ y
            torch.cuda.synchronize(); n += 20
            if not q_done and time.time() - t0 > 12:
                with open(os.path.join(logs, "host_gate_smi_q.txt"), "w") as f:
                    f.write(subprocess.run(["nvidia-smi", "-q", "-i", str(idx)], capture_output=True, text=True, timeout=30).stdout)
                q_done = True
        dt = time.time() - t0
        tflops = n * 2 * 8192 ** 3 / dt / 1e12
        phase["p"] = "B"
        h = torch.empty(256 * 2 ** 20, dtype=torch.uint8).pin_memory(); d = torch.empty_like(h, device="cuda")
        nb = 0; t0 = time.time()
        while time.time() - t0 < 8:
            d.copy_(h, non_blocking=True); torch.cuda.synchronize(); nb += h.numel()
        h2d = nb / (time.time() - t0) / 1e9
    except Exception as e:  # noqa: BLE001
        stop.set(); json.dump({"verdict": "UNMEASURED", "error": str(e)[:500]}, open(os.path.join(logs, "host_gate.json"), "w"))
        print("UNMEASURED", e); return 2
    phase["p"] = "end"; time.sleep(1.5); stop.set(); th.join(timeout=15)
    a = [s for s in samples if s.get("phase") == "A" and "err" not in s]
    b = [s for s in samples if s.get("phase") == "B" and "err" not in s]
    if not a:                                                     # QA: no readable sample for -i N = unmeasured
        return unmeasured(logs, f"no valid nvidia-smi -i {idx} sample in phase A ({len(samples)} samples, all errors)")
    verdict, checks = judge(a, b, tflops, h2d)
    json.dump({"verdict": verdict, "checks": checks, "samples": samples, "gpu_index": idx, "card": card},
              open(os.path.join(logs, "host_gate.json"), "w"), indent=1)
    print(verdict, json.dumps(checks))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
