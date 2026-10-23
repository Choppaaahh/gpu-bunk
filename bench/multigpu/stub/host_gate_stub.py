#!/usr/bin/env python3
# stub host gate (CPU host smoke only): writes a PASS gate file in the real file's format, exit 0
import json, os, sys
os.makedirs(sys.argv[1], exist_ok=True)
json.dump({"verdict": "PASS", "checks": {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}, "samples": [],
           "gpu_index": int(sys.argv[3]), "stub": True}, open(os.path.join(sys.argv[1], "host_gate.json"), "w"))
print("PASS stub")
