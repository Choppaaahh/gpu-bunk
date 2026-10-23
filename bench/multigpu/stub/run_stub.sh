#!/bin/bash
# CPU-only smoke of gpu_driver.py on a stub vLLM + stub nvidia-smi + stub host gate with the REAL
# beta_bench.py in --cpu mode on the 'tiny' geometry, then the scorer on its output.
# Usage: run_stub.sh <empty scratch dir>. Exit 0 = DONE starts with OK, every ARM_DONE_* and row file the scorer reads
# exists, and gpu_score.py exits 0 on the stub run; else 1. Writes only inside the scratch dir.
set -o pipefail
S=$(cd "$(dirname "$0")" && pwd); D=$(cd "$S/.." && pwd); R=${1:?scratch dir}
mkdir -p "$R" || exit 1
[ -z "$(ls -A "$R")" ] || { echo "REFUSE: $R is not empty"; exit 1; }
cp "$D/gpu_driver.py" "$D/beta_bench.py" "$D/inputs/prompts.jsonl" "$R/" || exit 1
cp "$S/host_gate_stub.py" "$R/host_gate.py" || exit 1
python3 - "$D/inputs/plan.json" "$R/plan.json" <<'EOF' || exit 1
import json, sys
p = json.load(open(sys.argv[1]))
p["beta"].update(geom="tiny", cells=[{"name": "Q", "B": 1, "L": 64}, {"name": "S", "B": 4, "L": 64},
                                     {"name": "LC", "B": 2, "L": 128}], passes=3, warm_steps=1, timed_steps=3)
p["ctrl"]["secs"] = 3
p["anchor"].update(out=8, passes=3, warm_rounds=1)
p["lender"]["settle_s"] = 2
p["item12"]["out"] = 8
p["item10"].update(out=6, passes=3)
p["item9"]["same"].update(n_live=3, live_gap_s=0.2, batch_conc=4, batch_out=64, batch_settle_s=1)
p["item9"]["other"].update(reps=2, batch_conc=2, batch_out=64, batch_settle_s=1)
# overload on purpose: stub cap = 9 running; 16 arrivals x 6 s each -> waiters pass the 1 s cancel (: a stress run
# that never contends passes vacuously; the first stub run cancelled 0 with out=400)
p["item8"].update(rate=4.0, secs=4, out=1500, timeout_s=1.0, join_cap_s=90, tail_s=3, drain_cap_s=60)
p["dcp"].update(out=8, passes=3)
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
mkdir -p "$R/pypkg/vllm/model_executor/models" || exit 1   # a models/ path is gitignored: written at run time (tkvp note)
printf '__version__ = "0.30.0"\n' > "$R/pypkg/vllm/__init__.py"
: > "$R/pypkg/vllm/model_executor/__init__.py"
: > "$R/pypkg/vllm/model_executor/models/__init__.py"
printf '_VLLM_MODELS = {"Qwen3ForCausalLM": 1, "DeepseekV2ForCausalLM": 1}\n' > "$R/pypkg/vllm/model_executor/models/registry.py"
cd "$R" || exit 1
PATH="$S/bin:$PATH" PYTHONPATH="$R/pypkg" MG_STUB=1 MG_ROOT="$R" MG_PLAN=plan.json MG_DPH=0.01 MG_CAP_GUARD=5 \
  MG_DEADLINE=$(( $(date +%s) + 3000 )) timeout 2400 python3 "$R/gpu_driver.py" > "$R/run.out" 2>&1
echo "driver rc=$? DONE=$(cat "$R/DONE" 2>/dev/null)"
grep -q '^OK' "$R/DONE" || { tail -20 "$R/logs/drive.log"; tail -20 "$R/run.out"; exit 1; }
for f in ARM_DONE_CTRL ARM_DONE_BETA_IDLE ARM_DONE_BETA_BUSY ARM_DONE_ITEM12 ARM_DONE_ITEM10 ARM_DONE_ITEM9 ARM_DONE_ITEM8 \
         ARM_DONE_DCP rows/ctrl.jsonl rows/anchor.jsonl rows/beta_IDLE.jsonl rows/beta_BUSY.jsonl rows/lender_busy.jsonl \
         rows/routing_B1.json rows/routing_B64.json rows/routing_meta.json rows/offload.jsonl rows/offload_meta.json \
         rows/item9_same.jsonl rows/item9_other.jsonl rows/item9_presleep.json rows/item8_Tinf.jsonl rows/item8_T10.jsonl \
         rows/item8_metrics_Tinf.jsonl rows/dcp.jsonl rows/dcp_meta.json logs/probe.json logs/g6_serving.json \
         logs/nvlink_gate.json logs/env.txt logs/gpu0/host_gate.json logs/gpu1/host_gate.json logs/warmup_engines.json; do
  [ -s "$R/$f" ] || { echo "MISSING $f"; exit 1; }
done
grep -q '^\$ nproc' "$R/logs/env.txt" || { echo "env.txt lacks the \$ nproc block"; exit 1; }
if [ -f "$D/gpu_score.py" ]; then
  python3 "$D/gpu_score.py" --run "$R" --plan "$R/plan.json" --out "$R/score.json" --allow-stub > "$R/score.out" 2>&1
  src=$?; echo "score rc=$src"; tail -40 "$R/score.out"; [ $src -eq 0 ] || exit 1
fi
echo "STUB-OK"
