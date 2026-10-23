#!/bin/bash
# run R3 orchestrator (Modal container, cwd = harness root). Prereg: (private report)
# (§2 frozen amendments A1, A2). env + artifact sha gate -> both models downloaded in parallel (background,
# one marker each) -> s0_r3.py (A2: S0-6/8/9 with the per-park push + D-STALE) -> rows/DONE.
set -u
cd "$(dirname "$0")"
mkdir -p rows logs
export HF_XET_HIGH_PERFORMANCE=1
{ date -u; nvidia-smi; nvidia-smi nvlink -s;
  python3 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'gpus',torch.cuda.device_count())";
  nproc; head -3 /proc/meminfo; df -h /root /tmp; } > logs/env.txt 2>&1
sha256sum -c sha_expected.txt > logs/artifact_sha.txt 2>&1 || { echo "INFRA-FAIL:artifact-sha" > rows/DONE; exit 0; }
dl() {
  python3 - "$1" "$2" "$3" > "logs/dl_$1.log" 2>&1 <<'PY'
import sys, time
from huggingface_hub import snapshot_download
key, repo, rev = sys.argv[1:4]
t0 = time.time()
pat = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja", "*.py"]
print(snapshot_download(repo, revision=rev, allow_patterns=pat, max_workers=16))
print(f"download_s {time.time() - t0:.1f}")
PY
  if [ $? -eq 0 ]; then echo ok > "logs/dl_$1.done"; else echo fail > "logs/dl_$1.fail"; fi
}
dl e8 Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 &
dl e14 AnastasiaFantasy/affine-ana11-3-5CJXygeziPM2F8C1bhupwAKpKmx28cw1zD15Eoa5QbFPSXXE 971185c6cf17f6c89168ad95b3c316ef27d0b459 &
S0_BACKEND=vllm python3 s0_r3.py > logs/r3.out 2>&1
echo "$(date -u +%T) r3 rc=$?" >> logs/progress.log
wait
nvidia-smi > logs/nvidia_smi_end.txt 2>&1
echo OK > rows/DONE
