#!/bin/bash
# S0 box orchestrator (Modal container, cwd = harness root). Prereg: (private report)
# (frozen). env + artifact sha gate -> both models downloaded in parallel (background, one marker each) ->
# s0_driver.py (S0-0..S0-11; it waits on the markers) -> s0_ctl.py (S0-12..S0-15) -> rows/DONE.
# An item's failure is recorded in rows/ and the run goes on; only the artifact-sha gate stops it.
set -u
cd "$(dirname "$0")"
mkdir -p rows logs
export HF_XET_HIGH_PERFORMANCE=1
{ date -u; nvidia-smi; nvidia-smi nvlink -s; nvidia-smi topo -m;
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
S0_BACKEND=vllm python3 s0_driver.py > logs/driver.out 2>&1
echo "$(date -u +%T) driver rc=$?" >> logs/progress.log
wait
S0_BACKEND=vllm python3 s0_ctl.py > logs/ctl.out 2>&1
echo "$(date -u +%T) ctl rc=$?" >> logs/progress.log
nvidia-smi > logs/nvidia_smi_end.txt 2>&1
echo OK > rows/DONE
