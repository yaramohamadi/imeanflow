#!/usr/bin/env bash
# Poll status of all 7 MeFT runs (5 DiT + 2 SiT). Read-only; prints a table and
# flags completion (best_fid saved / final_eval done) and post-relaunch errors.
set +u
REPO=/opt/dlami/nvme/meanflow/imeanflow
cd "$REPO"
DIT=(caltech101 artbench10 cub200 food101 stanfordcars)
SIT=(caltech101 artbench10)

step_of(){ grep -oE "\] loss=[0-9.a-znN]+" "$1" 2>/dev/null | tail -1; }
latest_step(){ grep -oE "logging_util.py:[0-9]+\] \[[0-9]+\]" "$1" 2>/dev/null | grep -oE "\[[0-9]+\]$" | tail -1; }
fid_of(){ grep -oE "FID[_a-z0-9]*=[0-9.]+" "$1" 2>/dev/null | tail -1; }
fdd_of(){ grep -oiE "FD_DINO[_a-z0-9]*=[0-9.]+" "$1" 2>/dev/null | tail -1; }
errs(){ awk '/17:[2-5][0-9]:|1[89]:[0-5][0-9]:/{s=1} s&&/Traceback|CUDNN_STATUS|XlaRuntimeError|EXECUTION_FAILED/{c++} END{print c+0}' "$1" 2>/dev/null; }

printf "%-6s %-14s %-6s %-10s %-22s %-22s %-4s %s\n" TYPE DATASET ALIVE STEP FID FD_DINO ERR DONE
for d in "${DIT[@]}"; do
  log="files/logs/ditmeft_${d}_run.log"
  screen -ls 2>/dev/null | grep -q "ditmeft_${d}[[:space:]]" && a=yes || a=NO
  done=$(ls -d files/logs/finetuning/*"$d"*DiT_DMF*/final_eval_metrics.csv 2>/dev/null | head -1)
  [ -n "$done" ] && dn=FINAL || { ls -d files/logs/finetuning/*"$d"*DiT_DMF*/best_fid/checkpoint_* >/dev/null 2>&1 && dn=best_fid || dn=-; }
  printf "%-6s %-14s %-6s %-10s %-22s %-22s %-4s %s\n" DiT "$d" "$a" "$(latest_step "$log")" "$(fid_of "$log")" "$(fdd_of "$log")" "$(errs "$log")" "$dn"
done
for d in "${SIT[@]}"; do
  log="files/logs/sitmeft_${d}_run.log"
  screen -ls 2>/dev/null | grep -q "sitmeft_${d}[[:space:]]" && a=yes || a=NO
  done=$(ls -d files/logs/finetuning/*"$d"*[Ss]iT*/final_eval_metrics.csv 2>/dev/null | head -1)
  [ -n "$done" ] && dn=FINAL || dn=-
  printf "%-6s %-14s %-6s %-10s %-22s %-22s %-4s %s\n" SiT "$d" "$a" "$(latest_step "$log")" "$(fid_of "$log")" "$(fdd_of "$log")" "$(errs "$log")" "$dn"
done
echo "GPUs:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
