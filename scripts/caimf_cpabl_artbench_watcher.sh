#!/usr/bin/env bash
# =============================================================================
# CA-iMF CP-ablation watcher -- ARTBENCH-10 (detached: screen caimf_cpabl_artb).
#
# Extends the CP-regularization ablation (previously CUB-200 only) to ArtBench.
# FT+CAMF: initialize from the ArtBench TARGET-FINETUNED iMF/XL-2 checkpoint
# (plain_iMF_finetune_artbench10 best_fid/checkpoint_7500), then pure-adversarial
# CA-iMF post-train, cp_mode overridden per-variant on the CLI.
#
# Reuses the CUB cpablation config (cub_imf_caimf_cpablation) but overrides the
# dataset section for ArtBench (root, num_classes=10 x3, FID/FDD cache_ref).
#
# 6 variants: full_sum2, afm_sum2, zt, zt_zhatr, zt_zr, none.
# "Ours" (full = D_zt^2+D_zr^2+D_zhatr^2) is NOT run here -- reused from the main
# ArtBench CAMF run (artbench10_iMF_CAIMF_puresadv_20260726).
#
# 150k batch cap, eval every 5000 steps, 10000 samples, save_best_fid_only.
# Integrated early-stop PATIENCE=5 (consecutive 4-step-FID rises from running
# min). best_fid retained. Report best FID/FDD up to the stop.
#
# GPU pool RESTRICTED to 6 7 so it never touches the imnet-CAMF 150k->300k
# resumes on GPUs 0/2/3/4/5 (or the AFM eval on GPU 1). Claims each as free.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/caimf_cpabl_artbench_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=3          # free-poll gate before claiming a GPU
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 2 6 7"
PATIENCE=5               # consecutive 4-step-FID rises -> early stop
STAMP=20260728
CKPT=$MAIN/files/logs/finetuning/plain_iMF_finetune_artbench10_h100_20260722_225848_jt6v93/best_fid/checkpoint_7500

# --- ArtBench dataset overrides (config is CUB by default) ---
ART_ROOT=/opt/dlami/nvme/meanflow/datasets/artbench-10_processed_latents
ART_NC=10
ART_FID=$MAIN/files/fid_stats/artbench-10_processed-fid_stats.npz
ART_FDD=$MAIN/files/fdd_stats/artbench-10-fd_dino-vitb14_stats.npz

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MODES=(afm_sum2 none zt zt_zr zt_zhatr full_sum2)
workdir_for(){ echo "$ADV/files/logs/finetuning/artbench10_iMF_CAIMF_cpabl_${1}_${STAMP}"; }

train_done(){  # $1=mode
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "cpablart_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=mode
  local gpu="$1" mode="$2"
  [[ -d "$CKPT" ]] || { log "ERROR ckpt missing: $CKPT"; return 1; }
  local wd; wd=$(workdir_for "$mode")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$mode] already has best_fid"; return 1; fi
  local sess="cpablart_gpu${gpu}_${mode}"
  log "LAUNCH cp-ablation(ArtBench) $mode on GPU $gpu (screen $sess) wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export REQUIRE_FD_DINO=1 MPLCONFIGDIR=/tmp/mpl-cpablart-$mode
    $PY main_caimf.py \
      --workdir='$wd' \
      --config=$ADV/configs/load_config.py:cub_imf_caimf_cpablation \
      --config.load_from='$CKPT' \
      --config.caimf.cp_mode=$mode \
      --config.dataset.root='$ART_ROOT' \
      --config.dataset.num_classes=$ART_NC \
      --config.model.num_classes=$ART_NC \
      --config.sampling.num_classes=$ART_NC \
      --config.fid.cache_ref='$ART_FID' \
      --config.fd_dino.cache_ref='$ART_FDD' \
      2>&1 | tee -a $ADV/files/logs/caimf_cpabl_artbench_${mode}.log
  "
}

fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

log "=== CP-ablation ArtBench watcher: GPUs {$ALLOWED}, ${#MODES[@]} variants, 150k cap, gate $STABLE_NEEDED, patience $PATIENCE, ckpt=$CKPT ==="

declare -A STABLE=() CLAIMED=() DONE=()
for m in "${MODES[@]}"; do if train_done "$m"; then DONE[$m]=1; log "SKIP $m (best_fid present)"; fi; done

# adopt in-flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ cpablart_gpu([0-9]+)_([a-z0-9_]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gmode="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gmode"; log "ADOPT in-flight $gmode on GPU $gidx"
done < <(screen -ls 2>/dev/null | grep -oE "cpablart_gpu[0-9]+_[a-z0-9_]+")

remaining(){ local n=0; for m in "${MODES[@]}"; do [[ -z "${DONE[$m]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    mode="${CLAIMED[$idx]}"
    if screen -ls | grep -qE "cpablart_gpu${idx}_${mode}\b"; then
      csv=$(workdir_for "$mode")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $mode: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing cpablart_gpu${idx}_${mode}."
        screen -S "cpablart_gpu${idx}_${mode}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    if ! screen -ls | grep -qE "cpablart_gpu${idx}_${mode}\b"; then
      if train_done "$mode"; then DONE[$mode]=1; log "COMPLETE $mode (GPU $idx freed)"; else log "WARN $mode screen gone, no best_fid -- check caimf_cpabl_artbench_${mode}.log"; DONE[$mode]=1; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for mode in "${MODES[@]}"; do
      [[ -n "${DONE[$mode]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$mode" ]] && inflight=1; done
      (( inflight )) && continue
      if launch "$idx" "$mode"; then CLAIMED[$idx]="$mode"; log "PROGRESS $mode -> GPU $idx; remaining $(remaining)"; sleep 20; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all 6 ArtBench CP-ablation variants complete; watcher exiting ==="
