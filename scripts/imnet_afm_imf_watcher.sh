#!/usr/bin/env bash
# =============================================================================
# AFM (discrete endpoint-adversarial, ablation=afm_only) starting DIRECTLY from
# the IMAGENET iMF checkpoint (iMF-XL-2-full) -- NOT from the domain-finetuned
# iMF. Sibling of the ImageNet-CAMF sweep (imnet_caimf_imf_watcher.sh): same
# ImageNet cold-start, but AFM regime instead of CA-iMF. Detached: screen
# imnet_afm_watcher.
#
# Reuses the finetuned-AFM path unchanged (configs/caltech_imf_afm_posttrain_
# config.yml + scripts/run_imf_afm.sh, already parameterised on
# <dataset> <load_from> <workdir>); the ONLY difference is every dataset
# loads_from the single ImageNet base checkpoint. partial_load: True (in the
# config) restores 757 tensors and skips the class-embed (1000 -> target N),
# so class conditioning is a fresh cold-start.
#
#   ablation=afm_only lambda_adv=1.0 lambda_imf=0.0 lambda_cp=0.01
#   D-warmup 5000 + 4:1 D:G ; max_posttrain_batches=150000 (=30k G-updates)
#   gen/dis LR=1e-5 (=0.1x iMF base FT LR 1e-4)
#   fid@5k every 5k, metric_num_steps=[4], 10000 samples, save_best_fid_only
#   native iMF op-point (omega=7.5, t in [0.4,0.65]). Final eval NFE 1&2.
#
# SCHEDULING: opportunistic -- claims GPUs one at a time AS THEY FREE (no hard
#   gate waiting for everything). STABLE_NEEDED=6 (~9min stably-free) so it
#   DEFERS to the retro-eval driver (STABLE_NEEDED=3): evals always win a
#   contested freed GPU, AFM fills any GPU no pending eval wants.
# EARLY-STOP: DISABLED -- run full 150k (matches the ImageNet-CAMF sibling,
#   user request 2026-07-27). One GPU/run. Shares the atomic GPU lock.
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
IMF=/opt/dlami/nvme/meanflow/imeanflow
PY=$IMF/.venv/bin/python
cd "$ADV"
LOG="$ADV/files/logs/imnet_afm_watcher.log"
mkdir -p "$ADV/files/logs"

POLL_S=90
STABLE_NEEDED=6           # defer to retro-eval driver (STABLE_NEEDED=3)
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
FINAL_EVAL_STEPS="1 2"
CONFIG_MODE=caltech_imf_afm_posttrain
STAMP=20260727_imnet
PATIENCE=999999           # early-stop DISABLED -- run full 150k (matches imnet-CAMF)
RUNNER="$ADV/scripts/run_imf_afm.sh"

# The single ImageNet iMF base checkpoint every dataset starts from.
IMNET_CKPT="$IMF/files/weights/iMF-XL-2-full"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }
consec_rises_from_min(){ awk '{ if(NR==1){mn=$1;s=0;next} if($1<=mn){mn=$1;s=0}else{s++} } END{print s+0}'; }

source "$ADV/scripts/gpu_lock.inc.sh"

# ds : load_from  -- ALL point at the ImageNet iMF base. All 5 datasets
# (caltech101 INCLUDED here per user request 2026-07-27: "all datasets").
QUEUE=(
  "artbench10:$IMNET_CKPT"
  "caltech101:$IMNET_CKPT"
  "cub200:$IMNET_CKPT"
  "food101:$IMNET_CKPT"
  "stanfordcars:$IMNET_CKPT"
)

workdir_for(){ echo "$ADV/files/logs/finetuning/${1}_iMF_imnet_AFM_puresadv_${STAMP}"; }

train_done(){  # $1=ds
  local wd; wd=$(workdir_for "$1")
  ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1 || return 1
  screen -ls | grep -qE "imnetafm_gpu[0-9]+_${1}\b" && return 1
  return 0
}

launch(){  # $1=gpu $2=entry
  local gpu="$1"; IFS=':' read -r ds ckpt <<< "$2"
  [[ -d "$ckpt" ]] || { log "ERROR [$ds] ckpt missing: $ckpt"; return 1; }
  local wd; wd=$(workdir_for "$ds")
  if ls -d "$wd/best_fid/checkpoint_"* >/dev/null 2>&1; then log "SKIP [$ds] already has best_fid"; return 1; fi
  local sess="imnetafm_gpu${gpu}_${ds}"
  log "LAUNCH imnet-AFM $ds on GPU $gpu (screen $sess) ckpt=$ckpt wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export CONFIG_MODE=$CONFIG_MODE
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS='$FINAL_EVAL_STEPS'
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-imnetafm-$ds
    bash $RUNNER $ds '$ckpt' '$wd' \
      2>&1 | tee -a $ADV/files/logs/imnetafm_${ds}.log
  "
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== imnet-AFM watcher: ${#QUEUE[@]} datasets from ImageNet iMF ($IMNET_CKPT), pool {$ALLOWED}, 150k batches (NO early-stop), eval@5k NFE4 10k-samples, best-fid-only, final NFE{$FINAL_EVAL_STEPS}, gate $STABLE_NEEDED (defers to retro evals), GPU-locked; opportunistic -- grabs GPUs as they free ==="

for entry in "${QUEUE[@]}"; do ds="${entry%%:*}"; if train_done "$ds"; then DONE[$ds]=1; log "SKIP $ds (best_fid already present)"; fi; done

# adopt any of OUR runs already in flight (watcher restart)
while read -r sess; do
  [[ "$sess" =~ imnetafm_gpu([0-9]+)_([a-z0-9]+) ]] || continue
  gidx="${BASH_REMATCH[1]}"; gds="${BASH_REMATCH[2]}"
  CLAIMED[$gidx]="$gds"; gpu_try_claim "$gidx" "imnetafm_gpu${gidx}_${gds}" >/dev/null 2>&1
  log "ADOPT in-flight $gds on GPU $gidx (screen imnetafm_gpu${gidx}_${gds})"
done < <(screen -ls 2>/dev/null | grep -oE "imnetafm_gpu[0-9]+_[a-z0-9]+")

remaining(){ local n=0; for e in "${QUEUE[@]}"; do ds="${e%%:*}"; [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  gpu_gc_locks
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  for line in "${SMI[@]}"; do
    idx=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$3);print $3}')
    [[ " $ALLOWED " == *" $idx "* ]] || continue
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    if gpu_locked "$idx"; then STABLE[$idx]=0; continue; fi
    if (( mem < MEM_MB && util < UTIL_PCT )); then STABLE[$idx]=$(( ${STABLE[$idx]:-0} + 1 )); else STABLE[$idx]=0; fi
  done
  for idx in "${!CLAIMED[@]}"; do
    ds="${CLAIMED[$idx]}"
    if screen -ls | grep -qE "imnetafm_gpu${idx}_${ds}\b"; then
      csv=$(workdir_for "$ds")/eval_metrics.csv
      [[ -f "$csv" ]] || continue
      series=$(fid_series "$csv"); n=$(echo "$series" | grep -c .)
      (( n < PATIENCE + 1 )) && continue
      streak=$(echo "$series" | consec_rises_from_min)
      if (( streak >= PATIENCE )); then
        mn=$(echo "$series" | sort -g | head -1); last=$(echo "$series" | tail -1)
        log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing imnetafm_gpu${idx}_${ds}."
        screen -S "imnetafm_gpu${idx}_${ds}" -X quit 2>/dev/null; sleep 3
      else
        continue
      fi
    fi
    if ! screen -ls | grep -qE "imnetafm_gpu${idx}_${ds}\b"; then
      if train_done "$ds"; then DONE[$ds]=1; log "COMPLETE $ds (GPU $idx freed)"; else log "WARN $ds screen gone but no best_fid ckpt -- check imnetafm_${ds}.log"; DONE[$ds]=1; fi
      gpu_release "$idx"; unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for entry in "${QUEUE[@]}"; do
      ds="${entry%%:*}"
      [[ -n "${DONE[$ds]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$ds" ]] && inflight=1; done
      (( inflight )) && continue
      gpu_try_claim "$idx" "imnetafm_gpu${idx}_${ds}" || { STABLE[$idx]=0; break; }
      if launch "$idx" "$entry"; then CLAIMED[$idx]="$ds"; log "PROGRESS $ds -> GPU $idx (locked); remaining $(remaining)"; sleep 20; else gpu_release "$idx"; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all imnet-AFM runs complete; watcher exiting ==="
