#!/usr/bin/env bash
# =============================================================================
# 10k-sample REDO eval watcher -- detached (screen redo10k_watcher).
#
# Re-evaluates every table cell that was computed at 5k, now at 10k samples,
# for ALL FOUR families, using the EXACT proven per-family recipe (only
# fid.num_samples -> 10000 changed). Read-only against saved best_fid ckpts;
# safe to run alongside the CAIMF training on GPUs 1-5.
#
#   SiT MF-A : NFE 4          eval_best_fid_steps.sh  (caltech_sit_dmf_finetune)
#   DiT MF-A : NFE 1 2 4 250  eval_best_fid_steps.sh  (caltech_dit_dmf_ddpmv,
#                              null-class/dit_native/noise/ema_vc=False)
#   JiT DMF  : NFE 1 2 4      main_imf_jit.py --config.eval_only  (num_samples=10000)
#   JiT FT   : NFE 1 2 4 50   eval_best_fid_steps_plain_jit.sh  (-- num_samples=10000)
#
# GPU pool: all 0-7 (1-6 rule retired). Gate = STABLE_NEEDED free polls so we
# never grab a GPU mid train->eval gap. One job per GPU. Idempotent: skips a
# job whose result CSV already exists.
# =============================================================================
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
PY=$REPO/.venv/bin/python
cd "$REPO"
LOG="$REPO/files/logs/redo10k_watcher.log"
mkdir -p "$REPO/files/logs"

POLL_S=90
STABLE_NEEDED=3
MEM_MB=3000
UTIL_PCT=15
ALLOWED="0 1 2 3 4 5 6 7"
NS=10000

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- per-dataset static params -------------------------------------------------
declare -A NC=( [caltech101]=101 [artbench10]=10 [cub200]=200 [food101]=101 [stanfordcars]=196 )
declare -A LAT=( [caltech101]=caltech-101_processed_latents [artbench10]=artbench-10_processed_latents [cub200]=cub-200-2011_processed_latents [food101]=food-101_processed_latents [stanfordcars]=stanford-cars_processed_latents )
declare -A IMG=( [caltech101]=caltech-101_images [artbench10]=artbench-10_images [cub200]=cub-200-2011_images [food101]=food-101_images [stanfordcars]=stanford-cars_images )
declare -A FID=( [caltech101]=caltech-101-fid_stats.npz [artbench10]=artbench-10_processed-fid_stats.npz [cub200]=cub-200-2011_processed-fid_stats.npz [food101]=food-101_processed-fid_stats.npz [stanfordcars]=stanford_cars_processed-fid_stats.npz )
declare -A FDD=( [caltech101]=caltech-101-fd_dino-vitb14_stats.npz [artbench10]=artbench-10-fd_dino-vitb14_stats.npz [cub200]=cub-200-2011-fd_dino-vitb14_stats.npz [food101]=food-101-fd_dino-vitb14_stats.npz [stanfordcars]=stanford-cars-fd_dino-vitb14_stats.npz )

# ---- checkpoint resolvers ------------------------------------------------------
sit_run(){  ls -dt $REPO/files/logs/finetuning/${1}_SiT_DMF_plain_meanflow_meft_online_*/ 2>/dev/null | grep -v smoke | head -1; }
dit_run(){  ls -dt $REPO/files/logs/finetuning/${1}_DiT_DMF_meanflow_taylor_plain_online_clip1_${1}_*/ 2>/dev/null | head -1; }
jitft_run(){ ls -dt $REPO/files/logs/finetuning/plain_JiT_finetune_${1}_online_*/ 2>/dev/null | grep -v smoke | head -1; }
jitdmf_ckpt(){ ls -d $REPO/files/workdirs/jitdmfclip_${1}/best_fid/checkpoint_* 2>/dev/null | tail -1; }

# ---- job queue: "family:ds" ----------------------------------------------------
DSES=(caltech101 artbench10 cub200 food101 stanfordcars)
QUEUE=()
for ds in "${DSES[@]}"; do QUEUE+=("sit:$ds"); done
for ds in "${DSES[@]}"; do QUEUE+=("dit:$ds"); done
for ds in "${DSES[@]}"; do QUEUE+=("jitdmf:$ds"); done
for ds in "${DSES[@]}"; do QUEUE+=("jitft:$ds"); done

# ---- done test: expected result CSV(s) already present at 10k ------------------
# We tag 10k eval workdirs with a _10k suffix so they never collide with the old 5k dirs.
job_done(){  # $1=family $2=ds
  local fam="$1" ds="$2"
  case "$fam" in
    sit)    [[ -f "$REPO/files/logs/redo10k/sit_${ds}/eval_metrics.csv" ]] ;;
    dit)    [[ -f "$REPO/files/logs/redo10k/dit_${ds}/eval_metrics.csv" ]] ;;
    jitdmf) [[ -f "$REPO/files/logs/redo10k/jitdmf_${ds}/eval_metrics.csv" ]] ;;
    jitft)  [[ -f "$REPO/files/logs/redo10k/jitft_${ds}/eval_metrics.csv" ]] ;;
  esac
}

launch(){  # $1=gpu $2=job
  local gpu="$1"; IFS=':' read -r fam ds <<< "$2"
  local out="$REPO/files/logs/redo10k/${fam}_${ds}"; mkdir -p "$out"
  local sess="redo_gpu${gpu}_${fam}_${ds}"
  local nc="${NC[$ds]}" fid="$REPO/files/fid_stats/${FID[$ds]}" fdd="$REPO/files/fdd_stats/${FDD[$ds]}"

  case "$fam" in
    sit)
      local rd; rd=$(sit_run "$ds"); rd=${rd%/}
      [[ -z "$rd" || ! -d "$rd/best_fid" ]] && { log "ERROR [sit $ds] no run dir"; return 1; }
      log "LAUNCH sit $ds GPU$gpu NFE4 ckpt=$(basename "$rd")"
      screen -dmS "$sess" bash -c "
        cd $REPO
        CONFIG_MODE=caltech_sit_dmf_finetune PYTHON=$PY USE_WANDB=False \
        MODEL_STR=imfSiT_DMF_XL_2 MODEL_USE_DOGFIT=False \
        TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 \
        DATASET_ROOT=$DATA/${LAT[$ds]} DATASET_NUM_CLASSES=$nc \
        FID_CACHE_REF=$fid FD_DINO_CACHE_REF=$fdd FID_NUM_SAMPLES=$NS \
        CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
        bash scripts/eval_best_fid_steps.sh '$rd' 4 2>&1 | tee -a $out/run.log
        # collect the produced NFE4 csv into a stable place
        cp \$(ls -t $rd/eval_best_fid_4steps/eval_metrics.csv 2>/dev/null | head -1) $out/eval_metrics.csv 2>/dev/null
      "
      ;;
    dit)
      local rd; rd=$(dit_run "$ds"); rd=${rd%/}
      [[ -z "$rd" || ! -d "$rd/best_fid" ]] && { log "ERROR [dit $ds] no run dir"; return 1; }
      log "LAUNCH dit $ds GPU$gpu NFE[1 2 4 250] ckpt=$(basename "$rd")"
      screen -dmS "$sess" bash -c "
        cd $REPO
        CONFIG_MODE=caltech_dit_dmf_ddpmv PYTHON=$PY USE_WANDB=False \
        MODEL_STR=imfDiT_DMF_XL_2 MODEL_USE_DOGFIT=False \
        TARGET_USE_NULL_CLASS=True CLASS_DROPOUT_PROB=0.1 \
        TARGET_OUTPUT_PREDICTION_SPACE=noise TARGET_VELOCITY_MAP_MODE=dit_native USE_EMA_VC=False \
        DATASET_ROOT=$DATA/${LAT[$ds]} DATASET_NUM_CLASSES=$nc \
        FID_CACHE_REF=$fid FD_DINO_CACHE_REF=$fdd FID_NUM_SAMPLES=$NS \
        CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \
        bash scripts/eval_best_fid_steps.sh '$rd' 1 2 4 250 2>&1 | tee -a $out/run.log
        : > $out/eval_metrics.csv
        for s in 1 2 4 250; do
          f=$rd/eval_best_fid_\${s}steps/eval_metrics.csv
          [[ -f \$f ]] && { [[ -s $out/eval_metrics.csv ]] && tail -n +2 \$f >> $out/eval_metrics.csv || cat \$f > $out/eval_metrics.csv; }
        done
      "
      ;;
    jitdmf)
      local ck; ck=$(jitdmf_ckpt "$ds")
      [[ -z "$ck" ]] && { log "ERROR [jitdmf $ds] no ckpt"; return 1; }
      log "LAUNCH jitdmf $ds GPU$gpu NFE[1 2 4] ckpt=$(basename "$ck")"
      screen -dmS "$sess" bash -c "
        cd $REPO
        export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore MPLCONFIGDIR=/tmp/mpl-redo-$ds
        $PY main_imf_jit.py \
          --workdir=$out \
          --config=configs/load_config.py:caltech_jit_dmf_meft \
          --config.eval_only=True \
          --config.load_from=$ck \
          --config.dataset.root=$DATA/${IMG[$ds]} \
          --config.dataset.num_classes=$nc \
          --config.model.num_classes=$nc \
          --config.fid.cache_ref=$fid \
          --config.fd_dino.cache_ref=$fdd \
          --config.logging.use_wandb=False \
          --config.training.force_metric_num_steps='1 2 4' \
          --config.fid.num_samples=$NS 2>&1 | tee -a $out/run.log
      "
      ;;
    jitft)
      local rd; rd=$(jitft_run "$ds"); rd=${rd%/}
      [[ -z "$rd" || ! -d "$rd/best_fid" ]] && { log "ERROR [jitft $ds] no run dir"; return 1; }
      log "LAUNCH jitft $ds GPU$gpu NFE[1 2 4 50] ckpt=$(basename "$rd")"
      screen -dmS "$sess" bash -c "
        cd $REPO
        export CUDA_VISIBLE_DEVICES=$gpu TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
        CONFIG_MODE=plain_jit_finetune PYTHON=$PY USE_WANDB=False \
        bash scripts/eval_best_fid_steps_plain_jit.sh '$rd' 1 2 4 50 -- \
          --config.dataset.root=$DATA/${IMG[$ds]} \
          --config.dataset.num_classes=$nc \
          --config.model.num_classes=$nc \
          --config.fid.cache_ref=$fid \
          --config.fd_dino.cache_ref=$fdd \
          --config.fid.num_samples=$NS 2>&1 | tee -a $out/run.log
        # plain_jit eval writes a combined summary at <run_root>/final_eval_metrics.csv
        if [[ -f $rd/final_eval_metrics.csv ]]; then
          cp $rd/final_eval_metrics.csv $out/eval_metrics.csv
        else
          : > $out/eval_metrics.csv
          for s in 1 2 4 50; do
            f=$rd/eval_best_fid_\${s}steps/eval_metrics.csv
            [[ -f \$f ]] && { [[ -s $out/eval_metrics.csv ]] && tail -n +2 \$f >> $out/eval_metrics.csv || cat \$f > $out/eval_metrics.csv; }
          done
        fi
      "
      ;;
  esac
}

declare -A STABLE=() CLAIMED=() DONE=()
log "=== 10k REDO watcher: ${#QUEUE[@]} jobs (sit4, dit1/2/4/250, jitdmf1/2/4, jitft1/2/4/50), pool {$ALLOWED}, gate $STABLE_NEEDED ==="
for job in "${QUEUE[@]}"; do IFS=':' read -r fam ds <<< "$job"; if job_done "$fam" "$ds"; then DONE[$job]=1; log "SKIP $job (10k csv present)"; fi; done
remaining(){ local n=0; for j in "${QUEUE[@]}"; do [[ -z "${DONE[$j]:-}" ]] && n=$((n+1)); done; echo "$n"; }

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
  # release finished
  for idx in "${!CLAIMED[@]}"; do
    job="${CLAIMED[$idx]}"; IFS=':' read -r fam ds <<< "$job"
    if ! screen -ls | grep -qE "redo_gpu${idx}_${fam}_${ds}\b"; then
      if job_done "$fam" "$ds"; then DONE[$job]=1; log "COMPLETE $job (GPU $idx freed)"; else log "WARN $job screen gone, no csv -- check redo10k/${fam}_${ds}/run.log"; DONE[$job]=1; fi
      unset 'CLAIMED[$idx]'; STABLE[$idx]=0
    fi
  done
  # assign
  for idx in $ALLOWED; do
    [[ -n "${CLAIMED[$idx]:-}" ]] && continue
    (( ${STABLE[$idx]:-0} >= STABLE_NEEDED )) || continue
    for job in "${QUEUE[@]}"; do
      [[ -n "${DONE[$job]:-}" ]] && continue
      inflight=0; for c in "${CLAIMED[@]}"; do [[ "$c" == "$job" ]] && inflight=1; done
      (( inflight )) && continue
      if launch "$idx" "$job"; then CLAIMED[$idx]="$job"; log "PROGRESS $job -> GPU $idx; remaining $(remaining)"; sleep 20; fi
      break
    done
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all 10k redo evals complete; watcher exiting ==="
