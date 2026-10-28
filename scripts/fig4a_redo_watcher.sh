#!/usr/bin/env bash
# Fig. 4(a) redo: frozen ImageNet DiT-XL/2, DDPM-eps (p_sample), DDPM-v (native_velocity_data), Transport-v
# (transport_velocity_diff2flow) at 1,2,4,8,16,32,64,250 steps, 10k samples, FID (full-train stats) + FDD
# (DINOv2-B/14 stats from the OpenAI ADM ImageNet-256 10k reference batch). Same modes and config as the original
# fid5k sweep (notebooks/plot_lines.ipynb); only NUM_SAMPLES and the FDD reference change.
# Claims a GPU only when it has been empty (<3 GB, <15% util) for 3 polls in a row; one job per GPU; longest first.
set -uo pipefail
M=/opt/dlami/nvme/meanflow/imeanflow; cd $M
OUT=$M/files/debug/fig4a_redo10k; mkdir -p $OUT; LOG=$OUT/watcher.log
FDD=/files/fdd_stats/imagenet256_adm_ref10k-fd_dino-vitb14_stats.npz
[ -f "$FDD" ] || { echo "missing FDD stats $FDD" | tee -a $LOG; exit 1; }
JOBS=(); for n in 250 64 32 16 8 4 2 1; do for m in p_sample native_velocity_data transport_velocity_diff2flow; do JOBS+=("$m:$n"); done; done
declare -A STABLE=() CLAIMED=()
log(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
done_(){ [ -f "$OUT/${1%%:*}/${1%%:*}_${1##*:}steps/eval_metrics.csv" ]; }
remaining(){ local n=0; for j in "${JOBS[@]}"; do done_ $j || n=$((n+1)); done; echo $n; }
log "=== Fig.4a redo watcher: ${#JOBS[@]} jobs ==="
while (( $(remaining) > 0 )); do
  mapfile -t SMI < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for line in "${SMI[@]}"; do IFS=', ' read -r i mem util <<< "$line"
    if [ -n "${CLAIMED[$i]:-}" ]; then screen -ls | grep -q "\.fig4a_g${i}\b" || unset "CLAIMED[$i]"; STABLE[$i]=0; continue; fi
    if (( mem < 3000 && util < 15 )); then STABLE[$i]=$(( ${STABLE[$i]:-0} + 1 )); else STABLE[$i]=0; fi
  done
  for i in "${!STABLE[@]}"; do
    (( ${STABLE[$i]} >= 3 )) || continue; [ -n "${CLAIMED[$i]:-}" ] && continue
    for j in "${JOBS[@]}"; do
      done_ $j && continue; busy=0; for c in "${CLAIMED[@]}"; do [ "$c" = "$j" ] && busy=1; done; (( busy )) && continue
      m=${j%%:*}; n=${j##*:}
      log "LAUNCH $m $n steps on GPU $i"
      screen -dmS fig4a_g$i bash -c "cd $M; CUDA_VISIBLE_DEVICES=$i PYTHON=$M/.venv/bin/python USE_WANDB=False NUM_SAMPLES=10000 MODES=$m WORKDIR_ROOT=$OUT/$m FD_DINO_CACHE_REF=$FDD bash scripts/sweep_fid5k_plain_dit_modes_imagenet.sh $n >> $OUT/${m}_${n}.log 2>&1"
      CLAIMED[$i]=$j; STABLE[$i]=0; sleep 15; break
    done
  done
  sleep 90
done
log "=== all Fig.4a jobs done ==="
