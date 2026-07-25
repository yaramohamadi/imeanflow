#!/usr/bin/env bash
# For each ddpm-e run, wait until its current (1 2 250) final-eval screen exits,
# then run a 4-step eval off the SAME best_fid checkpoint on the SAME GPU.
# 4-step off a 16-step-selected checkpoint (no retraining) — adds the paper's
# 4-step column for the ddpm-e (plain SiT) baseline. Idempotent: skips if the
# 4-step eval dir already has metrics.
set -uo pipefail
REPO=/opt/dlami/nvme/meanflow/imeanflow
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$REPO"
CONFIG_MODE=caltech_plain_sit_ddpme
PY="$REPO/.venv/bin/python"

# ds : gpu : latent_basename : num_classes : fid : fdd
RUNS=(
  "artbench:2:artbench-10_processed_latents:10:artbench-10_processed-fid_stats.npz:artbench-10-fd_dino-vitb14_stats.npz"
  "cub:3:cub-200-2011_processed_latents:200:cub-200-2011_processed-fid_stats.npz:cub-200-2011-fd_dino-vitb14_stats.npz"
  "stanford:5:stanford-cars_processed_latents:196:stanford_cars_processed-fid_stats.npz:stanford-cars-fd_dino-vitb14_stats.npz"
)

one() {
  IFS=':' read -r DS GPU LAT NC FID FDD <<< "$1"
  local rd; rd=$(ls -dt files/logs/finetuning/*${DS}*ddpme*/ 2>/dev/null | head -1); rd=${rd%/}
  [ -z "$rd" ] && { echo "[$DS] no run dir"; return; }
  local out4="$rd/eval_best_fid_4steps/eval_metrics.csv"
  if [ -f "$out4" ]; then echo "[$DS] 4-step already done"; return; fi
  # wait for this run's existing final-eval screen to finish (frees GPU $GPU)
  while screen -ls | grep -q "ddpme_${DS}"; do sleep 60; done
  echo "[$DS] ddpme_${DS} screen gone; launching 4-step on GPU $GPU"
  local sh="files/logs/run_ddpme4_${DS}.sh"
  cat > "$sh" <<EOF
#!/bin/bash
cd $REPO
CONFIG_MODE=$CONFIG_MODE PYTHON=$PY USE_WANDB=False \\
CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore \\
bash scripts/eval_best_fid_steps_plain_sit.sh "$rd" 4 \\
  -- --config.load_from=$REPO/files/weights/DiT-XL-2-256x256.pt \\
     --config.dataset.num_classes=$NC \\
     --config.model.num_classes=$NC \\
     --config.dataset.name=$DS \\
     --config.dataset.root=$DATA/$LAT \\
     --config.fid.cache_ref=$REPO/files/fid_stats/$FID \\
     --config.fd_dino.cache_ref=$REPO/files/fdd_stats/$FDD
EOF
  chmod +x "$sh"
  screen -dmS "ddpme4_${DS}" bash -c "bash $sh 2>&1 | tee -a $REPO/files/logs/ddpme4_${DS}.log"
  echo "[$DS] launched screen ddpme4_${DS}"
}

for r in "${RUNS[@]}"; do one "$r" & done
wait
echo "ALL 4-STEP WATCHERS DISPATCHED"
