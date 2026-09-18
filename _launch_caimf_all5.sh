#!/usr/bin/env bash
# Direct launch of all 5 CAIMF pure-adversarial SiT-MeFT runs, one per GPU (1-5).
# Fixes baked in: (1) train_caimf.py single-device save_state (no OOM);
# (2) files/weights symlink -> DINOv2 loads -> FDD computes at every 4-step eval.
# REQUIRE_FD_DINO=1 => hard-fail if FDD weights ever unloadable (no silent empty FDD).
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
STAMP=20260725
cd "$ADV"

# ds:gpu:ckpt
RUNS=(
  "caltech101:1:$MAIN/files/logs/finetuning/caltech101_SiT_DMF_plain_meanflow_meft_online_20260722_172432_pxp35w/best_fid/checkpoint_15000"
  "artbench10:2:$MAIN/files/logs/finetuning/artbench10_SiT_DMF_plain_meanflow_meft_online_20260722_172435_pe502i/best_fid/checkpoint_20000"
  "cub200:3:$MAIN/files/logs/finetuning/cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8/best_fid/checkpoint_20000"
  "food101:4:$MAIN/files/logs/finetuning/food101_SiT_DMF_plain_meanflow_meft_online_20260722_232709_tv7xmz/best_fid/checkpoint_20000"
  "stanfordcars:5:$MAIN/files/logs/finetuning/stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo/best_fid/checkpoint_30000"
)

for entry in "${RUNS[@]}"; do
  IFS=":" read -r ds gpu ckpt <<< "$entry"
  wd=$ADV/files/logs/finetuning/${ds}_SiT_MeFT_CAIMF_puresadv_${STAMP}
  if [[ ! -d "$ckpt" ]]; then echo "ERROR [$ds] ckpt missing: $ckpt"; continue; fi
  sess="caimf_gpu${gpu}_${ds}"
  echo "LAUNCH $ds -> GPU $gpu (screen $sess) wd=$wd"
  screen -dmS "$sess" bash -c "
    cd $ADV
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHON=$PY
    export REPO=$ADV
    export USE_WANDB=False
    export RUN_FINAL_EVAL=True
    export FINAL_EVAL_STEPS=\"1 2 4\"
    export REQUIRE_FD_DINO=1
    export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MPLCONFIGDIR=/tmp/mpl-caimf-$ds
    bash scripts/run_sit_meft_adversarial.sh caimf $ds \"$ckpt\" \"$wd\" \
      2>&1 | tee -a $ADV/files/logs/caimf_${ds}.log
  "
  sleep 8
done
echo "=== all 5 launched ==="
