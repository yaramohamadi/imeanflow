#!/usr/bin/env bash
# MF-T + CAMF from the FINAL MF-T checkpoint, trained to the full budget (150k batches = 30k G updates),
# NO early stop, then 10K-sample eval of the FINAL CAMF checkpoint at NFE 1 and 4 (reviewer 8oSC: no stage
# selected on the eval metric). Recipe otherwise = Tab. 2 (configs *_caimf_posttrain).
# Usage: bash scripts/camf_final_ckpt.sh <imf|sit> <dataset> <mft_final_ckpt_dir> <gpu>
set -uo pipefail
bb=$1 ds=$2 ck=$3 gpu=$4
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial; IMF=/opt/dlami/nvme/meanflow/imeanflow; PY=$IMF/.venv/bin/python
cd $ADV
wd=$ADV/files/logs/finetuning/${ds}_${bb}_CAIMF_finalckpt_20261028; log=$ADV/files/logs/camf_final_${bb}_${ds}.log
export CUDA_VISIBLE_DEVICES=$gpu PYTHON=$PY REPO=$ADV USE_WANDB=False RUN_FINAL_EVAL=False \
       TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/mpl-camffinal-$bb-$ds
echo "[$(date "+%F %T")] START $bb $ds from $ck -> $wd" | tee -a $log
if [ $bb = imf ]; then CONFIG_MODE=caltech_imf_caimf_posttrain bash scripts/run_imf_caimf.sh $ds $ck $wd 2>&1 | tee -a $log
else bash scripts/run_sit_meft_adversarial.sh caimf $ds $ck $wd 2>&1 | tee -a $log; fi
final=$(ls -d $wd/checkpoint_* 2>/dev/null | sort -V | tail -1)
[ -n "$final" ] || { echo "ERROR no final checkpoint in $wd" | tee -a $log; exit 1; }
echo "[$(date "+%F %T")] FINAL $final" | tee -a $log
ed=${wd}_eval10k_final; mkdir -p $ed/best_fid; [ -e $ed/best_fid/$(basename $final) ] || cp -al $final $ed/best_fid/
case $ds in cub200) LAT=cub-200-2011_processed_latents NC=200 FID=cub-200-2011_processed-fid_stats.npz FDD=cub-200-2011-fd_dino-vitb14_stats.npz;;
  stanfordcars) LAT=stanford-cars_processed_latents NC=196 FID=stanford_cars_processed-fid_stats.npz FDD=stanford-cars-fd_dino-vitb14_stats.npz;; esac
ARGS=(--config.dataset.name=${ds}_latent --config.dataset.root=$ADV/../datasets/$LAT --config.dataset.class_mapping_root=
      --config.dataset.num_classes=$NC --config.model.num_classes=$NC --config.sampling.num_classes=$NC
      --config.fid.cache_ref=$ADV/files/fid_stats/$FID --config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD)
if [ $bb = imf ]; then CONFIG_MODE=caltech_imf_caimf_posttrain bash scripts/eval_best_fid_steps_plain_imf.sh $ed 1 4 -- "${ARGS[@]}" 2>&1 | tee -a $log
else CONFIG_MODE=caltech_sit_meft_caimf_posttrain bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh $ed 1 4 -- "${ARGS[@]}" 2>&1 | tee -a $log; fi
echo "[$(date "+%F %T")] === CAMF FINAL DONE $bb $ds ===" | tee -a $log
