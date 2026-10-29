#!/usr/bin/env bash
# 250-NFE eval of the Tab. 2 iMF MF-T+CAMF / MF-T+AFM best_fid checkpoints (EXP-392 / EXP-393), on dev6.
# Usage: eval250_tab2_imf.sh <camf|afm> <dataset> <gpu> [sample_device_batch_size]
# Same eval path and overrides as the original final evals (run_imf_caimf.sh / run_imf_afm.sh), only
# num_steps=250 and a larger sampling batch (the training-time setting is 1). Writes <run>/eval_best_fid_250steps.
set -euo pipefail
METHOD=$1; DATASET=$2; GPU=$3; SBS=${4:-32}
REPO=/opt/dlami/nvme/meanflow/imeanflow_adversarial; L=$REPO/files/logs/finetuning
case "$METHOD" in
  camf) RUN=$L/${DATASET}_iMF_CAIMF_puresadv_20260726; CONFIG_MODE=caltech_imf_caimf_posttrain ;;
  afm)  RUN=$L/${DATASET}_iMF_AFM_puresadv_20260727;   CONFIG_MODE=caltech_imf_afm_posttrain ;;
  *) echo "method must be camf|afm" >&2; exit 2 ;;
esac
case "$DATASET" in
  caltech101)   LATENT_DIR="caltech-101_processed_latents"; NC=101; FID="caltech-101-fid_stats.npz"; FDD="caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   LATENT_DIR="artbench-10_processed_latents"; NC=10;  FID="artbench-10_processed-fid_stats.npz"; FDD="artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       LATENT_DIR="cub-200-2011_processed_latents"; NC=200; FID="cub-200-2011_processed-fid_stats.npz"; FDD="cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      LATENT_DIR="food-101_processed_latents"; NC=101; FID="food-101_processed-fid_stats.npz"; FDD="food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) LATENT_DIR="stanford-cars_processed_latents"; NC=196; FID="stanford_cars_processed-fid_stats.npz"; FDD="stanford-cars-fd_dino-vitb14_stats.npz" ;;
  *) echo "Unknown dataset" >&2; exit 2 ;;
esac
# the eval script deletes <run>/final_eval_metrics.csv; keep the dev5 NFE 1/2 summary
[[ -f $RUN/final_eval_metrics.csv && ! -f $RUN/final_eval_metrics_nfe12_dev5.csv ]] && cp $RUN/final_eval_metrics.csv $RUN/final_eval_metrics_nfe12_dev5.csv
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU CONFIG_MODE=$CONFIG_MODE PYTHON=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python USE_WANDB=False \
  bash scripts/eval_best_fid_steps_plain_imf.sh "$RUN" 250 -- \
    --config.dataset.name="${DATASET}_latent" \
    --config.dataset.root="/opt/dlami/nvme/meanflow/datasets/$LATENT_DIR" \
    --config.dataset.class_mapping_root="" \
    --config.dataset.num_classes="$NC" \
    --config.model.num_classes="$NC" \
    --config.sampling.num_classes="$NC" \
    --config.fid.cache_ref="$REPO/files/fid_stats/$FID" \
    --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD" \
    --config.fid.sample_device_batch_size="$SBS"
