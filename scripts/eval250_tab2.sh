#!/usr/bin/env bash
# 250-NFE (or any NFE) eval of a Tab. 2 best_fid checkpoint on dev6. Same eval path + overrides as the original
# runner final evals; only num_steps and a larger sampling batch (training-time setting is 1) differ.
# Usage: eval250_tab2.sh <kind> <dataset> <gpu> [nfe=250] [sample_device_batch_size=32]
#   kind: imnetcamf  -> <ds>_iMF_imnet_CAIMF_puresadv_20260727_imnet_300k   (iMF CAMF standalone, Tab. 2)
#         imnetafm   -> <ds>_iMF_imnet_AFM_puresadv_20261029                (iMF AFM standalone, retrained 2026-10-29)
#         sitcamf    -> <ds>_SiT_MeFT_CAIMF_puresadv_20260725               (SiT MF-T + CAMF, Tab. 2)
#         sitafm     -> <ds>_SiT_MeFT_AFM_tab2_20261028                     (SiT MF-T + AFM, EXP-382..386)
set -euo pipefail
KIND=$1; DATASET=$2; GPU=$3; NFE=${4:-250}; SBS=${5:-32}
REPO=/opt/dlami/nvme/meanflow/imeanflow_adversarial; L=$REPO/files/logs/finetuning
case "$KIND" in
  imnetcamf) RUN=$L/${DATASET}_iMF_imnet_CAIMF_puresadv_20260727_imnet_300k; CM=caltech_imf_caimf_posttrain; ES=eval_best_fid_steps_plain_imf.sh ;;
  imnetafm)  RUN=$L/${DATASET}_iMF_imnet_AFM_puresadv_20261029;           CM=caltech_imf_afm_posttrain;   ES=eval_best_fid_steps_plain_imf.sh ;;
  sitcamf)   RUN=$L/${DATASET}_SiT_MeFT_CAIMF_puresadv_20260725;          CM=caltech_sit_meft_caimf_posttrain; ES=eval_best_fid_steps_sit_meft_adversarial.sh ;;
  sitafm)    RUN=$L/${DATASET}_SiT_MeFT_AFM_tab2_20261028;                CM=sit_meft_afm_tab2;           ES=eval_best_fid_steps_sit_meft_adversarial.sh ;;
  *) echo "unknown kind $KIND" >&2; exit 2 ;;
esac
case "$DATASET" in
  caltech101)   LD="caltech-101_processed_latents"; NC=101; FID="caltech-101-fid_stats.npz"; FDD="caltech-101-fd_dino-vitb14_stats.npz" ;;
  artbench10)   LD="artbench-10_processed_latents"; NC=10;  FID="artbench-10_processed-fid_stats.npz"; FDD="artbench-10-fd_dino-vitb14_stats.npz" ;;
  cub200)       LD="cub-200-2011_processed_latents"; NC=200; FID="cub-200-2011_processed-fid_stats.npz"; FDD="cub-200-2011-fd_dino-vitb14_stats.npz" ;;
  food101)      LD="food-101_processed_latents"; NC=101; FID="food-101_processed-fid_stats.npz"; FDD="food-101-fd_dino-vitb14_stats.npz" ;;
  stanfordcars) LD="stanford-cars_processed_latents"; NC=196; FID="stanford_cars_processed-fid_stats.npz"; FDD="stanford-cars-fd_dino-vitb14_stats.npz" ;;
  *) echo "Unknown dataset" >&2; exit 2 ;;
esac
# both eval scripts delete <run>/final_eval_metrics.csv; keep the earlier summary once
if [[ -f $RUN/final_eval_metrics.csv && ! -f $RUN/final_eval_metrics_before_dev6.csv ]]; then cp $RUN/final_eval_metrics.csv $RUN/final_eval_metrics_before_dev6.csv; fi
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU CONFIG_MODE=$CM PYTHON=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python USE_WANDB=False \
  bash scripts/$ES "$RUN" $NFE -- \
    --config.dataset.name="${DATASET}_latent" --config.dataset.root="/opt/dlami/nvme/meanflow/datasets/$LD" \
    --config.dataset.class_mapping_root="" --config.dataset.num_classes="$NC" --config.model.num_classes="$NC" \
    --config.sampling.num_classes="$NC" --config.fid.cache_ref="$REPO/files/fid_stats/$FID" \
    --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD" --config.fid.sample_device_batch_size="$SBS"
