#!/usr/bin/env bash
# NFE sweep eval of an iMF best_fid checkpoint for the headline figure (CAMF / AFM, with and without MF-T).
# Same eval path + overrides as eval250_tab2.sh / eval250_tab2_imf.sh (eval_best_fid_steps_plain_imf.sh, native
# iMF sampler and guidance from the run's config); only the NFE list differs. Writes <run>/eval_best_fid_<n>steps.
# Usage: eval_sweep_imf.sh <camf|afm|imnetcamf|imnetafm> <dataset> <gpu> "<nfe list>" [sample_device_batch_size=32]
set -euo pipefail
KIND=$1; DATASET=$2; GPU=$3; NFES=$4; SBS=${5:-32}
REPO=/opt/dlami/nvme/meanflow/imeanflow_adversarial; L=$REPO/files/logs/finetuning
case "$KIND" in
  camf)      RUN=$L/${DATASET}_iMF_CAIMF_puresadv_20260726;                CM=caltech_imf_caimf_posttrain ;;   # MF-T + CAMF (Tab. 2)
  afm)       RUN=$L/${DATASET}_iMF_AFM_puresadv_20260727;                  CM=caltech_imf_afm_posttrain ;;     # MF-T + AFM  (Tab. 2, as EXP-393)
  imnetcamf) RUN=$L/${DATASET}_iMF_imnet_CAIMF_puresadv_20260727_imnet_300k; CM=caltech_imf_caimf_posttrain ;; # CAMF standalone (Tab. 2)
  imnetafm)  RUN=$L/${DATASET}_iMF_imnet_AFM_puresadv_20261029;            CM=caltech_imf_afm_posttrain ;;     # AFM standalone (EXP-394..398)
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
[[ "$NFES" == print ]] && { echo "$RUN"; exit 0; }
[[ -f $RUN/final_eval_metrics.csv && ! -f $RUN/final_eval_metrics_before_sweep.csv ]] && cp $RUN/final_eval_metrics.csv $RUN/final_eval_metrics_before_sweep.csv
cd $REPO
CUDA_VISIBLE_DEVICES=$GPU CONFIG_MODE=$CM PYTHON=/opt/dlami/nvme/meanflow/imeanflow/.venv/bin/python USE_WANDB=False \
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
  bash scripts/eval_best_fid_steps_plain_imf.sh "$RUN" $NFES -- \
    --config.dataset.name="${DATASET}_latent" --config.dataset.root="/opt/dlami/nvme/meanflow/datasets/$LD" \
    --config.dataset.class_mapping_root="" --config.dataset.num_classes="$NC" --config.model.num_classes="$NC" \
    --config.sampling.num_classes="$NC" --config.fid.cache_ref="$REPO/files/fid_stats/$FID" \
    --config.fd_dino.cache_ref="$REPO/files/fdd_stats/$FDD" --config.fid.sample_device_batch_size="$SBS"
