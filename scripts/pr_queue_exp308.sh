#!/usr/bin/env bash
# EXP-308 job queue: one screen per GPU, jobs run in order on that GPU.
set -uo pipefail
MAIN=/opt/dlami/nvme/meanflow/imeanflow
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
L=$MAIN/files/logs/finetuning; A=$ADV/files/logs/finetuning
D=/opt/dlami/nvme/meanflow/datasets; O=$MAIN/files/logs/pr_eval
PY=$MAIN/.venv/bin/python
real_cub="cd $MAIN && $PY scripts/pr_real_features.py $D/cub-200-2011_images/train $O/real_cub200/feats $MAIN/files/fid_stats/cub-200-2011_processed-fid_stats.npz $MAIN/files/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz 2>&1 | tee $O/real_cub200.log"
real_cars="cd $MAIN && $PY scripts/pr_real_features.py $D/stanford-cars_images/train $O/real_stanfordcars/feats $MAIN/files/fid_stats/stanford_cars_processed-fid_stats.npz $MAIN/files/fdd_stats/stanford-cars-fd_dino-vitb14_stats.npz 2>&1 | tee $O/real_stanfordcars.log"
ev(){ echo "bash $MAIN/scripts/pr_eval_one.sh $1 $2 $3 4 $4"; }
imf_mft_cub=$(ev $MAIN $L/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6 $L/plain_iMF_finetune_cub200_h100_20260722_230203_q9w1n6/eval_best_fid_1steps/output.log imf_mft_cub200)
imf_mft_cars=$(ev $MAIN $L/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh $L/plain_iMF_finetune_stanfordcars_h100_20260722_233147_5fimmh/eval_best_fid_1steps/output.log imf_mft_stanfordcars)
imf_camf_cub=$(ev $ADV $A/cub200_iMF_CAIMF_puresadv_20260726 $A/cub200_iMF_CAIMF_puresadv_20260726/eval_best_fid_1steps/output.log imf_camf_cub200)
imf_camf_cars=$(ev $ADV $A/stanfordcars_iMF_CAIMF_puresadv_20260726 $A/stanfordcars_iMF_CAIMF_puresadv_20260726/eval_best_fid_1steps/output.log imf_camf_stanfordcars)
sit_mft_cub=$(ev $MAIN $L/cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8 $L/cub200_SiT_DMF_plain_meanflow_meft_online_20260722_232439_9et5j8/eval_best_fid_4steps/output.log sit_mft_cub200)
sit_mft_cars=$(ev $MAIN $L/stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo $L/stanfordcars_SiT_DMF_plain_meanflow_meft_online_20260722_232940_wl1dqo/eval_best_fid_4steps/output.log sit_mft_stanfordcars)
sit_camf_cub=$(ev $ADV $A/cub200_SiT_MeFT_CAIMF_puresadv_20260725 $A/cub200_SiT_MeFT_CAIMF_puresadv_20260725/eval_best_fid_1steps/output.log sit_camf_cub200)
sit_camf_cars=$(ev $ADV $A/stanfordcars_SiT_MeFT_CAIMF_puresadv_20260725 $A/stanfordcars_SiT_MeFT_CAIMF_puresadv_20260725/eval_best_fid_1steps/output.log sit_camf_stanfordcars)
mkdir -p $O
run(){ local gpu=$1; shift; local cmds=""; for c in "$@"; do cmds+="$c; "; done
  screen -dmS "pr_gpu$gpu" bash -c "export CUDA_VISIBLE_DEVICES=$gpu; $cmds echo QUEUE_DONE; exec bash"; echo "GPU $gpu: $# jobs"; }
run 5 "$real_cub" "$real_cars" "$imf_mft_cub"
run 6 "$imf_camf_cub" "$imf_mft_cars" "$imf_camf_cars"
run 7 "$sit_mft_cub" "$sit_camf_cub" "$sit_mft_cars" "$sit_camf_cars"
