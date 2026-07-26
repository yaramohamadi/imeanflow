#!/usr/bin/env bash
# =============================================================================
# CAIMF adaptive early-stop watcher -- detached (screen caimf_earlystop).
#
# For each of the 5 CAIMF SiT-MeFT runs, watches the 4-step FID column of its
# eval_metrics.csv. Kills that run's TRAINING screen once the 4-step FID has
# risen for 3 CONSECUTIVE evals relative to the running minimum (user-set
# patience = 3). save_best_fid_only=True already retains the best checkpoint,
# so stopping only frees the GPU sooner for the redo queue -- the true minimum
# is never lost.
#
# NOTE: CSV training_step counts ALL batches (4D:1G after a 5000-batch D-only
# warmup), so a "step" here is ~1/5 of a generator update. Runs are young; this
# watcher will idle until a run genuinely turns over.
#
# Idempotent / safe: only ever kills a screen that matches caimf_gpu<N>_<ds>.
# Exits when all 5 runs are done (peaked-and-killed, or finished on their own).
# =============================================================================
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
LOG="$ADV/files/logs/caimf_earlystop.log"
mkdir -p "$ADV/files/logs"
PATIENCE=3
POLL_S=300
DSES=(caltech101 artbench10 cub200 food101 stanfordcars)

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

csv_for(){ echo "$ADV/files/logs/finetuning/${1}_SiT_MeFT_CAIMF_puresadv_20260725/eval_metrics.csv"; }
screen_alive(){ screen -ls | grep -qE "caimf_gpu[0-9]+_${1}\b"; }
screen_name(){ screen -ls | grep -oE "caimf_gpu[0-9]+_${1}\b" | head -1; }

# 4-step FID series (space-separated), chronological
fid_series(){ awk -F, 'NR>1 && $4==4 {print $8}' "$1" 2>/dev/null; }

# count consecutive rises at the tail relative to the running minimum:
# walk the series, track min; a value strictly greater than the current min
# increments the streak, a new min (<=) resets it to 0.
consec_rises_from_min(){
  awk '{
    if (NR==1){ mn=$1; streak=0; next }
    if ($1 <= mn){ mn=$1; streak=0 } else { streak++ }
  } END { print streak+0 }'
}

declare -A DONE=()
log "=== CAIMF early-stop watcher: patience=$PATIENCE consecutive 4-step-FID rises, poll ${POLL_S}s ==="
# mark already-finished runs as done at start
for ds in "${DSES[@]}"; do
  if ! screen_alive "$ds"; then DONE[$ds]=1; log "SKIP $ds (no training screen; already finished)"; fi
done

remaining(){ local n=0; for ds in "${DSES[@]}"; do [[ -z "${DONE[$ds]:-}" ]] && n=$((n+1)); done; echo "$n"; }

while (( $(remaining) > 0 )); do
  for ds in "${DSES[@]}"; do
    [[ -n "${DONE[$ds]:-}" ]] && continue
    if ! screen_alive "$ds"; then DONE[$ds]=1; log "DONE $ds (screen gone on its own; best_fid retained)"; continue; fi
    csv=$(csv_for "$ds")
    [[ -f "$csv" ]] || continue
    series=$(fid_series "$csv")
    n=$(echo "$series" | grep -c .)
    (( n < PATIENCE + 1 )) && continue    # need at least PATIENCE+1 points
    streak=$(echo "$series" | consec_rises_from_min)
    mn=$(echo "$series" | sort -g | head -1)
    last=$(echo "$series" | tail -1)
    if (( streak >= PATIENCE )); then
      sess=$(screen_name "$ds")
      log "EARLY-STOP $ds: 4-step FID rose $streak consecutive evals (min=$mn last=$last, n=$n). Killing $sess."
      screen -S "$sess" -X quit 2>/dev/null
      sleep 3
      screen_alive "$ds" && { log "WARN $ds screen $sess still alive after quit; retrying"; pkill -f "CUDA_VISIBLE.*${ds}_SiT_MeFT_CAIMF" 2>/dev/null; }
      DONE[$ds]=1
      log "STOPPED $ds; best_fid checkpoint intact, GPU freed for redo queue."
    fi
  done
  (( $(remaining) > 0 )) && sleep "$POLL_S"
done
log "=== all 5 CAIMF runs resolved (peaked-and-stopped or self-finished); watcher exiting ==="
