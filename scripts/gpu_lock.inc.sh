# --- shared atomic GPU lock (sourced by caimf_dit_watcher + caimf_jit_watcher) -
# Prevents two independent gated watchers from double-claiming the same idle GPU.
# Lock = a directory (atomic mkdir); owner file holds the screen session name so
# any watcher can garbage-collect a stale lock whose screen has exited.
LOCKDIR="${LOCKDIR:-/opt/dlami/nvme/meanflow/imeanflow_adversarial/files/gpu_locks}"
mkdir -p "$LOCKDIR"

gpu_try_claim(){  # $1=idx $2=sessname -> 0 if we won the lock
  if mkdir "$LOCKDIR/gpu_$1" 2>/dev/null; then
    echo "$2" > "$LOCKDIR/gpu_$1/owner"
    return 0
  fi
  return 1
}
gpu_release(){ rm -rf "$LOCKDIR/gpu_$1" 2>/dev/null; }
gpu_locked(){ [[ -d "$LOCKDIR/gpu_$1" ]]; }
gpu_gc_locks(){  # drop locks whose owner screen is gone (any watcher may clean)
  local d own
  for d in "$LOCKDIR"/gpu_*; do
    [[ -d "$d" ]] || continue
    own=$(cat "$d/owner" 2>/dev/null)
    if [[ -n "$own" ]] && screen -ls 2>/dev/null | grep -qE "$own\b"; then continue; fi
    rm -rf "$d"
  done
}
