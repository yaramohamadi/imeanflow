#!/usr/bin/env bash
# Resume an iMF AFM run (moved from dev6 on 2026-10-29) from a FULL-state checkpoint in its own workdir.
# Same as scripts/run_imf_afm.sh except: no "workdir already has a checkpoint" refusal, and --config.afm.resume_from.
# Usage: RESUME_FROM=<workdir>/checkpoint_N bash scripts/run_imf_afm_resume.sh <dataset> <load_from> <workdir>
set -euo pipefail
[[ -n "${RESUME_FROM:-}" && -d "$RESUME_FROM" ]] || { echo "RESUME_FROM must be an existing checkpoint dir" >&2; exit 3; }
SRC="$(cd "$(dirname "$0")" && pwd)/run_imf_afm.sh"
TMP=$(mktemp /tmp/run_imf_afm_resume.XXXX.sh)
python3 - "$SRC" "$TMP" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = 'if [[ -d "$WORKDIR" ]] && find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name \'checkpoint_*\' -print -quit | grep -q .; then'
i = s.index(a); j = s.index("fi\n", i) + 3; s = s[:i] + s[j:]
b = '  "${COMMON_ARGS[@]}" \\\n  --workdir="$WORKDIR"'
assert s.count(b) == 1
s = s.replace(b, '  "${COMMON_ARGS[@]}" \\\n  --config.afm.resume_from="$RESUME_FROM" \\\n  --workdir="$WORKDIR"')
s = s.replace('REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"', 'REPO="${REPO:-/opt/dlami/nvme/meanflow/imeanflow_adversarial}"')
open(sys.argv[2], "w").write(s)
PY
exec bash "$TMP" "$@"
