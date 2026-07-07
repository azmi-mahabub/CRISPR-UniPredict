#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run ALL CRISPR-UniPredict trainings sequentially: 3 off-target + 2 on-target.
# Continues to the next run even if one fails; timestamps + logs everything.
# (train_on_target_focused.py is intentionally excluded — older joint-era /
#  superseded, ~4-6 h. Add a line for it yourself if you really want it.)
#
#   Run from WSL with the `oneclick` conda env active:
#       bash run_all_training.sh
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
LOG="train_all_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $(pwd)/$LOG"

run () {
  echo | tee -a "$LOG"
  echo "========== [$(date '+%F %T')] START:  $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local code=${PIPESTATUS[0]}
  echo "========== [$(date '+%F %T')] DONE (exit $code):  $*" | tee -a "$LOG"
}

# ---- Off-target (3) -------------------------------------------------------
run python train_off_target_dedicated.py
run python run_offtarget_multiseed.py
rm -f models/checkpoints/off_target_dedicated/best_model_seqsplit_offtarget.pt
run python run_offtarget_seqsplit.py

# ---- On-target (2; seed 42 already covered by the dedicated run) ----------
run python train_on_target_dedicated.py
run python train_ensemble_on_target.py 1 2 3 4

echo | tee -a "$LOG"
echo "ALL RUNS COMPLETE — $(date '+%F %T').  Per-run status:" | tee -a "$LOG"
grep 'DONE (exit' "$LOG"
