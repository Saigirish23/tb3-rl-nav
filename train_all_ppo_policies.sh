
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/sim-real-rl"
source .venv/bin/activate

CONFIG="configs/main_16h_4policies.yaml"
LOGDIR="runs/overnight_logs"
RUN_TAG="seq_$(date +%Y%m%d_%H%M%S)"
POLICIES=(P1 P2 P3 P4)
TOTAL=${#POLICIES[@]}
START=$(date +%s)

mkdir -p "$LOGDIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "=============================================="
echo " PPO SEQUENTIAL TRAINING"
echo " Policies: P1 -> P2 -> P3 -> P4"
echo " Steps per policy: 60,000"
echo " Config: $CONFIG"
echo " Started: $(date)"
echo " Logs: $LOGDIR"
echo "=============================================="

for i in "${!POLICIES[@]}"; do
    POLICY="${POLICIES[$i]}"
    NUM=$((i + 1))
    REMAINING=$((TOTAL - i))
    RUN_DIR="runs/${RUN_TAG}_${POLICY}_seed0_60k"
    LOG="$LOGDIR/${RUN_TAG}_${POLICY}.log"
    POLICY_START=$(date +%s)

    echo
    echo ">>> POLICY $POLICY | Run $NUM/$TOTAL | Remaining: $REMAINING"
    echo ">>> Started: $(date)"
    echo ">>> Output log: $LOG"
    echo ">>> Run directory: $RUN_DIR"

    if ! OMP_NUM_THREADS=1 \
         MKL_NUM_THREADS=1 \
         OPENBLAS_NUM_THREADS=1 \
         NUMEXPR_NUM_THREADS=1 \
         python -u rl/train/train_ppo.py \
           --policy "$POLICY" \
           --seed 0 \
           --config "$CONFIG" \
           --run-dir "$RUN_DIR" \
           --gazebo \
           --workers 1 \
           --steps 60000 2>&1 | tee "$LOG"; then
        STATUS=${PIPESTATUS[0]}
        echo "FAILED: $POLICY (exit $STATUS) at $(date)"
        echo "Sequence stopped. Check $LOG"
        exit "$STATUS"
    fi

    POLICY_END=$(date +%s)
    echo "$POLICY completed in $((POLICY_END - POLICY_START)) seconds at $(date)"
    echo "Completed: $NUM/$TOTAL | Remaining: $((TOTAL - NUM))"
done

END=$(date +%s)
echo
echo "ALL FOUR POLICIES COMPLETED"
echo "Total elapsed: $(((END - START) / 3600))h $((((END - START) % 3600) / 60))m"
echo "Finished: $(date)"
