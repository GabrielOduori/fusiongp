#!/bin/bash
# Overnight batch: remaining 8 LOSO folds (station 121_61/M=300 already done)
# plus the full 7-point M-sweep (50,100,150,200,300,400,500) on station 121_61
# (M=300 already done, reused). Config matches today's verified, non-collapsing
# outputscale setup (PRIOR_MEAN_SOURCE=atmo + reparameterized kernel fix).
set -uo pipefail
cd "$(dirname "$0")/.."

MIN_FREE_GB=1
LOG_DIR=outputs/overnight_logs
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"
: > "$SUMMARY"

check_space() {
    avail_gb=$(df --output=avail -BG "$(pwd)" | tail -1 | tr -dc '0-9')
    if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
        echo "$(date '+%F %T') ABORT: only ${avail_gb}G free (< ${MIN_FREE_GB}G), stopping" | tee -a "$SUMMARY"
        exit 1
    fi
}

run_one() {
    local tag="$1"; shift
    check_space
    echo "$(date '+%F %T') START $tag" | tee -a "$SUMMARY"
    env RUN_TAG="$tag" \
        PRIOR_MEAN_SOURCE=atmo \
        PRE_CALIBRATE_TROPOMI=1 \
        LEARN_SATELLITE_CALIBRATION=0 \
        SATELLITE_NOISE_STD=20.0 \
        EPA_NOISE_STD=2.0 \
        EPA_TRAIN_FRACTION=1.0 \
        BIAS_CORRECTION_SOURCE=none \
        "$@" \
        python experiments/run_demo_pipeline.py --device cuda > "$LOG_DIR/${tag}.log" 2>&1
    status=$?
    rundir="outputs/demo_run_${tag}"
    rm -f "$rundir/merged_with_wind_weighted.csv"
    if [ $status -eq 0 ]; then
        echo "$(date '+%F %T') DONE  $tag (exit 0)" | tee -a "$SUMMARY"
    else
        echo "$(date '+%F %T') FAIL  $tag (exit $status) -- see $LOG_DIR/${tag}.log" | tee -a "$SUMMARY"
    fi
}

# Remaining 8 LOSO folds at paper-grade M=300, 200 epochs
for station in 10_94 115_50 23_44 52_41 61_52 76_51 80_26 88_52; do
    run_one "loso_paper_${station}" \
        EPA_HOLDOUT_MODE=station EPA_HOLDOUT_GRID_IDS="$station" \
        N_INDUCING=300 N_EPOCHS=200
done

# Full 7-point M-sweep on station 121_61 (M=300 already done in the pilot run)
for m in 50 100 150 200 400 500; do
    run_one "msweep_121_61_m${m}" \
        EPA_HOLDOUT_MODE=station EPA_HOLDOUT_GRID_IDS=121_61 \
        N_INDUCING="$m" N_EPOCHS=200
done

echo "$(date '+%F %T') ALL DONE" | tee -a "$SUMMARY"
