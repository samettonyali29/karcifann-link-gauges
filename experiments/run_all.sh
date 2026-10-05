#!/usr/bin/env bash
# Reproduce every number in this repository, from nothing.
#
# This file is the reproduction contract: running every queue below, then the
# "figures" pass, regenerates every CSV in results/ and every figure in
# figures/, and `python3 experiments/update_readme.py --check` then passes.
# tests/test_experiments.py checks that it stays that way -- an earlier version
# of this script had accreted into a log of repair passes and would have
# produced a strictly smaller result set than the one committed here.
#
# Queues are independent and may run in parallel, one core each.  Pass a queue
# name, or "figures" for the aggregation pass, which must run last.
#
#     for q in 1 2 3 4 5 6 7 8; do experiments/run_all.sh $q & done; wait
#     experiments/run_all.sh figures
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=python3

# A failure must be visible in the exit status, not only in the log: the queue
# previously printed FAILED and then reported "complete".
failures=0
run() {
    echo "### $(date +%H:%M:%S) $*"
    if ! "$@"; then
        echo "### FAILED: $*"
        failures=$((failures + 1))
    fi
}

# Step-multiplier grid for the gauge families.  Wide enough that no winner
# lands on a boundary; karcifann/audit.py enforces that, and tests/test_audit.py
# fails the build if it stops being true.
W="0.5 1 2 4 8 16 32 64 128"
# Vehicle is the one dataset whose optima run below that floor.
W_VEHICLE="0.03 0.0625 0.125 0.25 0.5 1 2 4 8 16 32 64 128"

# Every dataset gets the same three passes: the gauge-family sweep first (E1
# reads its winners back), then the full E1-E5 analysis, then headroom.
full() {
    local d=$1 grid=$2
    run $P -u experiments/gauge_family.py --dataset "$d" --seeds 16 --scales $grid
    run $P -u experiments/run_analysis.py --dataset "$d" --seeds 16
    run $P -u experiments/headroom.py --dataset "$d" --seeds 16
}

case "${1:?queue name 1-8 or 'figures'}" in
1)  full mnist "$W" ;;
2)  full letter "$W" ;;
3)  full dry_bean "$W" ;;
4)  full pendigits "$W"
    full segment "$W" ;;
5)  full optdigits "$W"
    full satimage "$W" ;;
6)  full vehicle "$W_VEHICLE" ;;
7)  # The standalone studies: XOR, digits, the three-operator comparison and
    # the chained-vs-terminal constructions.  None of these depend on a sweep.
    run $P -u experiments/xor_experiment.py
    run $P -u experiments/digits_experiment.py
    run $P -u experiments/fod_comparison.py
    run $P -u experiments/mnist_experiment.py --epochs 10 --tune
    run $P -u experiments/chained_vs_terminal.py --dataset digits
    run $P -u experiments/chained_vs_terminal.py --dataset mnist \
        --epochs 10 --batch-size 64 --seed 1
    run $P -u experiments/loss_scaling.py ;;
8)  # The nested-split study: tuning is re-run inside each split, so this is
    # the one queue whose cost is dominated by tuning rather than confirmation.
    for d in vehicle satimage segment letter; do
        run $P -u experiments/split_robustness.py --dataset "$d" \
            --splits 10 --seeds 8
        run $P -u experiments/headroom_splits.py --dataset "$d" \
            --splits 10 --seeds 8
    done
    # The paired follow-up, on the three splits that carried the positive
    # means and one control apiece.  It re-tunes on three seeds instead of
    # one, so each split costs three times what it costs above; Letter has
    # no outlier to test and is not included.
    for d in vehicle satimage segment; do
        run $P -u experiments/multiseed_tuning.py --dataset "$d" \
            --tune-seeds 3 --seeds 8
    done ;;
figures)
    run $P -u experiments/draw_architecture.py
    # Restore the current MNIST design after drawing the general collection.
    run $P -u experiments/draw_mnist_architecture.py
    run $P -u experiments/make_figures.py
    run $P -u experiments/update_readme.py ;;
*)  echo "unknown queue: $1"; exit 2 ;;
esac
if [ "$failures" -gt 0 ]; then
    echo "### $(date +%H:%M:%S) queue $1 finished with $failures FAILED step(s)"
    exit 1
fi
echo "### $(date +%H:%M:%S) queue $1 complete"
