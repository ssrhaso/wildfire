#!/bin/bash
# Copyright (c) Hasaan Ahmad et Al. 2026. All rights reserved.
# Licensed under the MIT License.
set -e

SEEDS="0 5 10 15 20"
CONFIGS="freeze_none freeze_conv1 freeze_conv1_layer1 freeze_conv1_layer1-2 freeze_conv1_layer1-3 freeze_conv1_layer1-4"

SEED_ARR=($SEEDS)
CONFIG_ARR=($CONFIGS)
TOTAL_RUNS=$(( ${#CONFIG_ARR[@]} * ${#SEED_ARR[@]} ))

echo "ResNet-50 Freezing Ablation"
echo "  Configs: ${#CONFIG_ARR[@]}  Seeds: ${#SEED_ARR[@]}  Total runs: $TOTAL_RUNS"
echo ""

RUN_INDEX=0
SKIPPED=0
FAILED=0
FAILED_LIST=""
START_TIME=$(date +%s)

for config in $CONFIGS; do
    for seed in $SEEDS; do
        RUN_INDEX=$((RUN_INDEX + 1))
        result="results/resnet/${config}/seed_${seed}.json"

        if [ -f "$result" ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "[$RUN_INDEX/$TOTAL_RUNS] Skipping (done): $config | seed $seed"
            continue
        fi

        # Elapsed and ETA
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        COMPLETED=$((RUN_INDEX - SKIPPED - 1))
        if [ "$COMPLETED" -gt 0 ]; then
            AVG=$((ELAPSED / COMPLETED))
            REMAINING=$(( AVG * (TOTAL_RUNS - RUN_INDEX - SKIPPED + 1) / 60 ))
            echo "[$RUN_INDEX/$TOTAL_RUNS] Running: $config | seed $seed  (ETA ~${REMAINING}m)"
        else
            echo "[$RUN_INDEX/$TOTAL_RUNS] Running: $config | seed $seed"
        fi

        if ! python src/run_experiment.py \
            --model resnet \
            --freeze-config "$config" \
            --seed "$seed" \
            --epochs 20 \
            --batch-size 16 \
            --grad-accum-steps 2 \
            --num-workers 4 \
            --wandb-project wildfire-freezing; then
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST  $config/seed_$seed\n"
            echo "  FAILED: $config | seed $seed"
        fi
    done
done

# Summary
END_TIME=$(date +%s)
TOTAL_TIME=$(( (END_TIME - START_TIME) / 60 ))
COMPLETED_RUNS=$((TOTAL_RUNS - SKIPPED - FAILED))

echo ""
echo "Done in ${TOTAL_TIME}m  |  completed: $COMPLETED_RUNS  skipped: $SKIPPED  failed: $FAILED"

if [ "$FAILED" -gt 0 ]; then
    echo "Failed runs:"
    echo -e "$FAILED_LIST"
fi

echo ""
echo "Running analysis"
python src/analyse_results.py --model resnet
