#!/bin/bash
# Copyright (c) Hasaan Ahmad 2026. All rights reserved.
# Licensed under the MIT License.
set -e

CONFIG_FILE="configs/vit.yaml"

# Read seeds and freeze configs from YAML
SEEDS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(str(s) for s in cfg['seeds']))")
CONFIGS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(cfg['freeze_configs']))")

SEED_ARR=($SEEDS)
CONFIG_ARR=($CONFIGS)
TOTAL_RUNS=$(( ${#CONFIG_ARR[@]} * ${#SEED_ARR[@]} ))

echo "ViT-B/16 Freezing Ablation"
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
        result="results/vit/${config}/seed_${seed}.json"

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
            --config "$CONFIG_FILE" \
            --freeze-config "$config" \
            --seed "$seed"; then
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
python src/analyse_results.py --model vit
