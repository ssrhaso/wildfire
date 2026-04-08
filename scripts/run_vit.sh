#!/bin/bash
# Copyright (c) Hasaan Ahmad 2026. All rights reserved.
# Licensed under the MIT License.
set -e

CONFIG_FILE="configs/vit.yaml"

# Read seeds and freeze configs from YAML
SEEDS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(str(s) for s in cfg['seeds']))")
CONFIGS=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(cfg['freeze_configs']))")

echo "ViT-B/16 Freezing Ablation"
echo ""

for config in $CONFIGS; do
    for seed in $SEEDS; do
        result="results/vit/${config}/seed_${seed}.json"
        if [ -f "$result" ]; then
            echo "Skipping (done): vit | $config | seed $seed"
            continue
        fi
        echo "Running: vit | $config | seed $seed"
        python src/run_experiment.py \
            --config "$CONFIG_FILE" \
            --freeze-config "$config" \
            --seed "$seed"
    done
done

echo ""
echo "Running analysis"
python src/analyse_results.py --model vit
