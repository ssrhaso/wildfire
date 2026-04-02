#!/bin/bash
set -e

SEEDS="0 5 10 15 20"
CONFIGS="freeze_none freeze_patch freeze_patch_blocks0-3 freeze_patch_blocks0-5 freeze_patch_blocks0-8 freeze_patch_blocks0-11"

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
            --model vit \
            --freeze-config "$config" \
            --seed "$seed" \
            --epochs 20 \
            --batch-size 16 \
            --grad-accum-steps 2 \
            --num-workers 4 \
            --wandb-project wildfire-freezing
    done
done

echo ""
echo "Running analysis"
python src/analyse_results.py --model vit
