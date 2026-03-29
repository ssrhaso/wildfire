#!/bin/bash
SEEDS="42 123 456 789 1024"

for config in freeze_none freeze_patch freeze_patch_blocks0-3 freeze_patch_blocks0-5 freeze_patch_blocks0-8 freeze_patch_blocks0-11; do
    for seed in $SEEDS; do
        echo "=== Running: vit | $config | seed $seed ==="
        python src/run_experiment.py \
            --model vit \
            --freeze-config "$config" \
            --seed "$seed" \
            --epochs 20 \
            --batch-size 16 \
            --num-workers 4 \
            --wandb-project wildfire-freezing
    done
done

echo "=== Running analysis ==="
python src/analyse_results.py --model vit
