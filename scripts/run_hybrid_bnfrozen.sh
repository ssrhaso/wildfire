#!/bin/bash
set -e

SEEDS="0 5 10 15 20"
CONFIGS="freeze_none_bnfrozen freeze_transformer_only_bnfrozen freeze_transformer_proj_bnfrozen freeze_blocks0-3_bnfrozen freeze_blocks0-5_bnfrozen freeze_blocks0-8_bnfrozen freeze_blocks0-11_bnfrozen"

echo "Hybrid CNN-ViT Freezing Ablation (BN Frozen)"
echo ""
for config in $CONFIGS; do
    for seed in $SEEDS; do
        result="results/hybrid/${config}/seed_${seed}.json"
        if [ -f "$result" ]; then
            echo "Skipping (done): hybrid | $config | seed $seed"
            continue
        fi
        echo "Running: hybrid | $config | seed $seed"
        python src/run_experiment.py \
            --model hybrid \
            --freeze-config "$config" \
            --seed "$seed" \
            --epochsd 20 \
            --batch-size 64 \
            --no-amp \
            --num-workers 4 \
            --wandb-project wildfire-freezing
    done
done