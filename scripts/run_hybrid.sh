#!/bin/bash
SEEDS="0 5 10 15 20"
CONFIGS="freeze_none freeze_backbone freeze_backbone_proj freeze_transformer_only freeze_transformer_proj freeze_backbone_proj_transformer freeze_backbone_proj_blocks0-3 freeze_backbone_proj_blocks0-5 freeze_backbone_proj_blocks0-8 freeze_backbone_proj_blocks0-11 freeze_blocks0-3 freeze_blocks0-5 freeze_blocks0-8 freeze_blocks0-11 freeze_none_bnfrozen freeze_transformer_only_bnfrozen freeze_transformer_proj_bnfrozen freeze_blocks0-3_bnfrozen freeze_blocks0-5_bnfrozen freeze_blocks0-8_bnfrozen freeze_blocks0-11_bnfrozen"

echo "Hybrid CNN-ViT Freezing Ablation"
echo ""

for config in $CONFIGS; do
    for seed in $SEEDS; do
        echo "Running: hybrid | $config | seed $seed"
        python src/run_experiment.py \
            --model hybrid \
            --freeze-config "$config" \
            --seed "$seed" \
            --epochs 20 \
            --batch-size 16 \
            --grad-accum-steps 4 \
            --num-workers 4 \
            --wandb-project wildfire-freezing
    done
done

echo ""
echo "Running analysis"
python src/analyse_results.py --model hybrid
