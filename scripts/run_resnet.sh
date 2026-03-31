#!/bin/bash
SEEDS="0 5 10 15 20"
CONFIGS="freeze_none freeze_conv1 freeze_conv1_layer1 freeze_conv1_layer1-2 freeze_conv1_layer1-3 freeze_conv1_layer1-4"

echo "ResNet-50 Freezing Ablation"
echo ""

for config in $CONFIGS; do
    for seed in $SEEDS; do
        echo "Running: resnet | $config | seed $seed"
        python src/run_experiment.py \
            --model resnet \
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
python src/analyse_results.py --model resnet
