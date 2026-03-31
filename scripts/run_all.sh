#!/bin/bash
echo "Wildfire Freezing Ablation  All Models"
echo ""

echo "ViT-B/16 (30 runs)"
bash "$(dirname "$0")/run_vit.sh"

echo ""
echo "ResNet-50 (30 runs)"
bash "$(dirname "$0")/run_resnet.sh"

echo ""
echo "Hybrid CNN-ViT (25 runs)"
bash "$(dirname "$0")/run_hybrid.sh"

echo ""
echo "All experiments complete"
echo "  Results: results/"
echo "  Plots:   results/analysis/"
