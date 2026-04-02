#!/bin/bash
set -e

echo "Wildfire Freezing Ablation - All Models"
echo ""

start=$SECONDS

echo "ViT-B/16 (30 runs)"
bash "$(dirname "$0")/run_vit.sh"
echo "  ViT done in $(( (SECONDS - start) / 60 ))m"

echo ""
mid=$SECONDS
echo "ResNet-50 (30 runs)"
bash "$(dirname "$0")/run_resnet.sh"
echo "  ResNet done in $(( (SECONDS - mid) / 60 ))m"

echo ""
mid=$SECONDS
echo "Hybrid CNN-ViT (25 runs)"
bash "$(dirname "$0")/run_hybrid.sh"
echo "  Hybrid done in $(( (SECONDS - mid) / 60 ))m"

echo ""
echo "All experiments complete in $(( (SECONDS - start) / 60 ))m"
echo "  Results: results/"
echo "  Plots:   results/analysis/"
