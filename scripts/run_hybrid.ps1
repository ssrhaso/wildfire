$Seeds = @(0, 5, 10, 15, 20)
$Configs = @(
    "freeze_none",
    "freeze_backbone",
    "freeze_backbone_proj",
    "freeze_transformer_only",
    "freeze_backbone_proj_transformer"
)

Write-Host "Hybrid CNN-ViT Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $($Configs.Count * $Seeds.Count)" -ForegroundColor Green
Write-Host ""

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        Write-Host "Running: hybrid | $config | seed $seed" -ForegroundColor Cyan
        python src/run_experiment.py `
            --model hybrid `
            --freeze-config $config `
            --seed $seed `
            --epochs 20 `
            --batch-size 8 `
            --grad-accum-steps 4 `
            --num-workers 0 `
            --wandb-project wildfire-freezing
    }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse_results.py --model hybrid
