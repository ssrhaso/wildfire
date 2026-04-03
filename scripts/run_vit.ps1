$Seeds = @(0, 5, 10, 15, 20)
$Configs = @(    "freeze_patch_blocks0-11"
)

Write-Host "ViT-B/16 Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $($Configs.Count * $Seeds.Count)" -ForegroundColor Green
Write-Host ""

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        Write-Host "Running: vit | $config | seed $seed" -ForegroundColor Cyan
        python src/run_experiment.py `
            --model vit `
            --freeze-config $config `
            --seed $seed `
            --epochs 20 `
            --batch-size 16 `
            --grad-accum-steps 2 `
            --num-workers 0 `
            --no-wandb
roject wildfire-freezing
    }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse
_results.py --model vit
