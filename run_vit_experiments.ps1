$Seeds = @(42, 123, 456, 789, 1024)
$Configs = @(
    "freeze_none",
    "freeze_patch",
    "freeze_patch_blocks0-3",
    "freeze_patch_blocks0-5",
    "freeze_patch_blocks0-8",
    "freeze_patch_blocks0-11"
)

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        Write-Host "=== Running: vit | $config | seed $seed ===" -ForegroundColor Cyan
        python src/run_experiment.py `
            --model vit `
            --freeze-config $config `
            --seed $seed `
            --epochs 20 `
            --batch-size 16 `
            --num-workers 0 `
            --wandb-project wildfire-freezing
    }
}

Write-Host "=== Running analysis ===" -ForegroundColor Green
python src/analyse_results.py --model vit
