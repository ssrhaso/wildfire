$Seeds = @(0, 5, 10, 15, 20)
$Configs = @(
    "freeze_none",
    "freeze_conv1",
    "freeze_conv1_layer1",
    "freeze_conv1_layer1-2",
    "freeze_conv1_layer1-3",
    "freeze_conv1_layer1-4"
)

Write-Host "ResNet-50 Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $($Configs.Count * $Seeds.Count)" -ForegroundColor Green
Write-Host ""

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        Write-Host "Running: resnet | $config | seed $seed" -ForegroundColor Cyan
        python src/run_experiment.py `
            --model resnet `
            --freeze-config $config `
            --seed $seed `
            --epochs 20 `
            --batch-size 16 `
            --grad-accum-steps 2 `
            --num-workers 0 `
            --wandb-project wildfire-freezing
    }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse_results.py --model resnet
