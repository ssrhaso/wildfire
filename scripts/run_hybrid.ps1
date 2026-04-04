$Seeds = @(0, 5, 10, 15, 20)
$Configs = @(
    "freeze_none",
    "freeze_backbone",
    "freeze_backbone_proj",
    "freeze_transformer_only",
    "freeze_transformer_proj",
    "freeze_backbone_proj_transformer",
    "freeze_backbone_proj_blocks0-3",
    "freeze_backbone_proj_blocks0-5",
    "freeze_backbone_proj_blocks0-8",
    "freeze_backbone_proj_blocks0-11",
    "freeze_blocks0-3",
    "freeze_blocks0-5",
    "freeze_blocks0-8",
    "freeze_blocks0-11",
    "freeze_none_bnfrozen",
    "freeze_transformer_only_bnfrozen",
    "freeze_transformer_proj_bnfrozen",
    "freeze_blocks0-3_bnfrozen",
    "freeze_blocks0-5_bnfrozen",
    "freeze_blocks0-8_bnfrozen",
    "freeze_blocks0-11_bnfrozen"
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
            --batch-size 32 `
            --amp `
            --num-workers 4 `
            --wandb-project wildfire-freezing
    }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse_results.py --model hybrid
