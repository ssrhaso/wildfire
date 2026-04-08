# Copyright (c) Hasaan Ahmad 2026. All rights reserved.
# Licensed under the MIT License.

$ConfigFile = "configs/vit.yaml"

# Read seeds and freeze configs from YAML
$Seeds = python -c "import yaml; cfg=yaml.safe_load(open('$ConfigFile')); print(' '.join(str(s) for s in cfg['seeds']))"
$Seeds = $Seeds -split ' '
$Configs = python -c "import yaml; cfg=yaml.safe_load(open('$ConfigFile')); print(' '.join(cfg['freeze_configs']))"
$Configs = $Configs -split ' '

Write-Host "ViT-B/16 Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $($Configs.Count * $Seeds.Count)" -ForegroundColor Green
Write-Host ""

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        $result = "results/vit/$config/seed_$seed.json"
        if (Test-Path $result) {
            Write-Host "Skipping (done): vit | $config | seed $seed" -ForegroundColor Yellow
            continue
        }
        Write-Host "Running: vit | $config | seed $seed" -ForegroundColor Cyan
        python src/run_experiment.py `
            --config $ConfigFile `
            --freeze-config $config `
            --seed $seed `
            --no-wandb
    }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse_results.py --model vit
