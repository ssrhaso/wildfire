# Copyright (c) Hasaan Ahmad 2026. All rights reserved.
# Licensed under the MIT License.

$ConfigFile = "configs/vit.yaml"

# Read seeds and freeze configs from YAML
$Seeds = python -c "import yaml; cfg=yaml.safe_load(open('$ConfigFile')); print(' '.join(str(s) for s in cfg['seeds']))"
$Seeds = $Seeds -split ' '
$Configs = python -c "import yaml; cfg=yaml.safe_load(open('$ConfigFile')); print(' '.join(cfg['freeze_configs']))"
$Configs = $Configs -split ' '

$TotalRuns = $Configs.Count * $Seeds.Count

Write-Host "ViT-B/16 Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $TotalRuns" -ForegroundColor Green
Write-Host ""

$RunIndex = 0
$Skipped = 0
$Failed = @()
$StartTime = Get-Date

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        $RunIndex++
        $result = "results/vit/$config/seed_$seed.json"
        if (Test-Path $result) {
            $Skipped++
            Write-Host "[$RunIndex/$TotalRuns] Skipping (done): $config | seed $seed" -ForegroundColor Yellow
            continue
        }

        # Elapsed and ETA
        $Elapsed = (Get-Date) - $StartTime
        $Completed = $RunIndex - $Skipped - 1
        if ($Completed -gt 0) {
            $AvgSec = $Elapsed.TotalSeconds / $Completed
            $Remaining = [math]::Round($AvgSec * ($TotalRuns - $RunIndex - $Skipped + 1) / 60, 1)
            Write-Host "[$RunIndex/$TotalRuns] Running: $config | seed $seed  (ETA ~${Remaining}m)" -ForegroundColor Cyan
        } else {
            Write-Host "[$RunIndex/$TotalRuns] Running: $config | seed $seed" -ForegroundColor Cyan
        }

        python src/run_experiment.py `
            --config $ConfigFile `
            --freeze-config $config `
            --seed $seed `
            --no-wandb

        if ($LASTEXITCODE -ne 0) {
            $Failed += "$config/seed_$seed"
            Write-Host "  FAILED: $config | seed $seed" -ForegroundColor Red
        }
    }
}

# Summary
$TotalTime = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
$Completed = $TotalRuns - $Skipped - $Failed.Count
Write-Host ""
Write-Host "Done in ${TotalTime}m  |  completed: $Completed  skipped: $Skipped  failed: $($Failed.Count)" -ForegroundColor Green

if ($Failed.Count -gt 0) {
    Write-Host "Failed runs:" -ForegroundColor Red
    foreach ($f in $Failed) { Write-Host "  $f" -ForegroundColor Red }
}

Write-Host ""
Write-Host "Running analysis" -ForegroundColor Green
python src/analyse_results.py --model vit
