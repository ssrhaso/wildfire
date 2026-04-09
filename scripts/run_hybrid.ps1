# Copyright (c) Hasaan Ahmad et Al. 2026. All rights reserved.
# Licensed under the MIT License.

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

$TotalRuns = $Configs.Count * $Seeds.Count

Write-Host "Hybrid CNN-ViT Freezing Ablation" -ForegroundColor Green
Write-Host "  Configs: $($Configs.Count)  Seeds: $($Seeds.Count)  Total runs: $TotalRuns" -ForegroundColor Green
Write-Host ""

$RunIndex = 0
$Skipped = 0
$Failed = @()
$StartTime = Get-Date

foreach ($config in $Configs) {
    foreach ($seed in $Seeds) {
        $RunIndex++
        $result = "results/hybrid/$config/seed_$seed.json"

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
            --model hybrid `
            --freeze-config $config `
            --seed $seed `
            --epochs 20 `
            --batch-size 32 `
            --amp `
            --num-workers 4 `
            --wandb-project wildfire-freezing

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
python src/analyse_results.py --model hybrid
