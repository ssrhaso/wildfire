Write-Host "Wildfire Freezing Ablation  All Models" -ForegroundColor Yellow
Write-Host ""

Write-Host "ViT-B/16 (30 runs)" -ForegroundColor Green
& "$PSScriptRoot\run_vit.ps1"

Write-Host ""
Write-Host "ResNet-50 (30 runs)" -ForegroundColor Green
& "$PSScriptRoot\run_resnet.ps1"

Write-Host ""
Write-Host "Hybrid CNN-ViT (25 runs)" -ForegroundColor Green
& "$PSScriptRoot\run_hybrid.ps1"

Write-Host ""
Write-Host "All experiments complete" -ForegroundColor Yellow
Write-Host "  Results: results/" -ForegroundColor Yellow
Write-Host "  Plots:   results/analysis/" -ForegroundColor Yellow
