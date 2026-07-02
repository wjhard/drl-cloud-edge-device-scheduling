$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON) { $env:PYTHON } else { "py" }
$PythonArgs = @()
if (-not $env:PYTHON) {
    $PythonArgs += "-s"
}

function Invoke-Phase2Step {
    param(
        [string]$Name,
        [string[]]$Args
    )

    Write-Host "==== $Name ===="
    & $PythonExe @PythonArgs @Args
    if ($LASTEXITCODE -ne 0) {
        throw "ERROR: $Name failed"
    }
}

Invoke-Phase2Step -Name "train MaskablePPO" -Args @(
    "training/train_ppo.py",
    "--config",
    "training/configs/ppo_mlp_baseline.yaml"
)
Invoke-Phase2Step -Name "generate validation scenarios" -Args @(
    "evaluation/generate_validation_scenarios.py"
)
Invoke-Phase2Step -Name "evaluate RL vs HEFT" -Args @(
    "evaluation/evaluate.py",
    "--config",
    "training/configs/ppo_mlp_baseline.yaml",
    "--model-path",
    "training/checkpoints/ppo_mlp_baseline"
)

