$ErrorActionPreference = "Continue"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = "training/configs/ppo_mlp_residual.yaml"
$ModelPath = "training/checkpoints/ppo_mlp_residual"
$ScenarioDir = "evaluation/scenarios"
$ResultsPath = "evaluation/results/summary_mlp_residual_bestof64.json"
$RequiredImports = "import importlib.util; import gymnasium, matplotlib, networkx, numpy, pulp, pytest, rich, torch, tqdm, yaml; assert all(importlib.util.find_spec(name) is not None for name in ('sb3_contrib', 'stable_baselines3', 'tensorboard'))"

if ($env:PYTHON) {
    $PythonExe = $env:PYTHON
    $PythonPrefixArgs = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonPrefixArgs = @("-3.12", "-s")
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonExe = "python3"
    $PythonPrefixArgs = @()
} else {
    $PythonExe = "python"
    $PythonPrefixArgs = @()
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    Write-Host "==== $StepName ===="
    & $PythonExe @PythonPrefixArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "ERROR: $StepName failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "FINAL_PIPELINE_START"
    Write-Host "project_root=$ProjectRoot"
    Write-Host "python_command=$PythonExe $($PythonPrefixArgs -join ' ')"

    Write-Host "==== check dependencies ===="
    & $PythonExe @PythonPrefixArgs -m pip check
    $PipCheckExit = $LASTEXITCODE
    & $PythonExe @PythonPrefixArgs -c $RequiredImports
    $ImportCheckExit = $LASTEXITCODE

    if ($PipCheckExit -ne 0) {
        Write-Warning "pip check reported conflicts; project import check will determine readiness"
    }

    if ($ImportCheckExit -ne 0) {
        Write-Host "dependencies_ready=false; installing requirements"
        Invoke-Python -StepName "upgrade pip" -Arguments @("-m", "pip", "install", "--upgrade", "pip")

        & $PythonExe @PythonPrefixArgs -c "import torch"
        if ($LASTEXITCODE -ne 0) {
            Invoke-Python -StepName "install CPU PyTorch" -Arguments @(
                "-m", "pip", "install", "torch==2.12.1+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            )
        }

        Invoke-Python -StepName "install project requirements" -Arguments @(
            "-m", "pip", "install", "-r", "requirements.txt"
        )
        Write-Host "==== verify dependency consistency ===="
        & $PythonExe @PythonPrefixArgs -m pip check
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "pip check still reports conflicts outside the project dependency set"
        }
        Invoke-Python -StepName "verify required imports" -Arguments @("-c", $RequiredImports)
    } else {
        Write-Host "dependencies_ready=true; skipping installation"
    }

    if ((Test-Path $ModelPath) -or (Test-Path "$ModelPath.zip")) {
        Write-Host "checkpoint_exists=true; skipping training"
        Write-Host "checkpoint_path=$ModelPath.zip"
    } else {
        Write-Host "checkpoint_exists=false; starting full 200000-step training"
        Invoke-Python -StepName "train final residual model" -Arguments @(
            "training/train_ppo.py", "--config", $ConfigPath
        )
    }

    $ScenarioFiles = @()
    if (Test-Path $ScenarioDir) {
        $ScenarioFiles = @(Get-ChildItem $ScenarioDir -Filter "scenario_*.json" -File -ErrorAction SilentlyContinue)
    }
    if ($ScenarioFiles.Count -gt 0) {
        Write-Host "validation_scenarios_exist=true; count=$($ScenarioFiles.Count); skipping generation"
    } else {
        Write-Host "validation_scenarios_exist=false; generating fixed validation scenarios"
        Invoke-Python -StepName "generate validation scenarios" -Arguments @(
            "evaluation/generate_validation_scenarios.py", "--config", $ConfigPath
        )
    }

    Invoke-Python -StepName "evaluate Residual best-of-64" -Arguments @(
        "evaluation/evaluate_bestofn.py",
        "--config", $ConfigPath,
        "--model-path", $ModelPath,
        "--num-samples", "64"
    )

    if (-not (Test-Path $ResultsPath)) {
        throw "ERROR: expected results file was not created: $ResultsPath"
    }
    $Summary = Get-Content $ResultsPath -Raw | ConvertFrom-Json
    $MeanRatio = [double]$Summary.overall.mean_ratio
    $OutperformCount = @($Summary.scenarios | Where-Object { [double]$_.ratio -lt 1.0 }).Count
    $ScenarioCount = @($Summary.scenarios).Count

    Write-Host "FINAL_PIPELINE_RESULT"
    Write-Host ("overall_mean_ratio={0:F12}" -f $MeanRatio)
    Write-Host "outperform_heft_scenarios=$OutperformCount/$ScenarioCount"
    Write-Host "results_path=$ResultsPath"
    Write-Host "FINAL_PIPELINE_COMPLETE"
} finally {
    Pop-Location
}
