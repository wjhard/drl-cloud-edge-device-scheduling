$ErrorActionPreference = "Continue"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = "training/configs/ppo_mlp_residual.yaml"
$ModelPath = "training/checkpoints/ppo_mlp_residual"
$ScenarioDir = "evaluation/scenarios"
$ResultsDir = "evaluation/results/final_pipeline_lns_repeats"
$SummaryPath = "evaluation/results/final_pipeline_lns_summary.json"
$CanonicalSeeds = @(1565812275, 842234145, 386081360, 1117038938, 1760006972)
$RequiredImports = "import importlib.util; import gymnasium, matplotlib, networkx, numpy, pulp, pytest, rich, scipy, torch, tqdm, yaml; assert all(importlib.util.find_spec(name) is not None for name in ('sb3_contrib', 'stable_baselines3', 'tensorboard'))"

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
    & $PythonExe @PythonPrefixArgs -c $RequiredImports *> $null
    $ImportCheckExit = $LASTEXITCODE

    if ($PipCheckExit -ne 0) {
        Write-Warning "pip check reported conflicts; project import check will determine readiness"
    }

    if ($ImportCheckExit -ne 0) {
        Write-Host "dependencies_ready=false; installing requirements"
        Invoke-Python -StepName "upgrade pip" -Arguments @("-m", "pip", "install", "--upgrade", "pip")

        & $PythonExe @PythonPrefixArgs -c "import torch" *> $null
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

    Write-Host "==== evaluate final Residual best-of-64 + relocation + LNS ===="
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
    Get-ChildItem $ResultsDir -Filter "direct_repeat_*.json" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    for ($Index = 0; $Index -lt $CanonicalSeeds.Count; $Index++) {
        $Repeat = $Index + 1
        $Seed = $CanonicalSeeds[$Index]
        $RunPath = Join-Path $ResultsDir "direct_repeat_$Repeat.json"
        Invoke-Python -StepName "final LNS repeat $Repeat/5 (seed=$Seed)" -Arguments @(
            "evaluation/evaluate_residual_lns.py",
            "--config", $ConfigPath,
            "--model-path", $ModelPath,
            "--results-path", $RunPath,
            "--sampling-seed", "$Seed",
            "--num-samples", "64",
            "--local-max-passes", "3",
            "--lns-iterations", "64"
        )
    }

    Invoke-Python -StepName "analyze five paired final LNS repeats" -Arguments @(
        "evaluation/analyze_residual_lns_direct_repeats.py",
        "--input-dir", $ResultsDir,
        "--output-path", $SummaryPath
    )

    if (-not (Test-Path $SummaryPath)) {
        throw "ERROR: expected final summary was not created: $SummaryPath"
    }
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $MeanRatio = [double]$Summary.statistics.lns_mean_ratio.mean
    $SampleStd = [double]$Summary.statistics.lns_mean_ratio.sample_std
    $PValue = [double]$Summary.statistics.paired_t_test_two_sided.p_value
    $WinningRuns = @($Summary.paired_runs | Where-Object {
        [int]$_.lns_better_than_heft_count -eq 20
    }).Count

    Write-Host "FINAL_PIPELINE_RESULT"
    Write-Host "method=Residual Best-of-64 + topological relocation + best-only LNS"
    Write-Host "repeat_count=$($Summary.repeat_count)"
    Write-Host ("overall_mean_ratio={0:F12}" -f $MeanRatio)
    Write-Host ("overall_mean_ratio_6dp={0:F6}" -f $MeanRatio)
    Write-Host ("sample_std={0:F12}" -f $SampleStd)
    Write-Host ("sample_std_6dp={0:F6}" -f $SampleStd)
    Write-Host "outperform_heft_scenarios=20/20 in $WinningRuns/5 repeats"
    Write-Host ("paired_p_value={0:E12}" -f $PValue)
    Write-Host "results_path=$SummaryPath"
    Write-Host "FINAL_PIPELINE_COMPLETE"
} finally {
    Pop-Location
}
