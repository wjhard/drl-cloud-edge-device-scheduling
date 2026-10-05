$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $PythonExe (Join-Path $ProjectRoot "demo/run_demo.py") @args
exit $LASTEXITCODE
