param([string]$Skeleton = 'Skeleton 002')
$ErrorActionPreference = 'Stop'
$runtime = 'C:\Users\shuji\Documents\Codex\2026-09-06\i-m-creating-a-project-which\work\venv\Scripts\python.exe'
if ($env:DARTISTIC_PYTHON) { $runtime = $env:DARTISTIC_PYTHON }
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Set DARTISTIC_PYTHON to the Python executable where AI4AnimationPy works.' }
Push-Location $PSScriptRoot
try {
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'work/mpl'
    & $runtime camera/live_view.py --skeleton $Skeleton
    if ($LASTEXITCODE -ne 0) { throw "Live viewer exited with code $LASTEXITCODE" }
} finally { Pop-Location }
