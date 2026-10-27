param(
    [string]$Skeleton = 'Skeleton 002',
    [string]$Server = '127.0.0.1',
    [string]$Client = '127.0.0.1',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $pipeArguments = @('camera/motive.py', '--skeleton', $Skeleton, '--server', $Server, '--client', $Client)
    if ($Output) { $pipeArguments += @('--output', $Output) }
    python @pipeArguments
    if ($LASTEXITCODE -ne 0) { throw "Motive pipe exited with code $LASTEXITCODE" }
} finally { Pop-Location }
