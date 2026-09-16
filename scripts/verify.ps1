[CmdletBinding()]
param(
    [switch] $Benchmarks
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Program,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot
try {
    Invoke-Checked uv run pytest
    Invoke-Checked uv build
    $checkpoint = Join-Path $projectRoot "checkpoints/four-form-experiment-policy.pt"
    $expectedCheckpointHash = "CEBB08E5D939AAEF813B2670D1A615CBE2686151136CF79C05CCD594A9B51CB4"
    $actualCheckpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
    if ($actualCheckpointHash -ne $expectedCheckpointHash) {
        throw "Planner checkpoint hash mismatch: $actualCheckpointHash"
    }
    Invoke-Checked npm ci --prefix bridge/mineflayer
    Invoke-Checked npm audit --prefix bridge/mineflayer --omit=dev
    Invoke-Checked node --check bridge/mineflayer/server.cjs
    Invoke-Checked powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        mod/scientist-world-rules/build.ps1

    if ($Benchmarks) {
        Invoke-Checked uv run minecraft-scientist benchmark --suite mvp --seed 0 `
            --output runs/verification/mvp --overwrite
        Invoke-Checked uv run minecraft-scientist benchmark --suite full --seed 0 `
            --planner learned `
            --planner-checkpoint checkpoints/four-form-experiment-policy.pt `
            --output runs/verification/full --overwrite
        Invoke-Checked uv run minecraft-scientist benchmark-baselines --seed 0 `
            --seeds 10 --output runs/verification/baselines --overwrite
    }
} finally {
    Pop-Location
}
