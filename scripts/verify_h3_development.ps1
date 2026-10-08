# Recheck only this frozen phase. Later package edits must refuse old evidence.
param([string]$Repository = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$revisionNames = @('h3-development-windows-v1', 'h3-development-windows-v2', 'h3-development-wsl-v1')
$revisionSummaries = [ordered]@{}
$revisionManifests = @{}
foreach ($revisionName in $revisionNames) {
    $revisionRoot = Join-Path "$Repository/runs" $revisionName
    $revisionManifest = Get-Content -LiteralPath "$revisionRoot/manifest.json" -Raw | ConvertFrom-Json
    $revisionReport = Get-Content -LiteralPath "$revisionRoot/report.json" -Raw | ConvertFrom-Json
    if ($revisionManifest.schema -cne 'h3-artifact-manifest-v1' -or $revisionReport.schema -cne 'h3-development-benchmark-v1') { throw 'Unregistered artifacts' }
    if (-not $revisionReport.all_independent_audits_valid -or @($revisionReport.rows).Count -ne 28) { throw 'Incomplete report' }
    $revisionDigests = $revisionManifest.sha256
    foreach ($revisionProperty in $revisionDigests.PSObject.Properties) {
        $revisionDigest = (Get-FileHash -LiteralPath "$revisionRoot/$($revisionProperty.Name)" -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($revisionDigest -cne $revisionProperty.Value) { throw "Artifact mismatch: $($revisionProperty.Name)" }
    }
    $revisionFiles = @(Get-ChildItem -LiteralPath $revisionRoot -Recurse -File | Where-Object { $_.FullName -ne (Join-Path $revisionRoot 'manifest.json') })
    if ($revisionFiles.Count -ne @($revisionDigests.PSObject.Properties).Count) { throw 'Unmanifested artifacts' }
    $revisionAudits = @($revisionDigests.PSObject.Properties | Where-Object { $_.Name.EndsWith('/audit.json') })
    if ($revisionAudits.Count -ne 28) { throw 'Missing individual audits' }
    foreach ($revisionAuditEntry in $revisionAudits) {
        $revisionAudit = Get-Content -LiteralPath "$revisionRoot/$($revisionAuditEntry.Name)" -Raw | ConvertFrom-Json
        if (-not $revisionAudit.valid -or -not $revisionAudit.public_replay_valid -or -not $revisionAudit.private_truth_opened) { throw 'Invalid individual audit' }
    }
    $revisionManifests[$revisionName] = $revisionDigests
    $revisionSummaries[$revisionName] = [ordered]@{
        runs = @($revisionReport.rows).Count
        verified_artifacts = $revisionFiles.Count
        all_independent_audits_valid = $true
        report_sha256 = (Get-FileHash -LiteralPath "$revisionRoot/report.json").Hash.ToLowerInvariant()
        manifest_sha256 = (Get-FileHash -LiteralPath "$revisionRoot/manifest.json").Hash.ToLowerInvariant()
    }
}
$revisionReference = $revisionManifests[$revisionNames[0]]
foreach ($revisionName in $revisionNames[1..2]) {
    if (@($revisionManifests[$revisionName].PSObject.Properties).Count -ne @($revisionReference.PSObject.Properties).Count) { throw 'Artifact count mismatch' }
    foreach ($revisionProperty in $revisionReference.PSObject.Properties) {
        if ($revisionManifests[$revisionName].($revisionProperty.Name) -cne $revisionProperty.Value) { throw 'Cross-platform artifact difference' }
    }
}
$revisionRegistration = Get-Content -LiteralPath "$Repository/runs/h3-development-windows-v1/registration.json" -Raw | ConvertFrom-Json
$revisionSources = $revisionRegistration.source_sha256
foreach ($revisionProperty in $revisionSources.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath "$Repository/src/minecraft_scientist/$($revisionProperty.Name)").Hash.ToLowerInvariant() -cne $revisionProperty.Value) { throw 'Current source differs from frozen phase' }
}
if (@(Get-ChildItem -LiteralPath "$Repository/src/minecraft_scientist" -Filter '*.py' -Recurse -File).Count -ne @($revisionSources.PSObject.Properties).Count) { throw 'Unregistered source file' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$revisionWheelPath = "$Repository/runs/h3-source-v1/minecraft_scientist-0.1.0-py3-none-any.whl"
$revisionArchive = [System.IO.Compression.ZipFile]::OpenRead($revisionWheelPath)
$revisionSnapshotCount = 0
try {
    foreach ($revisionEntry in $revisionArchive.Entries) {
        if ($revisionEntry.FullName.StartsWith('minecraft_scientist/') -and $revisionEntry.FullName.EndsWith('.py')) {
            $revisionRelative = $revisionEntry.FullName.Substring(20)
            $revisionStream = $revisionEntry.Open()
            $revisionHasher = [System.Security.Cryptography.SHA256]::Create()
            try {
                $revisionDigest = [BitConverter]::ToString($revisionHasher.ComputeHash($revisionStream)).Replace('-', '').ToLowerInvariant()
            } finally {
                $revisionStream.Dispose()
                $revisionHasher.Dispose()
            }
            if ($revisionSources.$revisionRelative -cne $revisionDigest) { throw 'Wheel differs from registered source' }
            $revisionSnapshotCount++
        }
    }
} finally { $revisionArchive.Dispose() }
if ($revisionSnapshotCount -ne 127 -or $revisionSnapshotCount -ne @($revisionSources.PSObject.Properties).Count) { throw 'Incomplete source snapshot' }
$revisionSuites = [ordered]@{}
$revisionCases = @{}
foreach ($revisionName in @('h3-focused-windows-v3.xml', 'h3-focused-wsl-v1.xml')) {
    [xml]$revisionXml = Get-Content -LiteralPath "$Repository/runs/$revisionName" -Raw
    $revisionSuite = $revisionXml.testsuites.testsuite
    if ([int]$revisionSuite.tests -ne 261 -or [int]$revisionSuite.failures -ne 0 -or [int]$revisionSuite.errors -ne 0 -or [int]$revisionSuite.skipped -ne 0) { throw 'Failed or incomplete focused suite' }
    $revisionSuites[$revisionName] = [ordered]@{
        tests = [int]$revisionSuite.tests; failures = [int]$revisionSuite.failures
        errors = [int]$revisionSuite.errors; skipped = [int]$revisionSuite.skipped
        seconds = [double]$revisionSuite.time
        sha256 = (Get-FileHash -LiteralPath "$Repository/runs/$revisionName").Hash.ToLowerInvariant()
    }
    $revisionCases[$revisionName] = @($revisionSuite.testcase | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object -CaseSensitive -Unique)
}
if (@(Compare-Object -ReferenceObject $revisionCases['h3-focused-windows-v3.xml'] -DifferenceObject $revisionCases['h3-focused-wsl-v1.xml'] -CaseSensitive).Count -ne 0) { throw 'Focused case sets differ' }
[ordered]@{
    schema = 'h3-cross-platform-verification-v1'
    runs = $revisionSummaries
    source_count = $revisionSnapshotCount
    source_sha256 = $revisionSources
    source_snapshot = [ordered]@{path = $revisionWheelPath; sha256 = (Get-FileHash -LiteralPath $revisionWheelPath).Hash.ToLowerInvariant(); python_sources = $revisionSnapshotCount}
    focused_suites = $revisionSuites
    identical_case_sets = $true
    identical_benchmark_artifacts = $true
    design_sha256 = (Get-FileHash -LiteralPath "$Repository/configs/revision-development-v1.json").Hash.ToLowerInvariant()
    scope = 'Bounded structural repair development and focused compatibility only; not structural holdouts, complete H3/H4, method superiority, stochastic diagnosis, Minecraft/RGB or full release.'
} | ConvertTo-Json -Depth 10
