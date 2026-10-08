# Frozen baseline phase only. Later package changes must reject this evidence.
param([string]$Repository = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$automataNames = @('apartness-development-windows-v1', 'apartness-development-windows-v2', 'apartness-development-wsl-v1')
$automataRuns = [ordered]@{}
$automataReference = $null
$automataSources = $null
foreach ($automataName in $automataNames) {
    $automataRoot = Join-Path "$Repository/runs" $automataName
    $automataManifest = Get-Content -LiteralPath "$automataRoot/manifest.json" -Raw | ConvertFrom-Json
    $automataReport = Get-Content -LiteralPath "$automataRoot/report.json" -Raw | ConvertFrom-Json
    $automataRegistration = Get-Content -LiteralPath "$automataRoot/registration.json" -Raw | ConvertFrom-Json
    if ($automataManifest.schema -cne 'apartness-artifact-manifest-v1' -or $automataReport.schema -cne 'apartness-development-benchmark-v1') { throw 'Unexpected baseline schema' }
    if (@($automataManifest.nondeterministic_artifacts).Count -ne 1 -or $automataManifest.nondeterministic_artifacts[0] -cne 'timing.json') { throw 'Unexpected nondeterministic artifact' }
    if (-not $automataReport.all_independent_audits_valid -or @($automataReport.rows).Count -ne 32) { throw 'Incomplete baseline cases' }
    if ($automataRegistration.protocol.budget -ne 240 -or $automataRegistration.protocol.horizon -ne 5 -or $automataRegistration.protocol.conformance_count -ne 8) { throw 'Unexpected development budget/protocol' }
    $automataDigests = $automataManifest.sha256
    foreach ($automataEntry in $automataDigests.PSObject.Properties) {
        if ([System.IO.Path]::IsPathRooted($automataEntry.Name) -or $automataEntry.Name.Contains('..')) { throw 'Manifest path escapes run' }
        if ((Get-FileHash -LiteralPath "$automataRoot/$($automataEntry.Name)").Hash.ToLowerInvariant() -cne $automataEntry.Value) { throw "Artifact changed: $($automataEntry.Name)" }
    }
    $automataFiles = @(Get-ChildItem -LiteralPath $automataRoot -Recurse -File)
    if (@($automataDigests.PSObject.Properties).Count -ne 162 -or $automataFiles.Count -ne 164) { throw 'Missing/unmanifested artifacts' }
    foreach ($automataRow in $automataReport.rows) {
        $automataDirectory = "$automataRoot/seed-$($automataRow.seed)/$($automataRow.family)/states-$($automataRow.private_states)"
        $automataAudit = Get-Content -LiteralPath "$automataDirectory/audit.json" -Raw | ConvertFrom-Json
        $automataRun = Get-Content -LiteralPath "$automataDirectory/run.json" -Raw | ConvertFrom-Json
        if (-not $automataAudit.valid -or -not $automataAudit.public_replay_valid -or -not $automataAudit.private_truth_opened -or -not $automataAudit.evaluation.actual_controls_match_private_reference) { throw 'Invalid individual audit' }
        if ($automataRun.all_allowed_queries_certified -or $automataRun.unique_causal_structure_claimed -or $automataRun.reference_models_used) { throw 'Unjustified learner claim' }
        if ($automataRow.spent -ne $automataRun.spent -or $automataRow.status -cne $automataRun.status -or $automataRun.spent -gt 240) { throw 'Individual ledger/status mismatch' }
        if ($automataRow.evaluation.provisional_behavior_correct_at_H -cne $automataAudit.evaluation.provisional_behavior_correct_at_H) { throw 'Hidden negative result' }
    }
    $automataCorrect = @($automataReport.rows | Where-Object { $_.evaluation.provisional_behavior_correct_at_H -ceq $true }).Count
    $automataIncorrect = @($automataReport.rows | Where-Object { $_.evaluation.provisional_behavior_correct_at_H -ceq $false }).Count
    $automataUnresolved = @($automataReport.rows | Where-Object { $_.status -ceq 'unresolved' }).Count
    if ($automataCorrect -ne 18 -or $automataIncorrect -ne 3 -or $automataUnresolved -ne 11) { throw 'Frozen positive/negative outcomes changed' }
    if ($null -eq $automataReference) {
        $automataReference = $automataDigests
        $automataSources = $automataRegistration.source_sha256
    } else {
        foreach ($automataEntry in $automataReference.PSObject.Properties) {
            if ($automataDigests.($automataEntry.Name) -cne $automataEntry.Value) { throw 'Cross-platform deterministic artifact mismatch' }
        }
    }
    $automataRuns[$automataName] = [ordered]@{
        runs = 32; verified_deterministic_artifacts = 162
        correct_provisional_at_H = $automataCorrect; incorrect_provisional_at_H = $automataIncorrect
        unresolved = $automataUnresolved; nondeterministic_artifacts = @('timing.json')
        report_sha256 = (Get-FileHash -LiteralPath "$automataRoot/report.json").Hash.ToLowerInvariant()
        manifest_sha256 = (Get-FileHash -LiteralPath "$automataRoot/manifest.json").Hash.ToLowerInvariant()
    }
}
foreach ($automataEntry in $automataSources.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath "$Repository/src/minecraft_scientist/$($automataEntry.Name)").Hash.ToLowerInvariant() -cne $automataEntry.Value) { throw 'Current source differs from frozen baseline phase' }
}
if (@($automataSources.PSObject.Properties).Count -ne 130 -or @(Get-ChildItem -LiteralPath "$Repository/src/minecraft_scientist" -Recurse -File -Filter '*.py').Count -ne 130) { throw 'Incomplete source registration' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$automataWheel = "$Repository/runs/apartness-source-v1/minecraft_scientist-0.1.0-py3-none-any.whl"
$automataArchive = [System.IO.Compression.ZipFile]::OpenRead($automataWheel)
$automataSnapshotCount = 0
try {
    foreach ($automataEntry in $automataArchive.Entries) {
        if ($automataEntry.FullName.StartsWith('minecraft_scientist/') -and $automataEntry.FullName.EndsWith('.py')) {
            $automataRelative = $automataEntry.FullName.Substring(20)
            $automataStream = $automataEntry.Open()
            $automataHasher = [System.Security.Cryptography.SHA256]::Create()
            try {
                $automataDigest = [BitConverter]::ToString($automataHasher.ComputeHash($automataStream)).Replace('-', '').ToLowerInvariant()
            } finally { $automataStream.Dispose(); $automataHasher.Dispose() }
            if ($automataSources.$automataRelative -cne $automataDigest) { throw 'Wheel differs from registered source' }
            $automataSnapshotCount++
        }
    }
} finally { $automataArchive.Dispose() }
if ($automataSnapshotCount -ne 130) { throw 'Incomplete source snapshot' }
$automataSuites = [ordered]@{}
$automataCaseSets = @{}
foreach ($automataName in @('apartness-focused-windows-v2.xml', 'apartness-focused-wsl-v1.xml')) {
    [xml]$automataXml = Get-Content -LiteralPath "$Repository/runs/$automataName" -Raw
    $automataSuite = $automataXml.testsuites.testsuite
    if ([int]$automataSuite.tests -ne 342 -or [int]$automataSuite.failures -ne 0 -or [int]$automataSuite.errors -ne 0 -or [int]$automataSuite.skipped -ne 0) { throw 'Failed/incomplete focused regression' }
    $automataCases = @($automataSuite.testcase | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object -Unique -CaseSensitive)
    if ($automataCases.Count -ne 342) { throw 'Duplicate/missing focused cases' }
    $automataCaseSets[$automataName] = $automataCases
    $automataSuites[$automataName] = [ordered]@{
        tests = 342; failures = 0; errors = 0; skipped = 0
        seconds = [double]$automataSuite.time
        sha256 = (Get-FileHash -LiteralPath "$Repository/runs/$automataName").Hash.ToLowerInvariant()
    }
}
if (@(Compare-Object -ReferenceObject $automataCaseSets['apartness-focused-windows-v2.xml'] -DifferenceObject $automataCaseSets['apartness-focused-wsl-v1.xml'] -CaseSensitive).Count -ne 0) { throw 'Windows/Linux focused cases differ' }
[ordered]@{
    schema = 'apartness-cross-platform-verification-v1'
    source_count = 130; source_sha256 = $automataSources
    source_snapshot = [ordered]@{path = $automataWheel; sha256 = (Get-FileHash -LiteralPath $automataWheel).Hash.ToLowerInvariant(); python_sources = 130}
    focused_suites = $automataSuites; identical_case_sets = $true
    runs = $automataRuns; identical_deterministic_artifacts = $true
    excluded_nondeterministic_artifacts = @('timing.json')
    design_sha256 = (Get-FileHash -LiteralPath "$Repository/configs/automata-development-v1.json").Hash.ToLowerInvariant()
    scope = 'Basic L# stationary development and focused compatibility only; preserves three wrong provisional models/eleven unresolved. Not full Adaptive L#, structural holdout, H3/LLM matrix, method superiority, H4, Minecraft/RGB or new full/default release.'
} | ConvertTo-Json -Depth 10
