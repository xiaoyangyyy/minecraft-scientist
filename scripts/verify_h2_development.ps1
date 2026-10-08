# Frozen H2 phase only. Later source edits must fail, not inherit old evidence.
param([string]$Repository = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$h2Runs=@('h2-development-windows-v1','h2-development-windows-v2','h2-development-wsl-v1')
$h2Summaries=[ordered]@{}
$h2Manifests=@{}
foreach($h2Name in $h2Runs) {
 $h2Root="$Repository/runs/"+$h2Name
 $h2Manifest=Get-Content -LiteralPath ($h2Root+'/manifest.json') -Raw -ErrorAction Stop | ConvertFrom-Json
 $h2Report=Get-Content -LiteralPath ($h2Root+'/report.json') -Raw -ErrorAction Stop | ConvertFrom-Json
 $h2Bad=@()
 foreach($h2Property in $h2Manifest.PSObject.Properties) {
  if ((Get-FileHash -LiteralPath ($h2Root+'/'+$h2Property.Name) -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant() -cne $h2Property.Value) { $h2Bad += $h2Property.Name }
 }
 if ($h2Bad.Count -ne 0) { throw 'Artifact digest mismatch' }

 if (-not $h2Report.all_independent_audits_valid -or @($h2Report.rows).Count -ne 108) { throw 'Incomplete benchmark report' }
 $h2ActualFiles=@(Get-ChildItem -LiteralPath $h2Root -Recurse -File | Where-Object { $_.FullName -ne (Join-Path $h2Root 'manifest.json') })
 if ($h2ActualFiles.Count -ne @($h2Manifest.PSObject.Properties).Count) { throw 'Unmanifested artifact' }
 $h2AuditEntries=@($h2Manifest.PSObject.Properties | Where-Object { $_.Name.EndsWith('/audit.json') })
 if ($h2AuditEntries.Count -ne 108) { throw 'Missing individual audits' }
 foreach($h2AuditEntry in $h2AuditEntries) {
  $h2Audit=Get-Content -LiteralPath ($h2Root+'/'+$h2AuditEntry.Name) -Raw | ConvertFrom-Json
  if (-not $h2Audit.valid -or -not $h2Audit.public_replay_valid -or -not $h2Audit.private_truth_opened) { throw 'Invalid individual audit' }
 }

 $h2Manifests[$h2Name]=$h2Manifest
 $h2Summaries[$h2Name]=[ordered]@{runs=@($h2Report.rows).Count;manifested_files=@($h2Manifest.PSObject.Properties).Count;verified_artifacts=@($h2Manifest.PSObject.Properties).Count;all_independent_audits_valid=$h2Report.all_independent_audits_valid;report_sha256=(Get-FileHash -LiteralPath ($h2Root+'/report.json')).Hash.ToLowerInvariant();manifest_sha256=(Get-FileHash -LiteralPath ($h2Root+'/manifest.json')).Hash.ToLowerInvariant()}
}
$h2Reference=$h2Manifests[$h2Runs[0]]
$h2Different=@()
foreach($h2Name in $h2Runs[1..2]) {
 if (@($h2Manifests[$h2Name].PSObject.Properties).Count -ne @($h2Reference.PSObject.Properties).Count) { throw 'Artifact count mismatch' }
 foreach($h2Property in $h2Reference.PSObject.Properties) {
  if ($h2Manifests[$h2Name].($h2Property.Name) -cne $h2Property.Value) { $h2Different += $h2Name+'/'+$h2Property.Name }
 }
}
$h2Reg=(Get-Content -LiteralPath "$Repository/runs/h2-development-windows-v1/registration.json" -Raw | ConvertFrom-Json)
$h2SourceBad=@()
foreach($h2Property in $h2Reg.source_sha256.PSObject.Properties) {
 if ((Get-FileHash -LiteralPath ("$Repository/src/minecraft_scientist/"+$h2Property.Name)).Hash.ToLowerInvariant() -cne $h2Property.Value) { $h2SourceBad += $h2Property.Name }
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$h2WheelPath="$Repository/runs/h2-source-v1/minecraft_scientist-0.1.0-py3-none-any.whl"
$h2Archive=[System.IO.Compression.ZipFile]::OpenRead($h2WheelPath)
$h2SnapshotCount=0
$h2SnapshotBad=@()
foreach($h2Entry in $h2Archive.Entries) {
 if ($h2Entry.FullName.StartsWith('minecraft_scientist/') -and $h2Entry.FullName.EndsWith('.py')) {
  $h2Relative=$h2Entry.FullName.Substring(20)
  $h2Stream=$h2Entry.Open()
  $h2Hasher=[System.Security.Cryptography.SHA256]::Create()
  $h2Digest=[BitConverter]::ToString($h2Hasher.ComputeHash($h2Stream)).Replace('-','').ToLowerInvariant()
  $h2Stream.Dispose()
  $h2Hasher.Dispose()
  $h2SnapshotCount++
  if ($h2Reg.source_sha256.$h2Relative -cne $h2Digest) { $h2SnapshotBad += $h2Relative }
 }
}
$h2Archive.Dispose()
$h2Suites=[ordered]@{}
$h2Cases=@{}
foreach($h2Name in @('h2-focused-windows-v2.xml','h2-focused-wsl-v1.xml')) {
 [xml]$h2Xml=Get-Content -LiteralPath ("$Repository/runs/"+$h2Name) -Raw
 $h2Suite=$h2Xml.testsuites.testsuite
 $h2Suites[$h2Name]=[ordered]@{tests=[int]$h2Suite.tests;failures=[int]$h2Suite.failures;errors=[int]$h2Suite.errors;skipped=[int]$h2Suite.skipped;seconds=[double]$h2Suite.time;sha256=(Get-FileHash -LiteralPath ("$Repository/runs/"+$h2Name)).Hash.ToLowerInvariant()}
 $h2Cases[$h2Name]=@($h2Suite.testcase | ForEach-Object { $_.classname+'::'+$_.name } | Sort-Object -CaseSensitive -Unique)
}
$h2CaseDifferences=@(Compare-Object -ReferenceObject $h2Cases['h2-focused-windows-v2.xml'] -DifferenceObject $h2Cases['h2-focused-wsl-v1.xml'] -CaseSensitive)

if ($h2Different.Count -ne 0 -or $h2SourceBad.Count -ne 0 -or $h2SnapshotBad.Count -ne 0 -or $h2CaseDifferences.Count -ne 0) { throw 'Cross-platform or source verification mismatch' }
if (@(Get-ChildItem -LiteralPath "$Repository/src/minecraft_scientist" -Filter '*.py' -Recurse -File).Count -ne @($h2Reg.source_sha256.PSObject.Properties).Count) { throw 'Unregistered source file' }
if ($h2SnapshotCount -ne @($h2Reg.source_sha256.PSObject.Properties).Count) { throw 'Incomplete wheel snapshot' }
foreach ($h2SuiteResult in $h2Suites.Values) {
 if ($h2SuiteResult.tests -ne 134 -or $h2SuiteResult.failures -ne 0 -or $h2SuiteResult.errors -ne 0 -or $h2SuiteResult.skipped -ne 0) { throw 'Focused suite failed or incomplete' }
}

[ordered]@{schema='h2-cross-platform-verification-v1';runs=$h2Summaries;source_count=@($h2Reg.source_sha256.PSObject.Properties).Count;source_mismatches=$h2SourceBad;artifact_differences=$h2Different;source_snapshot=[ordered]@{path=$h2WheelPath;sha256=(Get-FileHash -LiteralPath $h2WheelPath).Hash.ToLowerInvariant();python_sources=$h2SnapshotCount;mismatches=$h2SnapshotBad};focused_suites=$h2Suites;focused_case_differences=$h2CaseDifferences;design_sha256=(Get-FileHash -LiteralPath "$Repository/configs/diagnosis-development-v1.json").Hash.ToLowerInvariant();scope='Known deterministic joint H2 development only; no structural holdouts, stochastic faults, open model generation, actual online rule mutation, shadow validation, new-method superiority, Minecraft, RGB or full release.'} | ConvertTo-Json -Depth 8
