$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Read-ArtifactJson([string]$relative) {
    Get-Content -LiteralPath (Join-Path $repoRoot $relative) -Raw | ConvertFrom-Json
}

function Assert-Hash([string]$relative, [string]$expected) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $repoRoot $relative) -Algorithm SHA256).Hash.ToLower()
    if ($actual -cne $expected) { throw "SHA256 mismatch: $relative" }
}

$proof = Read-ArtifactJson 'runs/adaptive-rules-cross-platform-verification-v1.json'
$registration = Read-ArtifactJson 'runs/adaptive-rules-source-registration-v1.json'
Assert-Hash 'runs/adaptive-rules-source-registration-v1.json' $proof.registration_sha256
foreach ($entry in $registration.source_sha256.PSObject.Properties) {
    Assert-Hash ('research/adaptive_automata/' + $entry.Name) $entry.Value
}
$actualPrototypeFiles = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'research/adaptive_automata') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object Name | Sort-Object)
$registeredPrototypeFiles = @($registration.source_sha256.PSObject.Properties.Name | Sort-Object)
if (@(Compare-Object $actualPrototypeFiles $registeredPrototypeFiles).Count -ne 0) { throw 'Prototype file set changed' }
Assert-Hash $registration.main_registration $registration.main_registration_sha256
$mainFreeze = Read-ArtifactJson $registration.main_registration
foreach ($entry in $mainFreeze.source_sha256.PSObject.Properties) { Assert-Hash ('src/minecraft_scientist/' + $entry.Name) $entry.Value }
foreach ($entry in $mainFreeze.test_sha256.PSObject.Properties) { Assert-Hash ('tests/' + $entry.Name) $entry.Value }

$platformCases = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proof.$platform
    Assert-Hash $record.xml $record.sha256
    if ($record.exit_code -ne 0) { throw "No passing terminal record: $platform" }
    [xml]$xml = Get-Content -LiteralPath (Join-Path $repoRoot $record.xml) -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 107 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw "Unexpected actual JUnit counts: $platform" }
    $platformCases[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
}
if (@(Compare-Object $platformCases.windows $platformCases.linux).Count -ne 0) { throw 'Platform test-case sets differ' }

Assert-Hash 'scripts/export_adaptive_fixture_evidence.py' $proof.fixture_exports.exporter_sha256
foreach ($run in $proof.fixture_exports.runs) {
    if ($run.exit_code -ne 0) { throw 'Fixture export did not exit successfully' }
    Assert-Hash ($run.path + '/manifest.json') $run.manifest_sha256
    Assert-Hash ($run.path + '/report.json') $run.report_sha256
    $manifest = Read-ArtifactJson ($run.path + '/manifest.json')
    $fixtureRegistration = Read-ArtifactJson ($run.path + '/registration.json')
    if (@($manifest.sha256.PSObject.Properties).Count -ne 26) { throw 'Fixture manifest count changed' }
    if (@(Get-ChildItem -LiteralPath (Join-Path $repoRoot $run.path) -Recurse -File).Count -ne 27) { throw 'Fixture file set changed' }
    foreach ($entry in $manifest.sha256.PSObject.Properties) { Assert-Hash ($run.path + '/' + $entry.Name) $entry.Value }
    foreach ($entry in $fixtureRegistration.source_sha256.PSObject.Properties) { Assert-Hash $entry.Name $entry.Value }
    $report = Read-ArtifactJson ($run.path + '/report.json')
    if (@($report.rows).Count -ne 8 -or @($report.rows | Where-Object { $_.fixture_evaluator_behavior_correct -ne $true }).Count -ne 0) { throw 'Declared fixture evaluator outcomes changed' }
}
Write-Output 'Verified 107 focused cases per platform; 26 deterministic artifacts plus identical manifest in each of 3 fixture exports. Plan checks only, not whole-run audit, fair baseline, structural holdout or full-project completion.'
