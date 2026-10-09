$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$archiveRoot = Join-Path $repoRoot 'runs/adaptive-audit-source-v1'

function Read-RunJson([string]$relative) {
    Get-Content -LiteralPath (Join-Path $repoRoot $relative) -Raw | ConvertFrom-Json
}
function Assert-RunHash([string]$relative, [string]$expected) {
    if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $relative) -Algorithm SHA256).Hash.ToLower() -cne $expected) { throw "Source/artifact drift: $relative" }
}
function Assert-HistoricalSourceHash([string]$relative, [string]$expected) {
    if ($relative.StartsWith('research/adaptive_automata/') -or $relative -ceq 'scripts/run_adaptive_audit_development.py') {
        $path = Join-Path $archiveRoot $relative
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -cne $expected) { throw "Historical source drift: $relative" }
    } else { Assert-RunHash $relative $expected }
}
$proof = Read-RunJson 'runs/adaptive-audit-cross-platform-verification-v1.json'
$registration = Read-RunJson 'runs/adaptive-audit-source-registration-v1.json'
Assert-RunHash 'runs/adaptive-audit-source-registration-v1.json' $proof.registration_sha256
foreach ($entry in $registration.source_sha256.PSObject.Properties) { Assert-HistoricalSourceHash $entry.Name $entry.Value }
$registeredPrototype = @($registration.source_sha256.PSObject.Properties.Name | Where-Object { $_.StartsWith('research/adaptive_automata/') } | Sort-Object)
$actualPrototype = @(Get-ChildItem (Join-Path $archiveRoot 'research/adaptive_automata') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object { 'research/adaptive_automata/' + $_.Name } | Sort-Object)
if (@(Compare-Object $registeredPrototype $actualPrototype).Count -ne 0) { throw 'Prototype file set changed' }
Assert-RunHash $registration.main_registration $registration.main_registration_sha256
$mainFreeze = Read-RunJson $registration.main_registration
foreach ($entry in $mainFreeze.source_sha256.PSObject.Properties) { Assert-RunHash ('src/minecraft_scientist/' + $entry.Name) $entry.Value }
foreach ($entry in $mainFreeze.test_sha256.PSObject.Properties) { Assert-RunHash ('tests/' + $entry.Name) $entry.Value }

$names = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proof.$platform
    if ($record.exit_code -ne 0) { throw 'Focused pytest has no passing terminal record' }
    Assert-RunHash $record.xml $record.sha256
    [xml]$xml = Get-Content -LiteralPath (Join-Path $repoRoot $record.xml) -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 182 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw 'Unexpected focused test outcome' }
    $names[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if (@($names[$platform] | Select-Object -Unique).Count -ne 182) { throw 'Duplicate test identity' }
}
if (@(Compare-Object $names.windows $names.linux).Count -ne 0) { throw 'Platform test-case sets differ' }

$baselineManifest = Read-RunJson ($proof.runs[0].path + '/manifest.json')
foreach ($run in $proof.runs) {
    if ($run.exit_code -ne 0) { throw 'Development run has no passing terminal record' }
    Assert-RunHash ($run.path + '/manifest.json') $run.manifest_sha256
    Assert-RunHash ($run.path + '/report.json') $run.report_sha256
    $manifest = Read-RunJson ($run.path + '/manifest.json')
    if (@($manifest.sha256.PSObject.Properties).Count -ne 162 -or @(Get-ChildItem (Join-Path $repoRoot $run.path) -Recurse -File).Count -ne 164) { throw 'Artifact file set differs' }
    foreach ($entry in $baselineManifest.sha256.PSObject.Properties) {
        if ($manifest.sha256.($entry.Name) -cne $entry.Value) { throw 'Cross-run manifest differs' }
        Assert-RunHash ($run.path + '/' + $entry.Name) $entry.Value
    }
    $runRegistration = Read-RunJson ($run.path + '/registration.json')
    if (@($runRegistration.source_sha256.PSObject.Properties).Count -ne 144) { throw 'Unexpected complete development-source count' }
    foreach ($entry in $runRegistration.source_sha256.PSObject.Properties) { Assert-HistoricalSourceHash $entry.Name $entry.Value }
    $report = Read-RunJson ($run.path + '/report.json')
    $correct = @($report.rows | Where-Object { $_.evaluation.provisional_behavior_correct_at_H -eq $true }).Count
    $incorrect = @($report.rows | Where-Object { $_.evaluation.provisional_behavior_correct_at_H -eq $false }).Count
    $unresolved = @($report.rows | Where-Object status -eq 'unresolved').Count
    if (@($report.rows).Count -ne 32 -or $correct -ne 19 -or $incorrect -ne 5 -or $unresolved -ne 8) { throw 'Outcomes/negative results changed' }
    foreach ($row in $report.rows) {
        $casePath = $run.path + '/seed-' + $row.seed + '/' + $row.family + '/states-' + $row.private_states
        $audit = Read-RunJson ($casePath + '/audit.json')
        if (-not $audit.valid -or -not $audit.raw_public_ledger_valid -or -not $audit.public_replay_valid -or -not $audit.private_truth_opened) { throw 'A development case lacks full stationary audit' }
        if ($audit.evaluation.provisional_behavior_correct_at_H -ne $row.evaluation.provisional_behavior_correct_at_H) { throw 'Report disagrees with private evaluation' }
    }
}
Write-Output 'Verified HISTORICAL stationary adaptive audit phase from runs/adaptive-audit-source-v1: Windows/Linux 182 focused cases each; three identical 162-artifact runs, 32 valid audits, 19 correct/5 wrong/8 unresolved. NOT evidence for the changed current learner/auditor. Provided-prior development, not fair comparison, online transfer, structural holdout, default/full regression or full release.'
