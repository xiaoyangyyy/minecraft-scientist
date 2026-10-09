param([string]$Proof = 'C:\Users\Administrator.DESKTOP-HG1CIML\.codex\research-run-cache\structural-campaign-development-cross-platform-verification-v1.json')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function ReadJson($Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
function Hash($Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function EqualJson($Left, $Right, $Label) { if (($Left | ConvertTo-Json -Depth 100 -Compress) -cne ($Right | ConvertTo-Json -Depth 100 -Compress)) { throw ('JSON mismatch: ' + $Label) } }
function Sources($Path, $ExpectedHash, $Count) {
    if ((Hash $Path) -cne $ExpectedHash) { throw 'Source registration hash mismatch' }
    $registration = ReadJson $Path
    if (@($registration.source_sha256.PSObject.Properties).Count -ne $Count) { throw 'Source count mismatch' }
    foreach ($entry in $registration.source_sha256.PSObject.Properties) { if ((Hash $entry.Name) -cne $entry.Value) { throw ('Source drift: ' + $entry.Name) } }
}
function CaseIds($Record, $Count) {
    if ($Record.exit_code -ne 0 -or $Record.cases -ne $Count -or (Hash $Record.xml) -cne $Record.sha256) { throw 'JUnit receipt/hash mismatch' }
    [xml]$xml = Get-Content -LiteralPath $Record.xml -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    $ids = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if ($cases.Count -ne $Count -or ($ids | Sort-Object -Unique).Count -ne $Count -or @($xml.SelectNodes('//failure|//error|//skipped')).Count -ne 0) { throw 'JUnit cases/status mismatch' }
    return ,$ids
}
$proofRecord = ReadJson $Proof
if ($proofRecord.schema -ne 'structural-campaign-development-cross-platform-verification-v1') { throw 'Wrong proof schema' }
Sources $proofRecord.source_registration $proofRecord.source_registration_sha256 188
Sources $proofRecord.failure_diagnostics_tests.source_registration $proofRecord.failure_diagnostics_tests.source_registration_sha256 15
$win = CaseIds $proofRecord.windows 36
$linux = CaseIds $proofRecord.linux 36
if (@(Compare-Object $win $linux).Count -gt 0) { throw '36 case sets differ' }
$win = CaseIds $proofRecord.failure_diagnostics_tests.windows 75
$linux = CaseIds $proofRecord.failure_diagnostics_tests.linux 75
if (@(Compare-Object $win $linux).Count -gt 0) { throw '75 case sets differ' }
$main = ReadJson 'runs/matched-h3-source-registration-v1.json'
foreach ($entry in $main.test_sha256.PSObject.Properties) { if ((Hash $entry.Name) -cne $entry.Value) { throw 'Main default test drift' } }
if ($proofRecord.runs.Count -ne 3 -or $proofRecord.development_run.full_audit.exit_code -ne 0) { throw 'Missing terminal/audit evidence' }
$baseline = $proofRecord.runs[0].path
$baselineManifest = ReadJson (Join-Path $baseline 'manifest.json')
$baselineFiles = @(Get-ChildItem -LiteralPath $baseline -Recurse -File | ForEach-Object { $_.FullName.Substring($baseline.Length + 1).Replace('\','/') } | Sort-Object)
foreach ($run in $proofRecord.runs) {
    if ($run.exit_code -ne 0 -or (Hash (Join-Path $run.path 'manifest.json')) -cne $run.manifest_sha256 -or (Hash (Join-Path $run.path 'report.json')) -cne $run.report_sha256) { throw 'Run not terminal or receipt mismatch' }
    $manifest = ReadJson (Join-Path $run.path 'manifest.json')
    EqualJson $manifest $baselineManifest 'cross-platform manifest'
    if (@($manifest.sha256.PSObject.Properties).Count -ne 386) { throw 'Wrong artifact count' }
    $files = @(Get-ChildItem -LiteralPath $run.path -Recurse -File | ForEach-Object { $_.FullName.Substring($run.path.Length + 1).Replace('\','/') } | Sort-Object)
    if ($files.Count -ne 388 -or @(Compare-Object $baselineFiles $files).Count -gt 0) { throw 'Missing/extra run artifacts' }
    foreach ($entry in $manifest.sha256.PSObject.Properties) { if ((Hash (Join-Path $run.path $entry.Name)) -cne $entry.Value) { throw ('Run bytes drift: ' + $entry.Name) } }
    $timing = ReadJson (Join-Path $run.path 'timing.json')
    if ($timing.rows.Count -ne 64) { throw 'Timing denominator mismatch' }
    foreach ($row in $timing.rows) { foreach ($phase in @($row.complete_learning,$row.complete_audit)) { foreach ($duration in @($phase.wall_seconds,$phase.cpu_seconds)) { if ($null -eq $duration -or $duration -lt 0 -or [double]::IsNaN($duration) -or [double]::IsInfinity($duration)) { throw 'Invalid compute timing' } } } }
}
$report = ReadJson (Join-Path $baseline 'report.json')
$registration = $report.registration
$fixtures = @($registration.selection.fixture_ids)
$groups = @($registration.selection.groups.PSObject.Properties.Value | Sort-Object -Unique)
if ($registration.selection.split -ne 'development' -or $fixtures.Count -ne $proofRecord.expected_fixtures -or $groups.Count -ne $proofRecord.expected_independent_groups -or $report.rows.Count -ne $proofRecord.expected_campaigns) { throw 'False independent group/fixture/campaign count' }
if ($fixtures.Count -ne 16 -or $groups.Count -ne 1 -or $report.rows.Count -ne 64) { throw 'Probe scope drift' }
if ($report.summary.fixtures -ne 16 -or $report.summary.independent_development_groups -ne 1 -or $report.summary.campaigns -ne 64) { throw 'Report denominator mismatch' }
$keys = @($report.rows | ForEach-Object { $_.fixture_id + ':' + $_.method })
if (($keys | Sort-Object -Unique).Count -ne 64) { throw 'Duplicate campaign' }
foreach ($row in $report.rows) {
    $directory = Join-Path $baseline ($row.fixture_id + '/' + $row.method)
    $run = ReadJson (Join-Path $directory 'run.json')
    $audit = ReadJson (Join-Path $directory 'audit.json')
    if (-not $audit.valid -or $row.spent -ne $run.spent -or $run.spent -gt $run.budget -or $row.status -cne $run.status -or $row.independent_group -cne $registration.selection.groups.($row.fixture_id)) { throw 'Cost/status/group ledger mismatch' }
    EqualJson $row.raw_score $audit.raw_score 'raw score'
    EqualJson $row.evaluation $audit.evaluation 'evaluation'
}
foreach ($method in $registration.methods) {
    $rows = @($report.rows | Where-Object { $_.method -ceq $method })
    $summary = $report.summary.methods.$method
    if ($rows.Count -ne 16 -or $summary.runs -ne 16 -or $summary.old_correct_at_H -ne @($rows | Where-Object {$_.raw_score.old_correct_at_H -eq $true}).Count -or $summary.new_correct_at_H -ne @($rows | Where-Object {$_.raw_score.promoted_correct_at_H -eq $true}).Count -or $summary.new_wrong_at_H -ne @($rows | Where-Object {$null -ne $_.raw_score.promoted_correct_at_H -and $_.raw_score.promoted_correct_at_H -eq $false}).Count -or $summary.unresolved -ne @($rows | Where-Object {$_.status -ceq 'unresolved'}).Count -or $summary.bounded_no_anomaly -ne @($rows | Where-Object {$_.status -ceq 'bounded_monitoring_completed_no_anomaly'}).Count -or $summary.actual_total_cost -ne ($rows | Measure-Object -Property spent -Sum).Sum) { throw 'Report method denominator mismatch' }
}
Write-Output 'Verified DEVELOPMENT: 36 + 75 cases each platform in their registered scopes; three terminal 64-campaign runs, 386 identical deterministic files each, ONE independent development group. Wrong promotions retained. NOT formal holdout/statistical novelty, prospective oracle ability, unknown noise/control, H4 or Minecraft/RGB completion.'
