# Construction/leakage evidence only; not a held-out learner score or innovation.
param([string]$Proof = 'runs/structural-corpus-cross-platform-verification-v1.json')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
function Read-Json([string]$relative) { Get-Content -Raw -LiteralPath (Join-Path $repoRoot $relative) | ConvertFrom-Json }
function Assert-Hash([string]$relative, [string]$expected) {
    if ([System.IO.Path]::IsPathRooted($relative) -or $relative.Contains('..')) { throw 'Unsafe artifact path' }
    if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $relative)).Hash.ToLower() -cne $expected) { throw "Source/artifact drift: $relative" }
}
function Assert-SameJson($left, $right, [string]$label) {
    if (($left | ConvertTo-Json -Depth 100 -Compress) -cne ($right | ConvertTo-Json -Depth 100 -Compress)) { throw "JSON mismatch: $label" }
}
$proofRecord = Read-Json $Proof
if ($proofRecord.schema -cne 'structural-corpus-cross-platform-verification-v1') { throw 'Unexpected proof schema' }
if ($null -ne $proofRecord.complete_verifier) { Assert-Hash 'scripts/verify_structural_corpus.ps1' $proofRecord.complete_verifier.sha256 }
$registration = Read-Json 'runs/structural-corpus-source-registration-v1.json'
Assert-Hash 'runs/structural-corpus-source-registration-v1.json' $proofRecord.registration_sha256
if (@($registration.source_sha256.PSObject.Properties).Count -ne 9) { throw 'Unexpected source inventory' }
foreach ($entry in $registration.source_sha256.PSObject.Properties) { Assert-Hash $entry.Name $entry.Value }
$actualFiles = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'research/structural_holdouts') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object { 'research/structural_holdouts/' + $_.Name }) + @('scripts/build_structural_corpus.py')
if (@(Compare-Object $actualFiles @($registration.source_sha256.PSObject.Properties.Name)).Count -ne 0) { throw 'Corpus source inventory changed' }
Assert-Hash $registration.main_registration $registration.main_registration_sha256
$main = Read-Json $registration.main_registration
foreach ($field in @('source_sha256','test_sha256')) { foreach ($entry in $main.$field.PSObject.Properties) { Assert-Hash $entry.Name $entry.Value } }
$historical = Read-Json 'runs/holdout-identity-source-registration-v1.json'
foreach ($entry in $historical.source_sha256.PSObject.Properties) { Assert-Hash ($registration.historical_identity_archive + '/' + $entry.Name) $entry.Value }
$identities = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proofRecord.$platform
    if ($record.exit_code -ne 0 -or $record.cases -ne 128 -or [string]::IsNullOrWhiteSpace($record.terminal_chunk)) { throw 'Focused process lacks passing terminal receipt' }
    Assert-Hash $record.xml $record.sha256
    [xml]$xml = Get-Content -Raw -LiteralPath (Join-Path $repoRoot $record.xml)
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 128 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw 'Unexpected focused result' }
    $identities[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if (@($identities[$platform] | Select-Object -Unique).Count -ne 128) { throw 'Duplicate test ID' }
}
if (@(Compare-Object $identities.windows $identities.linux).Count -ne 0 -or $proofRecord.case_id_sets_equal -ne $true) { throw 'Focused platform identities differ' }
$expectedRuns = @('runs/structural-corpus-development-windows-v1','runs/structural-corpus-development-windows-v2','runs/structural-corpus-development-wsl-v1')
if (@($proofRecord.runs).Count -ne 3 -or @(Compare-Object $expectedRuns @($proofRecord.runs.path)).Count -ne 0) { throw 'Expected two Windows and one Linux construction' }
$baseline = Read-Json ($expectedRuns[0] + '/manifest.json')
foreach ($record in $proofRecord.runs) {
    if ($record.exit_code -ne 0 -or [string]::IsNullOrWhiteSpace($record.terminal_chunk)) { throw 'Corpus construction lacks passing terminal receipt' }
    Assert-Hash ($record.path + '/manifest.json') $record.manifest_sha256
    Assert-Hash ($record.path + '/report.json') $record.report_sha256
    $manifest = Read-Json ($record.path + '/manifest.json')
    if ($manifest.schema -cne 'new-structure-corpus-manifest-v1' -or @($manifest.sha256.PSObject.Properties).Count -ne 3075 -or @(Get-ChildItem -LiteralPath (Join-Path $repoRoot $record.path) -Recurse -File).Count -ne 3077 -or $manifest.nondeterministic_artifacts.Count -ne 1 -or $manifest.nondeterministic_artifacts[0] -cne 'timing.json') { throw 'Corpus artifact coverage changed' }
    Assert-SameJson $manifest $baseline 'three deterministic construction manifests'
    foreach ($entry in $manifest.sha256.PSObject.Properties) { Assert-Hash ($record.path + '/' + $entry.Name) $entry.Value }
    $runtime = Read-Json ($record.path + '/registration.json')
    foreach ($field in @('seed','base_count','renamings','boundary','salt')) { if ($runtime.config.$field -cne $registration.build_protocol.$field) { throw 'Frozen constructor inputs differ' } }
    Assert-SameJson $runtime.config.states $registration.build_protocol.states 'frozen state-count schedule'
    if (@($runtime.source_sha256.PSObject.Properties).Count -ne 9) { throw 'Constructor source dependency inventory changed' }
    foreach ($entry in $runtime.source_sha256.PSObject.Properties) { if ($registration.source_sha256.($entry.Name) -cne $entry.Value) { throw 'Constructor source dependency digest differs' } }
    $report = Read-Json ($record.path + '/report.json')
    if ($report.fixture_count -ne 1024 -or $report.independent_group_count -ne 60) { throw 'Independent sample count changed' }
    Assert-SameJson $report.summary $record.summary 'terminal/report summary'
    $timing = Read-Json ($record.path + '/timing.json')
    if ($timing.schema -cne 'new-structure-construction-timing-v1' -or @($timing.rows).Count -ne 1024) { throw 'Missing construction time records' }
    foreach ($row in $timing.rows) { if ($null -eq $row.construction_seconds -or $row.construction_seconds -lt 0) { throw 'Invalid construction time' } }
}
# Each byte is identical. Recompute all unique-grid grouping and denominators once.
$root = $expectedRuns[0]
$runtime = Read-Json ($root + '/registration.json')
$partition = Read-Json ($root + '/partitions.json')
$report = Read-Json ($root + '/report.json')
if (@($runtime.fixture_inputs).Count -ne 1024 -or @($partition.assignments.PSObject.Properties).Count -ne 1024 -or @($partition.groups.PSObject.Properties).Count -ne 1024 -or @($partition.groups.PSObject.Properties.Value | Select-Object -Unique).Count -ne 60) { throw 'Incomplete/duplicate dataset groups' }
$owners = @{}; $bodies = @{}; $rows = [System.Collections.Generic.List[object]]::new()
foreach ($entry in $partition.assignments.PSObject.Properties) {
    $id = $entry.Name; $split = $entry.Value
    if ($split -cnotin @('development','tuning','evaluation')) { throw 'Invalid split' }
    $metadata = Read-Json ($root + '/' + $id + '/metadata.json')
    if ($metadata.fixture_id -cne $id -or $metadata.inputs.seed -ne 0 -or $metadata.inputs.base_index -lt 0 -or $metadata.inputs.base_index -ge 64 -or $metadata.inputs.renaming_index -cnotin @(0,1)) { throw 'Fixture ID/input mismatch' }
    foreach ($key in $metadata.diagnostics.leakage_keys) {
        if ($owners.ContainsKey($key) -and $owners[$key] -cne $split) { throw 'Cross-split prior/target/pair leakage' }
        $owners[$key] = $split
    }
    foreach ($bodyEntry in $metadata.diagnostics.prior_identity_bodies.PSObject.Properties) {
        if ($bodies.ContainsKey($bodyEntry.Name) -and $bodies[$bodyEntry.Name] -cne $bodyEntry.Value) { throw 'Same digest/different canonical body' }
        $bodies[$bodyEntry.Name] = $bodyEntry.Value
    }
    foreach ($identity in $metadata.diagnostics.identities.PSObject.Properties) {
        $key = $identity.Name.Replace('target-','') + ':' + $identity.Value.sha256
        if ($bodies.ContainsKey($key) -and $bodies[$key] -cne $identity.Value.canonical) { throw 'Target/pair canonical body collision' }
        $bodies[$key] = $identity.Value.canonical
    }
    $rows.Add([pscustomobject]@{ split=$split; group=$partition.groups.$id; metadata=$metadata })
}
foreach ($split in @('development','tuning','evaluation')) {
    $selected = @($rows | Where-Object split -CEQ $split)
    $measured = [ordered]@{
        base_graphs = @($selected | ForEach-Object { $_.metadata.inputs.base_index } | Select-Object -Unique).Count
        behavior_change_at_H = @($selected | Where-Object { $_.metadata.diagnostics.behavior_change_at_H }).Count
        behavior_change_unbounded = @($selected | Where-Object { $_.metadata.diagnostics.behavior_change_unbounded }).Count
        cases = [ordered]@{}
        fixtures = $selected.Count
        independent_groups = @($selected.group | Select-Object -Unique).Count
        outside_declared_mechanism_edit_family = @($selected | Where-Object { -not $_.metadata.diagnostics.within_declared_mechanism_edit_family }).Count
        target_exact_in_public_family = @($selected | Where-Object { $_.metadata.diagnostics.target_exact_in_public_family }).Count
    }
    foreach ($case in @('composed','controller_change','insert','invisible_insert','out_of_family','output','redirect','unchanged')) { $measured.cases[$case] = @($selected | Where-Object { $_.metadata.inputs.case -ceq $case }).Count }
    Assert-SameJson $measured $report.summary.$split 'recomputed independent groups/negative conditions'
}
Write-Output 'Verified construction DEVELOPMENT only: Windows/Linux 128 cases, three identical 3075-file datasets, 1024 fixtures but 60 independent leakage groups (27 development/15 tuning/18 evaluation), prior-target-pair separation and full denominators. NOT 1024 independent worlds, formal method/statistical results, unknown controls/noise, H4, innovation or Minecraft/RGB/release completion.'
