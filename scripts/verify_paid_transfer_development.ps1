$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Read-RunJson([string]$relative) {
    Get-Content -LiteralPath (Join-Path $repoRoot $relative) -Raw | ConvertFrom-Json
}
function Assert-RunHash([string]$relative, [string]$expected) {
    $sourceRoot = if ($relative.StartsWith('src/minecraft_scientist/') -or $relative.StartsWith('tests/')) { Join-Path $repoRoot 'runs/apartness-main-source-v1' } else { $repoRoot }
    if ((Get-FileHash -LiteralPath (Join-Path $sourceRoot $relative) -Algorithm SHA256).Hash.ToLower() -cne $expected) { throw "Source/artifact drift: $relative" }
}
function Assert-SameJson($left, $right, [string]$label) {
    if (($left | ConvertTo-Json -Depth 100 -Compress) -cne ($right | ConvertTo-Json -Depth 100 -Compress)) { throw "JSON mismatch: $label" }
}
$proof = Read-RunJson 'runs/paid-transfer-cross-platform-verification-v1.json'
$registration = Read-RunJson 'runs/paid-transfer-source-registration-v1.json'
Assert-RunHash 'runs/paid-transfer-source-registration-v1.json' $proof.registration_sha256
if (@($registration.source_sha256.PSObject.Properties).Count -ne 20) { throw 'Unexpected registered source set' }
foreach ($entry in $registration.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
$registeredPrototype = @($registration.source_sha256.PSObject.Properties.Name | Where-Object { $_.StartsWith('research/adaptive_automata/') } | Sort-Object)
$actualPrototype = @(Get-ChildItem (Join-Path $repoRoot 'research/adaptive_automata') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object { 'research/adaptive_automata/' + $_.Name } | Sort-Object)
if (@(Compare-Object $registeredPrototype $actualPrototype).Count -ne 0) { throw 'Prototype file set changed' }
Assert-RunHash $registration.main_registration $registration.main_registration_sha256
$mainFreeze = Read-RunJson $registration.main_registration
foreach ($entry in $mainFreeze.source_sha256.PSObject.Properties) { Assert-RunHash ('src/minecraft_scientist/' + $entry.Name) $entry.Value }
foreach ($entry in $mainFreeze.test_sha256.PSObject.Properties) { Assert-RunHash ('tests/' + $entry.Name) $entry.Value }

$names = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proof.$platform
    if ($record.exit_code -ne 0) { throw 'Focused pytest lacks a passing terminal record' }
    Assert-RunHash $record.xml $record.sha256
    [xml]$xml = Get-Content -LiteralPath (Join-Path $repoRoot $record.xml) -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 235 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw 'Unexpected focused test outcome' }
    $names[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if (@($names[$platform] | Select-Object -Unique).Count -ne 235) { throw 'Duplicate test identity' }
}
if (@(Compare-Object $names.windows $names.linux).Count -ne 0) { throw 'Platform test identities differ' }

$baselineManifest = Read-RunJson ($proof.runs[0].path + '/manifest.json')
if (@($proof.runs).Count -ne 3) { throw 'Expected two Windows runs and one Linux run' }
foreach ($run in $proof.runs) {
    if ($run.exit_code -ne 0) { throw 'Development run lacks a passing terminal record' }
    Assert-RunHash ($run.path + '/manifest.json') $run.manifest_sha256
    Assert-RunHash ($run.path + '/report.json') $run.report_sha256
    $manifest = Read-RunJson ($run.path + '/manifest.json')
    if (@($manifest.sha256.PSObject.Properties).Count -ne 450 -or @(Get-ChildItem (Join-Path $repoRoot $run.path) -Recurse -File).Count -ne 452) { throw 'Artifact file set differs' }
    Assert-SameJson $baselineManifest $manifest 'cross-run manifests'
    if ($manifest.nondeterministic_artifacts.Count -ne 1 -or $manifest.nondeterministic_artifacts[0] -cne 'timing.json') { throw 'Unexpected determinism exclusions' }
    foreach ($entry in $baselineManifest.sha256.PSObject.Properties) { Assert-RunHash ($run.path + '/' + $entry.Name) $entry.Value }
    $runRegistration = Read-RunJson ($run.path + '/registration.json')
    if (@($runRegistration.source_sha256.PSObject.Properties).Count -ne 150) { throw 'Unexpected complete development-source count' }
    foreach ($entry in $runRegistration.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
    $report = Read-RunJson ($run.path + '/report.json')
    if ($report.all_campaign_audits_valid -ne $true -or @($report.rows).Count -ne 64 -or @($report.pairs).Count -ne 32) { throw 'Incomplete paired grid' }
    foreach ($method in @('from_scratch','adaptive')) {
        $rows = @($report.rows | Where-Object method -CEQ $method)
        $correct = @($rows | Where-Object { $_.evaluation.both_contexts_correct_at_H -eq $true }).Count
        $incorrect = @($rows | Where-Object { $_.evaluation.both_contexts_correct_at_H -eq $false }).Count
        $unresolved = @($rows | Where-Object { $null -eq $_.evaluation.both_contexts_correct_at_H }).Count
        if ($rows.Count -ne 32 -or $correct -ne 19 -or $incorrect -ne 5 -or $unresolved -ne 8) { throw 'Outcome denominator/negative results changed' }
        foreach ($row in $rows) {
            $casePath = $run.path + '/seed-' + $row.seed + '/' + $row.family + '/states-' + $row.private_new_states + '/' + $method
            $audit = Read-RunJson ($casePath + '/audit.json')
            $campaign = Read-RunJson ($casePath + '/run.json')
            $config = Read-RunJson ($casePath + '/config.json')
            if (-not $audit.valid -or -not $audit.all_contexts_public_valid -or -not $audit.private_truth_read_attempted) { throw 'Missing complete public campaign audit' }
            if (-not $audit.old_audit.valid -or -not $audit.old_audit.raw_public_ledger_valid -or -not $audit.old_audit.public_replay_valid -or -not $audit.old_audit.private_truth_opened) { throw 'Old-context audit missing' }
            if ($null -ne $audit.new_audit -and (-not $audit.new_audit.valid -or -not $audit.new_audit.raw_public_ledger_valid -or -not $audit.new_audit.public_replay_valid -or -not $audit.new_audit.private_truth_opened)) { throw 'New-context audit missing' }
            Assert-SameJson $row.evaluation $audit.evaluation 'report/private scoring'
            if ($campaign.unannounced_online_mutation_claimed -ne $false -or $campaign.unique_causal_structure_claimed -ne $false -or $campaign.all_allowed_queries_certified -ne $false) { throw 'Unsupported identification/online claim' }
            if ($row.spent -ne $row.old_spent + $row.new_spent -or $row.spent -ne $campaign.spent -or $row.old_spent -le 0 -or $row.old_spent -ne $campaign.old.run.spent -or $row.spent -gt $config.protocol.budget -or $row.status -cne $campaign.status -or $row.stop_reason -cne $campaign.stop_reason) { throw 'Combined paid cost/status disagreement' }
        }
    }
    $both = 0; $adaptiveCheaper = 0; $scratchCheaper = 0; $tied = 0
    foreach ($pair in $report.pairs) {
        $pairPath = $run.path + '/seed-' + $pair.seed + '/' + $pair.family + '/states-' + $pair.private_new_states
        $scratch = Read-RunJson ($pairPath + '/from_scratch/run.json')
        $adaptive = Read-RunJson ($pairPath + '/adaptive/run.json')
        Assert-SameJson $scratch.old $adaptive.old 'same paid old evidence'
        Assert-SameJson $scratch.public_old_theory $adaptive.public_old_theory 'same learned old theory'
        $difference = $adaptive.spent - $scratch.spent
        if ($pair.same_paid_old_history -ne $true -or $pair.adaptive_minus_scratch_total_cost -ne $difference) { throw 'Pair prior/cost disagreement' }
        if ($pair.both_methods_both_contexts_correct) {
            if ($pair.from_scratch_outcome -ne $true -or $pair.adaptive_outcome -ne $true) { throw 'False success conditioning' }
            $both++
            if ($difference -lt 0) { $adaptiveCheaper++ } elseif ($difference -gt 0) { $scratchCheaper++ } else { $tied++ }
        }
    }
    if ($both -ne 19 -or $adaptiveCheaper -ne 7 -or $scratchCheaper -ne 12 -or $tied -ne 0) { throw 'Success-conditioned cost comparison changed' }
    if ($report.summary.both_methods_correct_pairs -ne $both -or $report.summary.adaptive_cheaper_among_both_correct -ne $adaptiveCheaper -or $report.summary.scratch_cheaper_among_both_correct -ne $scratchCheaper -or $report.summary.tied_cost_among_both_correct -ne $tied) { throw 'Summary/pair disagreement' }
}
Write-Output 'Verified HISTORICAL archived-main paid explicit-context development: Windows/Linux 235 focused cases each; three identical 450-artifact runs, 64 valid campaign audits. Each method: 19 correct/5 wrong/8 unresolved. Both-correct cost wins: adaptive 7, scratch 12. NOT stable transfer advantage, hidden online mutation, causal identification, learned symbol transfer, formal structural holdout, default/full regression, novelty or full release.'
