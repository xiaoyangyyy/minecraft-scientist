$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
function Read-RunJson([string]$relative) { Get-Content -LiteralPath (Join-Path $repoRoot $relative) -Raw | ConvertFrom-Json }
function Assert-RunHash([string]$relative, [string]$expected) {
    if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $relative)).Hash.ToLower() -cne $expected) { throw "Source/artifact drift: $relative" }
}
function Assert-SameJson($left, $right, [string]$label) {
    if (($left | ConvertTo-Json -Depth 100 -Compress) -cne ($right | ConvertTo-Json -Depth 100 -Compress)) { throw "JSON mismatch: $label" }
}
$proof = Read-RunJson 'runs/shared-bootstrap-cross-platform-verification-v1.json'
$registration = Read-RunJson 'runs/shared-bootstrap-source-registration-v1.json'
Assert-RunHash 'runs/shared-bootstrap-source-registration-v1.json' $proof.registration_sha256
if (@($registration.source_sha256.PSObject.Properties).Count -ne 14) { throw 'Unexpected phase source set' }
foreach ($entry in $registration.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
$registered = @($registration.source_sha256.PSObject.Properties.Name | Where-Object { $_.StartsWith('research/online_automata/') } | Sort-Object)
$actual = @(Get-ChildItem (Join-Path $repoRoot 'research/online_automata') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object { 'research/online_automata/' + $_.Name } | Sort-Object)
if (@(Compare-Object $registered $actual).Count -ne 0) { throw 'Prototype source set changed' }
Assert-RunHash $registration.upstream_registration $registration.upstream_registration_sha256
$upstream = Read-RunJson $registration.upstream_registration
foreach ($entry in $upstream.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
Assert-RunHash $registration.main_registration $registration.main_registration_sha256
$main = Read-RunJson $registration.main_registration
foreach ($entry in $main.source_sha256.PSObject.Properties) { Assert-RunHash ('src/minecraft_scientist/' + $entry.Name) $entry.Value }
foreach ($entry in $main.test_sha256.PSObject.Properties) { Assert-RunHash ('tests/' + $entry.Name) $entry.Value }
$names = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proof.$platform
    if ($record.exit_code -ne 0) { throw 'No passing focused terminal record' }
    Assert-RunHash $record.xml $record.sha256
    [xml]$xml = Get-Content -LiteralPath (Join-Path $repoRoot $record.xml) -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 362 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw 'Unexpected focused outcome' }
    $names[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if (@($names[$platform] | Select-Object -Unique).Count -ne 362) { throw 'Duplicate test identity' }
}
if (@(Compare-Object $names.windows $names.linux).Count -ne 0) { throw 'Platform case identities differ' }
if (@($proof.runs).Count -ne 3) { throw 'Expected two Windows and one Linux runs' }
$baseline = Read-RunJson ($proof.runs[0].path + '/manifest.json')
foreach ($run in $proof.runs) {
    if ($run.exit_code -ne 0) { throw 'No passing development terminal record' }
    Assert-RunHash ($run.path + '/manifest.json') $run.manifest_sha256
    Assert-RunHash ($run.path + '/report.json') $run.report_sha256
    $manifest = Read-RunJson ($run.path + '/manifest.json')
    if (@($manifest.sha256.PSObject.Properties).Count -ne 650 -or @(Get-ChildItem (Join-Path $repoRoot $run.path) -Recurse -File).Count -ne 652) { throw 'Artifact file set changed' }
    Assert-SameJson $baseline $manifest 'cross-run manifests'
    if ($manifest.nondeterministic_artifacts.Count -ne 1 -or $manifest.nondeterministic_artifacts[0] -cne 'timing.json') { throw 'Unexpected determinism exclusions' }
    foreach ($entry in $manifest.sha256.PSObject.Properties) { Assert-RunHash ($run.path + '/' + $entry.Name) $entry.Value }
    $runRegistration = Read-RunJson ($run.path + '/registration.json')
    if (@($runRegistration.source_sha256.PSObject.Properties).Count -ne 164 -or $runRegistration.controlled_bootstrap_mode -cne 'known-family') { throw 'Unexpected runtime source/knowledge condition' }
    foreach ($entry in $runRegistration.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
    $report = Read-RunJson ($run.path + '/report.json')
    if (@($report.rows).Count -ne 108 -or @($report.pairs).Count -ne 54 -or -not $report.all_campaign_audits_valid) { throw 'Incomplete online grid' }
    Assert-SameJson $report.summary $run.summary 'proof/report summary'
    foreach ($row in $report.rows) {
        $path = $run.path + '/seed-' + $row.seed + '/' + $row.family + '/' + $row.case + '/boundary-' + $row.private_boundary + '/' + $row.method
        $campaign = Read-RunJson ($path + '/run.json')
        $config = Read-RunJson ($path + '/config.json')
        $audit = Read-RunJson ($path + '/audit.json')
        if (-not $audit.valid -or -not $audit.all_public_valid -or -not $audit.private_truth_read_attempted -or -not $audit.private_truth_opened) { throw 'Missing public-first audit' }
        Assert-SameJson $row.evaluation $audit.evaluation 'row/private behavior score'
        if ($campaign.mechanism_change_warranted -ne $false -or $campaign.unique_causal_structure_claimed -ne $false -or $campaign.all_allowed_queries_certified -ne $false -or $campaign.mechanism_attribution_status -cne 'unresolved') { throw 'Unsupported causal/complete claim' }
        $monitorSpent = 0
        foreach ($batch in $campaign.monitoring) { foreach ($query in $batch.queries) { $monitorSpent += $query.spent } }
        $repairSpent = if ($null -eq $campaign.new) { 0 } else { $campaign.new.run.spent }
        if ($row.spent -ne $row.old_spent + $row.monitor_spent + $row.repair_spent -or $row.spent -ne $campaign.spent -or $row.old_spent -ne $campaign.old.run.spent -or $row.monitor_spent -ne $monitorSpent -or $row.repair_spent -ne $repairSpent -or $row.spent -gt $config.protocol.budget -or $row.status -cne $campaign.status -or $row.stop_reason -cne $campaign.stop_reason) { throw 'Paid campaign cost/status disagreement' }
        if ($null -ne $campaign.new -and $campaign.new.config.protocol.budget -ne $config.protocol.budget - $row.old_spent - $monitorSpent) { throw 'Repair received extra budget' }
        if ($row.family -ceq 'controlled') {
            $old = $campaign.old.run
            if ($row.bootstrap_mode -cne 'known-family' -or $config.schema -cne 'online-relearning-config-v2' -or $old.schema -cne 'public-family-bootstrap-run-v1' -or $old.status -cne 'provisional_validation_passed' -or $row.evaluation.old_behavior_correct_at_H -ne $true) { throw 'Missing correct paid supplied-family bootstrap' }
            if ($old.spent -ne $old.discovery.spent + $old.validation.spent -or $old.validation.runs.Count -ne 3 -or $old.validation.status -cne 'passed' -or $old.validation.plan.all_public_actions_covered -ne $true -or $old.validation_query_ids.Count -ne 3) { throw 'Incomplete/unpaid old independent validation' }
            $discoveryWords = @($old.discovery.events | Where-Object kind -CEQ 'plan' | ForEach-Object { $_.actions | ConvertTo-Json -Compress })
            foreach ($query in $old.validation.plan.queries) { if ($discoveryWords -ccontains ($query.actions | ConvertTo-Json -Compress)) { throw 'Old discovery query reused as validation' } }
            $final = $campaign.validation_assignments[0]
            if (@($final.discovery_ids | Where-Object { $final.validation_ids -ccontains $_ }).Count -ne 0 -or $final.validation_ids.Count -ne 3) { throw 'Old evidence split mismatch' }
        } elseif ($row.bootstrap_mode -cne 'alphabet' -or $config.schema -cne 'online-relearning-config-v1') { throw 'Automata knowledge condition changed' }
    }
    foreach ($method in @('from_scratch','adaptive')) {
        $rows = @($report.rows | Where-Object method -CEQ $method)
        $measured = [ordered]@{
            anomaly_without_in_H_change_or_before_boundary = @($rows | Where-Object { $_.evaluation.anomaly_without_in_H_change_or_before_boundary }).Count
            bounded_monitoring_no_anomaly = @($rows | Where-Object status -CEQ 'bounded_monitoring_completed_no_anomaly').Count
            correct_replacements_at_H = @($rows | Where-Object { $_.evaluation.new_behavior_correct_at_H -eq $true }).Count
            finite_replacements = @($rows | Where-Object status -CEQ 'revision_finitely_validated').Count
            post_boundary_anomalies_detected = @($rows | Where-Object { $_.evaluation.post_boundary_anomaly_detected }).Count
            runs = $rows.Count
            unresolved = @($rows | Where-Object status -CEQ 'unresolved').Count
            wrong_old_models_at_H = @($rows | Where-Object { $_.evaluation.old_behavior_correct_at_H -eq $false }).Count
            wrong_replacements_at_H = @($rows | Where-Object { $_.evaluation.new_behavior_correct_at_H -eq $false }).Count
        }
        Assert-SameJson $measured $report.summary.methods.$method 'all-case denominators'
    }
    $both = 0; $adaptiveCheaper = 0; $scratchCheaper = 0; $tied = 0
    foreach ($pair in $report.pairs) {
        $path = $run.path + '/seed-' + $pair.seed + '/' + $pair.family + '/' + $pair.case + '/boundary-' + $pair.private_boundary
        $scratch = Read-RunJson ($path + '/from_scratch/run.json')
        $adaptive = Read-RunJson ($path + '/adaptive/run.json')
        foreach ($key in @('old','public_old_theory','monitoring','anomaly')) { Assert-SameJson $scratch.$key $adaptive.$key ('same paid ' + $key) }
        $left = Read-RunJson ($path + '/from_scratch/audit.json')
        $right = Read-RunJson ($path + '/adaptive/audit.json')
        $correct = $left.evaluation.new_behavior_correct_at_H -eq $true -and $right.evaluation.new_behavior_correct_at_H -eq $true
        $difference = $adaptive.spent - $scratch.spent
        if ($pair.same_paid_bootstrap_and_monitoring -ne $true -or $pair.both_methods_repaired_behavior_correct -ne $correct -or $pair.cost_difference_is_repair_success_conditioned -ne $correct -or $pair.adaptive_minus_scratch_total_cost -ne $difference) { throw 'Paired costs/conditioning disagree' }
        if ($correct) { $both++; if ($difference -lt 0) { $adaptiveCheaper++ } elseif ($difference -gt 0) { $scratchCheaper++ } else { $tied++ } }
    }
    if ($both -ne 16 -or $adaptiveCheaper -ne 0 -or $scratchCheaper -ne 16 -or $tied -ne 0 -or $report.summary.both_repaired_correct_pairs -ne $both -or $report.summary.adaptive_cheaper_among_both_repaired_correct -ne $adaptiveCheaper -or $report.summary.scratch_cheaper_among_both_repaired_correct -ne $scratchCheaper -or $report.summary.tied_cost_among_both_repaired_correct -ne $tied) { throw 'Repair-conditioned costs changed' }
}
Write-Output 'Verified shared public-family bootstrap phase: Windows/Linux 362 focused cases each; three identical 650-artifact/108-campaign runs; each method all 14 controlled old models correct at H. New controlled models: both 4 correct; scratch 5 wrong/adaptive 2 wrong plus 3 unresolved. Overall both-repair-correct cost wins adaptive 0/scratch 16. NOT full H3 fair matrix, causal attribution, structural holdout, stochastic-control solution, symbol transfer, stable advantage, novelty or Minecraft/RGB/release completion.'
