# Current-source, shared-lifecycle DEVELOPMENT evidence only. No novelty/holdout claim.
param([string]$Proof = 'runs/matched-h3-cross-platform-verification-v1.json')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
function Read-RunJson([string]$relative) {
    Get-Content -LiteralPath (Join-Path $repoRoot $relative) -Raw | ConvertFrom-Json
}
function Assert-RunHash([string]$relative, [string]$expected) {
    if ([System.IO.Path]::IsPathRooted($relative) -or $relative.Contains('..')) { throw 'Unsafe evidence path' }
    if ((Get-FileHash -LiteralPath (Join-Path $repoRoot $relative) -Algorithm SHA256).Hash.ToLower() -cne $expected) { throw "Source/artifact drift: $relative" }
}
function Assert-SameJson($left, $right, [string]$label) {
    if (($left | ConvertTo-Json -Depth 100 -Compress) -cne ($right | ConvertTo-Json -Depth 100 -Compress)) { throw "JSON mismatch: $label" }
}
function Measure-Rows($rows) {
    [ordered]@{
        actual_total_cost = [long]($rows | Measure-Object -Property spent -Sum).Sum
        bounded_no_anomaly = @($rows | Where-Object status -CEQ 'bounded_monitoring_completed_no_anomaly').Count
        detected_post_boundary = @($rows | Where-Object { $_.evaluation.post_boundary_anomaly_detected }).Count
        false_or_early_anomaly = @($rows | Where-Object { $_.evaluation.anomaly_without_in_H_change_or_before_boundary }).Count
        finite_replacements = @($rows | Where-Object status -CEQ 'revision_finitely_validated').Count
        new_correct_at_H = @($rows | Where-Object { $_.evaluation.new_behavior_correct_at_H -eq $true }).Count
        new_wrong_at_H = @($rows | Where-Object { $_.evaluation.new_behavior_correct_at_H -eq $false }).Count
        old_correct_at_H = @($rows | Where-Object { $_.evaluation.old_behavior_correct_at_H -eq $true }).Count
        runs = @($rows).Count
        unresolved = @($rows | Where-Object status -CEQ 'unresolved').Count
    }
}
$proofRecord = Read-RunJson $Proof
if ($proofRecord.schema -cne 'matched-h3-cross-platform-verification-v1') { throw 'Unexpected evidence schema' }
if ($null -ne $proofRecord.complete_verifier) { Assert-RunHash 'scripts/verify_matched_h3_development.ps1' $proofRecord.complete_verifier.sha256 }
$registration = Read-RunJson 'runs/matched-h3-source-registration-v1.json'
Assert-RunHash 'runs/matched-h3-source-registration-v1.json' $proofRecord.registration_sha256
if (@($registration.source_sha256.PSObject.Properties).Count -ne 152 -or @($registration.test_sha256.PSObject.Properties).Count -ne 132) { throw 'Current phase source/test inventory changed' }
foreach ($field in @('source_sha256','test_sha256')) {
    foreach ($entry in $registration.$field.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
}
$sourcePaths = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'src/minecraft_scientist') -Recurse -File -Filter '*.py' | ForEach-Object { 'src/minecraft_scientist/' + $_.FullName.Substring((Join-Path $repoRoot 'src/minecraft_scientist').Length + 1).Replace('\','/') } | Sort-Object)
$testPaths = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'tests') -Recurse -File -Filter '*.py' | ForEach-Object { 'tests/' + $_.FullName.Substring((Join-Path $repoRoot 'tests').Length + 1).Replace('\','/') } | Sort-Object)
$onlinePaths = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'research/online_automata') -File | Where-Object { $_.Extension -in '.py','.md' } | ForEach-Object { 'research/online_automata/' + $_.Name } | Sort-Object)
if ($sourcePaths.Count -ne 133 -or $testPaths.Count -ne 132 -or $onlinePaths.Count -ne 17) { throw 'Unregistered runtime/test files' }
if (@(Compare-Object $sourcePaths @($registration.source_sha256.PSObject.Properties.Name | Where-Object { $_.StartsWith('src/minecraft_scientist/') } | Sort-Object)).Count -ne 0 -or @(Compare-Object $testPaths @($registration.test_sha256.PSObject.Properties.Name | Sort-Object)).Count -ne 0 -or @(Compare-Object $onlinePaths @($registration.source_sha256.PSObject.Properties.Name | Where-Object { $_.StartsWith('research/online_automata/') } | Sort-Object)).Count -ne 0) { throw 'Source file-set drift' }
Assert-RunHash $registration.upstream_registration $registration.upstream_registration_sha256
$upstream = Read-RunJson $registration.upstream_registration
if (@($upstream.source_sha256.PSObject.Properties).Count -ne 20) { throw 'Upstream inventory changed' }
foreach ($entry in $upstream.source_sha256.PSObject.Properties) { Assert-RunHash $entry.Name $entry.Value }
$expectedRuntime = @{}
foreach ($entry in @($registration.source_sha256.PSObject.Properties) + @($upstream.source_sha256.PSObject.Properties)) { $expectedRuntime[$entry.Name] = $entry.Value }
if ($expectedRuntime.Count -ne 172) { throw 'Unexpected runtime dependency set' }
$identities = @{}
foreach ($platform in @('windows','linux')) {
    $record = $proofRecord.$platform
    if ($record.exit_code -ne 0 -or $record.cases -ne 560 -or [string]::IsNullOrWhiteSpace($record.terminal_chunk)) { throw 'Focused process not recorded as terminal/pass' }
    Assert-RunHash $record.xml $record.sha256
    [xml]$xml = Get-Content -LiteralPath (Join-Path $repoRoot $record.xml) -Raw
    $cases = @($xml.SelectNodes('//testcase'))
    if ($cases.Count -ne 560 -or $xml.SelectNodes('//failure | //error | //skipped').Count -ne 0) { throw 'Unexpected focused result' }
    $identities[$platform] = @($cases | ForEach-Object { $_.classname + '::' + $_.name } | Sort-Object)
    if (@($identities[$platform] | Select-Object -Unique).Count -ne 560) { throw 'Duplicate test identity' }
}
if (@(Compare-Object $identities.windows $identities.linux).Count -ne 0 -or $proofRecord.case_id_sets_equal -ne $true) { throw 'Platform case identities differ' }
$runPaths = @('runs/matched-h3-development-windows-v1','runs/matched-h3-development-windows-v2','runs/matched-h3-development-wsl-v1')
if (@($proofRecord.runs).Count -ne 3 -or @(Compare-Object $runPaths @($proofRecord.runs.path)).Count -ne 0) { throw 'Expected two Windows/one Linux development runs' }
$baselineManifest = Read-RunJson ($runPaths[0] + '/manifest.json')
foreach ($record in $proofRecord.runs) {
    if ($record.exit_code -ne 0 -or [string]::IsNullOrWhiteSpace($record.terminal_chunk)) { throw 'Development process not recorded as terminal/pass' }
    Assert-RunHash ($record.path + '/manifest.json') $record.manifest_sha256
    Assert-RunHash ($record.path + '/report.json') $record.report_sha256
    $manifest = Read-RunJson ($record.path + '/manifest.json')
    if ($manifest.schema -cne 'matched-h3-manifest-v1' -or @($manifest.sha256.PSObject.Properties).Count -ne 1010 -or @(Get-ChildItem -LiteralPath (Join-Path $repoRoot $record.path) -Recurse -File).Count -ne 1012) { throw 'Missing/extra artifacts' }
    if (@($manifest.nondeterministic_artifacts).Count -ne 1 -or $manifest.nondeterministic_artifacts[0] -cne 'timing.json') { throw 'Unexpected determinism exclusion' }
    Assert-SameJson $baselineManifest $manifest 'three deterministic manifests'
    foreach ($entry in $manifest.sha256.PSObject.Properties) { Assert-RunHash ($record.path + '/' + $entry.Name) $entry.Value }
    $runtime = Read-RunJson ($record.path + '/registration.json')
    if ($runtime.schema -cne 'matched-h3-development-registration-v1' -or @($runtime.source_sha256.PSObject.Properties).Count -ne 172) { throw 'Unbound development runtime' }
    foreach ($entry in $runtime.source_sha256.PSObject.Properties) {
        if ($expectedRuntime[$entry.Name] -cne $entry.Value) { throw 'Runtime source not in registered dependency union' }
        Assert-RunHash $entry.Name $entry.Value
    }
    $report = Read-RunJson ($record.path + '/report.json')
    Assert-SameJson $report.registration $runtime 'report registration'
    if ($record.summary.worlds -ne $report.summary.worlds -or $record.summary.campaigns -ne $report.summary.campaigns -or @($record.summary.methods.PSObject.Properties).Count -ne 4 -or @($record.summary.by_boundary.PSObject.Properties).Count -ne 3) { throw 'Terminal proof/report summary inventory differs' }
    foreach ($summaryMethod in @('from_scratch','adaptive','structural_witness','structural_ig')) {
        Assert-SameJson $report.summary.methods.$summaryMethod $record.summary.methods.$summaryMethod 'terminal proof/method summary'
        foreach ($summaryBoundary in @('6','25','100')) { Assert-SameJson $report.summary.by_boundary.$summaryBoundary.$summaryMethod $record.summary.by_boundary.$summaryBoundary.$summaryMethod 'terminal proof/boundary summary' }
    }
    $timing = Read-RunJson ($record.path + '/timing.json')
    if ($timing.schema -cne 'matched-h3-timing-v1' -or @($timing.rows).Count -ne 168) { throw 'Incomplete computation-cost record' }
    foreach ($row in $timing.rows) {
        foreach ($stage in @('complete_learning_monitoring_and_validation','complete_audit_and_private_score')) {
            foreach ($clock in @('wall_seconds','cpu_seconds')) { if ($row.$stage.$clock -lt 0 -or $null -eq $row.$stage.$clock) { throw 'Missing/invalid computation time' } }
        }
    }
}
# The manifest covers every byte of all three runs; examine the unique identical grid once.
$runRoot = $runPaths[0]
$report = Read-RunJson ($runRoot + '/report.json')
$methods = @('from_scratch','adaptive','structural_witness','structural_ig')
Assert-SameJson @($report.registration.methods) $methods 'method inventory'
if ($report.schema -cne 'matched-h3-development-report-v1' -or @($report.rows).Count -ne 168 -or @($report.comparisons).Count -ne 126 -or @($report.registration.private_grid).Count -ne 42 -or $report.all_public_first_audits_valid -ne $true -or $report.summary.worlds -ne 42 -or $report.summary.campaigns -ne 168) { throw 'Incomplete development grid' }
if ($report.registration.protocol.budget -ne 600 -or $report.registration.protocol.old_budget -ne 250 -or $report.registration.protocol.horizon -ne 5 -or $report.registration.repair_policy.validation -cne 'random-wp-plus-coverage-v1' -or $report.registration.repair_policy.shadow_count -ne 3) { throw 'Changed scientific protocol' }
$seen = @{}
$preludes = @{}
$rowIndex = @{}
foreach ($row in $report.rows) {
    $key = [string]$row.seed + '/' + $row.case + '/' + $row.private_boundary
    $fullKey = $key + '/' + $row.method
    if ($seen.ContainsKey($fullKey) -or $row.method -cnotin $methods) { throw 'Duplicate/invalid campaign' }
    $seen[$fullKey] = $true
    $rowIndex[$fullKey] = $row
    $path = $runRoot + '/seed-' + $row.seed + '/' + $row.case + '/boundary-' + $row.private_boundary + '/' + $row.method
    $campaign = Read-RunJson ($path + '/run.json')
    $config = Read-RunJson ($path + '/config.json')
    $audit = Read-RunJson ($path + '/audit.json')
    if ($campaign.schema -cne 'paid-online-relearning-v3' -or $config.schema -cne 'online-relearning-config-v3' -or $config.method -cne $row.method -or $campaign.method -cne $row.method) { throw 'Campaign identity/schema mismatch' }
    if ($audit.valid -ne $true -or $audit.all_public_valid -ne $true -or $audit.private_truth_read_attempted -ne $true -or $audit.private_truth_opened -ne $true) { throw 'Missing public-first audit' }
    Assert-SameJson $row.evaluation $audit.evaluation 'row/independent evaluation'
    Assert-SameJson $config.protocol $report.registration.protocol 'campaign protocol'
    Assert-SameJson $config.repair_policy $report.registration.repair_policy 'common final policy'
    if ($campaign.mechanism_change_warranted -ne $false -or $campaign.unique_causal_structure_claimed -ne $false -or $campaign.all_allowed_queries_certified -ne $false -or $campaign.mechanism_attribution_status -cne 'unresolved') { throw 'Unsupported causal/completeness claim' }
    $prelude = [ordered]@{ old=$campaign.old; public_old_theory=$campaign.public_old_theory; monitoring=$campaign.monitoring; anomaly=$campaign.anomaly }
    if ($preludes.ContainsKey($key)) { Assert-SameJson $prelude $preludes[$key] 'identical paid prelude/monitor/anomaly' } else { $preludes[$key] = $prelude }
    $monitorSpent = 0
    foreach ($batch in $campaign.monitoring) { foreach ($query in $batch.queries) { $monitorSpent += $query.spent } }
    $repairSpent = if ($null -eq $campaign.new) { 0 } else { $campaign.new.run.spent }
    if ($row.spent -ne $campaign.spent -or $row.spent -ne $row.old_spent + $row.monitor_spent + $row.repair_spent -or $row.old_spent -ne $campaign.old.run.spent -or $row.monitor_spent -ne $monitorSpent -or $row.repair_spent -ne $repairSpent -or $row.spent -gt $config.protocol.budget -or $row.status -cne $campaign.status -or $row.stop_reason -cne $campaign.stop_reason) { throw 'Paid total/status mismatch' }
    $old = $campaign.old.run
    if ($old.status -cne 'provisional_validation_passed' -or $old.spent -ne $old.discovery.spent + $old.validation.spent -or $old.validation.status -cne 'passed' -or @($old.validation.runs).Count -ne 3 -or @($old.validation_query_ids).Count -ne 3 -or $old.validation.plan.all_public_actions_covered -ne $true -or $row.evaluation.old_behavior_correct_at_H -ne $true) { throw 'Old paid/validated theory missing' }
    $groups = @($campaign.old)
    foreach ($batch in $campaign.monitoring) { $groups += @($batch.queries) }
    if ($null -ne $campaign.new) { $groups += @($campaign.new) }
    $resets = 0; $steps = 0; $failures = 0; $rawSpent = 0
    $expectedEvents = [System.Collections.Generic.List[object]]::new()
    foreach ($group in $groups) {
        foreach ($control in $group.raw_controls) {
            if ($control.kind -ceq 'reset') { $resets++; $rawSpent++ }
            elseif ($control.kind -ceq 'step') {
                $steps++
                $rawSpent += $config.costs.($control.transition.executed_action.target_id)
                if (-not $control.transition.intervention_compliance -or [bool]$control.transition.controller_failure -or -not $control.transition.event.success) { $failures++ }
            } else { throw 'Unknown raw control type' }
        }
        $epoch = if ($null -ne $group.epoch) { $group.epoch } else { $group.config.protocol.epoch }
        $events = if ($null -ne $group.run) { $group.run.events } else { $group.events }
        foreach ($event in $events) { $expectedEvents.Add([pscustomobject][ordered]@{ epoch=$epoch; event=$event }) }
    }
    if ($row.reset_count -ne $resets -or $row.atomic_action_count -ne $steps -or $row.known_failed_dispatch_count -ne $failures -or $campaign.spent -ne $rawSpent) { throw 'Reset/action/failure/raw-cost ledger mismatch' }
    $eventLines = @(Get-Content -LiteralPath (Join-Path $repoRoot ($path + '/events.jsonl')) | ForEach-Object { $_ | ConvertFrom-Json })
    if ($eventLines.Count -ne $expectedEvents.Count) { throw 'Exported events incomplete' }
    # Re-serialized dictionary order differs between runtime JSON and JSONL; compare fields separately.
    for ($i = 0; $i -lt $eventLines.Count; $i++) {
        if ($eventLines[$i].epoch -cne $expectedEvents[$i].epoch) { throw 'Exported event epoch mismatch' }
        Assert-SameJson $eventLines[$i].event $expectedEvents[$i].event 'exported event receipt'
    }
    $theoryLines = @(Get-Content -LiteralPath (Join-Path $repoRoot ($path + '/theories.jsonl')) | ForEach-Object { $_ | ConvertFrom-Json })
    Assert-SameJson $theoryLines @($campaign.theory_events) 'exported theory history'
    $snapshots = @{}
    foreach ($event in $campaign.theory_events) {
        if ($event.created_by -cne 'shared-paid-scientific-lifecycle') { throw 'Wrong theory provenance' }
        if ($snapshots.ContainsKey([string]$event.version)) { Assert-SameJson $event.ir $snapshots[[string]$event.version] 'immutable theory content' } else { $snapshots[[string]$event.version] = $event.ir }
        if (($event.version -eq 1 -and $null -ne $event.parent_version) -or ($event.version -eq 2 -and $event.parent_version -ne 1)) { throw 'Wrong theory parent' }
    }
    foreach ($assignment in $campaign.validation_assignments) {
        if (@($assignment.discovery_ids | Where-Object { $assignment.validation_ids -ccontains $_ }).Count -ne 0 -or @($assignment.validation_ids | Select-Object -Unique).Count -ne @($assignment.validation_ids).Count) { throw 'Discovery/final evidence overlap or duplicate' }
    }
    if ($null -ne $campaign.new) {
        $new = $campaign.new.run
        if ($new.schema -cne 'matched-repair-run-v1' -or $campaign.new.config.protocol.budget -ne $config.protocol.budget - $row.old_spent - $row.monitor_spent) { throw 'Wrong repair budget/lifecycle' }
        Assert-SameJson $campaign.new.config.policy $config.repair_policy 'repair final policy'
        if ($row.status -ceq 'revision_finitely_validated') {
            if ($new.status -cne 'provisional_matched_validation_passed' -or @($new.validation_query_ids).Count -ne 11 -or @($new.random_validation.queries).Count -ne 8 -or $new.random_validation.status -cne 'passed' -or @($new.coverage_validation.runs).Count -ne 3 -or $new.coverage_validation.status -cne 'passed' -or @($new.validation_counterexample_ids).Count -ne 0) { throw 'Incomplete common independent validation' }
            if (@($campaign.theory_events | Where-Object { $_.version -eq 2 -and $_.status -ceq 'accepted' }).Count -ne 1 -or $campaign.theory_events[-1].status -cne 'superseded' -or $campaign.theory_events[-1].version -ne 1) { throw 'Missing accepted/superseded history' }
        } elseif ($null -ne $new.proposed_ir) {
            $last = $campaign.theory_events[-1]
            $expectedStatus = if (@($new.validation_counterexample_ids).Count -gt 0) { 'rejected' } else { 'provisional' }
            if ($last.version -ne 2 -or $last.status -cne $expectedStatus -or @($campaign.theory_events | Where-Object { $_.status -ceq 'superseded' }).Count -ne 0) { throw 'Failed candidate history discarded or misclassified' }
        }
    }
}
foreach ($grid in $report.registration.private_grid) {
    foreach ($method in $methods) { if (-not $seen.ContainsKey(([string]$grid[0] + '/' + $grid[1] + '/' + $grid[2] + '/' + $method))) { throw 'Private grid/campaign mismatch' } }
}
foreach ($method in $methods) {
    Assert-SameJson (Measure-Rows @($report.rows | Where-Object method -CEQ $method)) $report.summary.methods.$method 'all-case method denominators/costs'
    foreach ($boundary in @('6','25','100')) {
        Assert-SameJson (Measure-Rows @($report.rows | Where-Object { $_.method -ceq $method -and [string]$_.private_boundary -ceq $boundary })) $report.summary.by_boundary.$boundary.$method 'boundary denominators/costs'
    }
}
$comparisonSeen = @{}
foreach ($comparison in $report.comparisons) {
    $key = [string]$comparison.seed + '/' + $comparison.case + '/' + $comparison.private_boundary
    $comparisonKey = $key + '/' + $comparison.method
    if ($comparisonSeen.ContainsKey($comparisonKey) -or $comparison.method -cnotin $methods[1..3]) { throw 'Duplicate/invalid comparison' }
    $comparisonSeen[$comparisonKey] = $true
    $reference = $rowIndex[$key + '/from_scratch']; $row = $rowIndex[$comparisonKey]
    $correct = $reference.evaluation.new_behavior_correct_at_H -eq $true -and $row.evaluation.new_behavior_correct_at_H -eq $true
    if ($comparison.reference -cne 'from_scratch' -or $comparison.same_paid_prelude_and_monitoring -ne $true -or $comparison.same_final_protocol -ne $true -or $comparison.both_repaired_correct -ne $correct -or $comparison.cost_difference_is_success_conditioned -ne $correct -or $comparison.method_minus_scratch_total_cost -ne $row.spent - $reference.spent) { throw 'Conditioned paired cost mismatch' }
}
Write-Output 'Verified current shared-lifecycle DEVELOPMENT: Windows/Linux 560 cases each, three identical 1010-artifact grids, 42 worlds/168 campaigns, paid prelude/monitor/final validation, complete costs and version history. NOT formal structural holdouts, stable/statistical superiority, full strong-baseline matrix, unknown stochastic controls, H4, causal uniqueness, innovation or Minecraft/RGB/release completion.'
