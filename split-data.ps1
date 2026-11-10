param([int]$Seed = 42)
$ErrorActionPreference = 'Stop'
$inventory = Join-Path $PSScriptRoot 'data/audit/image_inventory.csv'
$output = Join-Path $PSScriptRoot 'data/splits'
if (Test-Path -LiteralPath $output) { throw 'data/splits already exists; preserve it or choose a new output before rerunning.' }
$all = @(Import-Csv -LiteralPath $inventory)
$rows = @($all | Where-Object label -in 'NORMAL','PNEUMONIA','TUBERCULOSIS' | Sort-Object path)
if ($rows.Count -eq 0) { throw 'No supported images in inventory.' }
$parents = @{}
function Find-Root([string]$key) {
    $current = $key
    while ($parents[$current] -ne $current) { $current = $parents[$current] }
    while ($parents[$key] -ne $key) {
        $next = $parents[$key]; $parents[$key] = $current; $key = $next
    }
    return $current
}
function Join-Group([string]$a, [string]$b) {
    $ra = Find-Root $a
    $rb = Find-Root $b
    if ($ra -ne $rb) {
        if ([string]::CompareOrdinal($ra, $rb) -lt 0) { $parents[$rb] = $ra }
        else { $parents[$ra] = $rb }
    }
}
$hashOwners = @{}
$hintOwners = @{}
foreach ($row in $rows) {
    if (-not (Test-Path -LiteralPath (Join-Path "$PSScriptRoot/data/data" $row.path))) {
        throw "Missing image: $($row.path)"
    }
    $parents[$row.path] = $row.path
    if ($hashOwners.ContainsKey($row.sha256)) { Join-Group $row.path $hashOwners[$row.sha256] }
    else { $hashOwners[$row.sha256] = $row.path }
    if ($row.patient_hint) {
        $hint = "$($row.source):$($row.patient_hint)"
        if ($hintOwners.ContainsKey($hint)) { Join-Group $row.path $hintOwners[$hint] }
        else { $hintOwners[$hint] = $row.path }
    }
}
$groups = @{}
foreach ($row in $rows) {
    $key = Find-Root $row.path
    if (-not $groups.ContainsKey($key)) { $groups[$key] = [System.Collections.Generic.List[object]]::new() }
    $groups[$key].Add($row)
}
$sha = [System.Security.Cryptography.SHA256]::Create()
function Stable-Hash([string]$value) {
    return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($value)))).Replace('-','').ToLowerInvariant()
}
$units = @(foreach ($key in ($groups.Keys | Sort-Object)) {
    $members = $groups[$key]
    if (@($members.label | Sort-Object -Unique).Count -ne 1) { throw "Conflicting labels in group $key" }
    [PSCustomObject]@{
        id=(Stable-Hash $key); order=(Stable-Hash "$Seed|$key"); members=$members
        stratum=($members[0].label + ':' + (($members.source | Sort-Object -Unique) -join '+'))
        count=$members.Count
    }
})
$assignments = [System.Collections.Generic.List[object]]::new()
$splitNames = @('train','val','test')
$fractions = @{train=0.70; val=0.15; test=0.15}
foreach ($stratum in ($units | Group-Object stratum | Sort-Object Name)) {
    $total = ($stratum.Group | Measure-Object count -Sum).Sum
    $used = @{train=0; val=0; test=0}
    # Large groups first; deterministic seeded hashes break ties.
    foreach ($unit in ($stratum.Group | Sort-Object @{Expression='count';Descending=$true},order)) {
        $chosen = 'train'
        $best = [double]::PositiveInfinity
        foreach ($candidate in $splitNames) {
            $cost = 0.0
            foreach ($s in $splitNames) {
                $n = $used[$s]
                if ($s -eq $candidate) { $n += $unit.count }
                $target = $total * $fractions[$s]
                $cost += [math]::Pow($n - $target, 2) / [math]::Max($target, 1)
            }
            if ($cost -lt $best) { $best = $cost; $chosen = $candidate }
        }
        $used[$chosen] += $unit.count
        foreach ($row in $unit.members) {
            $assignments.Add([PSCustomObject]@{
                path=('../data/' + $row.path); label=$row.label.ToLowerInvariant()
                patient_id=$(if ($row.patient_hint) { "$($row.source):$($row.patient_hint)" } else { '' })
                patient_id_status=$(if ($row.patient_hint) { 'filename_inferred_unverified' } else { 'unavailable' })
                source=$row.source; split=$chosen; group_id=$unit.id
                original_split=$row.split; original_name=$row.name; sha256=$row.sha256
            })
        }
    }
}
foreach ($field in @('sha256','group_id','patient_id')) {
    $bad = @($assignments | Where-Object { $_.$field } | Group-Object $field | Where-Object {
        @($_.Group.split | Sort-Object -Unique).Count -gt 1
    })
    if ($bad.Count) { throw "Overlap remains for $field" }
}
foreach ($s in $splitNames) {
    foreach ($label in @('normal','pneumonia','tuberculosis')) {
        if (-not @($assignments | Where-Object { $_.split -eq $s -and $_.label -eq $label }).Count) {
            throw "Missing class $label in $s"
        }
    }
}
if ($assignments.Count -ne $rows.Count) { throw 'Assignment count mismatch' }
New-Item -ItemType Directory -Path $output | Out-Null
$assignments | Sort-Object split,label,path | Export-Csv -LiteralPath "$output/manifest.csv" -NoTypeInformation
foreach ($s in $splitNames) {
    $assignments | Where-Object split -eq $s | Sort-Object label,path | Export-Csv -LiteralPath "$output/$s.csv" -NoTypeInformation
}
$summary = [ordered]@{
    seed=$Seed; included_images=$assignments.Count
    excluded_unknown=@($all | Where-Object label -eq 'UNKNOWN').Count
    groups=$units.Count; target_fractions=$fractions
    counts=@($assignments | Group-Object split,label | Select-Object Name,Count)
    source_counts=@($assignments | Group-Object split,source,label | Select-Object Name,Count)
    unavailable_patient_id=@($assignments | Where-Object patient_id_status -eq 'unavailable').Count
    validation='No cross-split SHA256, inferred patient ID, or connected-group overlap. All three classes present in every split.'
    limitations='Patient IDs inferred from filenames are unverified; missing IDs stay blank. Hashes reused from prior audit; image bytes assumed unchanged. Near-duplicates and unnamed patient overlap remain unchecked.'
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$output/summary.json" -Encoding UTF8
$sha.Dispose()
$summary | ConvertTo-Json -Depth 6
