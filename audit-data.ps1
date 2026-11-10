param([string]$Dataset = "$PSScriptRoot/data/data", [string]$Report = "$PSScriptRoot/data/audit")
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Dataset).Path
$rows = @(Import-Csv -LiteralPath (Join-Path $root 'dataset_splits_with_metadata.csv'))
New-Item -ItemType Directory -Path $Report -Force | Out-Null
$records = @()
$missing = @()
$expected = @{}
foreach ($row in $rows) {
    $relative = "$($row.split)/$($row.label)/$($row.name)"
    $expected[$relative] = $true
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $missing += $relative
        continue
    }
    # Filename hints only: these patterns are not verified patient metadata.
    $hint = ''
    if ($row.name -match '^(person\d+)_') { $hint = $Matches[1] }
    elseif ($row.name -match '^(NORMAL2-IM-\d+|IM-\d+)-') { $hint = $Matches[1] }
    elseif ($row.name -match '^((?:CHNCXR|MCUCXR)_\d+)_') { $hint = $Matches[1] }
    $records += [PSCustomObject]@{
        path=$relative; split=$row.split; label=$row.label; source=$row.source
        name=$row.name; patient_hint=$hint
        sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    }
}
$extras = @(Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object {
    $_.Extension.ToLowerInvariant() -in '.jpg','.jpeg','.png'
} | ForEach-Object {
    $relative = $_.FullName.Substring($root.Length + 1).Replace('\','/')
    if (-not $expected.ContainsKey($relative)) { $relative }
})
$hashGroups = @($records | Group-Object sha256 | Where-Object Count -gt 1)
$crossHashes = @($hashGroups | Where-Object { @($_.Group.split | Sort-Object -Unique).Count -gt 1 })
$nameGroups = @($records | Group-Object name | Where-Object {
    $_.Count -gt 1 -and @($_.Group.split | Sort-Object -Unique).Count -gt 1
})
$hintGroups = @($records | Where-Object patient_hint | Group-Object source,patient_hint | Where-Object {
    @($_.Group.split | Sort-Object -Unique).Count -gt 1
})
$conflicts = @($hashGroups | Where-Object { @($_.Group.label | Sort-Object -Unique).Count -gt 1 })
$records | Export-Csv -LiteralPath (Join-Path $Report 'image_inventory.csv') -NoTypeInformation
$details = [ordered]@{
    exact_duplicates_across_splits=@($crossHashes | ForEach-Object { ,@($_.Group) })
    filenames_across_splits=@($nameGroups | ForEach-Object { ,@($_.Group) })
    possible_patients_across_splits=@($hintGroups | ForEach-Object { ,@($_.Group) })
    conflicting_labels=@($conflicts | ForEach-Object { ,@($_.Group) })
    missing_files=$missing; extra_images=$extras
}
$details | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Report 'overlaps.json') -Encoding UTF8
$summary = [ordered]@{
    metadata_rows=$rows.Count; images_checked=$records.Count
    missing_files=$missing.Count; extra_images=$extras.Count
    duplicate_hash_groups=$hashGroups.Count; cross_split_duplicate_groups=$crossHashes.Count
    cross_split_filename_groups=$nameGroups.Count
    cross_split_possible_patient_groups=$hintGroups.Count
    label_conflict_groups=$conflicts.Count
    images_with_patient_filename_hints=@($records | Where-Object patient_hint).Count
    counts=@($records | Group-Object split,label | Select-Object Name,Count)
    sources=@($records | Group-Object source,label | Select-Object Name,Count)
    limitations='SHA256 detects identical files only. Filename patient hints are unverified and scoped by source. Resized/recompressed images and patients with different naming patterns may escape detection. No patient independence is established.'
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Report 'summary.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
