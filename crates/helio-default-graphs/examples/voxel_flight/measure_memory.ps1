# Windows process-memory diagnostic. Run separately from performance timing.
# Counter samples include the whole process, graph and driver, not terrain alone.
# Dedicated/shared GPU usage includes cross-process shared allocations; do not
# sum different processes. See https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/
param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$Width = 1280,
    [int]$Height = 720,
    [ValidateSet('native', 'quality')][string]$Quality = 'native'
)
$ErrorActionPreference = 'Stop'
$taskBinary = (Resolve-Path -LiteralPath $Executable).Path
$taskDirectory = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $taskDirectory) { throw "Use a new output directory: $taskDirectory" }
New-Item -ItemType Directory -Path $taskDirectory | Out-Null
$taskFlags = @{}
foreach ($taskFlag in @('HELIO_VOXEL_INLINE_ADMISSION', 'HELIO_VOXEL_ASYNC_ADMISSION', 'HELIO_VOXEL_RETARGETING',
    'HELIO_VOXEL_FLIGHT_PROFILE', 'HELIO_VOXEL_FLIGHT_RECORD', 'HELIO_VOXEL_FLIGHT_SUN')) {
    $taskFlags[$taskFlag] = [Environment]::GetEnvironmentVariable($taskFlag)
}
$taskManifest = [ordered]@{
    executable = $taskBinary
    sha256 = (Get-FileHash -LiteralPath $taskBinary -Algorithm SHA256).Hash.ToLowerInvariant()
    output = $taskDirectory; width = $Width; height = $Height; quality = $Quality
    environment = $taskFlags; start_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'whole process; sampled OS counters; not isolated terrain allocations or frame-time acceptance'
}
$taskProcess = Start-Process -FilePath $taskBinary -WindowStyle Hidden -PassThru `
    -WorkingDirectory (Get-Location).Path `
    -ArgumentList @(('"' + $taskDirectory + '"'), $Width, $Height, $Quality) `
    -RedirectStandardOutput (Join-Path $taskDirectory 'stdout.log') `
    -RedirectStandardError (Join-Path $taskDirectory 'stderr.log')
$taskManifest['pid'] = $taskProcess.Id
$taskManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskDirectory 'memory-run.json') -Encoding utf8
$taskRows = [System.Collections.Generic.List[object]]::new()
$taskGpuRows = [System.Collections.Generic.List[object]]::new()
$taskClock = [Diagnostics.Stopwatch]::StartNew()
$taskInstance = 'pid_' + $taskProcess.Id + '_*'
$taskCounterNames = @('Dedicated Usage', 'Shared Usage', 'Local Usage', 'Non Local Usage', 'Total Committed')
$taskCounters = @($taskCounterNames | ForEach-Object { '\GPU Process Memory(' + $taskInstance + ')\' + $_ })
while (-not $taskProcess.HasExited) {
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) { break }
    $taskRow = [ordered]@{
        sample_utc = [DateTime]::UtcNow.ToString('o'); elapsed_ms = $taskClock.Elapsed.TotalMilliseconds
        working_set_bytes = $taskProcess.WorkingSet64; private_bytes = $taskProcess.PrivateMemorySize64
        peak_working_set_bytes = $taskProcess.PeakWorkingSet64
    }
    # A not-yet-created or already-destroyed GPU context yields missing values,
    # not zero. Record the actual timestamps and sampling gaps in the output.
    $taskSamples = @((Get-Counter -Counter $taskCounters -ErrorAction SilentlyContinue).CounterSamples |
        Where-Object { $_.Status -in 0, 1 })
    foreach ($taskName in $taskCounterNames) {
        $taskMatches = @($taskSamples | Where-Object { $_.Path.EndsWith(('\' + $taskName), [StringComparison]::OrdinalIgnoreCase) })
        $taskRow[$taskName] = if ($taskMatches.Count -gt 0) {
            [long](($taskMatches | Measure-Object -Property CookedValue -Sum).Sum)
        } else { $null }
    }
    foreach ($taskSample in $taskSamples) {
        $taskGpuRows.Add([pscustomobject]@{
            sample_utc = $taskSample.Timestamp.ToUniversalTime().ToString('o')
            instance = $taskSample.InstanceName; counter = $taskSample.Path; bytes = [long]$taskSample.CookedValue
        })
    }
    $taskRows.Add([pscustomobject]$taskRow)
    Start-Sleep -Milliseconds 500
}
$taskProcess.WaitForExit()
$taskRows | Export-Csv -LiteralPath (Join-Path $taskDirectory 'process-memory.csv') -NoTypeInformation -Encoding utf8
$taskGpuRows | Export-Csv -LiteralPath (Join-Path $taskDirectory 'gpu-process-memory.csv') -NoTypeInformation -Encoding utf8
$taskManifest['exit_code'] = $taskProcess.ExitCode
$taskManifest['elapsed_ms'] = $taskClock.Elapsed.TotalMilliseconds
$taskManifest['samples'] = $taskRows.Count
$taskManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskDirectory 'memory-run.json') -Encoding utf8
Write-Output "VOXEL_MEMORY_RUN samples=$($taskRows.Count) exit=$($taskProcess.ExitCode) output=$taskDirectory"
exit $taskProcess.ExitCode
