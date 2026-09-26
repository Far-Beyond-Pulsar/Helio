$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $root
Set-Location -LiteralPath $workspace
if (Get-Process cargo,rustc -ErrorAction SilentlyContinue) {
    throw 'Finish compilation before running this trial.'
}
$cases = @(
    @('trial-control-720p', 'target/voxel-flight/voxel_flight-before-batch.exe', '1280', '720', 'native'),
    @('trial-1024-720p', 'target/release/examples/voxel_flight.exe', '1280', '720', 'native'),
    @('trial-control-1080p-quality', 'target/voxel-flight/voxel_flight-before-batch.exe', '1920', '1080', 'quality'),
    @('trial-1024-1080p-quality', 'target/release/examples/voxel_flight.exe', '1920', '1080', 'quality')
)
foreach ($case in $cases) {
    $name = $case[0]
    $binary = $case[1]
    Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 3 |
        ForEach-Object { $_.CounterSamples | Select-Object Timestamp,InstanceName,CookedValue } |
        Export-Csv -NoTypeInformation -LiteralPath "$PSScriptRoot/$name-gpu-before.csv"
    $started = Get-Date -Format o
    & $binary "target/voxel-flight/$name" $case[2] $case[3] $case[4] *> "$PSScriptRoot/$name.log"
    $flightExit = $LASTEXITCODE
    [pscustomobject]@{ name=$name; started=$started; finished=(Get-Date -Format o); exit=$flightExit; binarySha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash } |
        ConvertTo-Json | Set-Content -LiteralPath "$PSScriptRoot/$name-run.json"
    Write-Output "$name exit=$flightExit"
    if ($flightExit -ne 0) { exit $flightExit }
}
