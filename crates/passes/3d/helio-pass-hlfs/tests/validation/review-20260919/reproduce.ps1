param(
    [ValidateSet('Quality','Timing','Regression','Capture')][string]$Mode = 'Quality',
    [string]$Output = 'target/rt-review-reproduction'
)
$ErrorActionPreference = 'Stop'
$root = git -C $PSScriptRoot rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'Run from a Helio checkout' }
Push-Location $root
try {
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    # Isolate each run from settings left in the calling shell.
    Get-ChildItem Env: | Where-Object Name -Like 'HLFS_RT_*' | ForEach-Object { Remove-Item -LiteralPath "Env:$($_.Name)" }
    if ($Mode -eq 'Capture') {
        cargo build -p examples --release --bin indoor_cathedral_hlfs
        if ($LASTEXITCODE -ne 0) { throw 'Example build failed' }
        $env:HLFS_RT='1'; $env:HLFS_PRESAMPLED='1'
        $env:HLFS_RESOLUTION='1440p'; $env:HLFS_CAPTURE_FRAMES='100'
        Remove-Item Env:HLFS_REFERENCE,Env:HLFS_SAMPLE_COUNT,Env:HLFS_GLASS_ALPHA -ErrorAction SilentlyContinue
        & target/release/indoor_cathedral_hlfs.exe --capture "$Output/cathedral"
        if ($LASTEXITCODE -ne 0) { throw 'Cathedral capture failed' }
        $env:HLFS_REFERENCE='1'
        & target/release/indoor_cathedral_hlfs.exe --capture "$Output/cathedral-reference"
        if ($LASTEXITCODE -ne 0) { throw 'Reference capture failed' }
        return
    }
    $artifacts = cargo test -p helio-pass-hlfs --release --test gpu_hlfs_rt --no-run --message-format=json
    if ($LASTEXITCODE -ne 0) { throw 'GPU test build failed' }
    $binary = $artifacts | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable -and $_.target.name -eq 'gpu_hlfs_rt' } |
        Select-Object -Last 1 -ExpandProperty executable
    if (!$binary) { throw 'Missing GPU test executable' }
    if ($Mode -eq 'Regression') {
        & $binary --ignored --skip benchmark --test-threads=1
        if ($LASTEXITCODE -ne 0) { throw 'RT regression failed' }
    } elseif ($Mode -eq 'Quality') {
        $env:HLFS_RT_QUALITY_SETTING='2:2'; $env:HLFS_RT_QUALITY_DISCOVERY='1'
        $env:HLFS_RT_QUALITY_PRESAMPLE='1'; $env:HLFS_RT_QUALITY_REACTIVE='1'
        $env:HLFS_RT_QUALITY_HELD_OUT='1'
        foreach ($seedSet in @('development','review')) {
            if ($seedSet -eq 'review') { $env:HLFS_RT_QUALITY_REVIEW_SEEDS='1' }
            foreach ($surface in @('rough','glossy')) {
                if ($surface -eq 'glossy') { $env:HLFS_RT_QUALITY_GLOSSY_MOTION='1' }
                else { Remove-Item Env:HLFS_RT_QUALITY_GLOSSY_MOTION -ErrorAction SilentlyContinue }
                $env:HLFS_RT_QUALITY_OUTPUT="$Output/$seedSet-$surface"
                & $binary benchmark_rt_quality_frontier --exact --ignored --nocapture
                if ($LASTEXITCODE -ne 0) { throw "Quality failed: $seedSet/$surface" }
            }
        }
        $env:HLFS_RT_QUALITY_REVIEW_SEEDS='1'
        $env:HLFS_RT_QUALITY_SWITCH_KEY='1'
        $env:HLFS_RT_QUALITY_OUTPUT="$Output/replacement-key-glossy"
        & $binary benchmark_rt_quality_frontier --exact --ignored --nocapture
        if ($LASTEXITCODE -ne 0) { throw 'Dominant emitter replacement quality failed' }
        Remove-Item Env:HLFS_RT_QUALITY_SWITCH_KEY
        $env:HLFS_RT_QUALITY_REVIEW_SEEDS='1'
        $env:HLFS_RT_QUALITY_OUTPUT="$Output/primary1440-direct-glossy"
        $env:HLFS_RT_QUALITY_RESOLUTION='1440p'; $env:HLFS_RT_QUALITY_SEED='701'
        $env:HLFS_RT_QUALITY_DIRECT_ONLY='1'
        & $binary benchmark_rt_quality_frontier --exact --ignored --nocapture
        if ($LASTEXITCODE -ne 0) { throw '1440p direct-only quality failed' }
    } else {
        $env:HLFS_RT_PROBE_SAMPLES='2'; $env:HLFS_RT_PROBE_CANDIDATES='2'
        $env:HLFS_RT_PROBE_DISCOVERY='1'; $env:HLFS_RT_PROBE_PRESAMPLE='1'
        $env:HLFS_RT_PROBE_REACTIVE='1'; $env:HLFS_RT_PROBE_DENSE_GEOMETRY='1'
        foreach ($case in @('1440p-run1','1440p-run2','1440p-run3','native','4k','glossy-key')) {
            $env:HLFS_RT_PROBE_FOCUS=if ($case -eq '4k') { '4k-reconstructed' } elseif ($case -eq 'native') { '1440p-native' } else { '1440p-reconstructed' }
            if ($case -eq 'glossy-key') { $env:HLFS_RT_PROBE_GLOSSY='1'; $env:HLFS_RT_PROBE_DOMINANT='1' }
            $env:HLFS_RT_PROBE_OUTPUT="$Output/$case"
            & $binary benchmark_rt_resolution_and_acceleration --exact --ignored --nocapture
            if ($LASTEXITCODE -ne 0) { throw "Timing failed: $case" }
            # Only the primary rough-surface workload has the 4ms/5ms gate.
            if ($case -like '1440p-*') {
                $csv = Get-ChildItem "$Output/$case/*.csv" | Select-Object -First 1
                $values = Import-Csv $csv.FullName | ForEach-Object { [double]::Parse($_.gpu_sum_ms, [cultureinfo]::InvariantCulture) } | Sort-Object
                if ($values[300] -gt 4 -or $values[570] -gt 5) { throw "GPU budget missed: $case" }
            }
        }
    }
} finally { Pop-Location }
