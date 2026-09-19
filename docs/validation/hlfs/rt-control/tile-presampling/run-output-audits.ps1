$ErrorActionPreference = 'Stop'
foreach ($samples in 1..4) {
    $env:HLFS_RT_AUDIT_SAMPLES="$samples"
    foreach ($variant in @('A','U')) {
        $env:HLFS_RT_AUDIT_OUTPUT="$PWD/target/rt-presampled-audit-$samples-$variant"
        $exe=if($variant -eq 'A') {'target/rt-stencil-A.exe'} else {'target/rt-presampled-final.exe'}
        & $exe benchmark_candidate_output_audit --exact --ignored --nocapture *> "target/rt-presampled-audit-$samples-$variant.log"
        if ($LASTEXITCODE -ne 0) { throw "Audit failed: $samples $variant" }
        Write-Output "AUDIT complete spp=$samples variant=$variant"
    }
}
