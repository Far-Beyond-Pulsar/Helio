$ErrorActionPreference='Stop'
$env:HLFS_RT_PROBE_FOCUS='1440p-reconstructed'
$env:HLFS_RT_PROBE_SAMPLES='1'
$env:HLFS_RT_PROBE_CANDIDATES='8'
$env:HLFS_RT_PROBE_DISCOVERY='1'
$env:HLFS_RT_PROBE_PRESAMPLE='1'
$env:HLFS_RT_PROBE_REACTIVE='1'
$telemetry=Start-Process nvidia-smi -ArgumentList @('--query-gpu=timestamp,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader','-lms','200','-f',"$PWD/target/rt-isolated-telemetry.csv") -WindowStyle Hidden -PassThru
try {
 $ordinal=0
 foreach ($block in 1..3) {
  foreach ($variant in @('A','U','U','A')) {
   $ordinal++
   $stem="rt-isolated-$block-$ordinal-$variant"
   $env:HLFS_RT_PROBE_OUTPUT="$PWD/target/$stem"
   $exe=if($variant -eq 'A') {'target/rt-stencil-A.exe'} else {'target/rt-presampled-final.exe'}
   & $exe benchmark_rt_resolution_and_acceleration --exact --ignored --nocapture *> "target/$stem.log"
   if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $stem" }
   Get-Content "target/$stem.log" | Select-String RT_PROBE
  }
 }
 foreach ($focus in @('1440p-native','4k-reconstructed')) {
  $env:HLFS_RT_PROBE_FOCUS=$focus
  $env:HLFS_RT_PROBE_OUTPUT="$PWD/target/rt-isolated-$focus"
  & target/rt-presampled-final.exe benchmark_rt_resolution_and_acceleration --exact --ignored --nocapture *> "target/rt-isolated-$focus.log"
  if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $focus" }
  Get-Content "target/rt-isolated-$focus.log" | Select-String RT_PROBE
 }
} finally { Stop-Process -Id $telemetry.Id -ErrorAction SilentlyContinue }
