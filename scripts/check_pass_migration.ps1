# Check pass migration status for the helio-v3 graph executor.
# Identifies passes that need manual review.
#
# Usage: pwsh scripts/check_pass_migration.ps1

$crates = "crates\helio-pass-*"

Write-Host "=== Passes with render_pass_descriptor (migrated path) ===" -ForegroundColor Cyan
$migrated = @{}
Select-String -Path "$crates\src\lib.rs" -Pattern "fn render_pass_descriptor" | ForEach-Object {
    $file = $_.Path
    $line = $_.LineNumber
    $name = ($_ -replace '.*impl RenderPass for (\w+).*', '$1')
    if (-not $name -or $name -eq $_) { $name = Split-Path $file -Parent | Split-Path -Leaf }
    Write-Host "  $name at $file : $line" -ForegroundColor Green
    $migrated[$file] = $name
}

Write-Host ""
Write-Host "=== Passes WITHOUT render_pass_descriptor (legacy path — cannot chain) ===" -ForegroundColor Yellow
Select-String -Path "$crates\src\lib.rs" -Pattern "impl RenderPass for" | ForEach-Object {
    $file = $_.Path
    $imp_line = $_.LineNumber
    $name = $_ -match 'for (\w+)' ? $Matches[1] : "?"
    $has_desc = Select-String -Path $file -Pattern "fn render_pass_descriptor" -SimpleMatch -Quiet
    if (-not $has_desc) {
        Write-Host "  $name at $file : $imp_line" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Migrated-path passes using encoder_ptr for compute (POTENTIAL BUG) ===" -ForegroundColor Red
foreach ($file in $migrated.Keys) {
    $content = Get-Content $file -Raw
    $has_encoder_compute = $content -match "encoder_ptr.*begin_compute_pass"
    $has_compute_encoder = $content -match "compute_encoder_ptr.*begin_compute_pass"
    if ($has_encoder_compute -and -not $has_compute_encoder) {
        Write-Host "  $($migrated[$file]) at $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Migrated-path passes using encoder_ptr for clear/copy (POTENTIAL BUG) ===" -ForegroundColor Red
foreach ($file in $migrated.Keys) {
    $content = Get-Content $file -Raw
    $lines = Select-String -Path $file -Pattern "encoder_ptr.*(clear_buffer|copy_buffer_to_buffer)" | ForEach-Object {
        "    line $($_.LineNumber): $($_.Line.Trim())"
    }
    if ($lines) {
        Write-Host "  $($migrated[$file]) at $file" -ForegroundColor Red
        $lines | ForEach-Object { Write-Host $_ }
    }
}

Write-Host ""
Write-Host "=== All other render passes ===" -ForegroundColor Cyan
Select-String -Path "$crates\src\lib.rs" -Pattern "impl RenderPass for" | ForEach-Object {
    $file = $_.Path
    $name = $_ -match 'for (\w+)' ? $Matches[1] : "?"
    if (-not $migrated.ContainsKey($file)) {
        Write-Host "  $name at $_" -ForegroundColor Gray
    }
}
