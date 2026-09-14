# Enforce the Helio 3.0 scene-authority boundary.
#
# Usage: pwsh scripts/check_scene_authority.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$guard = (Resolve-Path $MyInvocation.MyCommand.Path).Path

# Scan the complete tracked Rust/TOML surface, including examples and targets
# that are outside the default Cargo build. The guard itself is excluded so
# the forbidden-symbol inventory below does not trip its own check.
$tracked = @(git -C $repo ls-files -- '*.rs' '*.toml')
$files = @(
    $tracked |
        ForEach-Object { Join-Path $repo $_ } |
        Where-Object { (Resolve-Path -LiteralPath $_).Path -ne $guard }
)

$forbiddenSymbols = @(
    'FrameResources',
    'MainSceneResources',
    'frame_resources',
    'scene_for_legacy_mut',
    'add_foliage_type',
    'add_foliage_layer',
    'add_foliage_interactor',
    'add_portal',
    'FoliageFrameData',
    'PortalsFrameData',
    'FoliageTypeDescriptor',
    'PortalDescriptor',
    'FoliageTypeId',
    'FoliageLayerId',
    'FoliageInteractorId',
    'PortalId',
    'helio-foliage-core',
    'helio-portal-core',
    'helio_foliage_core',
    'helio_portal_core'
)

$violations = [System.Collections.Generic.List[string]]::new()
foreach ($filePath in $files) {
    if (-not (Test-Path -LiteralPath $filePath)) { continue }
    $lines = Get-Content -LiteralPath $filePath
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($symbol in $forbiddenSymbols) {
            if ($lines[$i] -match [regex]::Escape($symbol)) {
                $violations.Add("$($filePath):$($i + 1): retired Helio 3.0 symbol '$symbol': $($lines[$i].Trim())")
                break
            }
        }
    }
}

# Shared core manifests may not depend on primitive-specific implementation
# crates. Pass contracts live with the pass that consumes them.
$coreManifests = @(
    (Join-Path $repo 'crates\helio-core\Cargo.toml'),
    (Join-Path $repo 'crates\libhelio\Cargo.toml')
)
$primitiveCorePattern = '(?i)(?:foliage|portal|voxel|planet|billboard|corona|virtual[-_]?geometry|vg|primitive)[-_]core'
foreach ($manifest in $coreManifests) {
    if (-not (Test-Path -LiteralPath $manifest)) { continue }
    $lines = Get-Content -LiteralPath $manifest
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $primitiveCorePattern) {
            $violations.Add("$($manifest):$($i + 1): primitive-specific core dependency: $($lines[$i].Trim())")
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Helio 3.0 scene-authority guard: FAILED ($($violations.Count) violation(s))" -ForegroundColor Red
    $violations | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host "Helio 3.0 scene-authority guard: PASS" -ForegroundColor Green
