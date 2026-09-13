# Enforce the Helio 3.0 scene-authority boundary.
#
# This guard is intentionally source-based. It checks production code, including
# examples that are outside the default Cargo build, so a stale compatibility
# seam cannot be hidden behind a feature flag or an unbuilt target.
#
# Usage: pwsh scripts/check_scene_authority.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# These are the production roots in this nested Helio workspace. The
# helio-component crate is excluded from the nested workspace and is therefore
# checked here only when its source is present; the full Pulsar-Native workspace
# remains the authority for its cross-repository inventory wiring.
$sourceRoots = @(
    "crates\helio-core\src",
    "crates\libhelio\src",
    "crates\helio\src",
    "crates\helio-bake\src",
    "crates\helio-android-demos\src",
    "crates\examples",
    "crates\helio-web-demos\src",
    "crates\helio-web-demos\examples-wasm"
)

# Retired production contracts and renderer-owned scene mutation APIs. These
# names must remain absent; SceneDB/SceneBufferProjection are the only authored
# CPU/GPU world-state authority after the Helio 3.0 migration.
$forbiddenSymbols = @(
    "FrameResources",
    "MainSceneResources",
    "frame_resources",
    "scene_for_legacy_mut",
    "place_static_object",
    "update_static_object_transform",
    "remove_static_object",
    "transient_scene_mut"
)

$violations = [System.Collections.Generic.List[string]]::new()
foreach ($root in $sourceRoots) {
    $path = Join-Path $repo $root
    if (-not (Test-Path -LiteralPath $path)) { continue }
    Get-ChildItem -LiteralPath $path -Recurse -File -Include *.rs,*.toml |
        Where-Object { $_.FullName -notmatch "[\\/]target[\\/]" } |
        ForEach-Object {
            $file = $_
            $lines = Get-Content -LiteralPath $file.FullName
            for ($i = 0; $i -lt $lines.Count; $i++) {
                foreach ($symbol in $forbiddenSymbols) {
                    if ($lines[$i] -match [regex]::Escape($symbol)) {
                        $violations.Add("$($file.FullName):$($i + 1): retired Helio 3.0 symbol '$symbol': $($lines[$i].Trim())")
                        break
                    }
                }
            }
        }
}

# The open registry is a graph/transient binding mechanism, not authored scene
# storage. Keep primitive-specific dependencies out of the two shared core
# crates so foliage/portal/voxel/planet implementations cannot leak into the
# generic graph/resource contracts.
$coreManifests = @(
    "crates\helio-core\Cargo.toml",
    "crates\libhelio\Cargo.toml"
)
$primitiveCorePattern = '(?i)(?:foliage|portal|voxel|planet|billboard|corona|virtual[-_]?geometry|vg|primitive)[-_]core'
foreach ($manifest in $coreManifests) {
    $path = Join-Path $repo $manifest
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $lines = Get-Content -LiteralPath $path
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $primitiveCorePattern) {
            $violations.Add("$($path):$($i + 1): primitive-specific core dependency: $($lines[$i].Trim())")
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Helio 3.0 scene-authority guard: FAILED ($($violations.Count) violation(s))" -ForegroundColor Red
    $violations | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host "Helio 3.0 scene-authority guard: PASS" -ForegroundColor Green
