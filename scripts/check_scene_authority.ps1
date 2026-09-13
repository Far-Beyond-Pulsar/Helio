# Enforce the Helio 3.0 scene-authority boundary.
#
# Production callers must author persistent scene content in the shared SceneDB
# World and pass only its GPU projection to Helio. This guard is deliberately
# source-based: it runs before Cargo can hide a stale caller behind a feature
# flag, and it also catches examples that are not part of the default build.
#
# Usage: pwsh scripts/check_scene_authority.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$callerRoots = @(
    "crates\helio-component\src",
    "crates\helio-bake\src",
    "crates\helio-android-demos\src",
    "crates\helio-controls\src",
    "crates\helio-wasm\src",
    "crates\helio-xr\src",
    "crates\helio-default-graphs\src",
    "crates\examples",
    "crates\helio-web-demos\src",
    "crates\helio-web-demos\examples-wasm"
)

# These are the broad actor/vector seams that let an application accidentally
# create a second CPU scene authority. Asset uploads (mesh/material bytes) are
# intentionally not listed: they are resource-pool operations and contain no
# placement or entity lifecycle.
$callerPatterns = @(
    "scene_for_legacy_mut",
    "SceneEntity::",
    "\.insert_entity\s*\(",
    "set_billboard_instances",
    "set_corona_emitters",
    "set_virtual_geometry",
    "rebuild_light_instances",
    "rebuild_static_mesh_instances"
)

$violations = [System.Collections.Generic.List[string]]::new()
foreach ($root in $callerRoots) {
    $path = Join-Path $repo $root
    if (-not (Test-Path -LiteralPath $path)) { continue }
    Get-ChildItem -LiteralPath $path -Recurse -File -Include *.rs,*.toml |
        Where-Object { $_.FullName -notmatch "[\\/]tests?[\\/]" } |
        ForEach-Object {
            $file = $_
            $lines = Get-Content -LiteralPath $file.FullName
            for ($i = 0; $i -lt $lines.Count; $i++) {
                # Documentation may describe the retired API. Guard executable
                # caller code, not prose examples in comments.
                $code = $lines[$i] -replace '//.*$', ''
                foreach ($pattern in $callerPatterns) {
                    if ($code -match $pattern) {
                        $violations.Add("$($file.FullName):$($i + 1): $($lines[$i].Trim())")
                        break
                    }
                }
            }
        }
}

# Helio itself may retain generic GPU resources, but never authored primitive
# records in vectors/arenas. Keep these patterns narrow so unrelated geometry
# scratch buffers remain legal.
$helioPath = Join-Path $repo "crates\helio\src"
$rendererPatterns = @(
    "Vec\s*<\s*(BillboardInstance|GpuCoronaEmitter|VirtualObjectDescriptor|SceneEntity)",
    "(billboard|corona|virtual_geometry|vg)_(instances|emitters|objects|arena)\s*:\s*(Vec|DenseArena|SparsePool)"
)
if (Test-Path -LiteralPath $helioPath) {
    Get-ChildItem -LiteralPath $helioPath -Recurse -File -Include *.rs | ForEach-Object {
        $file = $_
        $lines = Get-Content -LiteralPath $file.FullName
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $code = $lines[$i] -replace '//.*$', ''
            foreach ($pattern in $rendererPatterns) {
                if ($code -match $pattern) {
                    $violations.Add("$($file.FullName):$($i + 1): renderer-owned authored primitive storage: $($lines[$i].Trim())")
                    break
                }
            }
        }
    }
}

# No Helio consumer may reintroduce the retired primitive *-core crates. The
# suffix check is intentionally generic because these crates have been renamed
# during the staged migration; it catches both old and newly invented names.
Get-ChildItem -LiteralPath $repo -Recurse -File -Filter Cargo.toml |
    Where-Object { $_.FullName -notmatch "[\\/]target[\\/]" } |
    ForEach-Object {
        $file = $_
        $lines = Get-Content -LiteralPath $file.FullName
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '(?:billboard|corona|virtual[-_]?geometry|vg|primitive)[-_][^" ]*[-_]core') {
                $violations.Add("$($file.FullName):$($i + 1): retired primitive core dependency: $($lines[$i].Trim())")
            }
        }
    }

if ($violations.Count -gt 0) {
    Write-Host "Helio 3.0 scene-authority guard: FAILED ($($violations.Count) violation(s))" -ForegroundColor Red
    $violations | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host "Helio 3.0 scene-authority guard: PASS" -ForegroundColor Green
