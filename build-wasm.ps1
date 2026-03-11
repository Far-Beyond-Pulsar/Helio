#!/usr/bin/env pwsh
# build-wasm.ps1
#
# Pre-builds all helio-wasm-examples WASM demos and runs wasm-bindgen on each.
# Outputs land in target/wasm-prebuilt/<name>/ where build.rs copies them from.
#
# Usage:
#   ./build-wasm.ps1            # build all examples
#   ./build-wasm.ps1 sdf_blend  # build one example
#
# After this finishes, run:
#   cargo run --release --bin helio-wasm-server -p helio-wasm-app

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Push-Location $root

$wasm_target = "wasm32-unknown-unknown"

$all_examples = @(
    [pscustomobject]@{ name = "render_v2_basic"; title = "Basic Render"    },
    [pscustomobject]@{ name = "render_v2_sky";   title = "Volumetric Sky"  },
    [pscustomobject]@{ name = "sdf_blend";       title = "SDF Blend"       },
    [pscustomobject]@{ name = "sdf_boolean";     title = "SDF Boolean"     },
    [pscustomobject]@{ name = "sdf_demo";        title = "SDF Demo"        },
    [pscustomobject]@{ name = "sdf_grid";        title = "SDF Grid Wave"   },
    [pscustomobject]@{ name = "sdf_morph";       title = "SDF Morph"       },
    [pscustomobject]@{ name = "sdf_move";        title = "SDF Move"        },
    [pscustomobject]@{ name = "sdf_multi";       title = "SDF Multi"       },
    [pscustomobject]@{ name = "sdf_pulse";       title = "SDF Pulse"       },
    [pscustomobject]@{ name = "sdf_terrain";     title = "SDF Terrain"     }
)

# Filter to a specific example if one was given on the command line
if ($args.Count -gt 0) {
    $examples = $all_examples | Where-Object { $_.name -eq $args[0] }
    if (-not $examples) {
        Write-Error "Unknown example '$($args[0])'. Valid: $($all_examples.name -join ', ')"
        exit 1
    }
} else {
    $examples = $all_examples
}

$failed = @()

foreach ($ex in $examples) {
    $name = $ex.name
    Write-Host ""
    Write-Host "════ $name ════" -ForegroundColor Cyan

    $target_dir = Join-Path $root "target" "wasm-build" "ex" $name
    $out_dir    = Join-Path $root "target" "wasm-prebuilt" $name
    New-Item -ItemType Directory -Force -Path $target_dir | Out-Null
    New-Item -ItemType Directory -Force -Path $out_dir    | Out-Null

    Write-Host "  cargo build --release --lib -p helio-wasm-examples --features $name"
    cargo build --release --lib `
        -p helio-wasm-examples `
        --target $wasm_target `
        --no-default-features `
        --features $name `
        --target-dir $target_dir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED cargo build for $name" -ForegroundColor Red
        $failed += $name
        continue
    }

    $wasm_path = Join-Path $target_dir $wasm_target "release" "helio_wasm_examples.wasm"
    if (-not (Test-Path $wasm_path)) {
        Write-Host "  FAILED: wasm not found at $wasm_path" -ForegroundColor Red
        $failed += $name
        continue
    }
    $wasm_size = (Get-Item $wasm_path).Length
    Write-Host "  wasm OK ($wasm_size bytes)"

    Write-Host "  wasm-bindgen --out-dir $out_dir --target web"
    wasm-bindgen $wasm_path --out-dir $out_dir --target web
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED wasm-bindgen for $name" -ForegroundColor Red
        $failed += $name
        continue
    }

    Write-Host "  OK -> $out_dir" -ForegroundColor Green
}

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "FAILED examples: $($failed -join ', ')" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All examples built successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Now run:"
    Write-Host "  cargo run --release --bin helio-wasm-server -p helio-wasm-app" -ForegroundColor Yellow
}

Pop-Location
