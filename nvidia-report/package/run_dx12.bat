@echo off
rem NVIDIA bug 6939141 repro: one fixed-step descent on D3D12 (comparison), captures in out_dx12\
cd /d "%~dp0"
set WGPU_BACKEND=dx12
set HELIO_VOXEL_FLIGHT_SEED=75
set HELIO_VOXEL_STRESS_AT=0,1,0
set HELIO_VOXEL_FLIGHT_SUN_ELEVATION=32
set HELIO_VOXEL_FLIGHT_STRESS=descent
set HELIO_VOXEL_STRESS_STEP=1
set HELIO_VOXEL_FLIGHT_HOLD_SECONDS=16
set HELIO_VOXEL_STRESS_SAMPLE=0.25
if exist out_dx12 rmdir /s /q out_dx12
voxel_flight.exe out_dx12 2560 1440
echo.
echo Done. Captures are in out_dx12\ (stress_descent_NN.png, settled references *_ref.png).
echo Look for flat grey-purple 8x8 blocks on the ground, or run: python detect_lost_blocks.py out_dx12
pause
