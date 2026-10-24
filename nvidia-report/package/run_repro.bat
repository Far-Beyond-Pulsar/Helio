@echo off
rem NVIDIA bug 6939141 repro: one fixed-step descent on Vulkan, captures in out_vulkan\
cd /d "%~dp0"
set WGPU_BACKEND=vulkan
set HELIO_VOXEL_FLIGHT_SEED=75
set HELIO_VOXEL_STRESS_AT=0,1,0
set HELIO_VOXEL_FLIGHT_SUN_ELEVATION=32
set HELIO_VOXEL_FLIGHT_STRESS=descent
set HELIO_VOXEL_STRESS_STEP=1
set HELIO_VOXEL_FLIGHT_HOLD_SECONDS=16
set HELIO_VOXEL_STRESS_SAMPLE=0.25
if exist out_vulkan rmdir /s /q out_vulkan
voxel_flight.exe out_vulkan 2560 1440
echo.
echo Done. Captures are in out_vulkan\ (stress_descent_NN.png, settled references *_ref.png).
echo Look for flat grey-purple 8x8 blocks on the ground, or run: python detect_lost_blocks.py out_vulkan
pause
