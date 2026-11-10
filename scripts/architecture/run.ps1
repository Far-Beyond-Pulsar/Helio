# Runs one of the Python architecture scripts with an interpreter that works.
#
# `python` on a stock Windows machine is often the Microsoft Store stub (it
# prints a hint and exits non-zero), and the scripts need Python 3.11+
# (`tomllib`). Try the launcher, `python`, then `python3`, keep the first that
# can import `tomllib`, and fail loudly with an install hint if none can.
#
# Usage: pwsh scripts/architecture/run.ps1 <script.py> [args...]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)

$ErrorActionPreference = 'Stop'
$candidates = @(
    @{ Exe = 'py'; Pre = @('-3') },
    @{ Exe = 'python'; Pre = @() },
    @{ Exe = 'python3'; Pre = @() }
)

foreach ($c in $candidates) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    & $c.Exe @($c.Pre) -c 'import tomllib' 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { continue }
    $root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    & $c.Exe @($c.Pre) (Join-Path $root $Script) @ScriptArgs
    exit $LASTEXITCODE
}

[Console]::Error.WriteLine(@'
error: no usable Python 3.11+ found (tried: py -3, python, python3).
The Microsoft Store `python` stub does not count. Install Python and reopen the terminal:
    winget install Python.Python.3.12
or download it from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
'@)
exit 1
