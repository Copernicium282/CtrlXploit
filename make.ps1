# make.ps1 - Windows equivalent of the Makefile.
# The Makefile is POSIX-only (PY ?= .venv/bin/python, python3, .venv/bin/pip) and
# `make` is not installed on this machine, so this script mirrors it 1:1.
#
#   .\make.ps1 setup
#   .\make.ps1 all
#   .\make.ps1 ctu13
<#
.SYNOPSIS
  Run the ThreatAhead pipeline targets.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string[]]$Targets = @('all')
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) {
  throw "venv not found at $Py -- run: .\make.ps1 setup"
}

# Mirror of Makefile `PY ?= ...` invocation, including the -W ignore the Makefile uses.
function Invoke-Sih {
  param([string[]]$SihArgs, [switch]$SuppressWarnings)
  $exe = @('-m') + $SihArgs
  if ($SuppressWarnings) { $exe = @('-W', 'ignore') + $exe }
  & $Py @exe
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $Py $($exe -join ' ')" }
}

# Makefile line 12-14
function Target-Data {
  Write-Host "`n=== data ===" -ForegroundColor Cyan
  Invoke-Sih @('sih_v2', 'build_dataset', '--dataset', 'ctu13')
  Invoke-Sih @('sih_v2', 'build_dataset', '--dataset', 'cic2018')
  Invoke-Sih @('sih_v2', 'build_dataset', '--dataset', 'synthetic', '--synthetic')
}

# Makefile line 17-20
function Target-Train {
  Write-Host "`n=== train ===" -ForegroundColor Cyan
  foreach ($c in @(@('ctu13','temporal'), @('ctu13','family'), @('cic2018','temporal'), @('synthetic','temporal'))) {
    Invoke-Sih @('sih_v2','train','--dataset',$c[0],'--protocol',$c[1]) -SuppressWarnings
  }
}

# Makefile line 23-26
function Target-Evaluate {
  Write-Host "`n=== evaluate ===" -ForegroundColor Cyan
  foreach ($c in @(@('ctu13','temporal'), @('ctu13','family'), @('cic2018','temporal'), @('synthetic','temporal'))) {
    Invoke-Sih @('sih_v2','evaluate','--dataset',$c[0],'--protocol',$c[1]) -SuppressWarnings
  }
}

# Makefile line 29-30
function Target-Eval {
  Write-Host "`n=== eval ===" -ForegroundColor Cyan
  Invoke-Sih @('eval/compare_baselines.py')
  Invoke-Sih @('eval/ablation.py','--dataset','ctu13','--protocol','temporal')
}

# Makefile line 33
function Target-Benchmark {
  Write-Host "`n=== benchmark ===" -ForegroundColor Cyan
  Invoke-Sih @('sih_v2','benchmark','--dataset','ctu13','--protocol','temporal','--skip-eval') -SuppressWarnings
}

# Makefile line 36
function Target-Test {
  Write-Host "`n=== test ===" -ForegroundColor Cyan
  Invoke-Sih @('-m','pytest','-q')
}

# Makefile line 9
function Target-Setup {
  $base = (Get-Command python -ErrorAction SilentlyContinue)
  if (-not $base) { throw "no system python on PATH" }
  & $base.Source -m venv (Join-Path $PSScriptRoot '.venv')
  Invoke-Sih @('-m','pip','install','-r','requirements.txt')
  Invoke-Sih @('-m','pip','install','-e','.')
}

# Makefile line 38-39
function Target-App {
  Write-Host "`n=== app ===" -ForegroundColor Cyan
  Invoke-Sih @('-m','streamlit','run','app.py')
}

# Makefile line 42-44
function Target-One {
  param([string]$Dataset)
  Write-Host "`n=== $Dataset end to end ===" -ForegroundColor Cyan
  Invoke-Sih @('sih_v2','build_dataset','--dataset',$Dataset)
  Invoke-Sih @('sih_v2','train','--dataset',$Dataset) -SuppressWarnings
  Invoke-Sih @('sih_v2','evaluate','--dataset',$Dataset) -SuppressWarnings
}

$map = @{
  setup = { Target-Setup }
  data = { Target-Data }
  train = { Target-Train }
  evaluate = { Target-Evaluate }
  eval = { Target-Eval }
  benchmark = { Target-Benchmark }
  test = { Target-Test }
  app = { Target-App }
  ctu13 = { Target-One 'ctu13' }
  cic2018 = { Target-One 'cic2018' }
  synthetic = { Target-One 'synthetic' }
}

# Makefile line 6: all = data train evaluate eval benchmark test
if ($Targets -contains 'all') {
  $Targets = @('data','train','evaluate','eval','benchmark','test')
}

foreach ($t in $Targets) {
  if (-not $map.ContainsKey($t)) { throw "unknown target '$t' (have: $($map.Keys -join ', '), all)" }
  Write-Host "`n########## make $t ##########" -ForegroundColor Green
  & $map[$t]
}

Write-Host "`nAll Targets Complete" -ForegroundColor Green
