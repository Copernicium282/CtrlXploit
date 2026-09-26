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

# Windows consoles default to a legacy code page (cp1252); our reports are full of
# `≤ ± ⊕ → ·`. Force UTF-8 mode so any print()/open() anywhere in the stack is safe.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Resolved lazily: `setup` has to run before the venv exists, so the check cannot
# happen at load time or the bootstrap target would be unreachable.
function Get-Py {
  $py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
  if (-not (Test-Path $py)) {
    throw "venv not found at $py -- run: .\make.ps1 setup"
  }
  return $py
}

# Mirror of Makefile `PY ?= ...` invocation, including the -W ignore the Makefile uses.
# Takes the interpreter's full argument list, so it serves `-m sih_v2 ...` calls and
# plain script paths alike.
function Invoke-Py {
  param([string[]]$PyArgs, [switch]$SuppressWarnings)
  $exe = @()
  if ($SuppressWarnings) { $exe += @('-W', 'ignore') }
  $exe += $PyArgs
  & (Get-Py) @exe
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): python $($exe -join ' ')" }
}

# Makefile `fetch-data`: pull the prebuilt data/ tree from the Hugging Face dataset
# repo (260 MB, incl. the 123 MB CTU-13 cells.parquet that git cannot hold).
function Target-FetchData {
  Write-Host "`n=== fetch-data ===" -ForegroundColor Cyan
  Invoke-Py @('-m','pip','install','-q','-U','huggingface_hub')
  Invoke-Py @('-c', "import os; from huggingface_hub import snapshot_download; skip = os.path.exists('data/processed/ctu13/cells.parquet') and not os.environ.get('FORCE_FETCH'); snapshot_download(repo_id='ir192m2Cn282/ThreatAhead', repo_type='dataset', local_dir='.', allow_patterns=['data/**']) if not skip else print('fetch-data: data/ already present - set FORCE_FETCH=1 to re-sync from HF')")
}

# Makefile line 12-14
function Target-Data {
  Write-Host "`n=== data ===" -ForegroundColor Cyan
  Invoke-Py @('-m', 'sih_v2', 'build_dataset', '--dataset', 'ctu13')
  Invoke-Py @('-m', 'sih_v2', 'build_dataset', '--dataset', 'cic2018')
  Invoke-Py @('-m', 'sih_v2', 'build_dataset', '--dataset', 'synthetic', '--synthetic')
}

# Makefile line 17-20
function Target-Train {
  Write-Host "`n=== train ===" -ForegroundColor Cyan
  foreach ($c in @(@('ctu13','temporal'), @('ctu13','family'), @('cic2018','temporal'), @('synthetic','temporal'))) {
    Invoke-Py @('-m','sih_v2','train','--dataset',$c[0],'--protocol',$c[1]) -SuppressWarnings
  }
}

# Makefile line 23-26
function Target-Evaluate {
  Write-Host "`n=== evaluate ===" -ForegroundColor Cyan
  foreach ($c in @(@('ctu13','temporal'), @('ctu13','family'), @('cic2018','temporal'), @('synthetic','temporal'))) {
    Invoke-Py @('-m','sih_v2','evaluate','--dataset',$c[0],'--protocol',$c[1]) -SuppressWarnings
  }
}

# Makefile line 29-30
function Target-Eval {
  Write-Host "`n=== eval ===" -ForegroundColor Cyan
  Invoke-Py @('eval/compare_baselines.py')
  Invoke-Py @('eval/ablation.py','--dataset','ctu13','--protocol','temporal')
}

# Makefile line 33
function Target-Benchmark {
  Write-Host "`n=== benchmark ===" -ForegroundColor Cyan
  Invoke-Py @('-m','sih_v2','benchmark','--dataset','ctu13','--protocol','temporal','--skip-eval') -SuppressWarnings
}

# Makefile line 36
function Target-Test {
  Write-Host "`n=== test ===" -ForegroundColor Cyan
  Invoke-Py @('-m','pytest','-q')
}

# System interpreter, used only to create the venv. `python`/`python3`/`py` on PATH first,
# then the per-user install locations: a normal Windows install does not put python.exe on
# PATH for every shell, which used to make this target fail with "no system python on PATH".
function Get-BootstrapPython {
  foreach ($name in @('python', 'python3', 'py')) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    $args = @()
    if ($name -eq 'py') { $args = @('-3') }        # the launcher needs an explicit major version
    return [pscustomobject]@{ Exe = $cmd.Source; Args = $args }
  }
  foreach ($root in @((Join-Path $env:LOCALAPPDATA 'Programs\Python'), 'C:\')) {
    if (-not (Test-Path $root)) { continue }
    $hit = Get-ChildItem -Path $root -Filter 'python.exe' -Recurse -Depth 2 -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -notlike '*\.venv\*' } |
           Select-Object -First 1
    if ($hit) { return [pscustomobject]@{ Exe = $hit.FullName; Args = @() } }
  }
  throw "no system Python 3 found (looked for python/python3/py on PATH and the default per-user install dirs). Install Python 3.10+ or create .venv yourself, then re-run."
}

# Makefile line 9
function Target-Setup {
  $base = Get-BootstrapPython
  Write-Host "Using $($base.Exe) to create .venv" -ForegroundColor DarkGray
  & $base.Exe @($base.Args + @('-m', 'venv', (Join-Path $PSScriptRoot '.venv')))
  if ($LASTEXITCODE -ne 0) { throw "venv creation failed ($LASTEXITCODE)" }
  Invoke-Py @('-m','pip','install','-r','requirements.txt')
  Invoke-Py @('-m','pip','install','-e','.')
}

# Makefile line 38-39
function Target-App {
  Write-Host "`n=== app ===" -ForegroundColor Cyan
  Invoke-Py @('-m','streamlit','run','app.py')
}

# Makefile line 42-44
function Target-One {
  param([string]$Dataset)
  Write-Host "`n=== $Dataset end to end ===" -ForegroundColor Cyan
  Invoke-Py @('-m','sih_v2','build_dataset','--dataset',$Dataset)
  Invoke-Py @('-m','sih_v2','train','--dataset',$Dataset) -SuppressWarnings
  Invoke-Py @('-m','sih_v2','evaluate','--dataset',$Dataset) -SuppressWarnings
}

$map = @{
  setup = { Target-Setup }
  'fetch-data' = { Target-FetchData }
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

Write-Host "`nALL TARGETS COMPLETE" -ForegroundColor Green
