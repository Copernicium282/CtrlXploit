# Chains `make evaluate` + `make test` behind the in-flight `make data` build.
# Waits for the build_dataset process to exit AND for the cells parquet to exist,
# then runs the two downstream targets via make.ps1.
$ErrorActionPreference = 'Continue'
# Repo root = directory this script lives in (override with -RepoRoot if needed).
$r = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if (-not (Test-Path (Join-Path $r 'make.ps1'))) {
  Write-Error "run_after_data.ps1 must live at the repo root (make.ps1 not found under '$r')"
  exit 1
}
New-Item -ItemType Directory -Force -Path (Join-Path $r 'logs') | Out-Null
$log = Join-Path $r 'logs\win_eval_test.log'

function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
  Write-Host $line
  Add-Content -Path $log -Value $line
}

Say "waiter started; waiting for build_dataset to finish"

# Phase 1: wait for any running build_dataset to exit.
while ($true) {
  $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -like '*build_dataset*' }
  if (-not $busy) { break }
  Start-Sleep -Seconds 20
}
Say "build_dataset process exited"

# Phase 2: wait for processed cells to land (up to 10 min).
$deadline = (Get-Date).AddMinutes(10)
$cells = $null
while ((Get-Date) -lt $deadline) {
  $cells = Get-ChildItem (Join-Path $r 'data\processed') -Recurse -Filter '*.parquet' -ErrorAction SilentlyContinue
  if ($cells -and $cells.Count -gt 0) { break }
  Start-Sleep -Seconds 15
}

if (-not $cells -or $cells.Count -eq 0) {
  Say "ABORT: no processed cells found - data build did not produce output"
  exit 1
}
Say ("data ready: {0} cell parquet files" -f $cells.Count)
$cells | ForEach-Object { Say ("  {0,8:N1} MB  {1}" -f ($_.Length/1MB), $_.FullName.Substring($r.Length+1)) }

# Phase 3: evaluate, then test.
Say "=== running make evaluate ==="
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $r 'make.ps1') evaluate *>&1 |
  Tee-Object -FilePath (Join-Path $r 'logs\win_evaluate.log') | Out-Null
$evalExit = $LASTEXITCODE
Say "make evaluate exit=$evalExit"

Say "=== running make test ==="
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $r 'make.ps1') test *>&1 |
  Tee-Object -FilePath (Join-Path $r 'logs\win_test.log') | Out-Null
$testExit = $LASTEXITCODE
Say "make test exit=$testExit"

Say "CHAIN COMPLETE eval=$evalExit test=$testExit"
