# Sync: PowerShell helper for Windows (interactive MFA).
# Run in a normal PowerShell window where you can complete Narval MFA prompts.
#
#   cd c:\Users\meriemterki\Documents\Thesis\eeg_qgan
#   .\demo\sync_from_narval.ps1 -Optional

param(
  [string]$NarvalHost = "mrmterki@narval.computecanada.ca",
  [string]$RemoteScratch = "/scratch/linah03",
  [switch]$Optional
)

$ErrorActionPreference = "Stop"
$LocalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$Main = @(
  "eeg_qgan_results_thesis_figures/thesis_figure_capture/artifacts/seed0_fold0"
)

$Extra = @(
  "eeg_qgan_results_canonical/rematch_deadline/artifacts/seed0_fold0",
  "eeg_qgan_results_canonical/entanglement_deadline/artifacts/seed0_fold0",
  "eeg_qgan_results_canonical/depth_deadline/artifacts/seed0_fold0",
  "eeg_qgan_results_canonical/surrogate/artifacts/seed0_fold0"
)

function Sync-One([string]$Rel) {
  $remote = "$RemoteScratch/$Rel"
  $localDir = Join-Path $LocalRoot ($Rel -replace "/", [IO.Path]::DirectorySeparatorChar)
  $parent = Split-Path $localDir -Parent
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Write-Host "==> $Rel"
  Write-Host "    from ${NarvalHost}:${remote}"
  Write-Host "    to   $localDir"
  # scp creates the leaf folder name under parent
  scp -r "${NarvalHost}:${remote}" $parent
  if (Test-Path (Join-Path $localDir "generators")) {
    Write-Host "    generators:" (Get-ChildItem (Join-Path $localDir "generators") -Name)
  }
}

Write-Host "Local thesis root: $LocalRoot"
Write-Host "You will be prompted for Narval MFA / password."
Write-Host ""

foreach ($rel in $Main) { Sync-One $rel }

if ($Optional) {
  Write-Host ""
  Write-Host "=== Optional folds ==="
  foreach ($rel in $Extra) { Sync-One $rel }
} else {
  Write-Host ""
  Write-Host "Optional folds skipped. Re-run with -Optional to pull rematch / entanglement / Fourier."
}

Write-Host ""
Write-Host "Restart demo:  cd eeg_qgan; python demo/server.py"
Write-Host "Check status:  python demo/verify_studies.py"
