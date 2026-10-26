# Download Narval result trees with scp (Narval -> this Windows PC).
#
#   cd C:\Users\meriemterki\Documents\Thesis
#   ssh mrmterki@narval.computecanada.ca "echo `$SCRATCH"
#   $env:NARVAL_SCRATCH = "/scratch/mrmterki"   # paste that path
#   powershell -ExecutionPolicy Bypass -File eeg_qgan\scripts\sync_narval_to_windows.ps1
#
# Or copy-paste the scp lines printed at the end of this file's comment in PowerShell.
# scp cannot skip *.pt; those weight files are large. For figures only, use -CaptureOnly
# and accept that artifacts/*.pt will come along, or scp the decoded/ folder by itself.

param(
    [string]$UserHost = "mrmterki@narval.computecanada.ca",
    [string]$RemoteScratch = $env:NARVAL_SCRATCH,
    [string]$LocalThesis = "C:\Users\meriemterki\Documents\Thesis",
    [switch]$CaptureOnly
)

$ErrorActionPreference = "Stop"

if (-not $RemoteScratch) {
    Write-Host "Set the remote scratch path first (scp on Windows cannot expand `$SCRATCH)."
    Write-Host "  ssh $UserHost `"echo `$SCRATCH`""
    Write-Host "  `$env:NARVAL_SCRATCH = '/scratch/mrmterki'"
    exit 1
}

$RemoteScratch = $RemoteScratch.TrimEnd("/")
$trees = @(
    "eeg_qgan_results_thesis_figures"
    "eeg_qgan_results_paper_style_cpu"
    "eeg_qgan_results_paper_style_q4"
    "eeg_qgan_results_canonical"
    "eeg_qgan_results_dose_deadline_d256"
    "eeg_qgan_results_dose_deadline_d512"
    "eeg_qgan_results_dataeff_deadline_f0.25"
    "eeg_qgan_results_dataeff_deadline_f0.5"
)
if ($CaptureOnly) {
    $trees = @("eeg_qgan_results_thesis_figures")
}

Write-Host "Using scp -r"
Write-Host "Remote: ${UserHost}:${RemoteScratch}/<tree>"
Write-Host "Local:  $LocalThesis"
Write-Host ""

foreach ($tree in $trees) {
    Write-Host "=== $tree ==="
    $src = "${UserHost}:${RemoteScratch}/${tree}"
    scp -r $src $LocalThesis
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $tree (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "Then plot:"
Write-Host "  python eeg_qgan\scripts\plot_thesis_figures.py --results-root eeg_qgan_results_thesis_figures\thesis_figure_capture --pfe-copy"
