# multi-batch.ps1 — queue multiple batch.ps1 runs for unattended overnight processing.
#
# Edit $batches below to define your folder queue, then run:
#   .\multi-batch.ps1 -Monitor
#
# After all batches complete, outputs are automatically ingested into yogamann.db.
# Closes #26

param(
    [string]$OutputRoot      = "D:\Temp\yogamann-output",
    [string]$PipelineProfile = "yoga_asana",
    [string]$LogLevel        = "INFO",
    [int]$Limit              = 0,       # per-batch image limit (0 = unlimited)
    [switch]$DryRun,
    [switch]$Monitor,
    [switch]$Overwrite
)

# ── Batch queue — edit this list to define your overnight run ───────────────────
$batches = @(
    @{
        Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Asanas"
        SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways"
    }
    @{
        Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Pranayama"
        SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways"
    }
    # Add more entries as needed:
    # @{
    #     Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Mudras"
    #     SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways"
    # }
)
# ───────────────────────────────────────────────────────────────────────────────

$BatchScript = Join-Path $PSScriptRoot "batch.ps1"
$MakeScript  = Join-Path $PSScriptRoot "make.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║  nightrun — $($batches.Count) batch(es) queued                      ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host "   OutputRoot   : $OutputRoot"      -ForegroundColor DarkGray
Write-Host "   Profile      : $PipelineProfile" -ForegroundColor DarkGray
Write-Host "   Limit/batch  : $(if ($Limit -gt 0) { $Limit } else { 'unlimited' })" -ForegroundColor DarkGray
if ($DryRun)    { Write-Host "   [DRY RUN — no images will be generated]" -ForegroundColor Yellow }
if ($Overwrite) { Write-Host "   [OVERWRITE — will reprocess existing outputs]" -ForegroundColor Yellow }
Write-Host ""

# Launch GPU monitor once for the whole run
$monitorProc = $null
if ($Monitor) {
    $monitorScript = Join-Path $PSScriptRoot "monitor.ps1"
    if (Test-Path $monitorScript) {
        try {
            $monitorProc = Start-Process pwsh -ArgumentList "-NoExit", "-File", $monitorScript -PassThru -ErrorAction Stop
            Write-Host "[monitor] Launched in new terminal (PID $($monitorProc.Id))" -ForegroundColor DarkCyan
        } catch {
            Write-Host "[monitor] Failed to launch: $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[monitor] monitor.ps1 not found — skipping" -ForegroundColor Yellow
    }
}

$startTime  = Get-Date
$batchNum   = 0
$totalFailed = 0

foreach ($b in $batches) {
    $batchNum++
    Write-Host ""
    Write-Host "══ Batch $batchNum / $($batches.Count) ═══════════════════════════════════════" -ForegroundColor Magenta
    Write-Host "   Source     : $($b.Source)"     -ForegroundColor White
    Write-Host "   SourceRoot : $($b.SourceRoot)" -ForegroundColor DarkGray

    $batchArgs = @{
        Source          = $b.Source
        SourceRoot      = $b.SourceRoot
        OutputRoot      = $OutputRoot
        PipelineProfile = $PipelineProfile
        LogLevel        = $LogLevel
    }
    if ($Limit -gt 0) { $batchArgs['Limit']     = $Limit }
    if ($DryRun)       { $batchArgs['DryRun']    = $true }
    if ($Overwrite)    { $batchArgs['Overwrite'] = $true }
    # -Monitor intentionally omitted — launched once above

    & $BatchScript @batchArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Batch $batchNum exited with code $LASTEXITCODE" -ForegroundColor Yellow
        $totalFailed++
    }
}

$elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  All $($batches.Count) batch(es) complete — ${elapsed} min total              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green

if (-not $DryRun) {
    Write-Host ""
    Write-Host "==> Ingesting outputs into yogamann.db ..." -ForegroundColor Cyan
    & $MakeScript -Target ingest -OutputRoot $OutputRoot -LogLevel $LogLevel
    Write-Host ""
    Write-Host "==> Done. Launch the review gallery with:" -ForegroundColor Green
    Write-Host "    .\make.ps1 -Target review" -ForegroundColor White
}

if ($monitorProc -and -not $monitorProc.HasExited) {
    Write-Host "[monitor] Stopping monitor (PID $($monitorProc.Id))..." -ForegroundColor DarkCyan
    $monitorProc.Kill()
}

if ($totalFailed -gt 0) {
    Write-Host ""
    Write-Host "[WARN] $totalFailed batch(es) reported errors — check output above." -ForegroundColor Yellow
    exit 1
}
