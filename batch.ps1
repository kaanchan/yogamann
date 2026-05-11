param(
    # What to process: a single image file, or any subdirectory (walked recursively).
    # Defaults to the full data library root.
    [string]$Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways",

    # Top of the source library — used to compute relative paths for mirroring.
    # Always defaults to the data root so subfolder runs still produce the full structure:
    #   .\batch.ps1 -Source "...\Asanas\Vajrasana"
    #   → output: D:\Temp\yogamann-output\Asanas\Vajrasana\  (not \Vajrasana\)
    [string]$SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways",

    # Where outputs land, mirroring SourceRoot structure.
    [string]$OutputRoot = "D:\Temp\yogamann-output",

    [string]$PipelineProfile = "yoga_asana",
    [string]$LogLevel = "INFO",
    [int]$Limit = 0,     # max images to process this run (0 = unlimited)
    [switch]$DryRun,
    [switch]$Monitor     # spawn monitor.ps1 in a new terminal window
)

$Python   = ".\.venv\Scripts\python.exe"
$ImageExt = @(".jpg", ".jpeg", ".png", ".webp")

$env:YOGAMANN_LOG_LEVEL = $LogLevel
$env:HF_HUB_OFFLINE     = "1"

# ── Resolve paths ──────────────────────────────────────────────────────────────
$ResolvedSource = (Resolve-Path $Source -ErrorAction Stop).Path.TrimEnd('\', '/')
$SourceRoot     = $SourceRoot.TrimEnd('\', '/')
$OutputRoot     = $OutputRoot.TrimEnd('\', '/')
$isFile         = Test-Path $ResolvedSource -PathType Leaf

Write-Host ""
Write-Host "Source     : $ResolvedSource" -ForegroundColor White
Write-Host "SourceRoot : $SourceRoot"     -ForegroundColor DarkGray
Write-Host "OutputRoot : $OutputRoot"     -ForegroundColor DarkGray
if ($DryRun) { Write-Host "[DRY RUN — no images will be generated]" -ForegroundColor Yellow }

# ── Optional monitor window ────────────────────────────────────────────────────
if ($Monitor) {
    $monitorScript = Join-Path $PSScriptRoot "monitor.ps1"
    if (-not (Test-Path $monitorScript)) {
        Write-Host "[monitor] Script not found at $monitorScript — skipping" -ForegroundColor Yellow
    } else {
        try {
            Start-Process pwsh -ArgumentList "-NoExit", "-File", $monitorScript -ErrorAction Stop
            Write-Host "[monitor] Launched in new terminal" -ForegroundColor DarkCyan
        } catch {
            Write-Host "[monitor] Failed to launch: $_" -ForegroundColor Yellow
        }
    }
}

# ── PID tracking for clean abort ───────────────────────────────────────────────
$script:ActiveProcId = $null

function Invoke-Cleanup {
    if (-not $script:ActiveProcId) { return }
    Write-Host ""
    Write-Host "── Aborting — killing subprocess tree (PID $script:ActiveProcId) ──" -ForegroundColor Red
    taskkill /F /T /PID $script:ActiveProcId 2>$null | Out-Null
    $script:ActiveProcId = $null

    # Poll until VRAM drains or 15s timeout
    Write-Host "   Waiting for VRAM to release..." -ForegroundColor Yellow -NoNewline
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $free = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null).Trim()
        if ([int]$free -gt 10000) { break }  # >10 GB free = model unloaded
        Write-Host "." -NoNewline -ForegroundColor Yellow
    }
    $vram = (nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader,nounits 2>$null) -split ","
    Write-Host ""
    Write-Host "   VRAM free: $($vram[0].Trim()) MiB  used: $($vram[1].Trim()) MiB" -ForegroundColor Cyan
}

# ── Helpers ────────────────────────────────────────────────────────────────────
function Get-OutputDir([string]$absInputDir) {
    if ($absInputDir.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rel = $absInputDir.Substring($SourceRoot.Length).TrimStart('\', '/')
        if ($rel) { return "$OutputRoot\$rel" }
        return $OutputRoot
    }
    return "$OutputRoot\$([System.IO.Path]::GetFileName($absInputDir))"
}

function Get-ImageFiles([string]$dir) {
    return Get-ChildItem -LiteralPath $dir -File |
        Where-Object { $ImageExt -contains $_.Extension.ToLower() }
}

function Invoke-Python([string[]]$pyArgs) {
    # Quote args that contain spaces so Start-Process doesn't split them
    $argStr = ($pyArgs | ForEach-Object {
        if ($_ -match ' ') { "`"$_`"" } else { $_ }
    }) -join ' '
    $proc = Start-Process `
        -FilePath $Python `
        -ArgumentList $argStr `
        -NoNewWindow `
        -PassThru
    $script:ActiveProcId = $proc.Id
    $proc.WaitForExit()
    $script:ActiveProcId = $null
    return $proc.ExitCode
}

function Invoke-DirBatch([string]$dir, [System.IO.FileInfo[]]$images) {
    $outDir = Get-OutputDir $dir

    Write-Host ""
    Write-Host "==> $dir" -ForegroundColor Cyan
    Write-Host "    -> $outDir" -ForegroundColor DarkCyan
    foreach ($img in $images) {
        Write-Host "    $($img.Name)" -ForegroundColor Gray
    }

    if ($DryRun) { return }

    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    # Pass individual files when a subset is selected; use --folder for the full set
    if ((Get-ImageFiles $dir).Count -eq $images.Count) {
        $pyArgs = @("src/make_mannequin.py", "--folder", $dir,
                    "--output-dir", $outDir, "--profile", $PipelineProfile, "--log-level", $LogLevel)
    } else {
        $pyArgs = @("src/make_mannequin.py") + ($images | ForEach-Object { $_.FullName }) +
                  @("--output-dir", $outDir, "--profile", $PipelineProfile, "--log-level", $LogLevel)
    }

    $exitCode = Invoke-Python $pyArgs
    if ($exitCode -ne 0) {
        Write-Host "[FAILED] $dir (exit $exitCode)" -ForegroundColor Red
    }
}

# ── Main (wrapped in try/finally for clean Ctrl+C) ─────────────────────────────
try {
    if ($isFile) {
        $dir    = [System.IO.Path]::GetDirectoryName($ResolvedSource)
        $outDir = Get-OutputDir $dir

        Write-Host ""
        Write-Host "==> Single file: $ResolvedSource" -ForegroundColor Cyan
        Write-Host "    -> $outDir"                   -ForegroundColor DarkCyan

        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $outDir -Force | Out-Null
            Invoke-Python @(
                "src/make_mannequin.py",
                $ResolvedSource,
                "--output-dir", $outDir,
                "--profile", $PipelineProfile,
                "--log-level", $LogLevel
            ) | Out-Null
        }

    } else {
        $dirs = @($ResolvedSource) + @(
            Get-ChildItem -LiteralPath $ResolvedSource -Recurse -Directory |
                Where-Object { $_.FullName -notlike "$OutputRoot*" -and $_.Name -notlike "yogamann-output*" } |
                Select-Object -ExpandProperty FullName
        )

        $processed = 0; $skipped = 0; $totalImages = 0
        $remaining = $Limit   # 0 = unlimited

        foreach ($d in $dirs) {
            if ($Limit -gt 0 -and $remaining -le 0) { break }

            $imgs = Get-ImageFiles $d
            if (-not $imgs) { $skipped++; continue }

            # Trim to remaining budget if needed
            $batch = if ($Limit -gt 0 -and $imgs.Count -gt $remaining) {
                $imgs | Select-Object -First $remaining
            } else { $imgs }

            Invoke-DirBatch $d $batch
            $processed++
            $totalImages += $batch.Count
            if ($Limit -gt 0) { $remaining -= $batch.Count }
        }

        $limitNote = if ($Limit -gt 0) { "  (limit: $Limit)" } else { "" }
        Write-Host ""
        Write-Host "── Batch complete ──────────────────────────────" -ForegroundColor Green
        Write-Host "   Dirs with images : $processed"
        Write-Host "   Total images     : $totalImages$limitNote"
        Write-Host "   Dirs skipped     : $skipped (no images)"
    }

} finally {
    Invoke-Cleanup
}
