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
    [switch]$DryRun
)

$Python   = ".\.venv\Scripts\python.exe"
$ImageExt = @(".jpg", ".jpeg", ".png", ".webp")

$env:YOGAMANN_LOG_LEVEL = $LogLevel
$env:HF_HUB_OFFLINE     = "1"

# ── Resolve paths ──────────────────────────────────────────────────────────────
$ResolvedSource = (Resolve-Path $Source -ErrorAction Stop).Path.TrimEnd('\', '/')
$SourceRoot     = $SourceRoot.TrimEnd('\', '/')
$OutputRoot     = $OutputRoot.TrimEnd('\', '/')

$isFile = Test-Path $ResolvedSource -PathType Leaf

Write-Host ""
Write-Host "Source     : $ResolvedSource" -ForegroundColor White
Write-Host "SourceRoot : $SourceRoot"     -ForegroundColor DarkGray
Write-Host "OutputRoot : $OutputRoot"     -ForegroundColor DarkGray
if ($DryRun) { Write-Host "[DRY RUN — no images will be generated]" -ForegroundColor Yellow }

# ── Helpers ────────────────────────────────────────────────────────────────────
function Get-OutputDir([string]$absInputDir) {
    if ($absInputDir.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rel = $absInputDir.Substring($SourceRoot.Length).TrimStart('\', '/')
        if ($rel) { return "$OutputRoot\$rel" }
        return $OutputRoot
    }
    # Input is outside SourceRoot — map by leaf name
    return "$OutputRoot\$([System.IO.Path]::GetFileName($absInputDir))"
}

function Get-ImageFiles([string]$dir) {
    return Get-ChildItem -LiteralPath $dir -File |
        Where-Object { $ImageExt -contains $_.Extension.ToLower() }
}

function Invoke-DirBatch([string]$dir) {
    $images = Get-ImageFiles $dir
    if (-not $images) { return }

    $outDir = Get-OutputDir $dir

    Write-Host ""
    Write-Host "==> $dir" -ForegroundColor Cyan
    Write-Host "    -> $outDir" -ForegroundColor DarkCyan
    foreach ($img in $images) {
        Write-Host "    $($img.Name)" -ForegroundColor Gray
    }

    if ($DryRun) { return }

    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    $pyArgs = @(
        "src/make_mannequin.py",
        "--folder", $dir,
        "--output-dir", $outDir,
        "--profile", $PipelineProfile,
        "--log-level", $LogLevel
    )
    & $Python @pyArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] $dir (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

# ── Main ───────────────────────────────────────────────────────────────────────
if ($isFile) {
    $dir    = [System.IO.Path]::GetDirectoryName($ResolvedSource)
    $outDir = Get-OutputDir $dir

    Write-Host ""
    Write-Host "==> Single file: $ResolvedSource" -ForegroundColor Cyan
    Write-Host "    -> $outDir"                   -ForegroundColor DarkCyan

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        $pyArgs = @(
            "src/make_mannequin.py",
            $ResolvedSource,
            "--output-dir", $outDir,
            "--profile", $PipelineProfile,
            "--log-level", $LogLevel
        )
        & $Python @pyArgs
    }

} else {
    $dirs = @($ResolvedSource) + @(
        Get-ChildItem -LiteralPath $ResolvedSource -Recurse -Directory |
            Where-Object { $_.FullName -notlike "$OutputRoot*" -and $_.Name -notlike "yogamann-output*" } |
            Select-Object -ExpandProperty FullName
    )

    $processed = 0; $skipped = 0; $totalImages = 0

    foreach ($d in $dirs) {
        $imgs = Get-ImageFiles $d
        if ($imgs) {
            Invoke-DirBatch $d
            $processed++
            $totalImages += $imgs.Count
        } else {
            $skipped++
        }
    }

    Write-Host ""
    Write-Host "── Batch complete ──────────────────────────────" -ForegroundColor Green
    Write-Host "   Dirs with images : $processed"
    Write-Host "   Total images     : $totalImages"
    Write-Host "   Dirs skipped     : $skipped (no images)"
}
