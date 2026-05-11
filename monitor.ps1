param([int]$IntervalMs = 1500)

$ESC = [char]27

function Move-Up([int]$n) {
    for ($i = 0; $i -lt $n; $i++) {
        Write-Host "${ESC}[2K${ESC}[1A" -NoNewline
    }
    Write-Host "${ESC}[2K" -NoNewline
}

function Get-Bar([double]$pct, [int]$width = 24) {
    $filled = [Math]::Round($pct / 100 * $width)
    $empty  = $width - [Math]::Max(0, $filled)
    $bar    = ('█' * [Math]::Max(0, $filled)) + ('░' * $empty)
    $label  = "$([math]::Round($pct, 1))%".PadLeft(6)
    return "[$bar] $label"
}

function Write-Metric([string]$label, [double]$pct, [string]$extra = "") {
    $bar = Get-Bar $pct
    $color = if ($pct -gt 90) { "Red" } elseif ($pct -gt 70) { "Yellow" } else { "Green" }
    Write-Host "$($label.PadRight(5))" -NoNewline -ForegroundColor DarkGray
    Write-Host $bar -NoNewline -ForegroundColor $color
    if ($extra) { Write-Host "  $extra" -NoNewline -ForegroundColor DarkCyan }
    Write-Host ""
}

$linesOut = 0

Write-Host ""
Write-Host "  yogamann monitor  (Ctrl+C to stop)" -ForegroundColor White
Write-Host ""

while ($true) {
    # ── GPU (nvidia-smi) ──────────────────────────────────────────────────────
    $gpuRaw = & nvidia-smi `
        --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit `
        --format=csv,noheader,nounits 2>$null

    # ── CPU ───────────────────────────────────────────────────────────────────
    $cpuPct = (Get-CimInstance Win32_Processor |
        Measure-Object -Property LoadPercentage -Average).Average

    # ── RAM ───────────────────────────────────────────────────────────────────
    $os       = Get-CimInstance Win32_OperatingSystem
    $ramUsedGB  = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    $ramTotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $ramPct     = $ramUsedGB / $ramTotalGB * 100

    # ── Overwrite previous block ──────────────────────────────────────────────
    if ($linesOut -gt 0) { Move-Up $linesOut }

    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "── $ts " -NoNewline -ForegroundColor DarkGray
    Write-Host "─────────────────────────────────────────" -ForegroundColor DarkGray

    Write-Metric "CPU " $cpuPct
    Write-Metric "RAM " $ramPct "${ramUsedGB}/${ramTotalGB} GB"

    $gpuLines = 0
    foreach ($gpuLine in ($gpuRaw -split "`n" | Where-Object { $_.Trim() })) {
        $p = $gpuLine -split ","
        if ($p.Count -lt 7) { continue }

        $gpuName  = $p[0].Trim()
        $gpuUtil  = [double]($p[1].Trim())
        $memUsed  = [int]($p[3].Trim())
        $memTotal = [int]($p[4].Trim())
        $memPct   = if ($memTotal -gt 0) { $memUsed / $memTotal * 100 } else { 0 }
        $temp     = $p[5].Trim()
        $pwrD = 0.0; $pwrLimD = 0.0
        $pwr    = if ($p.Count -ge 7 -and [double]::TryParse($p[6].Trim(), [ref]$pwrD))   { [math]::Round($pwrD) }   else { "?" }
        $pwrLim = if ($p.Count -ge 8 -and [double]::TryParse($p[7].Trim(), [ref]$pwrLimD)) { [math]::Round($pwrLimD) } else { "?" }

        Write-Host "  $gpuName" -ForegroundColor White
        Write-Metric "  GPU" $gpuUtil "${temp}°C  ${pwr}/${pwrLim}W"
        Write-Metric " VRAM" $memPct "${memUsed}/${memTotal} MB"
        $gpuLines += 3
    }

    # header + CPU + RAM + GPU block
    $linesOut = 3 + $gpuLines

    Start-Sleep -Milliseconds $IntervalMs
}
