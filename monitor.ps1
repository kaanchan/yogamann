param([int]$IntervalMs = 1500)

$ESC = [char]27

# ── Cursor / drawing helpers ────────────────────────────────────────────────────
function Move-Up([int]$n) {
    for ($i = 0; $i -lt $n; $i++) {
        Write-Host "${ESC}[2K${ESC}[1A" -NoNewline
    }
    Write-Host "${ESC}[2K" -NoNewline
}

function Get-Bar([double]$pct, [int]$width = 20) {
    $filled = [Math]::Round($pct / 100 * $width)
    $empty  = $width - [Math]::Max(0, $filled)
    $bar    = ('█' * [Math]::Max(0, $filled)) + ('░' * $empty)
    $label  = "$([math]::Round($pct, 1))%".PadLeft(6)
    return "[$bar] $label"
}

function Write-Metric([string]$label, [double]$pct, [string]$extra = "") {
    $color = if ($pct -gt 90) { "Red" } elseif ($pct -gt 70) { "Yellow" } else { "Green" }
    Write-Host "$($label.PadRight(6))" -NoNewline -ForegroundColor DarkGray
    Write-Host (Get-Bar $pct) -NoNewline -ForegroundColor $color
    if ($extra) { Write-Host "  $extra" -NoNewline -ForegroundColor DarkCyan }
    Write-Host ""
}

# ── Per-process CPU% tracking ───────────────────────────────────────────────────
$script:CpuSamples = @{}   # PID -> [ticks, DateTime]

function Get-ProcCpuPct([CimInstance]$proc) {
    $procId   = $proc.ProcessId
    $ticks    = $proc.KernelModeTime + $proc.UserModeTime
    $now      = [DateTime]::UtcNow
    $pct      = 0.0

    if ($script:CpuSamples.ContainsKey($procId)) {
        $last       = $script:CpuSamples[$procId]
        $deltaTicks = $ticks - $last[0]
        $deltaSec   = ($now - $last[1]).TotalSeconds
        if ($deltaSec -gt 0) {
            $pct = [Math]::Round($deltaTicks / ($deltaSec * 1e7) * 100, 1)
        }
    }
    $script:CpuSamples[$procId] = @($ticks, $now)
    return $pct
}

function Get-ScriptName([string]$cmdLine) {
    if ($cmdLine -match 'sd_make\.py')          { return 'sd_make.py        ' }
    if ($cmdLine -match 'make_mannequin\.py')    { return 'make_mannequin.py ' }
    if ($cmdLine -match 'make_gallery\.py')      { return 'make_gallery.py   ' }
    return 'python            '
}

# ── Main loop ───────────────────────────────────────────────────────────────────
$linesOut = 0

Write-Host ""
Write-Host "  yogamann monitor  (Ctrl+C to stop)" -ForegroundColor White
Write-Host ""

while ($true) {
    # GPU
    $gpuRaw = & nvidia-smi `
        --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit `
        --format=csv,noheader,nounits 2>$null

    # CPU
    $cpuPct = (Get-CimInstance Win32_Processor |
        Measure-Object -Property LoadPercentage -Average).Average

    # RAM
    $os         = Get-CimInstance Win32_OperatingSystem
    $ramUsedGB  = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    $ramTotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $ramPct     = $ramUsedGB / $ramTotalGB * 100

    # Yogamann processes — match by command line
    $yogaProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'make_mannequin|sd_make|make_gallery' }

    # ── Overwrite previous block ──────────────────────────────────────────────
    if ($linesOut -gt 0) { Move-Up $linesOut }
    $lineCount = 0

    # Header
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "── $ts ─────────────────────────────────────────" -ForegroundColor DarkGray
    $lineCount++

    # System
    Write-Metric "CPU  " $cpuPct
    Write-Metric "RAM  " $ramPct "${ramUsedGB}/${ramTotalGB} GB"
    $lineCount += 2

    # GPU
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
        $pwr    = if ($p.Count -ge 7 -and [double]::TryParse($p[6].Trim(), [ref]$pwrD))    { "$([math]::Round($pwrD))W" }   else { "?W" }
        $pwrLim = if ($p.Count -ge 8 -and [double]::TryParse($p[7].Trim(), [ref]$pwrLimD)) { "/$([math]::Round($pwrLimD))W" } else { "" }

        Write-Host "  $gpuName" -ForegroundColor White
        Write-Metric "  GPU " $gpuUtil "${temp}°C  ${pwr}${pwrLim}"
        Write-Metric " VRAM " $memPct "${memUsed}/${memTotal} MiB"
        $lineCount += 3
    }

    # Yogamann processes
    Write-Host "── yogamann procs ──────────────────────────────" -ForegroundColor DarkGray
    $lineCount++

    if (-not $yogaProcs) {
        Write-Host "  (none running)" -ForegroundColor DarkGray
        $lineCount++
    } else {
        foreach ($proc in $yogaProcs) {
            $cpuP   = Get-ProcCpuPct $proc
            $ramMB  = [math]::Round($proc.WorkingSetSize / 1MB, 0)
            $script = Get-ScriptName $proc.CommandLine
            $cpuColor = if ($cpuP -gt 80) { "Green" } elseif ($cpuP -gt 10) { "Yellow" } else { "DarkGray" }

            Write-Host "  PID " -NoNewline -ForegroundColor DarkGray
            Write-Host "$($proc.ProcessId)  " -NoNewline -ForegroundColor Gray
            Write-Host $script -NoNewline -ForegroundColor White
            Write-Host "CPU " -NoNewline -ForegroundColor DarkGray
            Write-Host "$("$cpuP%".PadLeft(7))  " -NoNewline -ForegroundColor $cpuColor
            Write-Host "RAM " -NoNewline -ForegroundColor DarkGray
            Write-Host "${ramMB} MB" -ForegroundColor Cyan
            $lineCount++
        }
    }

    $linesOut = $lineCount
    Start-Sleep -Milliseconds $IntervalMs
}
