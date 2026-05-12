param(
    [int]$IntervalMs      = 1500,  # poll interval
    [int]$GpuThreshold    = 95,    # % above which GPU counts as active
    [int]$DipToleranceSec = 10     # seconds below threshold before counted as a real pause
)

$ESC       = [char]27
$StartTime = Get-Date
$IntervalSec = $IntervalMs / 1000.0

# ── GPU active-time tracker ────────────────────────────────────────────────────
# Dip buffer: time below threshold is held pending. If GPU recovers before
# tolerance expires it is committed as active; if not it is discarded (real pause).
$script:GpuActiveSecs = 0.0   # confirmed active seconds
$script:DipBufSecs    = 0.0   # pending below-threshold seconds

function Update-GpuTimer([double]$gpuPct) {
    if ($gpuPct -ge $GpuThreshold) {
        if ($script:DipBufSecs -le $DipToleranceSec) {
            # short dip — commit buffer + this interval as active
            $script:GpuActiveSecs += $script:DipBufSecs + $IntervalSec
        } else {
            # real pause just ended — only count this interval
            $script:GpuActiveSecs += $IntervalSec
        }
        $script:DipBufSecs = 0.0
    } else {
        $script:DipBufSecs += $IntervalSec
        if ($script:DipBufSecs -le $DipToleranceSec) {
            # still within tolerance — count optimistically (will be confirmed when GPU returns)
            $script:GpuActiveSecs += $IntervalSec
        }
        # beyond tolerance: stop adding until GPU recovers
    }
}

function Format-Dur([double]$secs) {
    $s = [int]$secs
    $h = [int]($s / 3600); $m = [int](($s % 3600) / 60); $s = $s % 60
    if ($h -gt 0) { return "${h}h ${m}m ${s}s" }
    if ($m -gt 0) { return "${m}m ${s}s" }
    return "${s}s"
}

# ── Cursor / drawing helpers ────────────────────────────────────────────────────
function Move-Up([int]$n) {
    for ($i = 0; $i -lt $n; $i++) { Write-Host "${ESC}[2K${ESC}[1A" -NoNewline }
    Write-Host "${ESC}[2K" -NoNewline
}

function Get-Bar([double]$pct, [int]$width = 20) {
    $filled = [Math]::Max(0, [Math]::Round($pct / 100 * $width))
    $empty  = $width - $filled
    return "[$('█' * $filled)$('░' * $empty)] $("$([math]::Round($pct,1))%".PadLeft(6))"
}

function Write-Metric([string]$label, [double]$pct, [string]$extra = "") {
    $color = if ($pct -gt 90) { "Red" } elseif ($pct -gt 70) { "Yellow" } else { "Green" }
    Write-Host "$($label.PadRight(6))" -NoNewline -ForegroundColor DarkGray
    Write-Host (Get-Bar $pct)          -NoNewline -ForegroundColor $color
    if ($extra) { Write-Host "  $extra" -NoNewline -ForegroundColor DarkCyan }
    Write-Host ""
}

function Write-Sep([string]$label) {
    Write-Host "── $label " -NoNewline -ForegroundColor DarkGray
    Write-Host ('─' * [Math]::Max(0, 46 - $label.Length)) -ForegroundColor DarkGray
}

# ── Per-process CPU% tracking ───────────────────────────────────────────────────
$script:CpuSamples = @{}

function Get-ProcCpuPct([CimInstance]$proc) {
    $procId     = $proc.ProcessId
    $ticks      = $proc.KernelModeTime + $proc.UserModeTime
    $now        = [DateTime]::UtcNow
    $pct        = 0.0
    if ($script:CpuSamples.ContainsKey($procId)) {
        $last       = $script:CpuSamples[$procId]
        $deltaTicks = $ticks - $last[0]
        $deltaSec   = ($now - $last[1]).TotalSeconds
        if ($deltaSec -gt 0) { $pct = [Math]::Round($deltaTicks / ($deltaSec * 1e7) * 100, 1) }
    }
    $script:CpuSamples[$procId] = @($ticks, $now)
    return $pct
}

function Get-ScriptName([string]$cmdLine) {
    if ($cmdLine -match 'sd_make\.py')       { return 'sd_make.py       ' }
    if ($cmdLine -match 'make_mannequin\.py') { return 'make_mannequin.py' }
    if ($cmdLine -match 'make_gallery\.py')   { return 'make_gallery.py  ' }
    return 'python           '
}

# ── Main loop ───────────────────────────────────────────────────────────────────
$linesOut = 0
Write-Host ""
Write-Host "  yogamann monitor  threshold=${GpuThreshold}%  dip-tol=${DipToleranceSec}s  (Ctrl+C to stop)" -ForegroundColor White
Write-Host ""

while ($true) {
    # Collect data
    $gpuRaw  = & nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>$null
    $cpuPct  = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
    $os      = Get-CimInstance Win32_OperatingSystem
    $ramUsedGB  = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    $ramTotalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $ramPct     = $ramUsedGB / $ramTotalGB * 100
    $yogaProcs  = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match 'make_mannequin|sd_make|make_gallery' }

    # Parse first GPU util for timer update
    $firstGpuUtil = 0.0
    $gpuLines = $gpuRaw -split "`n" | Where-Object { $_.Trim() }
    if ($gpuLines) {
        $p0 = $gpuLines[0] -split ","
        if ($p0.Count -ge 2) { [double]::TryParse($p0[1].Trim(), [ref]$firstGpuUtil) | Out-Null }
    }
    Update-GpuTimer $firstGpuUtil

    # Overwrite
    if ($linesOut -gt 0) { Move-Up $linesOut }
    $lc = 0

    # Header
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Sep $ts; $lc++

    # System
    Write-Metric "CPU  " $cpuPct
    Write-Metric "RAM  " $ramPct "${ramUsedGB}/${ramTotalGB} GB"
    $lc += 2

    # GPU rows
    foreach ($gl in $gpuLines) {
        $p = $gl -split ","
        if ($p.Count -lt 7) { continue }
        $gpuName = $p[0].Trim()
        $gpuUtil = [double]($p[1].Trim())
        $memUsed = [int]($p[3].Trim()); $memTotal = [int]($p[4].Trim())
        $memPct  = if ($memTotal -gt 0) { $memUsed / $memTotal * 100 } else { 0 }
        $temp    = $p[5].Trim()
        $pwrD = 0.0; $pwrLimD = 0.0
        $pwr    = if ($p.Count -ge 7 -and [double]::TryParse($p[6].Trim(), [ref]$pwrD))    { "$([math]::Round($pwrD))W" }   else { "?W" }
        $pwrLim = if ($p.Count -ge 8 -and [double]::TryParse($p[7].Trim(), [ref]$pwrLimD)) { "/$([math]::Round($pwrLimD))W" } else { "" }
        Write-Host "  $gpuName" -ForegroundColor White
        Write-Metric "  GPU " $gpuUtil "${temp}°C  ${pwr}${pwrLim}"
        Write-Metric " VRAM " $memPct "${memUsed}/${memTotal} MiB"
        $lc += 3
    }

    # Yogamann procs
    Write-Sep "yogamann procs"; $lc++
    if (-not $yogaProcs) {
        Write-Host "  (none running)" -ForegroundColor DarkGray; $lc++
    } else {
        foreach ($proc in $yogaProcs) {
            $cpuP     = Get-ProcCpuPct $proc
            $ramMB    = [math]::Round($proc.WorkingSetSize / 1MB, 0)
            $sname    = Get-ScriptName $proc.CommandLine
            $cpuColor = if ($cpuP -gt 80) { "Green" } elseif ($cpuP -gt 10) { "Yellow" } else { "DarkGray" }
            Write-Host "  PID " -NoNewline -ForegroundColor DarkGray
            Write-Host "$($proc.ProcessId)  " -NoNewline -ForegroundColor Gray
            Write-Host "$sname  " -NoNewline -ForegroundColor White
            Write-Host "CPU " -NoNewline -ForegroundColor DarkGray
            Write-Host "$("$cpuP%".PadLeft(7))  " -NoNewline -ForegroundColor $cpuColor
            Write-Host "RAM " -NoNewline -ForegroundColor DarkGray
            Write-Host "${ramMB} MB" -ForegroundColor Cyan
            $lc++
        }
    }

    # Timers
    Write-Sep "timers"; $lc++

    $runDur   = (Get-Date) - $StartTime
    $gpuLabel = if ($script:DipBufSecs -gt $DipToleranceSec) { "paused" } else { "active" }
    $gpuColor = if ($gpuLabel -eq "active") { "Green" } else { "Yellow" }

    Write-Host "  Run time        " -NoNewline -ForegroundColor DarkGray
    Write-Host (Format-Dur $runDur.TotalSeconds) -ForegroundColor White
    Write-Host "  GPU >${GpuThreshold}% time    " -NoNewline -ForegroundColor DarkGray
    Write-Host (Format-Dur $script:GpuActiveSecs) -NoNewline -ForegroundColor $gpuColor
    Write-Host "  [$gpuLabel  dip-tol: ${DipToleranceSec}s]" -ForegroundColor DarkGray
    $lc += 2

    $linesOut = $lc
    Start-Sleep -Milliseconds $IntervalMs
}
