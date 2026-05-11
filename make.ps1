param(
    [string]$Target   = "test",
    [string]$LogLevel = "INFO"
)

$Python  = ".\.venv\Scripts\python.exe"
$Sample  = "input\yoga-pose-sample-4.jpg"
$Profile = "yoga_asana"
$env:YOGAMANN_LOG_LEVEL = $LogLevel

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Host "`n==> $Label" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[FAILED] $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    "test" {
        Invoke-Step "Generate mannequin: $Sample" {
            & $Python src/make_mannequin.py $Sample --profile $Profile --log-level $LogLevel
        }
        Invoke-Step "Build gallery" { & $Python src/make_gallery.py }
        Start-Process "out\index.html"
    }
    "test-all" {
        Invoke-Step "Generate mannequins: all input images" {
            & $Python src/make_mannequin.py --folder input --profile $Profile --log-level $LogLevel
        }
        Invoke-Step "Build gallery" { & $Python src/make_gallery.py }
        Start-Process "out\index.html"
    }
    "diag" {
        Invoke-Step "Hardware + import diagnostics" {
            & $Python src/diagnostics/rtx5080-test.py
        }
    }
    "gallery" {
        Invoke-Step "Build gallery" { & $Python src/make_gallery.py }
        Start-Process "out\index.html"
    }
    "open" {
        Start-Process "out\index.html"
    }
    default {
        Write-Host "Usage: .\run.ps1 [-Target <target>] [-LogLevel <level>]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Targets:"
        Write-Host "  test      single image + open gallery (default)"
        Write-Host "  test-all  all input images + open gallery"
        Write-Host "  diag      hardware + import diagnostics"
        Write-Host "  gallery   regenerate HTML gallery + open browser"
        Write-Host "  open      open existing gallery"
        Write-Host ""
        Write-Host "Log levels: DEBUG  INFO  WARNING  ERROR  (default: INFO)"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\run.ps1                          # default test, INFO logging"
        Write-Host "  .\run.ps1 -LogLevel DEBUG          # verbose output"
        Write-Host "  .\run.ps1 -Target test-all         # all images"
    }
}
