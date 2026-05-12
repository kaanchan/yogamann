param(
    [string]$Target     = "test",
    [string]$LogLevel   = "INFO",
    [string]$OutputRoot = "D:\Temp\yogamann-output"
)

$Python  = ".\.venv\Scripts\python.exe"
$Sample  = "input\yoga-pose-sample-4.jpg"
$PipelineProfile = "yoga_asana"
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
        $env:HF_HUB_OFFLINE = "1"
        Invoke-Step "Generate mannequin: $Sample" {
            & $Python src/make_mannequin.py $Sample --profile $PipelineProfile --log-level $LogLevel
        }
        Invoke-Step "Build gallery" { & $Python src/make_gallery.py }
        Start-Process "out\index.html"
    }
    "test-all" {
        $env:HF_HUB_OFFLINE = "1"
        Invoke-Step "Generate mannequins: all input images" {
            & $Python src/make_mannequin.py --folder input --profile $PipelineProfile --log-level $LogLevel
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
    "ingest" {
        Invoke-Step "Ingest .metrics.json → yogamann.db ($OutputRoot)" {
            & $Python src/db.py --ingest --output-root $OutputRoot
        }
        Invoke-Step "DB stats" {
            & $Python src/db.py --stats --output-root $OutputRoot
        }
    }
    "review" {
        Write-Host "`n==> Launching Streamlit gallery" -ForegroundColor Cyan
        $env:HF_HUB_OFFLINE = "1"
        & ".\.venv\Scripts\streamlit.exe" run src/gallery.py -- --output-root $OutputRoot
    }
    "download" {
        if (-not $env:HF_TOKEN) {
            Write-Host "`n[WARN] HF_TOKEN not set — downloads will be slower (unauthenticated)." -ForegroundColor Yellow
            Write-Host "       Get a free read token at https://huggingface.co/settings/tokens" -ForegroundColor Yellow
            Write-Host "       Then: `$env:HF_TOKEN = 'hf_...'; .\make.ps1 -Target download`n" -ForegroundColor Yellow
        }
        Invoke-Step "Download all models (hf_transfer)" {
            & $Python src/download_models.py
        }
    }
    default {
        Write-Host "Usage: .\make.ps1 [-Target <target>] [-LogLevel <level>]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Targets:"
        Write-Host "  download  pre-fetch all HF model weights (~10 GB, one-time)"
        Write-Host "  test      single image + open gallery (default)"
        Write-Host "  test-all  all input images + open gallery"
        Write-Host "  diag      hardware + import diagnostics"
        Write-Host "  gallery   regenerate HTML gallery + open browser"
        Write-Host "  open      open existing gallery"
        Write-Host "  ingest    import .metrics.json files into yogamann.db"
        Write-Host "  review    launch Streamlit review gallery"
        Write-Host ""
        Write-Host "Log levels: DEBUG  INFO  WARNING  ERROR  (default: INFO)"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\make.ps1 -Target download        # first-time setup: cache all models"
        Write-Host "  .\make.ps1                         # run test, INFO logging"
        Write-Host "  .\make.ps1 -LogLevel DEBUG         # verbose output"
        Write-Host "  .\make.ps1 -Target test-all        # all images"
        Write-Host "  .\make.ps1 -Target ingest          # import outputs into DB"
        Write-Host "  .\make.ps1 -Target review          # open Streamlit gallery"
    }
}
