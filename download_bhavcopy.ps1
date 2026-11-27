# download_bhavcopy.ps1 - fetch NSE F&O UDiFF bhavcopies for the sample window
#
#   cd "C:\Users\sansk\OneDrive\Desktop\Academics\3-1\DOP"
#   powershell -ExecutionPolicy Bypass -File .\download_bhavcopy.ps1
#
# NSE rejects a bare HTTP request with 403. It accepts one that carries the
# cookies a browser would have picked up from the homepage, so the script
# visits the homepage first and reuses that session.
#
# This may stop working whenever NSE changes its site - data_sources.md
# section 3 says as much, and recommends manual download for exactly that
# reason. Treat it as a convenience: if it fails, the URLs it was trying are
# in data\raw\download_list.txt and the browser will fetch them one by one.
#
# Already-downloaded dates are skipped, so it is safe to re-run.

$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "data\raw\fo_bhavcopy"
$list = Join-Path $PSScriptRoot "data\raw\download_list.txt"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

if (-not (Test-Path $list)) { throw "Missing $list" }
$urls = Get-Content $list | Where-Object { $_.Trim() -ne "" }

$ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
$headers = @{
  "User-Agent"      = $ua
  "Accept"          = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
  "Accept-Language" = "en-US,en;q=0.9"
  "Referer"         = "https://www.nseindia.com/all-reports-derivatives"
}

Write-Host "Opening a session with nseindia.com ..." -ForegroundColor Cyan
try {
  Invoke-WebRequest -Uri "https://www.nseindia.com" -Headers $headers `
      -SessionVariable sess -UseBasicParsing -TimeoutSec 30 | Out-Null
  Invoke-WebRequest -Uri "https://www.nseindia.com/all-reports-derivatives" `
      -Headers $headers -WebSession $sess -UseBasicParsing -TimeoutSec 30 | Out-Null
  Write-Host "  session established" -ForegroundColor Green
} catch {
  Write-Host "  could not reach nseindia.com: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "  Fall back to downloading the URLs in data\raw\download_list.txt by hand."
  exit 1
}

$ok = 0; $skip = 0; $fail = @()
foreach ($url in $urls) {
  if ($url -notmatch "_(\d{8})_F_0000") { continue }
  $date = $Matches[1]
  $out  = Join-Path $dest "fo_$date.zip"

  if (Test-Path $out) {
    Write-Host ("  {0}  already have it" -f $date) -ForegroundColor DarkGray
    $skip++; continue
  }

  try {
    Invoke-WebRequest -Uri $url -Headers $headers -WebSession $sess `
        -OutFile $out -TimeoutSec 60
    $kb = [math]::Round((Get-Item $out).Length / 1KB)
    # A 403 or an error page can arrive as a small HTML file named .zip.
    # A real bhavcopy is well over 100 KB, so anything tiny is a failure.
    if ($kb -lt 100) {
      Remove-Item $out -Force
      throw "response was only $kb KB - probably an error page, not a bhavcopy"
    }
    Write-Host ("  {0}  downloaded, {1} KB" -f $date, $kb) -ForegroundColor Green
    $ok++
  } catch {
    Write-Host ("  {0}  FAILED: {1}" -f $date, $_.Exception.Message) -ForegroundColor Yellow
    $fail += $date
  }

  Start-Sleep -Milliseconds 900     # be polite to the archive
}

Write-Host ""
Write-Host ("Downloaded {0}, already had {1}, failed {2}." -f $ok, $skip, $fail.Count)
if ($fail.Count -gt 0) {
  Write-Host "Failed dates (fetch these by hand):" -ForegroundColor Yellow
  $fail | ForEach-Object {
    Write-Host ("  https://nsearchives.nseindia.com/content/fo/" +
                "BhavCopy_NSE_FO_0_0_0_${_}_F_0000.csv.zip")
  }
}
Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  python nse_data.py data\raw\fo_bhavcopy\fo_20261126.zip   # check one"
Write-Host "  python clean_options.py                                   # clean them all"
