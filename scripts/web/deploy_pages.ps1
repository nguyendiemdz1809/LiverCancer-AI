# Đưa giao diện LiverCancer AI (web/dist) lên Cloudflare Pages bằng Wrangler Direct Upload. Xem docs/DEPLOY_CLOUDFLARE.md.
#   powershell -ExecutionPolicy Bypass -File scripts\web\deploy_pages.ps1                       (giữ nguyên, chưa có máy chủ AI)
#   powershell -ExecutionPolicy Bypass -File scripts\web\deploy_pages.ps1 -Rebuild              (build lại giao diện trước)
#   powershell -ExecutionPolicy Bypass -File scripts\web\deploy_pages.ps1 -ApiBase https://x.trycloudflare.com
# Chỉ HTML/CSS/JS/SVG và runtime-config.json (chỉ địa chỉ máy chủ AI, không khóa/bí mật nào) được tải lên.
# Script dừng nếu thư mục tải lên có checkpoint, ảnh y khoa, .env, mã bệnh nhân hoặc localhost.
param([string]$ApiBase = "", [switch]$Rebuild, [switch]$Quiet)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$cfg = Get-Content (Join-Path $root "configs\web_deploy.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$env:WRANGLER_SEND_METRICS = "false"

function Say([string]$m, [string]$c = "Gray") { if (-not $Quiet) { Write-Host $m -ForegroundColor $c } }

# Native programs write progress to stderr; in Windows PowerShell 5.1 that must not count as a failure.
function Invoke-Native([string]$exe, [string[]]$argv) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $o = & $exe @argv 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $old }
    return [pscustomobject]@{ Code = $LASTEXITCODE; Text = (($o -join "`n") -replace '\x1b\[[0-9;]*m', '') }
}

if ($ApiBase -and $ApiBase -notmatch '^https://[a-z0-9.-]+(:\d+)?/?$') { throw "ApiBase phải là địa chỉ https:// (không có đường dẫn): $ApiBase" }

$wr = Join-Path $root "web\node_modules\.bin\wrangler.cmd"
if (-not (Test-Path $wr) -or $Rebuild -or -not (Test-Path "web\dist\index.html")) {
    Push-Location web
    try {
        if (-not (Test-Path $wr)) { Say "Cài thư viện giao diện (npm install)..." Cyan; $r = Invoke-Native "npm.cmd" @("install", "--no-audit", "--no-fund"); if ($r.Code) { throw $r.Text } }
        if ($Rebuild -or -not (Test-Path "dist\index.html")) { Say "Build giao diện (web\dist)..." Cyan; $r = Invoke-Native "npm.cmd" @("run", "build"); if ($r.Code) { throw $r.Text } }
    } finally { Pop-Location }
}

# ---- thư mục tải lên = bản sao web/dist + runtime-config.json
$stage = Join-Path $root "outputs\web_deploy\pages"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item (Join-Path $root "web\dist\*") $stage -Recurse
$conf = [ordered]@{ api_base = $(if ($ApiBase) { $ApiBase.TrimEnd("/") } else { $null }); updated_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") }
[IO.File]::WriteAllText((Join-Path $stage "runtime-config.json"), ($conf | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))

# ---- kiểm tra an toàn trước khi tải lên
$allowed = @(".html", ".js", ".css", ".svg", ".json", ".ico", ".png", ".webp", ".woff2", ".txt")
$bad = @()
foreach ($f in Get-ChildItem $stage -Recurse -File) {
    if ($f.Name -in @("_headers", "_redirects")) { continue }
    if ($f.Extension.ToLower() -notin $allowed) { $bad += "loại tệp không được phép: $($f.Name)" }
    if ($f.Length -gt 5MB) { $bad += "tệp quá lớn ($([math]::Round($f.Length / 1MB, 1)) MB): $($f.Name)" }
    if ($f.Name -match '\.(pt|pth|onnx|nii|gz|dcm|zip|env|pem|key)$|^\.env') { $bad += "tệp nhạy cảm: $($f.Name)" }
}
$hits = Get-ChildItem $stage -Recurse -File -Include *.js, *.json, *.html, *.css, *.txt |
    Select-String -Pattern 'LCAI_ACCESS_TOKEN|X-Access-Token|PRIVATE KEY|\bP0\d{3}\b|\bMR-\d{5,}|localhost|127\.0\.0\.1' -AllMatches
foreach ($h in $hits) { $bad += "nội dung bị chặn '$($h.Matches[0].Value)' trong $($h.Filename)" }
if ($bad) { $bad | ForEach-Object { Write-Host "  ✗ $_" -ForegroundColor Red }; throw "Dừng: thư mục tải lên có nội dung không được phép." }
$n = (Get-ChildItem $stage -Recurse -File).Count
Say "Kiểm tra an toàn: $n tệp, chỉ giao diện tĩnh, không checkpoint/dữ liệu/mã bí mật." Green

# ---- tải lên Cloudflare Pages (Direct Upload)
# ---- đúng tài khoản Cloudflare (không deploy nhầm tài khoản khác đang đăng nhập)
$env:CLOUDFLARE_ACCOUNT_ID = $cfg.cloudflare_account_id
$w = Invoke-Native $wr @("whoami")
if ($w.Text -notmatch [regex]::Escape($cfg.cloudflare_account_id)) {
    Write-Host ($w.Text -split "`n" | Select-String "logged|Account" | Out-String)
    throw "Wrangler không đăng nhập vào tài khoản Cloudflare $($cfg.cloudflare_account_id) (configs/web_deploy.json). Chạy: cd web; npx wrangler login  và cấp quyền bằng đúng tài khoản đó."
}
Say "Tài khoản Cloudflare: $($cfg.cloudflare_account_id)" Green
Say "Đang tải giao diện lên Cloudflare Pages ($($cfg.pages_project))..." Cyan
$r = Invoke-Native $wr @("pages", "deploy", $stage, "--project-name", $cfg.pages_project, "--branch", "main",
    "--commit-dirty=true", "--commit-message", "LiverCancer AI web")
if ($r.Code) {
    Write-Host $r.Text
    if ($r.Text -match "login|auth|Authentication|not logged") { Write-Host "Wrangler chưa đăng nhập. Chạy: cd web; npx wrangler login  (đăng nhập bằng tài khoản Cloudflare của bạn trên trình duyệt)." -ForegroundColor Yellow }
    throw "Không tải được giao diện lên Cloudflare Pages."
}
$pagesHost = ([uri]$cfg.pages_url).Host
$dep = [regex]::Match($r.Text, "https://[a-z0-9]+\.$([regex]::Escape($pagesHost))").Value
if (-not $dep) { Write-Host $r.Text; throw "Wrangler không trả về địa chỉ deployment của $pagesHost." }

# ---- xác nhận deployment có trong danh sách của project, môi trường Production
$depShort = ([uri]$dep).Host.Split(".")[0]
$l = Invoke-Native $wr @("pages", "deployment", "list", "--project-name", $cfg.pages_project, "--environment", "production")
$row = ($l.Text -split "`n") | Where-Object { $_ -match [regex]::Escape($depShort) } | Select-Object -First 1
$depId = [regex]::Match("$row", "$depShort-[0-9a-f-]{27}").Value
if (-not $depId -or "$row" -notmatch "Production") { Write-Host $l.Text; throw "Không thấy deployment $depShort ở môi trường Production của $($cfg.pages_project)." }
Say "Deployment Production: $depId" Green

# ---- xác nhận bản production đã nhận cấu hình mới
$ok = $false
for ($i = 0; $i -lt 20 -and -not $ok; $i++) {
    try {
        $live = Invoke-RestMethod "$($cfg.pages_url)/runtime-config.json?t=$([guid]::NewGuid().ToString('N'))" -Headers @{ "Cache-Control" = "no-cache" } -TimeoutSec 15
        $ok = ($live.updated_utc -eq $conf.updated_utc)
    } catch { }
    if (-not $ok) { Start-Sleep -Seconds 3 }
}
if (-not $ok) { Write-Host "Cảnh báo: chưa thấy cấu hình mới trên $($cfg.pages_url) (Cloudflare có thể cần thêm ít giây)." -ForegroundColor Yellow }
Say "Đã triển khai: $($cfg.pages_url)   (bản này: $dep)" Green
[pscustomobject]@{ AccountId = $cfg.cloudflare_account_id; Project = $cfg.pages_project; DeploymentId = $depId; PagesUrl = $cfg.pages_url; DeploymentUrl = $dep; ApiBase = $conf.api_base; Live = $ok }
