# =============================================================================
#  Render every .mmd file under figures/src to PNG via https://mermaid.ink
#  No install required (uses Invoke-WebRequest).
#  Run:   powershell -ExecutionPolicy Bypass -File .\render_all_online.ps1
# =============================================================================
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $here 'src'
$out  = $here

$files = Get-ChildItem -Path $src -Filter '*.mmd' | Sort-Object Name
Write-Host "Found $($files.Count) Mermaid source files."

# Force TLS 1.2 (some default PS5 setups still pick TLS 1.0)
[Net.ServicePointManager]::SecurityProtocol =
    [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13

$ok = 0; $fail = 0

foreach ($f in $files) {
    $name = $f.BaseName
    $png  = Join-Path $out "$name.png"
    $mmd  = Get-Content -Path $f.FullName -Raw -Encoding UTF8

    # mermaid.ink expects the diagram source as URL-safe base64.
    $bytes = [Text.Encoding]::UTF8.GetBytes($mmd)
    $b64   = [Convert]::ToBase64String($bytes)
    $b64   = $b64.TrimEnd('=').Replace('+','-').Replace('/','_')

    $url = "https://mermaid.ink/img/$b64`?type=png&bgColor=FFFFFF"
    Write-Host "[mermaid.ink] $($f.Name) -> $name.png"

    try {
        Invoke-WebRequest -Uri $url -OutFile $png `
            -UserAgent 'Mozilla/5.0' -TimeoutSec 60
        $size = (Get-Item $png).Length
        Write-Host "    ok  ($size bytes)"
        $ok++
    } catch {
        Write-Host "    FAIL  $($_.Exception.Message)"
        $fail++
    }
}

Write-Host ""
Write-Host "Done. ok=$ok  fail=$fail  out=$out"
