# Renders every .mmd file under figures/src into a PNG inside figures/
# Run with:  powershell -ExecutionPolicy Bypass -File .\render_all.ps1

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $here 'src'
$out  = $here
$config = Join-Path $here 'puppeteer-config.json'

if (-not (Test-Path $config)) {
    @{ args = @('--no-sandbox','--disable-setuid-sandbox') } |
        ConvertTo-Json | Set-Content -Path $config -Encoding UTF8
}

Get-ChildItem -Path $src -Filter '*.mmd' | ForEach-Object {
    $in  = $_.FullName
    $png = Join-Path $out ($_.BaseName + '.png')
    Write-Host "[mmdc] $($_.Name) -> $($_.BaseName).png"
    npx --yes -p @mermaid-js/mermaid-cli mmdc `
        -i $in `
        -o $png `
        -b white `
        -w 1600 `
        -s 2 `
        -p $config
}
Write-Host ""
Write-Host "All diagrams rendered into:  $out"
