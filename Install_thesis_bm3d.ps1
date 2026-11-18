# Official legacy implementation. Kept local because network redistribution is restricted.
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$target = Join-Path $taskRoot 'experiments\third_party'
[void][IO.Directory]::CreateDirectory($target)
$archive = Join-Path $target 'BM3D.zip'
$expected = '9E48A8152AB7C9CE69C18F0D0EAFFFB074BB0AA5696D86CF4F5C397A40C903A3'
if (-not (Test-Path -LiteralPath $archive)) {
    & curl.exe -L --fail --max-time 60 'https://webpages.tuni.fi/foi/GCF-BM3D/BM3D.zip' -o $archive
    if ($LASTEXITCODE -ne 0) { throw 'Official BM3D download failed.' }
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
    throw 'BM3D archive differs from the reviewed version. Inspect the update before use.'
}
Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $target 'bm3d') -Force
Write-Output 'Installed official legacy BM3D v2.01 locally. Read experiments/third_party/bm3d/LICENSE.'
