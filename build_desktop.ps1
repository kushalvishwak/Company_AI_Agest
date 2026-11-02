$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

py -3.11 -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "kushal.ai" `
    --icon "$PSScriptRoot\examples\web\kushal_ai.ico" `
    --add-data "$PSScriptRoot\examples\web;examples\web" `
    --add-data "$PSScriptRoot\data\hr_docs;data\hr_docs" `
    --collect-all webview `
    --collect-all clr_loader `
    --collect-all pythonnet `
    --exclude-module PIL `
    --exclude-module numpy `
    --distpath "dist" `
    --workpath "build\pyinstaller" `
    --specpath "build\pyinstaller" `
    "examples\web_app.py"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
