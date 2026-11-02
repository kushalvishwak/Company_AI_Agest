$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($env:GROQ_API_KEY)) {
    $secureKey = Read-Host "Groq API key (input is hidden)" -AsSecureString
    $env:GROQ_API_KEY = [System.Net.NetworkCredential]::new(
        "",
        $secureKey
    ).Password
}

$env:GROQ_BASE_URL = "https://api.groq.com/openai/v1"
$env:GROQ_MODEL = "openai/gpt-oss-120b"
$env:KUSHAL_AI_PROJECT_ROOT = $PSScriptRoot

$app = Join-Path $PSScriptRoot "dist\kushal.ai\kushal.ai.exe"
if (-not (Test-Path $app)) {
    throw "Desktop app not built. Run .\build_desktop.ps1 first."
}

Start-Process -FilePath $app -WorkingDirectory $PSScriptRoot -Wait
