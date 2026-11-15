param(
    [int]$Port = 8642
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$url = "http://127.0.0.1:$Port"
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

function Test-Server {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne(500)) { $client.Close(); return $false }
        $client.EndConnect($iar)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

# Already running: just open the browser.
if (Test-Server) {
    Start-Process $url
    exit 0
}

Set-Location $root

if (-not (Test-Path $venvPy)) {
    # First run: let run.bat create the venv, install deps and build frontend.
    Start-Process -FilePath 'cmd.exe' `
        -ArgumentList '/c', "`"$root\run.bat`" --port $Port" `
        -WorkingDirectory $root -WindowStyle Minimized
} else {
    $dist = Join-Path $root 'frontend\dist\index.html'
    if (-not (Test-Path $dist)) {
        Write-Host '[build] frontend dist missing, building...'
        Push-Location (Join-Path $root 'frontend')
        try {
            if (-not (Test-Path 'node_modules')) { npm install --no-audit --no-fund }
            npm run build
        } finally {
            Pop-Location
        }
    }

    $logDir = Join-Path $root 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Start-Process -FilePath $venvPy `
        -ArgumentList 'server.py', '--no-auth', '--port', "$Port" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'server.log') `
        -RedirectStandardError (Join-Path $logDir 'server.err.log')
}

# Wait until the server accepts connections, then open the browser.
for ($i = 0; $i -lt 120; $i++) {
    if (Test-Server) {
        Start-Process $url
        exit 0
    }
    Start-Sleep -Seconds 1
}

Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show(
    "FernLens failed to start within 120s. See logs\server.err.log",
    'FernLens Web',
    'OK',
    'Warning'
) | Out-Null
exit 1
