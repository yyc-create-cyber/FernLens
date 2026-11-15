param(
    [int]$Port = 8642
)

$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host "[info] no server listening on port $Port"
    exit 0
}

$conn.OwningProcess | Select-Object -Unique | ForEach-Object {
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    Write-Host "[stopped] PID $_"
}
exit 0
