@echo off
rem ==================================================================
rem  FernLens 一键打包：构建前端 -> PyInstaller 打服务端 -> electron-builder
rem  出安装包（Setup + Portable）并复制到桌面。
rem
rem  用法：双击运行，或在仓库根目录执行  package.bat
rem  产物：desktop\release\  以及桌面上的 FernLens-Setup-*.exe / FernLens-Portable-*.exe
rem  首次运行自动创建虚拟环境并安装依赖；Defender 锁 exe 时由
rem  desktop\afterpack.cjs 自动指数退避重试，耐心等待即可。
rem ==================================================================
setlocal enabledelayedexpansion
cd /d %~dp0

echo ==========================================================
echo   FernLens 一键打包（安装包输出到桌面）
echo ==========================================================

rem ---- 0. 环境自检 / 自愈 ----
where npm >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 npm，请先安装 Node.js（https://nodejs.org）并加入 PATH
    goto :fail
)

if not exist .venv\Scripts\python.exe (
    echo [初始化] 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败：请确认已安装 Python 并加入 PATH
        goto :fail
    )
)

.venv\Scripts\python.exe -c "import fastapi, uvicorn, cv2, numpy" 2>nul
if errorlevel 1 (
    echo [初始化] 安装后端依赖...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 后端依赖安装失败，请检查网络后重试
        goto :fail
    )
)

rem onnxruntime：本地分割引擎运行时。打包前装好才能内置进 exe，
rem 否则内置模型没有推理引擎可用（会退化为云端 VLM）。失败不阻断打包。
.venv\Scripts\python.exe -c "import onnxruntime" 2>nul
if errorlevel 1 (
    echo [初始化] 安装 onnxruntime（本地分割引擎）...
    .venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check onnxruntime
    if errorlevel 1 ( echo [警告] onnxruntime 安装失败，打包后本地蒙版将退化为云端 VLM )
)

if not exist .venv\Scripts\pyinstaller.exe (
    echo [初始化] 安装 PyInstaller...
    .venv\Scripts\python.exe -m pip install pyinstaller
    if errorlevel 1 ( echo [错误] PyInstaller 安装失败 & goto :fail )
)

rem ---- 1. 前端构建 ----
echo.
echo [1/4] 构建前端（Vue3 + Vite）...
pushd frontend
if not exist node_modules (
    call npm install
    if errorlevel 1 ( popd & echo [错误] npm install 失败 & goto :fail )
)
call npm run build
if errorlevel 1 ( popd & echo [错误] 前端构建失败 & goto :fail )
popd

rem ---- 2. PyInstaller 打包服务端 sidecar ----
echo.
echo [2/4] PyInstaller 打包服务端（含 rawpy / 前端静态资源，约 1 分钟）...
.venv\Scripts\pyinstaller.exe build\fernlens-server.spec --noconfirm --distpath backend-dist
if errorlevel 1 ( echo [错误] sidecar 打包失败 & goto :fail )

echo [2/4] sidecar 冒烟测试（启动后应打出就绪行）...
set READY=
for /f "delims=" %%i in ('powershell -NoProfile -Command "$p = Start-Process -FilePath '.\backend-dist\fernlens-server\fernlens-server.exe' -ArgumentList '--no-auth','--port','0' -RedirectStandardOutput \"$env:TEMP\pl_sidecar.out\" -PassThru; Start-Sleep -Seconds 6; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Select-String -Path \"$env:TEMP\pl_sidecar.out\" -Pattern 'FERNLENS_SERVER_READY' | Select-Object -First 1"') do set READY=%%i
if not defined READY (
    echo [警告] sidecar 未在 6 秒内打出就绪行，继续打包；安装后首次运行请留意
) else (
    echo [2/4] sidecar 就绪 OK
)

rem ---- 3. electron-builder 打安装包 ----
echo.
echo [3/4] electron-builder 打安装包（杀软锁文件时会自动重试，请耐心）...
pushd desktop
if not exist node_modules (
    call npm install
    if errorlevel 1 ( popd & echo [错误] desktop npm install 失败 & goto :fail )
)
call node node_modules\electron-builder\out\cli\cli.js --win
if errorlevel 1 ( popd & echo [错误] electron-builder 失败 & goto :fail )
popd

rem ---- 4. 复制安装包到桌面 ----
echo.
echo [4/4] 复制安装包到桌面...
set DESKTOP=
for /f "delims=" %%d in ('powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"') do set DESKTOP=%%d
if not defined DESKTOP set DESKTOP=%USERPROFILE%\Desktop

set COPIED=0
for %%f in (desktop\release\FernLens-Setup-*.exe desktop\release\FernLens-Portable-*.exe) do (
    copy /y "%%f" "%DESKTOP%\" >nul
    if errorlevel 1 (
        echo [错误] 复制失败：%%~nxf —— 目标可能被占用
        goto :fail
    )
    echo        已放到桌面：%%~nxf
    set /a COPIED+=1
)
if %COPIED%==0 (
    echo [错误] desktop\release 里没有找到安装包
    goto :fail
)

echo.
echo ==========================================================
echo   打包完成！共 %COPIED% 个文件已复制到桌面：
echo     %DESKTOP%
echo   完整产物目录：desktop\release\
echo   安装版双击安装；便携版单文件免安装直接运行。
echo   未签名发行，SmartScreen 提示“未知发布者”时选“仍要运行”。
echo ==========================================================
goto :end

:fail
echo.
echo [失败] 打包中断，请查看上方输出定位原因。
exit /b 1

:end
endlocal
pause
