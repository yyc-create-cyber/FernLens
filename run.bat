@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
rem ==================================================================
rem  PhotoLab 本地运行（Web UI，开发即用）
rem  用法：双击运行，或在仓库根目录执行  run.bat
rem        也可带参数透传给 server.py，例如：run.bat --port 8000
rem
rem  首次运行自动创建虚拟环境、安装依赖、构建前端，然后启动服务端；
rem  就绪后控制台会打印： PHOTOLAB_SERVER_READY http://127.0.0.1:<port>
rem  默认 --no-auth，浏览器打开上面的地址即可。
rem ==================================================================

cd /d "%~dp0"

rem 阿里云镜像：清华镜像对 pip 26 返回的索引解析不出候选版本
set "PY_MIRROR=https://mirrors.aliyun.com/pypi/simple"
set "VENV_PY=.venv\Scripts\python.exe"
set "NEED_IMPORTS=fastapi, uvicorn, cv2, numpy, pydantic, multipart"

rem ---- 0. 虚拟环境 ----
if not exist "%VENV_PY%" (
    echo [初始化] 未找到 .venv，正在创建虚拟环境...
    set "BASE_PY="
    where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "BASE_PY=py -3"
    if not defined BASE_PY (
        where python >nul 2>nul && set "BASE_PY=python"
    )
    if not defined BASE_PY (
        echo [错误] 未找到 Python 3，请先安装 Python 3.11+ 并加入 PATH
        goto :fail
    )
    %BASE_PY% -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败：请确认 Python 版本 >= 3.9 且自带 venv
        goto :fail
    )
)

rem ---- 1. 核心依赖 ----
"%VENV_PY%" -c "import %NEED_IMPORTS%" >nul 2>nul
if errorlevel 1 (
    echo [初始化] 安装后端依赖...
    "%VENV_PY%" -m pip install --disable-pip-version-check -q -i "%PY_MIRROR%" -r requirements.txt
    if errorlevel 1 (
        echo [提示] requirements.txt 安装失败，改用核心依赖列表重试...
        "%VENV_PY%" -m pip install --disable-pip-version-check -q -i "%PY_MIRROR%" numpy opencv-python fastapi "uvicorn[standard]" pydantic python-multipart
        if errorlevel 1 (
            echo [错误] 后端依赖安装失败，请检查网络后重试
            goto :fail
        )
    )
    "%VENV_PY%" -c "import %NEED_IMPORTS%" >nul 2>nul
    if errorlevel 1 (
        echo [错误] 依赖安装完成但仍无法导入，请手动检查 .venv
        goto :fail
    )
)

rem ---- 2. 可选依赖：rawpy / pyopencl / onnxruntime ----
call :opt rawpy
call :opt pyopencl
call :opt onnxruntime

rem ---- 3. 前端：dist 缺失才构建 ----
if not exist frontend\dist\index.html (
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [警告] 未找到 npm 且前端未构建，将仅启动 API（无 Web UI）
    ) else (
        echo [初始化] 构建前端（Vue3 + Vite）...
        pushd frontend
        if not exist node_modules (
            call npm install --no-audit --no-fund
            if errorlevel 1 (
                popd
                echo [警告] npm install 失败，将仅启动 API
                goto :serve
            )
        )
        call npm run build
        if errorlevel 1 (
            popd
            echo [警告] 前端构建失败，将仅启动 API
            goto :serve
        )
        popd
    )
)

:serve
echo.
echo ==========================================================
echo   PhotoLab 服务端启动中... 浏览器打开下面打印的地址
echo ==========================================================
"%VENV_PY%" server.py --no-auth %*
if errorlevel 1 (
    echo.
    echo [错误] 服务端异常退出，请查看上方报错
    pause
)
endlocal
exit /b 0

:opt
"%VENV_PY%" -c "import %~1" >nul 2>nul
if errorlevel 1 (
    echo [可选] 安装 %~1 ...
    "%VENV_PY%" -m pip install --disable-pip-version-check -q -i "%PY_MIRROR%" %~1 >nul 2>&1
    if errorlevel 1 echo [可选] %~1 安装失败，已跳过（不影响基本功能）
)
goto :eof

:fail
echo.
pause
exit /b 1
