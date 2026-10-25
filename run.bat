@echo off
chcp 65001 >nul
title FernLens
cd /d %~dp0
if not exist .venv (
    echo 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo 创建虚拟环境失败：请确认已安装 Python 并加入 PATH
        pause
        exit /b 1
    )
)
.venv\Scripts\python -c "import PySide6, numpy, cv2" 2>nul
if errorlevel 1 (
    echo 安装依赖...
    .venv\Scripts\python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖安装失败，请检查网络后重试
        pause
        exit /b 1
    )
)
rem 可选依赖：RAW 解码 rawpy / GPU 加速 pyopencl，失败不影响基本功能
for %%p in (rawpy pyopencl) do (
    .venv\Scripts\python -c "import %%p" 2>nul || (
        echo 安装可选依赖 %%p ...
        .venv\Scripts\python -m pip install --quiet --disable-pip-version-check %%p >nul 2>&1 || echo 可选依赖 %%p 安装失败，已跳过
    )
)
echo 启动 FernLens...
.venv\Scripts\python main.py
if errorlevel 1 pause
