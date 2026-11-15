# -*- mode: python ; coding: utf-8 -*-
# FernLens 服务端 sidecar 打包规格（PyInstaller onedir）
#
#   pyinstaller build/fernlens-server.spec --noconfirm --distpath backend-dist
#
# 产物：backend-dist/fernlens-server/（内含 fernlens-server.exe + _internal/）
# 随后由 desktop/package.json 的 extraResources 整体复制为安装包里的 backend/ 目录。
#
# 关键内置资源：
#   - frontend_dist : 前端静态产物，server.py 的 _dist_dir() 会定位到 _MEIPASS/frontend_dist
#   - models        : 本地分割模型（SAM2.1/U2Net/SkySeg/YOLOv8n），segment.py 的
#                     _models_dir() 会定位到 _MEIPASS/models；内置后无需再联网下载。
#                     若未内置，segment 会退化为 exe 同级 models/ 目录并从 hf-mirror 按需下载。

import os
import sys

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# ---- 内置数据 ----
datas = []

# 前端静态资源 -> _MEIPASS/frontend_dist
_frontend_dist = os.path.join(SPEC_DIR, '..', 'frontend', 'dist')
if os.path.isdir(_frontend_dist):
    datas.append((_frontend_dist, 'frontend_dist'))
else:
    print(f'[spec][警告] 未找到前端产物 {_frontend_dist}，打包后 Web UI 不可用')

# 本地分割模型 -> _MEIPASS/models（内置后免联网下载）
_models = os.path.join(SPEC_DIR, '..', 'models')
if os.path.isdir(_models):
    datas.append((_models, 'models'))
else:
    print(f'[spec][提示] 未找到 {_models}，打包后首次使用本地蒙版时会从 hf-mirror 下载')

# ---- 隐藏导入 ----
# uvicorn 的事件循环/协议/生命周期模块是动态加载的，PyInstaller 静态分析抓不到；
# onnxruntime 为本地推理引擎运行时（segment.py 在 try 中导入），务必内置。
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'onnxruntime',
]

# ---- 分析与打包 ----
a = Analysis(
    [os.path.join(SPEC_DIR, '..', 'server.py')],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Windows：防止旧版 wheel 附带的过时 MSVC 运行库（如 msvcp140.dll 14.26）覆盖系统新版，
# 导致 onnxruntime 等 C++20 编译产物在初始化 std::mutex 时直接 0xC0000005 崩溃。
if sys.platform == 'win32':
    _sys32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    _vc_targets = {'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
                   'vcruntime140.dll', 'vcruntime140_1.dll'}
    _fixed_binaries = []
    for _dest, _src, _kind in a.binaries:
        _base = os.path.basename(_dest).lower()
        if os.path.dirname(_dest) in ('', '.') and _base in _vc_targets:
            _sys_dll = os.path.join(_sys32, _base)
            if os.path.isfile(_sys_dll):
                _src = _sys_dll
        _fixed_binaries.append((_dest, _src, _kind))
    a.binaries = _fixed_binaries

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fernlens-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='fernlens-server',
)
