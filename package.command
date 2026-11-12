#!/bin/bash
# ==================================================================
# PhotoLab — macOS 打包脚本（在 Mac 上运行，需 macOS 11+）
#
# 用法：
#   chmod +x package.command
#   ./package.command              # 按当前机器架构构建 dmg + zip
#   ./package.command --universal  # 说明见下方“关于双架构”
#
# 产物：desktop/release/PhotoLab-Setup-<version>-<arch>.dmg 与 .zip
#
# 为什么必须在 Mac 上跑：
#   1. PyInstaller 不能交叉编译，sidecar 必须是本机 Mach-O 二进制；
#   2. dmg 打包与 codesign / notarytool 依赖 macOS 自带工具（hdiutil、codesign）。
#
# 关于双架构：PyInstaller 产物是单架构的，所以一次只能出 arm64 或 x64 的包。
#   Apple Silicon 用本脚本直接跑；Intel Mac 上也直接跑。要同时发布两种，
#   就在两台机器（或 CI 的 macos-13 / macos-14 矩阵）各跑一次。
# ==================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "=========================="
echo " PhotoLab macOS 打包"
echo "=========================="
echo "架构：$(uname -m)   系统：$(sw_vers -productVersion)"

# ---- 0. 环境检查 ----
command -v node >/dev/null || { echo "[错误] 未找到 node，请先安装 Node 18+"; exit 1; }
command -v python3 >/dev/null || { echo "[错误] 未找到 python3"; exit 1; }
if ! xcode-select -p >/dev/null 2>&1; then
  echo "[错误] 缺少 Xcode Command Line Tools：xcode-select --install"
  exit 1
fi

PY=python3
NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
if [ "$NODE_MAJOR" -lt 18 ]; then echo "[错误] Node 版本过低（需要 18+）"; exit 1; fi

# ---- 1. Python 环境 ----
echo
echo "[1/5] 准备 Python 虚拟环境..."
if [ ! -x .venv/bin/python3 ]; then
  "$PY" -m venv .venv
fi
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

echo "[2/5] 安装依赖（含 macOS 可选组件）..."
# PySide6 只服务旧版桌面 UI 与 tests.py --ui，服务端 sidecar 不 import 它；
# 排除后能少下 ~150MB，桌面版产物也更小。
grep -v -i '^PySide6' requirements.txt > /tmp/photolab-req.txt
python -m pip install -r /tmp/photolab-req.txt
# 可选组件：失败不阻断基础功能（RAW 解码 / GPU / 本地蒙版）
python -m pip install rawpy onnxruntime pyinstaller || \
  echo "[提示] rawpy/onnxruntime/pyinstaller 安装失败，稍后功能会相应降级"
# pyopencl 在 macOS 需编译 OpenCL 头文件；失败即回退 CPU（画面一致，仅变慢）
python -m pip install pyopencl || \
  echo "[提示] pyopencl 未安装：GPU 加速关闭，自动回退 CPU（不影响画质）"

# ---- 2. 前端构建 ----
echo
echo "[3/5] 构建前端（Vue3 + Vite）..."
pushd frontend >/dev/null
if [ ! -d node_modules ]; then npm install; fi
npm run build
popd >/dev/null

# ---- 3. 打包 Python sidecar ----
echo
echo "[4/5] PyInstaller 打包服务端 sidecar（约 1-3 分钟）..."
rm -rf backend-dist
.venv/bin/pyinstaller build/photolab-server.spec --noconfirm --distpath backend-dist

# PyInstaller 会自带执行位，这里再补一次，防止某些文件系统丢权限
chmod 755 backend-dist/photolab-server/photolab-server

# ---- 4. 打包 Electron 应用 ----
echo
echo "[5/5] 打包 .app / dmg..."
pushd desktop >/dev/null
if [ ! -d node_modules ]; then npm install; fi
ARCH_FLAG=""
case "$(uname -m)" in
  arm64) ARCH_FLAG="--arm64" ;;
  x86_64) ARCH_FLAG="--x64" ;;
esac
# 国内网络拉 electron / electron-builder 依赖慢时可加镜像：
#   export ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
#   export ELECTRON_BUILDER_BINARIES_MIRROR=https://npmmirror.com/mirrors/electron-builder-binaries/
npx electron-builder --mac $ARCH_FLAG
popd >/dev/null

echo
echo "=========================="
echo " 完成。产物：desktop/release/"
ls -lh desktop/release/*.dmg desktop/release/*.zip 2>/dev/null || true
echo
echo "未签名应用首次打开：右键 → 打开，或执行"
echo "  xattr -dr com.apple.quarantine /Applications/PhotoLab.app"
