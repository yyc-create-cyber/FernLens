# FernLens — 全能AI照片调色工作站

> 拍了一堆好照片，想修出胶片感、电影感或大光圈虚化，却不想每次都对着几十个滑块从零折腾？  
> **FernLens** 是一套跑在你自己电脑上的本地调色工具——不仅还原了Lightroom风格的大多数基础调色功能（比如曲线，蒙版，取色器等等）还使用更**AI 调色**帮你打下手：聊聊你的想法，或者丢给它一张参考剧照，剩下的滑块、曲线和色轮全都给你摆好。

---

##AI 调色全家桶

FernLens 的AI做的所有事情**最终都会落回原生滑块、RGB 曲线、颜色分级色轮与局部蒙版**——随时可微调，随时可撤回（`Ctrl+Z`），保留每一分原片质感。

### 1.AI 精修（AI Grade）
点一下顶栏的 **AI 精修**，FernLens 会把压缩后的预览图和画面亮度/色彩统计送给视觉大模型（支持任意 OpenAI 兼容接口，可以在「AI 设置」里配好，或者用环境变量 `FERNLENS_AI_BASE_URL` / `FERNLENS_AI_KEY` / `FERNLENS_AI_MODEL`）。
- 可以直接让它自由发挥，也可以写上一句人话需求，比如：*“要一点日系清透感，肤色干净但别过曝”*、*“黄昏电影感，暗部带点青”*。
- 模型返回的参数会被自动钳制并按你的 **调整强度**（默认 60%，在设置里 20%～100% 自由拉）温和缩放——告别一眼假的过度后期！

### 2.AI 调色助手（AI Chat）
打开右侧的 AI 聊天面板，你可以像跟摄影朋友聊天一样跟它对话：
- **纯聊思路**：问它*“这张图构图和光影还有什么救？”*，它会看懂当前渲染图，给你具体可执行的调色建议。
- **直接动手**：告诉它*“帮我压一点高光、天空加点蓝，整体再通透些”*，它直接生成可编辑的调色参数 JSON 一键套用上去。
- **独立接口支持**：对话/精修模型和图生图模型各自独立配置，想搭哪个厂商的模型都随你。

### 3. 虚化调整（Blur Adjust）
不需要连任何云端 API Key，点击顶栏 **虚化调整**，三大本地绝活全开：
- **AI 调节景深**：依托本地 **Depth Anything V2** 深度模型估算画面三维景深。直接在画面上点选你想对焦的主体，调一下“光圈”与“高光散景”，就能模拟出自然的大光圈焦外虚化。
- **AI 景深立体调色**：同样用深度图自动分出「近景」和「远景」蒙版，内置 *主体聚光*、*空气透视*、*冷暖景深*、*电影暗调* 等光影预设，还能顺手生成 `[AI 近景]` / `[AI 远景]` 蒙版让你接着精修。
- **AI 参考仿色**：存了一张超喜欢的电影剧照或大神样片？拖进「参考仿色」，FernLens 会提取它的 CIELAB 色彩、明暗曲线和三路色轮氛围，一键迁到你的照片上（还贴心支持*保护主体肤色/明度*）。

### 4.  AI 一键蒙版 
类似Lightroom的天空、人物、背景蒙版？
- **AI 一键蒙版**：点选 **主体 / 人物 / 背景 / 天空 / 近景 / 远景**，全部靠本地 HuggingFace ONNX 模型（SAM 2.1 + U²-Net + SkySeg + YOLOv8n）推理完成，**照片不出本机**，抠完就是精确像素掩码，还能随时配合 **自定义多边形**、画笔、线性/径向渐变叠加半透明红色选区微调局部影调。

---

##编译与启动

> 准备环境：**Python 3.11+**、**Node.js 18+**（Windows / macOS 通用）

### 方式一：最省事的桌面“开发/浏览器”直启（Windows）
如果你已经装好了 Python 和 Node.js，仓库根目录直接双击（或在终端运行）：

```bat
run.bat
```

它会自动帮你把 `.venv` 虚拟环境建好、装齐依赖、顺便编译好 `frontend/` 前端静态资源，然后在终端打印出本地地址（形如 `http://127.0.0.1:xxxx/?token=...`），复制进浏览器就能开修！  
喜欢一键弹出浏览器窗口的，也可以直接跑：

```bat
start-web.bat
```

### 方式二：手动编译 & 启动（Windows / macOS / Linux 通吃）

```pwsh
# 1. 建虚拟环境并安装 Python 依赖
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
pip install rawpy onnxruntime   # 可选：RAW 格式解码 & 本地 AI 蒙版/景深引擎
pip install pyopencl            # 可选：OpenCL GPU 硬件加速（装不上会自动回退 CPU）

# 2. 编译 Vue3 前端
cd frontend
npm install
npm run build
cd ..

# 3. 启动本地服务端，用浏览器开 http://127.0.0.1:8642
python server.py --no-auth --port 8642 或者 start-web.ps1直接启动
```

> 💡 想体验**带独立窗口 + 原生文件选择器**的 Electron 开发模式？在先编译好 `frontend` 并安装好 `desktop` 依赖后执行：
> ```pwsh
> cd desktop && npm install && npm start
> ```




---

## macOS 用户怎么打包

 PyInstaller 不支持跨平台交叉编译，加上 `.dmg` 与签名工具链依赖 macOS 原生命令（`hdiutil`、`codesign`），而我没有Mac。不过如果你是Mac用户，你可以：

### 具体步骤
确保已经装好 **Node 18+**、**Python 3** 和 **Xcode 命令行工具**（`xcode-select --install`），在项目根目录跑自带的一键脚本：

```bash
chmod +x package.command
./package.command
```

脚本会自动搞定 `.venv`（跳过用不上的 PySide6 顺便瘦身 150MB）、构建前端、打出本机架构（Apple Silicon `arm64` 或 Intel `x64`）的 `fernlens-server`，再打好 **`FernLens-Setup-<version>-<arch>.dmg`** 和 **`.zip`** 放在 `desktop/release/` 里！

> 如果国内下载 Electron 二进制较慢，运行脚本前先加两行环境变量镜像：
> ```bash
> export ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
> export ELECTRON_BUILDER_BINARIES_MIRROR=https://npmmirror.com/mirrors/electron-builder-binaries/
> ```
 

### 🔓 macOS 首次打开提示“无法验证开发者”怎么办？
因为开源发布默认未付苹果开发者签名年费，首次拖进 `Applications` 后打不开时：
- **点按方式**：右键 `FernLens.app` → 选 **打开**；
- **终端一行搞定**：
  ```bash
  xattr -dr com.apple.quarantine /Applications/FernLens.app
  ```

（这个是Gemini写的方法，实际上可能有点区别哈）

---

##跑一下自检（可选）

```bash
python tests.py          # 核心色彩流水线与自动调色自检
python tests_server.py   # FastAPI 服务端 + 蒙版/渲染逐位一致测试
```

