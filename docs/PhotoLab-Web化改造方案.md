# PhotoLab Web 化改造方案（Electron + Python Sidecar）

> **阅读对象**：本方案同时写给人看和给 AI Agent 执行用。Agent 请从头到尾读完再动手，
> 尤其是第 3 节（资产冻结清单）、第 5 节（协议设计）和第 11 节（工作守则）。
> 协议或架构如有变更，必须回写更新本文档对应章节（本文档是单一事实来源）。

---

## 1. 目标与非目标

**目标**

1. 用 Web 技术重写 UI，获得远超 Qt 样式表的观感（参考 Lightroom Web / Photopea 的深色风格）。
2. 打包成普通用户可一键安装的桌面应用（安装器 + 便携版），可公开发布到 GitHub Releases，
   带自动更新。**体积不是约束**（已确认），采用全量打包，不做首启下载环境。
3. **调色引擎零损失**：现有算法、性能指标（12MP 全图 1~4s）、GPU 加速、RAW 支持全部保留。

**非目标（明确不做）**

- 不重写图像处理引擎为 JS/WebGPU（成本以月计，且会丢性能与功能）。
- 不做 Linux/macOS 支持（现状仅 Windows，打包链先只管 Windows）。
- 不改任何算法行为。改造后同一参数、同一输入，输出必须与旧版逐位一致（有测试保证，见 §9）。

---

## 2. 总体架构

```
┌─ Electron 窗口（desktop/）────────────────────────┐
│  Vue 3 + TS Web UI（frontend/，Vite 构建）         │
│  - 主视口 <canvas>（缩放/平移/蒙版/图章/裁剪叠加）   │
│  - 调整面板（滑块/曲线/色轮/取色器）                 │
└──────┬─────────────────────────────────────────────┘
       │ REST（打开/保存/几何修改/自动调色）
       │ WebSocket（只传预览/全图帧，二进制 JPEG）
┌──────▼─────────────────────────────────────────────┐
│  server.py：FastAPI + uvicorn（新增，唯一入口）      │
│  - 文档会话：源图 + 图章工作副本 + 版本号（后端持有） │
│  - 任务循环：复刻 ProcessorThread 语义（见 §5.3）    │
├─────────────────────────────────────────────────────┤
│  core.py / gpu.py / auto.py —— 冻结不改             │
└─────────────────────────────────────────────────────┘
```

**进程模型**：Electron 主进程启动时拉起 sidecar（打包后是 PyInstaller 产物 `photolab-server.exe`，
开发时直接 `python server.py`），绑定 `127.0.0.1` 随机端口，等 `/api/meta` 就绪后创建
BrowserWindow 加载 UI。**Webview 只加载本地地址**，不加载任何远程内容。

**关键设计决定（已定，不要重新讨论）**：

| 决定 | 选择 | 一句话理由 |
| --- | --- | --- |
| 壳 | **Electron**（非 Tauri） | 全 JS 工具链、electron-builder/updater 最成熟；体积不敏感，Tauri 的小体积优势无意义 |
| 前端框架 | **Vue 3 + TypeScript + Vite**，组件库 Naive UI（深色主题） | 深色编辑器类 UI 开箱即用；自定义控件反正都要手写 Canvas |
| 后端框架 | **FastAPI + uvicorn** | 类型注解好、WebSocket 一等公民、与现有纯函数引擎贴合 |
| 状态模型 | **参数即文档**：完整 `Adjustments` JSON 随每次渲染请求由前端发来；后端只持有源图/图章副本/几何状态 | 后端接近无状态渲染器，与现有 `submit(params)` 语义完全一致 |
| 帧传输 | WebSocket 二进制帧：`[4 字节 rid][1 字节 kind][JPEG 数据]` | 预览仅用于显示，JPEG q≈90 足够；**保存不走 WS**，服务端渲染后直接落盘，无画质损失 |
| 参数序列化 | `Adjustments`/`MaskState` 等 dataclass ↔ dict，用 dataclasses.asdict/递归重建 | auto.py 已经在产出参数字典，方向现成 |

---

## 3. 现状资产盘点（哪些动、哪些不动）

仓库：`C:\Users\dy\Desktop\photolab`（PhotoLab，纯本地 Windows 照片调色器）。

### 3.1 冻结区（只许加胶水，不许改逻辑）

| 文件 | 行数 | 内容 | 处置 |
| --- | --- | --- | --- |
| `core.py` | 1676 | 全部图像处理，纯 NumPy/OpenCV，无 Qt 依赖 | **冻结**。关键入口：`process(params, rgb8, should_cancel=None)`（第 1676 行）、`process_linear`、`linear_histograms(rgb8)`（1478）、`sample_patch(rgb8,x,y)`（737）、`crop_rotate`（1437）、`rotate90`（1299）、`remap_mask_geometry`（1387）、`build_lut`（PCHIP，561）、`sanitize_points`（511）、`downscale_max_edge`（1524，预览 1200px）。参数模型：`Adjustments`（210）、`MaskState`（173）、`MaskAdjustments`（129）、`WheelState`（107）、`PickerState`（116） |
| `gpu.py` | 295 | OpenCL 融合内核，异常静默回退 CPU，`PHOTOLAB_NO_GPU=1` 可强制关闭 | **冻结**，sidecar 里原样可用 |
| `auto.py` | 403 | 自动调色（规则版 + AI 视觉模型版），只产出参数字典 | **冻结**。注意：AI 版的端点自动发现依赖本机环境，打包后行为可能不同，需用 `PHOTOLAB_AI_BASE_URL/KEY/MODEL` 显式指定 |
| `tests.py` | 1403 | 核心算法自检 + `--ui` 冒烟 | 保留并持续跑，见 §11 |

> 若 `Adjustments` 缺少 `to_dict()/from_dict()`，允许在 core.py 末尾**追加**纯序列化辅助代码
> （不碰任何现有函数）。这是对冻结区唯一的例外。

### 3.2 重写区（Qt UI → Web，工作量主体）

| 文件 | 行数 | 内容 | Web 对应物 |
| --- | --- | --- | --- |
| `ui_window.py` | 2091 | 主窗口、各调整面板、蒙版/图章/裁剪状态机 | `frontend/` 全部组件 + §6 映射表 |
| `ui_widgets.py` | 1614 | `SliderRow`（滚轮微调/双击复位）、`CollapsibleSection`、`PreviewWidget`（缩放/平移/取色/蒙版手柄/裁剪框/图章叠加）、`ProcessorThread` | 前两者→UI 组件；PreviewWidget→Canvas 视口组件；**ProcessorThread 语义移到 server.py**（§5.3） |
| `main.py` | 45 | 入口 | 由 server.py + Electron 取代 |

旧 Qt UI（`main.py`/`ui_window.py`/`ui_widgets.py`/`run.bat`）在 M4 验收前**保留不删**，
作为行为参照与回退方案；M4 验收通过后单独一次 commit 删除。

### 3.3 新增目录

```
photolab/
├── core.py  gpu.py  auto.py  tests.py     # 冻结
├── server.py                              # 新增：FastAPI 会话/任务/路由
├── tests_server.py                        # 新增：API 层测试（TestClient）
├── main.py  ui_window.py  ui_widgets.py   # 旧 UI：M4 验收后删除
├── frontend/                              # Vue3 + Vite + TS
│   ├── src/views/Editor.vue
│   ├── src/components/viewport/           # 主画布 + 蒙版/裁剪/图章叠加
│   ├── src/components/panels/             # 基础/效果/曲线/色轮/取色器
│   ├── src/components/widgets/            # SliderRow、CollapsibleSection 等
│   └── src/api/                           # REST/WS 客户端 + Adjustments TS 类型
├── desktop/                               # Electron main + electron-builder 配置
└── build/                                 # PyInstaller spec、打包脚本
```

---

## 4. 通信协议（实现以此为准）

### 4.1 REST（所有"变更"走 REST，WebSocket 只传帧）

| 方法/路径 | 请求 → 响应 | 对应现有代码 |
| --- | --- | --- |
| `GET /api/meta` | → `{version, rawpy: bool, gpu: bool, width, height}` | 启动探测/健康检查 |
| `POST /api/open` | `{path}` → `{width, height, serial}`；DNG/CR2/... 走 rawpy（`use_camera_wb=True, output_color=sRGB, output_bps=8, no_auto_bright=True, DemosaicAlgorithm.AHD`） | `ui_window.load_image` |
| `POST /api/open-upload` | multipart 上传文件 → `{width, height, serial}`；**浏览器模式专用**（拿不到本地路径），仅常规格式，RAW 走 `/api/open` | 同上 |
| `POST /api/save` | `{path, adjustments}` → `{rid, saved}`；走**紧急槽位**渲染全图后直接写盘；WS 推 `saved` 事件 | `ProcessorThread.submit(urgent=True)` + 保存路径 |
| `POST /api/render` | `{adjustments, format}` → 图像字节；**浏览器模式下载保存专用**（Electron 走 `/api/save`） | `core.process_linear` |
| `POST /api/pick` | `{x, y}` → `sample_patch` 取色结果 | `sample_patch` |
| `POST /api/stamp` | `{stroke}`（取样点/笔画），作用于图章工作副本，返回新版本号 | `ui_window` 图章逻辑 |
| `POST /api/stamp/undo` `POST /api/stamp/clear` | 撤销一笔 / 清空（原图重放剩余笔画） | 同上 |
| `POST /api/geometry/crop` | `{rect_norm, angle_deg}` → 应用裁剪/拉直，蒙版坐标同步，返回新尺寸 | `crop_rotate` + `remap_mask_geometry` |
| `POST /api/geometry/rotate` | `{quarter_turns}` → 90° 无损旋转 | `rotate90` + `remap_masks_rot90` |
| `POST /api/auto` | `{mode: "rule"\|"ai"}` → `{adjustments}`（完整参数字典，前端套用） | `auto.py` |
| `GET /api/histogram` | → `{r[], g[], b[], luma[]}`（64 bins，来自当前源图） | `linear_histograms` |

安全：只绑 `127.0.0.1`；启动时生成随机 token，WS 升级与 REST 均要求 `?token=` 匹配（本机应用防其它本地进程干扰，成本一行）。

### 4.2 WebSocket `/ws?token=...`

- C→S（JSON 文本）：`{"type":"render", "rid":123, "kind":"preview"|"full", "adjustments":{...}}`
  ——`adjustments` 是**完整**参数字典（含蒙版），一次几十 KB，不搞增量。
- S→C（二进制）：`[uint32 BE rid][uint8 kind: 0=preview 1=full][JPEG bytes]`。
- S→C（JSON 文本）：`{"type":"saved","rid":..,"path":..}` 等控制回执。
- 前端职责：拖动中发 `preview`（约 30–60Hz 节流）、松手或 400ms 空闲后发 `full`；
  收到帧按 rid 丢弃过期帧。**`rid` 由前端生成单调递增**，服务端不负责去重（服务端只保证最新覆盖）。

### 4.3 服务端任务循环（复刻 ProcessorThread 语义，这是行为一致性的关键）

在 server.py 中用 `asyncio` + `run_in_executor`（NumPy 会释放 GIL）复刻：

1. 普通任务槽**只保留最新一次**——连续 render 请求覆盖中间帧，永不堆积；
2. **紧急槽位独立**——`save` 不被随后的普通 render 覆盖；
3. 数据源按 kind 分开（preview = `downscale_max_edge(rgb8, 1200)`，full = 原图），线性光缓存按
   "数据源 + 版本号"失效（图章/几何修改会 bump 版本号）；
4. 渲染中传 `should_cancel` 协作取消（core 已支持分块间检查），后端停服/换图时调用；
5. 多线程渲染时把 OpenCV 线程数限制为 1（core 内部已处理，勿在 server 重复设置）。

---

## 5. 前端重写：组件映射表

工作量主体在这里。逐行对照旧 Qt 代码迁移行为，不要凭感觉重设计交互。

| Qt 现状（ui_widgets.py） | Web 组件 | 难度 | 备注 |
| --- | --- | --- | --- |
| `PreviewWidget`：滚轮以光标为中心缩放、左键拖拽平移、双击适应窗口、`0`/`1`/`+`/`-` | `viewport/CanvasViewport.vue`：双层 canvas（图像层 + 叠加层），图像帧 `drawImage` | 中 | 缩放用 CSS transform 会糊，必须 canvas 重绘；最近邻插值保持像素风 |
| 取色模式：点击图像 → `POST /api/pick` → 选中色相 | 视口点击状态机 | 低 | |
| 蒙版叠加（红色半透明）+ 手柄编辑：线性/径向（蓝色手柄拖动缩放、橙色旋转）/画笔 | `viewport/MaskOverlay.vue`，几何全部归一化 0..1（与后端一致） | **高** | 手柄命中测试逐个对照 `PreviewWidget.set_edit_mode` 相关代码 |
| 裁剪：8 手柄、三分网格、框外压暗、比例锁定（自由/原始/1:1/3:2/4:3/16:9 及转置）、±45° 拉直实时预览 | `viewport/CropOverlay.vue`；拉直角度变化时发 preview 请求 | **高** | 手柄命中对照 `_crop_hit`；约束逻辑对照 `_clamp_crop_rect`/`fit_rect_in_rotation`（后端负责约束算法，前端只发候选 rect） |
| 图章：Alt+点击取样、涂抹、源区十字标、Esc 退出 | `viewport/StampOverlay.vue`，笔画序列 `POST /api/stamp` | 中 | |
| `SliderRow`：滚轮微调、双击标签复位、右侧重置按钮、拖动中/松手两种提交 | `widgets/SliderRow.vue` 包装 Naive UI Slider | 低 | 拖动→preview，`@release`→full 的节流逻辑放全局 composable |
| `CollapsibleSection` | `widgets/Section.vue` | 低 | |
| 曲线编辑器：主+R/G/B 四通道、拖拽加点/右键删点/双击加点、背景直方图 | `panels/CurveEditor.vue`（Canvas） | **高** | 显示用 TS 实现同款 PCHIP（Fritsch–Carlson，照 `build_lut`/`_pchip_slopes` 抄 ~40 行）；**权威数据只有控制点**，发后端算 LUT；显示与后端允许 <1/255 误差 |
| 色轮 ×4（阴影/中间调/高光/全局）+ 饱和度/亮度滑块 + 混合 | `panels/ColorWheel.vue`（Canvas，输出 hue/strength → `WheelState`） | 中 | |
| 取色器面板：8 预制色相、±60° 微调、影响范围 5~180°、羽化 | `panels/PickerPanel.vue` | 中 | 选中蒙版时切换为编辑该蒙版局部参数（与旧版行为一致） |
| 蒙版列表面板（选中/添加/删除/反向/羽化/笔刷大小/撤销一笔） | `panels/MaskList.vue` | 中 | |
| 顶栏（打开/保存/重置/黑白/对比/缩放/裁剪/左转/右转/取色/图章）+ 状态栏 | `App toolbar`，快捷键对齐 README §1（Ctrl+O/S/R、`\`、Esc 等，浏览器 keydown 实现） | 低 | 打开/保存用 Electron 原生对话框（IPC 拿路径→REST） |

直方图、参数默认值、各滑块范围，从 `ui_window.py`/`Adjustments` 字段默认值抄，**不要自己发明数值**。

---

## 6. 打包与分发（已实现，实况说明）

> 2026-10 实施记录：以下流程已全部跑通并产出安装包。

### 6.1 后端 sidecar（build/photolab-server.spec）

```bash
cd frontend && npm install && npm run build          # 产出 frontend/dist（由 server 托管）
.venv/Scripts/pyinstaller build/photolab-server.spec --noconfirm --distpath backend-dist
```

- 产物 `backend-dist/photolab-server/`（onedir，约 178MB：numpy + opencv-headless + rawpy + fastapi/uvicorn + 前端 dist）。
- rawpy 已打包（RAW 开箱即用）；PySide6/桌面版 OpenCV 已排除；**pyopencl 未打包**（驱动 ICD 依赖，`gpu.py` 静默回退 CPU）。
- server.py 冻结态下从 `sys._MEIPASS/frontend_dist` 托管 UI（`_dist_dir()`）。

### 6.2 Electron（desktop/）

- `main.cjs`：拉起 sidecar（打包后 `resources/backend/photolab-server.exe`；开发时 `.venv` 的 python），
  解析 READY 行拿端口/token，BrowserWindow 加载 `http://127.0.0.1:<port>/?token=...`；
  单实例锁；`window-all-closed` 时 taskkill /T /F 杀进程树。
- `preload.cjs`：contextBridge 暴露 `photolabNative`（原生打开/保存对话框 + 落盘）。
  前端检测到该桥即走原生对话框（RAW 用路径打开），否则用浏览器上传/下载。
- electron-builder：NSIS（`PhotoLab-Setup-1.0.0.exe`，~138MB）+ portable（`PhotoLab-Portable-1.0.0.exe`）；
  electron-updater 对接 GitHub Releases（nothing1145141/photolab），未发布版本时静默。

### 6.3 打包环境注意（Windows 无开发者模式时）

- `winCodeSign` 解压需要符号链接权限，会报"客户端没有所需的特权"。绕过：把任一
  `%LOCALAPPDATA%\electron-builder\Cache\winCodeSign\<hash>` 解压结果复制为
  `winCodeSign-2.6.0` 目录（见本文档 Git 历史 / 或开启系统"开发者模式"一劳永逸）。
- rcedit 偶发 "Unable to commit changes" 是杀软短暂锁文件，重跑即可。
- npm 侧建议全程 `ELECTRON_MIRROR` / `ELECTRON_BUILDER_BINARIES_MIRROR` / registry 指 npmmirror。
- electron 的 postinstall 下载可能被 npm allow-scripts 拦截，需手动 `node install.js`。

### 6.4 开发/发布流程

```
开发：python server.py  +  cd frontend && npm run dev（纯浏览器调试）
桌面开发：cd desktop && npm start（拉起 .venv 后端）
发布：build backend → build frontend → cd desktop && npm run dist → GitHub Release（含 latest.yml）
```

---

## 7. 里程碑与验收标准

> 每个里程碑可独立提交验收；估时按熟悉前端的开发者， unfamiliar 时 ×2。

**M1 后端 + 浏览器最小可用版（约 1~2 周）**
- server.py：§4 全部 REST + WS + 任务循环；frontend：骨架 + Canvas 视口（缩放/平移）+
  基础影调/白平衡滑块 + 直方图 + 保存。
- 验收：**纯浏览器**完成 打开→调色→保存 全流程；`python tests_server.py` 通过——
  其中一条关键测试：同一参数下，`POST /api/save` 输出的 PNG 与直接调用
  `core.process(params, rgb8)` 的结果**逐位一致**（证明协议层无损）。

**M2 调整面板全量（约 1~2 周）**
- 曲线/色轮/取色器/黑白/效果（清晰度/降噪/锐度/去朦胧）/RAW 打开/自动+AI 调色/直方图联动。
- 验收：README §2 功能表"基础/效果/曲线/颜色分级/取色器/RAW/自动调色"逐项过；
  AI 调色走环境变量指定的端点能跑通。

**M3 交互工具（约 1~2 周）**
- 三种蒙版 + 蒙版局部调整全套（含面板联动切换）、图章、裁剪/拉直/90°、快捷键全对齐。
- 验收：README §1 的快捷键清单逐条过；蒙版 + 图章 + 裁剪组合操作后保存，与旧 Qt 版
  同操作结果一致（人工对比抽查）；`python tests.py --ui` 仍通过（旧 UI 未删）。

**M4 Electron 壳 + 打包发布（约 3~5 天）**
- 壳、sidecar 拉起、原生对话框、NSIS + portable、electron-updater、图标/版本信息。
- 验收：干净 Windows 10/11 虚拟机（无 Python、无开发环境）安装后全功能可用；
  关窗无残留进程；自动更新能收到一次测试发布。

**M5 收尾**
- 删除旧 Qt UI 与 run.bat，README 改写（新架构、新打包方式、GPU 现状），
  可选：环境包拆分 / MSI / 签名（均非必须，体积不敏感）。

---

## 8. 风险与已知坑

| 风险 | 对策 |
| --- | --- |
| WS JPEG 帧被误当"保存画质" | 协议上保存走服务端直写，帧仅显示；README 说明 |
| 前端 PCHIP 显示与后端 LUT 细微不一致 | 控制点是唯一权威；显示误差可接受，保存永远走后端 |
| PyInstaller 漏 hidden imports（rawpy、uvicorn 的 websockets/httptools） | M1 就用 tests_server.py 打包产物冒烟，不要等到 M4 |
| 图章/裁剪并发渲染撕裂 | 照抄 `ProcessorThread.set_source` 的拷贝语义 + 版本号失效缓存（§4.3） |
| AI 调色端点在打包后找不到 | 文档明确 `PHOTOLAB_AI_*` 三个环境变量；打包版不做端点自动发现 |
| Electron 安全默认配置 | contextIsolation 开、nodeIntegration 关、只加载 `http://127.0.0.1:<port>?token=` |
| 中文路径/文件名 | core 已支持（README 声明），M1 验收用例里必须包含中文路径 |

---

## 9. 工作守则（给执行 Agent）

1. **冻结区纪律**：不改 `core.py`/`gpu.py`/`auto.py` 的任何现有函数；唯一例外是按需在
   core.py 末尾追加序列化辅助。发现"必须改算法才能实现"的需求 → 停下来回报，不要动手。
2. **每次后端改动**跑 `python tests.py`；动到 API 层跑 `python tests_server.py`；
   声称完成前两个都必须绿。
3. **数值一律抄旧代码**：滑块范围、默认值、预设色相、LUT 尺寸、预览 1200px 等，
   从 `Adjustments`/`ui_window.py` 抄，禁止自己发明。
4. **交互行为逐条对照旧 Qt 实现**（`PreviewWidget`/`ui_window.py` 的状态机），先读懂再写。
5. 协议/架构变更必须同步更新本文档对应章节，保持文档与代码一致。
6. 每个里程碑结束输出：改动清单 + 测试结果 + 与 README 功能表的对照勾选。
7. 不引入未在本方案出现的重型依赖（状态管理用 Vue 原生 reactivity 即可，不需要 Pinia/Redux）。
