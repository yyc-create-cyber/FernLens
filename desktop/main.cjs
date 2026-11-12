/* PhotoLab Electron 主进程：
 *  - 拉起 Python sidecar（打包后 resources/backend/photolab-server[.exe]，开发时 .venv 的 python）
 *  - 解析 stdout 的 PHOTOLAB_SERVER_READY 行拿到端口与 token，加载本地 UI
 *  - 原生打开/保存对话框（渲染数据由后端 /api/render 生成，主进程负责落盘）
 *  - 生产环境尝试 electron-updater 自动更新（GitHub Releases）
 *
 * 跨平台：Windows 用 photolab-server.exe，macOS/Linux 用无扩展名的同目录二进制；
 * 开发态虚拟环境路径也按平台区分（Scripts/python.exe vs bin/python3）。 */
const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron')
const { spawn, exec } = require('child_process')
const path = require('path')
const fs = require('fs')

let backend = null
let mainWindow = null
let quitting = false
let lastUrl = ''
let lastToken = ''

const IS_WIN = process.platform === 'win32'
const IS_MAC = process.platform === 'darwin'

const IMAGE_FILTERS = [
  { name: '图像（含 RAW）', extensions: ['jpg', 'jpeg', 'png', 'webp', 'tif', 'tiff', 'bmp',
    'dng', 'cr2', 'cr3', 'crw', 'nef', 'nrw', 'arw', 'srf', 'sr2', 'raw', 'rw2',
    'raf', 'orf', 'pef', 'x3f', 'erf', 'kdc', 'dcr', 'mef', 'mos', 'mrw', 'fff', 'r3d'] },
]

function killBackend() {
  if (!backend) return
  const pid = backend.pid
  backend = null
  if (IS_WIN) {
    exec(`taskkill /pid ${pid} /T /F`, () => {})
  } else {
    try { process.kill(-pid) } catch { try { backend.kill() } catch {} }
  }
}

function startBackend() {
  return new Promise((resolve, reject) => {
    let exe, args, cwd
    if (app.isPackaged) {
      const dir = path.join(process.resourcesPath, 'backend')
      // PyInstaller 产物：Windows 带 .exe，macOS/Linux 无扩展名
      exe = path.join(dir, IS_WIN ? 'photolab-server.exe' : 'photolab-server')
      args = []
      cwd = dir
    } else {
      const root = path.join(__dirname, '..')
      const venvPython = IS_WIN
        ? path.join(root, '.venv', 'Scripts', 'python.exe')
        : path.join(root, '.venv', 'bin', 'python3')
      exe = fs.existsSync(venvPython) ? venvPython : (IS_WIN ? 'python' : 'python3')
      args = [path.join(root, 'server.py')]
      cwd = root
    }
    // POSIX 上 detached 让 sidecar 自成进程组，killBackend 的 kill(-pid) 才能连带子进程
    const child = spawn(exe, args, { cwd, windowsHide: true, detached: !IS_WIN })
    let buf = ''
    const timer = setTimeout(() => reject(new Error('后端启动超时')), 30000)
    child.stdout.on('data', (chunk) => {
      buf += chunk.toString()
      const m = buf.match(/PHOTOLAB_SERVER_READY (http:\/\/127\.0\.0\.1:(\d+))(?: token=(\S+))?/)
      if (m) {
        clearTimeout(timer)
        resolve({ url: m[1], token: m[3] || '', child })
      }
    })
    child.stderr.on('data', (chunk) => console.error('[backend]', chunk.toString().trim()))
    child.on('exit', (code) => {
      backend = null
      if (!quitting && mainWindow) {
        dialog.showErrorBox('PhotoLab', `渲染服务异常退出（code=${code}），应用将关闭。`)
        app.quit()
      }
    })
    resolve.pending = child
  })
}

function createWindow(url, token) {
  const win = new BrowserWindow({
    width: 1500,
    height: 940,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#101014',
    title: 'PhotoLab',
    icon: fs.existsSync(path.join(__dirname, '..', 'build', 'icon.png'))
      ? path.join(__dirname, '..', 'build', 'icon.png')
      : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  applyMenu()
  const target = token ? `${url}/?token=${encodeURIComponent(token)}` : `${url}/`
  win.loadURL(target)
  win.on('closed', () => { mainWindow = null })
  return win
}

/* 菜单：Windows/Linux 隐藏菜单栏；macOS 必须有应用菜单，否则 ⌘C/⌘V/⌘Z/⌘Q
 * 这些系统标准快捷键会全部失效（菜单是 macOS 快捷键的注册载体）。 */
function applyMenu() {
  if (!IS_MAC) {
    Menu.setApplicationMenu(null)
    return
  }
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {
      label: 'PhotoLab',
      submenu: [
        { role: 'about', label: '关于 PhotoLab' },
        { type: 'separator' },
        { role: 'hide', label: '隐藏 PhotoLab' },
        { role: 'hideOthers', label: '隐藏其他' },
        { role: 'unhide', label: '全部显示' },
        { type: 'separator' },
        { role: 'quit', label: '退出 PhotoLab' },
      ],
    },
    {
      label: '编辑',
      submenu: [
        { role: 'undo', label: '撤销' },
        { role: 'redo', label: '重做' },
        { type: 'separator' },
        { role: 'cut', label: '剪切' },
        { role: 'copy', label: '拷贝' },
        { role: 'paste', label: '粘贴' },
        { role: 'selectAll', label: '全选' },
      ],
    },
    {
      label: '显示',
      submenu: [
        { role: 'reload', label: '重新载入' },
        { role: 'togglefullscreen', label: '全屏' },
      ],
    },
    {
      label: '窗口',
      submenu: [
        { role: 'minimize', label: '最小化' },
        { role: 'zoom', label: '缩放' },
        { role: 'front', label: '前置全部窗口' },
      ],
    },
  ]))
}

const gotLock = app.requestSingleInstanceLock()
if (!gotLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })

  app.whenReady().then(async () => {
    try {
      const { url, token, child } = await startBackend()
      backend = child
      lastUrl = url
      lastToken = token
      mainWindow = createWindow(url, token)
    } catch (err) {
      dialog.showErrorBox('PhotoLab', '渲染服务启动失败：\n' + (err?.message || err))
      app.quit()
    }

    // 生产环境自动更新（无发布配置时静默跳过）
    if (app.isPackaged) {
      try {
        const { autoUpdater } = require('electron-updater')
        autoUpdater.checkForUpdatesAndNotify().catch(() => {})
      } catch { /* 未配置 publish 时忽略 */ }
    }
  })

  app.on('window-all-closed', () => {
    // macOS 习惯：关掉窗口不退出应用，⌘Q 才退出
    if (IS_MAC) return
    quitting = true
    killBackend()
    app.quit()
  })

  // macOS：点 Dock 图标时若无窗口则重建（后端仍在跑，直接复用地址）
  app.on('activate', () => {
    if (!mainWindow && lastUrl) mainWindow = createWindow(lastUrl, lastToken)
  })
  app.on('before-quit', () => { quitting = true; killBackend() })
  process.on('exit', () => killBackend())
}

// ------------------------------------------------------------ IPC：原生对话框
ipcMain.handle('open-image-dialog', async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    title: '打开图像（可多选）',
    filters: IMAGE_FILTERS,
    properties: ['openFile', 'multiSelections'],
  })
  return r.canceled || !r.filePaths.length ? null : r.filePaths
})

ipcMain.handle('save-image-dialog', async (_ev, { defaultName }) => {
  const r = await dialog.showSaveDialog(mainWindow, {
    title: '保存图像',
    defaultPath: defaultName || 'photolab.png',
    filters: [
      { name: 'PNG', extensions: ['png'] },
      { name: 'JPEG', extensions: ['jpg'] },
      { name: 'WebP', extensions: ['webp'] },
      { name: 'TIFF', extensions: ['tif'] },
    ],
  })
  return r.canceled || !r.filePath ? null : r.filePath
})

ipcMain.handle('write-file', async (_ev, { filePath, bytes }) => {
  await fs.promises.writeFile(filePath, Buffer.from(bytes))
  return true
})
