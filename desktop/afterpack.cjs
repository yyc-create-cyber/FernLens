/* afterPack 钩子：写入平台相关的可执行文件元信息 / 权限。
 *
 * Windows：用 rcedit 写版本串与图标，失败时回退纯 JS 的 resedit。
 *   rcedit 来自 electron-builder 的 winCodeSign 缓存，Defender 常会短暂锁住刚写出的
 *   exe，所以带指数退避重试；两种方式都失败只告警、不中断打包（安装包仍可用，
 *   只是缺图标与版本串）。
 * macOS：PyInstaller sidecar 从 Windows 拷过来会丢执行位，这里补 chmod 755，
 *   否则打包后的 .app 启动即报 “Permission denied”。
 */
const fs = require('fs')
const path = require('path')
const { execFileSync } = require('child_process')

const STRINGS = {
  FileDescription: 'PhotoLab — 本地照片调色器（Electron 壳）',
  ProductName: 'PhotoLab',
  LegalCopyright: 'Copyright © 2026 nothing1145141',
  InternalName: 'PhotoLab',
  CompanyName: 'nothing1145141',
}

function rceditPath() {
  const local = process.env.LOCALAPPDATA
    || path.join(require('os').homedir(), 'AppData', 'Local')
  return path.join(local, 'electron-builder', 'Cache', 'winCodeSign',
    'winCodeSign-2.6.0', 'rcedit-x64.exe')
}

async function viaRcedit(exe, icon) {
  const rcedit = rceditPath()
  if (!fs.existsSync(rcedit)) return false
  const args = [exe]
  for (const [k, v] of Object.entries(STRINGS)) {
    args.push('--set-version-string', k, v)
  }
  args.push('--set-file-version', '1.0.0', '--set-product-version', '1.0.0.0')
  if (fs.existsSync(icon)) args.push('--set-icon', icon)

  await new Promise((r) => setTimeout(r, 2000))
  let delay = 1500
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      execFileSync(rcedit, args)
      console.log(`  ✓ afterPack: rcedit OK (attempt ${attempt})`)
      return true
    } catch (err) {
      console.warn(`  · afterPack: rcedit attempt ${attempt} failed: ${err.status ?? err.message}`)
      await new Promise((r) => setTimeout(r, delay))
      delay = Math.min(delay * 2, 10000)
    }
  }
  return false
}

function viaResedit(exe, icon) {
  const R = require('resedit')
  const exeFile = R.NtExecutable.from(fs.readFileSync(exe))
  const res = R.NtExecutableResource.from(exeFile)

  const versions = R.Resource.VersionInfo.fromEntries(res.entries)
  if (versions.length) {
    const vi = versions[0]
    vi.setStringValues({ lang: 1033, codepage: 1200 }, STRINGS)
    vi.setFileVersion(1, 0, 0, 0)
    vi.setProductVersion(1, 0, 0, 0)
    vi.outputToResourceEntries(res.entries)
  }

  if (fs.existsSync(icon)) {
    const iconFile = R.Data.IconFile.from(fs.readFileSync(icon))
    const groups = R.Resource.IconGroupEntry.fromEntries(res.entries)
    const id = groups.length ? groups[0].id : 1
    R.Resource.IconGroupEntry.replaceIconsForResource(
      res.entries, id, 1033, iconFile.icons.map((i) => i.data))
  }

  res.outputResource(exeFile)
  fs.writeFileSync(exe, Buffer.from(exeFile.generate()))
  console.log('  ✓ afterPack: resedit OK（rcedit 不可用时的回退）')
}

/* macOS：确保 sidecar 与 .app 内可执行文件带执行位 */
function fixPosixModes(appOutDir) {
  const backendDir = path.join(appOutDir, 'PhotoLab.app', 'Contents', 'Resources', 'backend')
  const bin = path.join(backendDir, 'photolab-server')
  if (!fs.existsSync(bin)) {
    console.warn(`  · afterPack: 未找到 ${bin}，跳过 chmod`)
    return
  }
  fs.chmodSync(bin, 0o755)
  console.log('  ✓ afterPack: chmod 755 photolab-server')
  // _internal 里的动态库按需补权限（PyInstaller 有时会给 .dylib 保留 644）
  const internal = path.join(backendDir, '_internal')
  if (fs.existsSync(internal)) {
    const walk = (dir) => {
      for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, ent.name)
        if (ent.isDirectory()) walk(p)
        else if (ent.name.endsWith('.so') || ent.name.endsWith('.dylib')) {
          try { fs.chmodSync(p, 0o755) } catch { /* 只读卷忽略 */ }
        }
      }
    }
    try { walk(internal) } catch { /* 忽略 */ }
  }
}

exports.default = async function afterPack(context) {
  if (context.electronPlatformName === 'darwin') {
    fixPosixModes(context.appOutDir)
    return
  }
  if (context.electronPlatformName !== 'win32') return

  const exe = path.join(context.appOutDir, 'PhotoLab.exe')
  const icon = path.join(context.packager.projectDir, '..', 'build', 'icon.ico')
  if (!fs.existsSync(exe)) {
    console.warn(`  · afterPack: 未找到 ${exe}，跳过版本信息写入`)
    return
  }
  try {
    if (await viaRcedit(exe, icon)) return
    viaResedit(exe, icon)
  } catch (err) {
    console.warn(`  · afterPack: 写入版本信息失败（不影响打包）：${err.message}`)
  }
}
