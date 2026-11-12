<script setup lang="ts">
/** 主布局：顶栏 + （裁剪/图章条）+ 舞台（视口+胶片条）+ 右侧面板 + 状态栏。
 *  右侧面板用 Tab 分区（光线/色彩/效果/曲线/分级/蒙版/取色），一次只显示一组，
 *  避免整列折叠面板的视觉噪音。快捷键对照 README：
 *  Ctrl+O 打开 / Ctrl+S 保存 / Ctrl+R 重置 / `\` 对比原图 / Delete 删除取色点
 *  / Esc 退出编辑 / 0、1、+、- 缩放（Viewport 内）。 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { NDropdown, NConfigProvider, darkTheme } from 'naive-ui'
import Viewport from '@/components/Viewport.vue'
import Filmstrip from '@/components/Filmstrip.vue'
import Histogram from '@/components/Histogram.vue'
import SliderRow from '@/components/SliderRow.vue'
import MiniSlider from '@/components/MiniSlider.vue'
import CurveEditor from '@/components/CurveEditor.vue'
import GradingPanel from '@/components/GradingPanel.vue'
import PickerPanel from '@/components/PickerPanel.vue'
import MaskPanel from '@/components/MaskPanel.vue'
import HistoryPanel from '@/components/HistoryPanel.vue'
import FilterPanel from '@/components/FilterPanel.vue'
import AiSettings from '@/components/AiSettings.vue'
import AiChat from '@/components/AiChat.vue'
import Icon from '@/components/Icon'
import {
  useEditor, resetAll, notifyChanged, notifyCommitted,
  toggleBlackwhite, saveBrowser, autoRule, autoAI, cancelPick, deletePicker,
  targetAdj, setComparing, enterCrop, cancelCrop, applyCrop, rotate90,
  stampUndo, stampClear, exitMaskEdit, toggleMaskOverlay, openFiles, openPaths,
  initLibrary, openAiSettings, openAiChat, aiConfigured, rotateLinearMask, refreshLocalAI,
  undoHistory, redoHistory,
} from '@/stores/editor'
import { native, openDesktop, saveDesktop } from '@/api/client'

const store = useEditor()
const fileInput = ref<HTMLInputElement | null>(null)

// ---- 右侧面板 Tab ----
type TabKey = 'light' | 'color' | 'effects' | 'filter' | 'curve' | 'grading' | 'mask' | 'picker' | 'history'
const TABS: { key: TabKey; label: string }[] = [
  { key: 'light', label: '光线' },
  { key: 'color', label: '色彩' },
  { key: 'effects', label: '效果' },
  { key: 'filter', label: '滤镜' },
  { key: 'curve', label: '曲线' },
  { key: 'grading', label: '分级' },
  { key: 'mask', label: '蒙版' },
  { key: 'picker', label: '取色' },
  { key: 'history', label: '历史' },
]
const tab = ref<TabKey>('light')

// ---- 面板目标（选中蒙版 → 局部参数）----
const target = computed(() => targetAdj())
const isMaskTarget = computed(() => store.maskSelected !== null)
const maskName = computed(() => {
  const i = store.maskSelected
  return i === null ? '' : store.adjustments.masks[i]?.name || '蒙版'
})

// ---- 保存 ----
const saveOptions = [
  { label: 'PNG（无损）', key: 'png' },
  { label: 'JPEG（质量 95）', key: 'jpg' },
  { label: 'WebP', key: 'webp' },
]
const rotateOptions = [
  { label: '向左旋转 90°', key: 'left' },
  { label: '向右旋转 90°', key: 'right' },
]

function onOpenClick(): void {
  if (native) {
    void openDesktop().then((paths) => {
      if (!paths || !paths.length) return
      openPaths(paths).catch((err) => {
        store.status = `打开失败：${err instanceof Error ? err.message : err}`
      })
    })
  } else {
    fileInput.value?.click()
  }
}
async function onFileChosen(e: Event): Promise<void> {
  const input = e.target as HTMLInputElement
  const files = [...(input.files ?? [])]
  if (!files.length) return
  try {
    await openFiles(files)
  } catch (err) {
    store.status = `打开失败：${err instanceof Error ? err.message : err}`
  }
  input.value = ''
}

async function onSavePick(key: string): Promise<void> {
  try {
    if (native) {
      const path = await saveDesktop(store.adjustments, `photolab.${key}`)
      if (path) store.status = `已保存：${path}`
    } else {
      await saveBrowser(key)
    }
  } catch (err) {
    store.status = `保存失败：${err instanceof Error ? err.message : err}`
  }
}

async function onAuto(): Promise<void> {
  try { await autoRule() }
  catch (err) { store.status = `自动调色失败：${err instanceof Error ? err.message : err}` }
}

const aiRunning = ref(false)
async function onAutoAI(): Promise<void> {
  if (aiRunning.value) return
  aiRunning.value = true
  try { await autoAI() }
  catch (err) { store.status = `AI 调色失败：${err instanceof Error ? err.message : err}` }
  finally { aiRunning.value = false }
}

// ---- 工具栏状态 ----
const blackwhite = computed(() => store.adjustments.blackwhite > 0)
const cropActive = computed(() => store.editMode === 'crop')
const stampActive = computed(() => store.editMode === 'stamp')
const comparing = computed(() => store.comparing)

function setBlackwhite(on: boolean): void {
  if (blackwhite.value !== on) toggleBlackwhite()
}
function toggleCrop(): void {
  if (cropActive.value) cancelCrop()
  else enterCrop()
}
function toggleStamp(): void {
  if (stampActive.value) {
    store.editMode = null
    store.status = ''
  } else {
    store.editMode = 'stamp'
    store.status = '图章：Alt+拖动选源矩形，直接拖动覆盖目标区域'
  }
}
function toggleCompare(): void {
  setComparing(!store.comparing)
}
function openMaskTab(): void {
  tab.value = 'mask'
}
function onRotate(key: string): void {
  rotate90(key === 'right')
}

// ---- 裁剪条 ----
const aspectOptions = [
  { label: '自由', key: 'free' }, { label: '原始', key: 'orig' },
  { label: '1:1', key: '1:1' }, { label: '3:2', key: '3:2' },
  { label: '4:3', key: '4:3' }, { label: '16:9', key: '16:9' },
  { label: '2:3（转置）', key: '3:2:t' }, { label: '9:16（转置）', key: '16:9:t' },
]
const aspectLabel = computed(() =>
  aspectOptions.find(o => o.key === store.cropAspect)?.label ?? '自由')

// ---- 拖放打开 ----
const dragOver = ref(false)
let dragDepth = 0
function onDragEnter(e: DragEvent): void {
  if (!e.dataTransfer?.types.includes('Files')) return
  dragDepth++
  dragOver.value = true
}
function onDragOver(e: DragEvent): void {
  if (!e.dataTransfer?.types.includes('Files')) return
  e.preventDefault()
}
function onDragLeave(): void {
  dragDepth = Math.max(0, dragDepth - 1)
  if (dragDepth === 0) dragOver.value = false
}
async function onDrop(e: DragEvent): Promise<void> {
  const files = [...(e.dataTransfer?.files ?? [])]
  dragDepth = 0
  dragOver.value = false
  if (!files.length) return
  e.preventDefault()
  try {
    await openFiles(files)
  } catch (err) {
    store.status = `打开失败：${err instanceof Error ? err.message : err}`
  }
}

// ---- 快捷键 ----
function onKey(e: KeyboardEvent): void {
  if (e.target instanceof HTMLElement
      && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
          || e.target.isContentEditable)) return
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
    e.preventDefault(); onOpenClick(); return
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
    e.preventDefault(); void onSavePick('png'); return
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'r') {
    e.preventDefault(); resetAll(); return
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault()
    if (e.shiftKey) redoHistory(); else undoHistory()
    return
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
    e.preventDefault(); redoHistory(); return
  }
  if (e.key === '\\') { setComparing(true); return }
  if (e.key === 'Enter' && cropActive.value) {
    e.preventDefault(); void applyCrop(); return
  }
  if (e.key === 'Delete') {
    if (store.pickSelected !== null) deletePicker(store.pickSelected)
    return
  }
  if (e.key === 'Escape') {
    if (store.pickTarget !== null) { cancelPick(); return }
    if (cropActive.value) { cancelCrop(); return }
    if (stampActive.value) { toggleStamp(); return }
    if (store.editMode === 'mask') { exitMaskEdit(); return }
  }
  if ((e.key === 'o' || e.key === 'O') && !e.ctrlKey && !e.metaKey) {
    if (store.editMode === 'mask') toggleMaskOverlay()
  }
  // Q/E 旋转选中的线性渐变蒙版（±1°，Shift ±15°；Q=逆时针，E=顺时针）
  if ((e.key === 'q' || e.key === 'Q' || e.key === 'e' || e.key === 'E')
      && !e.ctrlKey && !e.metaKey) {
    const i = store.maskSelected
    if (i !== null && store.adjustments.masks[i]?.kind === 'linear') {
      e.preventDefault()
      const step = (e.shiftKey ? 15 : 1) * (e.key === 'q' || e.key === 'Q' ? 1 : -1)
      rotateLinearMask(i, step)
    }
  }
}
function onKeyUp(e: KeyboardEvent): void {
  if (e.key === '\\') setComparing(false)
}
onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKeyUp)
  void initLibrary()
  void refreshLocalAI()
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('keyup', onKeyUp)
})

const themeOverrides = {
  common: {
    primaryColor: '#7d8bff',
    primaryColorHover: '#9aa5ff',
    primaryColorPressed: '#6b78e6',
    primaryColorSuppl: '#7d8bff',
    bodyColor: '#0e1014',
    cardColor: '#12151a',
    modalColor: '#12151a',
    popoverColor: '#161a20',
    tableColor: '#0e1014',
    inputColor: '#0a0c10',
    borderColor: 'rgba(255,255,255,0.11)',
    dividerColor: 'rgba(255,255,255,0.07)',
    textColorBase: '#ecedf1',
    textColor1: '#ecedf1',
    textColor2: '#99a0ac',
    textColor3: '#616875',
    placeholderColor: '#5b626e',
    hoverColor: 'rgba(255,255,255,0.06)',
    borderRadius: '9px',
    fontSize: '12.5px',
  },
  Button: {
    colorSecondary: '#181c23',
    colorSecondaryHover: '#1f242c',
    colorSecondaryPressed: '#141821',
    textColorSecondary: '#c6cbd4',
    borderSecondary: '1px solid rgba(255,255,255,0.10)',
  },
  Switch: {
    railColor: 'rgba(255,255,255,0.14)',
    railColorActive: '#7d8bff',
    buttonColor: '#e9ebf0',
  },
  Slider: {
    railColor: 'rgba(255,255,255,0.10)',
    railColorHover: 'rgba(255,255,255,0.16)',
    fillColor: 'rgba(255,255,255,0.6)',
    fillColorHover: 'rgba(255,255,255,0.75)',
    handleColor: '#f1f2f5',
  },
  Internal: {
    color: '#12151a',
  },
}
</script>

<template>
  <NConfigProvider :theme="darkTheme" :theme-overrides="themeOverrides" class="provider">
    <div
      class="app"
      @dragenter="onDragEnter"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
    >
      <!-- 顶栏 -->
      <header class="topbar">
        <div class="brand">
          <svg class="mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5" />
            <path d="M12 3v9M20.4 7.5 12 12M20.4 16.5 12 12M12 21v-9M3.6 16.5 12 12M3.6 7.5 12 12"
                  stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
          </svg>
          <span class="brand-name">PhotoLab</span>
          <span v-if="store.busy" class="pulse" title="处理中…" />
        </div>

        <input ref="fileInput" type="file" accept="image/*,.dng,.cr2,.cr3,.nef,.arw,.rw2,.raf,.orf" hidden multiple
               @change="onFileChosen" />

        <button class="btn" :title="native ? '打开照片' : '打开照片（可多选）'" @click="onOpenClick">
          <Icon name="open" /> 打开
        </button>
        <NDropdown :options="saveOptions" trigger="click" @select="onSavePick">
          <button class="btn solid" :disabled="!store.loaded" title="导出（Ctrl+S 存 PNG）">
            <Icon name="save" /> 保存 <Icon name="chevron" :size="12" />
          </button>
        </NDropdown>
        <button class="btn icon" :disabled="!store.loaded" title="重置全部参数（Ctrl+R）" @click="resetAll">
          <Icon name="reset" />
        </button>

        <span class="sep" />

        <button class="btn" :disabled="!store.loaded" title="自动调整（规则版）" @click="onAuto">
          <Icon name="auto" /> 自动
        </button>
        <button class="btn" :disabled="!store.loaded || aiRunning" title="AI 自动调色" @click="onAutoAI">
          <Icon name="ai" /> {{ aiRunning ? '调色中…' : 'AI' }}
        </button>
        <button class="btn" :disabled="!store.loaded" title="AI 对话 / 参数 / 生图" @click="openAiChat">
          <Icon name="chat" /> 对话
        </button>

        <span class="sep" />

        <button class="btn" :class="{ on: cropActive }" :disabled="!store.loaded" title="裁剪 / 拉直" @click="toggleCrop">
          <Icon name="crop" /> 裁剪
        </button>
        <button class="btn" :class="{ on: stampActive }" :disabled="!store.loaded" title="仿制图章" @click="toggleStamp">
          <Icon name="stamp" /> 图章
        </button>
        <button class="btn" :class="{ on: tab === 'mask' }" :disabled="!store.loaded" title="蒙版" @click="openMaskTab">
          <Icon name="mask" /> 蒙版
        </button>
        <NDropdown :options="rotateOptions" trigger="click" @select="onRotate">
          <button class="btn icon" :disabled="!store.loaded" title="旋转 90°">
            <Icon name="rotateR" />
          </button>
        </NDropdown>

        <span class="grow" />

        <div class="seg" role="group" aria-label="色彩模式">
          <button :class="{ on: !blackwhite }" @click="setBlackwhite(false)">彩色</button>
          <button :class="{ on: blackwhite }" :disabled="!store.loaded" @click="setBlackwhite(true)">黑白</button>
        </div>
        <button class="btn" :class="{ on: comparing }" :disabled="!store.loaded || !store.originalBmp"
                title="按住与按住不放对比原图（\\）" @click="toggleCompare">
          <Icon name="compare" /> 对比
        </button>
        <button class="btn icon" :title="aiConfigured() ? 'AI 设置' : 'AI 未配置，点击设置端点'" @click="openAiSettings">
          <Icon name="settings" />
          <span v-if="!aiConfigured()" class="alert-dot" />
        </button>
      </header>

      <!-- 裁剪条 -->
      <div v-if="cropActive" class="subbar">
        <NDropdown :options="aspectOptions" trigger="click"
                   @select="(k: string) => store.cropAspect = k">
          <button class="btn outline">比例 · {{ aspectLabel }} <Icon name="chevron" :size="12" /></button>
        </NDropdown>
        <div class="sub-slider">
          <MiniSlider label="拉直" :value="store.cropAngle" :min="-45" :max="45" :step="0.1"
                      :decimals="1" suffix="°" :label-width="30"
                      @update:value="(v: number) => { store.cropAngle = v; notifyChanged() }" />
        </div>
        <span class="grow" />
        <span class="sub-hint">拖动裁剪框与手柄 · <span class="kbd">Enter</span> 应用 · <span class="kbd">Esc</span> 取消</span>
        <button class="btn solid" :disabled="store.cropPending" @click="applyCrop">应用</button>
        <button class="btn" @click="cancelCrop">取消</button>
      </div>

      <!-- 图章条 -->
      <div v-if="stampActive" class="subbar">
        <div class="sub-slider">
          <MiniSlider label="羽化" :value="store.stampFeather" :min="0" :max="100"
                      :label-width="30"
                      @update:value="(v: number) => store.stampFeather = v" />
        </div>
        <button class="btn outline" @click="store.stampSource = null; store.status = '按住 Alt 拖出新源矩形'">重选源区</button>
        <button class="btn outline" @click="stampUndo()">撤销一笔</button>
        <button class="btn outline" @click="stampClear()">全部清除</button>
        <span class="grow" />
        <span class="sub-hint"><span class="kbd">Alt</span> + 拖动 选源 · 直接拖动 覆盖</span>
      </div>

      <!-- 舞台 + 面板 -->
      <div class="body">
        <div class="stage">
          <Viewport />
          <Filmstrip />
        </div>

        <aside class="side">
          <div class="side-head">
            <Histogram :hist="store.histogram" />
          </div>

          <nav class="tabs">
            <button
              v-for="t in TABS"
              :key="t.key"
              class="tab"
              :class="{ on: tab === t.key }"
              @click="tab = t.key"
            >{{ t.label }}</button>
          </nav>

          <div class="side-body">
            <!-- 局部调整提示条 -->
            <div v-if="isMaskTarget" class="banner">
              <span class="banner-dot" />
              <span class="banner-text">{{ maskName }} · 局部调整中</span>
              <button class="mini" @click="exitMaskEdit()">完成</button>
            </div>

            <template v-if="!store.loaded">
              <div class="pane-empty">
                <Icon name="image" :size="26" :stroke="1.2" />
                <p>还没有打开照片</p>
                <span>把图片拖进窗口，或点左上角「打开」</span>
              </div>
            </template>

            <template v-else>
              <div v-if="tab === 'light'" class="stack">
                <SliderRow label="曝光" :min="-5" :max="5" :step="0.01" :decimals="2" suffix=" EV"
                           v-model="target.exposure" @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="对比度" :min="-100" :max="100" v-model="target.contrast"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="高光" :min="-100" :max="100" v-model="target.highlights"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="阴影" :min="-100" :max="100" v-model="target.shadows"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="白色" :min="-100" :max="100" v-model="target.whites"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="黑色" :min="-100" :max="100" v-model="target.blacks"
                           @input="notifyChanged" @release="notifyCommitted" />
              </div>

              <div v-else-if="tab === 'color'" class="stack">
                <SliderRow label="色温" :min="-100" :max="100" v-model="target.temperature"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="色调" :min="-100" :max="100" v-model="target.tint"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="饱和度" :min="-100" :max="100" v-model="target.saturation"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow v-if="!isMaskTarget" label="自然饱和度" :min="-100" :max="100"
                           v-model="store.adjustments.vibrance"
                           @input="notifyChanged" @release="notifyCommitted" />
              </div>

              <div v-else-if="tab === 'effects'" class="stack">
                <SliderRow label="清晰度" :min="-100" :max="100" v-model="target.clarity"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="降噪" :min="0" :max="100" v-model="target.denoise"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="锐度" :min="0" :max="100" v-model="target.sharpen"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow label="去朦胧" :min="-100" :max="100" v-model="target.dehaze"
                           @input="notifyChanged" @release="notifyCommitted" />
                <p class="pane-tip">降噪分亮/色两路：色度重滤去色斑，亮度轻滤保纹理。</p>
              </div>

              <FilterPanel v-else-if="tab === 'filter'" />

              <CurveEditor v-else-if="tab === 'curve'" :curves="target.curves" :hist="store.histogram"
                           @changed="() => notifyChanged()" @committed="() => notifyCommitted()" />

              <GradingPanel v-else-if="tab === 'grading'" :adj="target" />

              <MaskPanel v-else-if="tab === 'mask'" />

              <PickerPanel v-else-if="tab === 'picker'" :pickers="target.pickers" />

              <HistoryPanel v-else-if="tab === 'history'" />
            </template>
          </div>
        </aside>
      </div>

      <!-- 状态栏 -->
      <footer class="statusbar">
        <span v-if="store.loaded" class="num">{{ store.width }} × {{ store.height }}</span>
        <span v-if="store.meta" class="dim">{{ store.meta.gpu ? 'GPU 加速' : 'CPU 渲染' }}</span>
        <span v-if="cropActive" class="dim">裁剪模式</span>
        <span v-else-if="stampActive" class="dim">图章模式</span>
        <span v-else-if="isMaskTarget" class="dim">蒙版局部编辑</span>
        <span class="grow" />
        <span class="msg">{{ store.status }}</span>
      </footer>

      <!-- 拖放遮罩 -->
      <div v-if="dragOver" class="dropzone">
        <div class="dropzone-inner">
          <Icon name="image" :size="30" :stroke="1.2" />
          <span>松开以打开照片</span>
        </div>
      </div>
    </div>
    <AiSettings />
    <AiChat />
  </NConfigProvider>
</template>

<style scoped>
.provider { height: 100%; }

.app {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
  color: var(--text);
  overflow: hidden;
}

/* ---------- 顶栏 ---------- */
.topbar {
  flex: none;
  display: flex;
  align-items: center;
  gap: 4px;
  height: 52px;
  padding: 0 12px 0 18px;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
  z-index: 3;
  box-shadow: 0 1px 0 rgba(0, 0, 0, 0.45);
}
.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-right: 18px;
  user-select: none;
}
.mark {
  width: 21px;
  height: 21px;
  color: var(--accent);
  opacity: 0.95;
  filter: drop-shadow(0 0 6px rgba(224, 161, 99, 0.35));
}
.brand-name {
  font-family: var(--font-brand);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--text);
}
.pulse {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 0 var(--accent-soft);
  animation: pulse 1.5s var(--ease) infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(224, 161, 99, 0.55); }
  70% { box-shadow: 0 0 0 7px rgba(224, 161, 99, 0); }
  100% { box-shadow: 0 0 0 0 rgba(224, 161, 99, 0); }
}
.sep {
  width: 1px;
  height: 20px;
  margin: 0 9px;
  background: linear-gradient(180deg, transparent, var(--line-2), transparent);
  flex: none;
}
.grow { flex: 1; }
.alert-dot {
  position: absolute;
  top: 5px;
  right: 5px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--warn);
}
.btn.icon { position: relative; }

/* ---------- 子条（裁剪 / 图章） ---------- */
.subbar {
  flex: none;
  display: flex;
  align-items: center;
  gap: 10px;
  height: 42px;
  padding: 0 16px;
  background: var(--panel-2);
  border-bottom: 1px solid var(--line);
  box-shadow: inset 0 1px 0 rgba(255, 243, 228, 0.03);
  animation: slide-down 0.16s var(--ease);
}
@keyframes slide-down {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: none; }
}
.sub-slider { width: 210px; }
.sub-hint { font-size: 10.5px; color: var(--text-3); }

/* ---------- 主体 ---------- */
.body {
  flex: 1;
  display: flex;
  min-height: 0;
}
.stage {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.stage > :first-child { flex: 1; min-height: 0; }

/* ---------- 右侧面板 ---------- */
.side {
  flex: none;
  width: 340px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  background: var(--panel);
  border-left: 1px solid var(--line);
  box-shadow: -1px 0 0 rgba(0, 0, 0, 0.35);
}
.side-head {
  flex: none;
  padding: 14px 16px 12px;
  background: radial-gradient(120% 100% at 50% 0%, rgba(224, 161, 99, 0.05), transparent 70%);
}
.tabs {
  flex: none;
  display: flex;
  gap: 2px;
  padding: 0 8px;
  border-bottom: 1px solid var(--line);
}
.tab {
  position: relative;
  flex: 1;
  height: 32px;
  border: 0;
  background: transparent;
  color: var(--text-3);
  font-size: 10.5px;
  letter-spacing: 0.02em;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.14s var(--ease);
}
.tab:hover { color: var(--text-2); }
.tab.on { color: var(--text); }
.tab.on::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: 0;
  width: 16px;
  height: 2px;
  margin-left: -8px;
  border-radius: 2px;
  background: var(--accent);
  box-shadow: 0 0 8px rgba(224, 161, 99, 0.6);
}
.side-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 12px 16px 22px;
}

.stack { display: flex; flex-direction: column; }

/* 局部调整提示条 */
.banner {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  padding: 8px 11px;
  border-radius: var(--r-2);
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.banner-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  flex: none;
}
.banner-text {
  flex: 1;
  font-size: 11.5px;
  color: var(--accent-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 空态 / 提示 */
.pane-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 56px 10px;
  color: var(--text-3);
  text-align: center;
}
.pane-empty p { margin: 6px 0 0; font-size: 13px; color: var(--text-2); letter-spacing: 0.02em; }
.pane-empty span { font-size: 11px; color: var(--text-3); }
.pane-tip {
  margin: 10px 0 0;
  font-size: 10.5px;
  line-height: 1.6;
  color: var(--text-3);
}

/* ---------- 状态栏 ---------- */
.statusbar {
  flex: none;
  display: flex;
  align-items: center;
  gap: 14px;
  height: 28px;
  padding: 0 16px;
  background: var(--panel);
  border-top: 1px solid var(--line);
  font-size: 10.5px;
  letter-spacing: 0.03em;
  color: var(--text-2);
}
.dim { color: var(--text-3); }
.msg { color: var(--text-2); }

/* ---------- 拖放遮罩 ---------- */
.dropzone {
  position: absolute;
  inset: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(6, 5, 4, 0.74);
  backdrop-filter: blur(6px) saturate(0.9);
  pointer-events: none;
}
.dropzone-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 38px 52px;
  border-radius: 18px;
  border: 1px dashed var(--accent-line);
  background: radial-gradient(120% 120% at 50% 0%, rgba(224, 161, 99, 0.10), rgba(224, 161, 99, 0.03));
  color: var(--accent-text);
  font-size: 13px;
  letter-spacing: 0.04em;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.6);
}
</style>
