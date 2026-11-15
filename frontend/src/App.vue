<script setup lang="ts">
/** 主布局：顶栏 + （裁剪/图章条）+ 舞台（视口+胶片条）+ 右侧面板 + 状态栏。
 *  右侧面板用 Tab 分区（光线/色彩/效果/曲线/分级/蒙版/取色），一次只显示一组，
 *  避免整列折叠面板的视觉噪音。快捷键对照 README：
 *  Ctrl+O 打开 / Ctrl+S 保存 / Ctrl+R 重置 / `\` 对比原图 / Delete 删除取色点
 *  / Esc 退出编辑 / 0、1、+、- 缩放（Viewport 内）。 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { NDropdown, NConfigProvider } from 'naive-ui'
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
import AiStudio from '@/components/AiStudio.vue'
import AiChat from '@/components/AiChat.vue'
import PreferencesModal from '@/components/PreferencesModal.vue'
import Icon from '@/components/Icon'
import { HINT } from '@/lib/hints'
import { t, tx } from '@/lib/i18n'
import { naiveTheme, naiveThemeOverrides } from '@/lib/theme'
import {
  useEditor, resetAll, notifyChanged, notifyCommitted,
  toggleBlackwhite, saveBrowser, autoRule, autoAI, cancelPick, deletePicker,
  targetAdj, setComparing, enterCrop, cancelCrop, applyCrop, rotate90,
  stampUndo, stampClear, toggleStampPicking,
  exitMaskEdit, toggleMaskOverlay, maskOverlayVisible, setActiveTab,
  confirmPolygonMask, cancelPolygonDrawing, undoPolygonPoint,
  openFiles, openPaths,
  initLibrary, openAiSettings, openAiStudio, openPrefs, openAiChat, aiConfigured,
  rotateLinearMask, refreshLocalAI,
  undoHistory, redoHistory,
} from '@/stores/editor'
import type { EditorTab } from '@/stores/editor'
import { native, openDesktop, saveDesktop } from '@/api/client'

const store = useEditor()
const fileInput = ref<HTMLInputElement | null>(null)

// ---- 右侧面板 Tab ----
type TabKey = EditorTab
const TABS = computed<{ key: TabKey; label: string }[]>(() => [
  { key: 'light', label: tx('光线', 'Light') },
  { key: 'color', label: t('tab.basic') },
  { key: 'effects', label: tx('效果', 'Effects') },
  { key: 'filter', label: t('tab.filters') },
  { key: 'curve', label: t('tab.curve') },
  { key: 'grading', label: t('tab.grading') },
  { key: 'mask', label: t('tab.mask') },
  { key: 'picker', label: t('tab.picker') },
  { key: 'history', label: t('tab.history') },
])
const tab = computed<TabKey>({
  get: () => store.activeTab,
  set: (v) => setActiveTab(v),
})

// ---- 面板目标（选中蒙版 → 局部参数）----
const target = computed(() => targetAdj())
const isMaskTarget = computed(() => store.maskSelected !== null)
const maskName = computed(() => {
  const i = store.maskSelected
  return i === null ? '' : store.adjustments.masks[i]?.name || t('tab.mask')
})

// ---- 保存 ----
const saveOptions = computed(() => [
  { label: tx('PNG（无损）', 'PNG (Lossless)'), key: 'png' },
  { label: tx('JPEG（质量 95）', 'JPEG (95%)'), key: 'jpg' },
  { label: 'WebP', key: 'webp' },
])
const rotateOptions = computed(() => [
  { label: t('crop.rotateL'), key: 'left' },
  { label: t('crop.rotateR'), key: 'right' },
])

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
      const path = await saveDesktop(store.adjustments, `fernlens.${key}`)
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
    store.stampPickingSource = false
    store.status = ''
  } else {
    if (store.editMode === 'mask') exitMaskEdit()
    store.editMode = 'stamp'
    store.status = store.stampSourcePt ? t('stamp.hintReady') : t('stamp.hintNeedSrc')
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
const aspectOptions = computed(() => [
  { label: t('crop.ratioFree'), key: 'free' },
  { label: t('crop.ratioOrig'), key: 'orig' },
  { label: '1:1', key: '1:1' }, { label: '3:2', key: '3:2' },
  { label: '4:3', key: '4:3' }, { label: '16:9', key: '16:9' },
  { label: '2:3', key: '3:2:t' }, { label: '9:16', key: '16:9:t' },
])
const aspectLabel = computed(() =>
  aspectOptions.value.find(o => o.key === store.cropAspect)?.label ?? t('crop.ratioFree'))

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
  if (e.key === 'Enter' && store.polyDrawing) {
    e.preventDefault(); confirmPolygonMask(); return
  }
  if (e.key === 'Enter' && cropActive.value) {
    e.preventDefault(); void applyCrop(); return
  }
  if (e.key === 'Backspace' && store.polyDrawing) {
    e.preventDefault(); undoPolygonPoint(); return
  }
  if (e.key === 'Delete') {
    if (store.polyDrawing) { e.preventDefault(); undoPolygonPoint(); return }
    if (store.pickSelected !== null) deletePicker(store.pickSelected)
    return
  }
  if (e.key === 'Escape') {
    if (store.polyDrawing) { cancelPolygonDrawing(); return }
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

</script>

<template>
  <NConfigProvider :theme="naiveTheme" :theme-overrides="naiveThemeOverrides" class="provider">
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
          <span class="brand-name">FernLens</span>
          <span v-if="store.busy" class="pulse" />
        </div>

        <input ref="fileInput" type="file" accept="image/*,.dng,.cr2,.cr3,.nef,.arw,.rw2,.raf,.orf" hidden multiple
               @change="onFileChosen" />

        <button class="btn" :title="t('app.openTip')" @click="onOpenClick">
          <Icon name="open" /> {{ t('app.open') }}
        </button>
        <NDropdown :options="saveOptions" trigger="click" @select="onSavePick">
          <button class="btn solid" :disabled="!store.loaded" :title="t('app.saveAsTip')">
            <Icon name="save" /> {{ t('app.saveAs') }} <Icon name="chevron" :size="12" />
          </button>
        </NDropdown>
        <button class="btn icon" :disabled="!store.loaded" :title="t('app.resetAllTip')" @click="resetAll">
          <Icon name="reset" />
        </button>

        <span class="sep" />

        <button class="btn" :disabled="!store.loaded" :title="t('app.autoTip')" @click="onAuto">
          <Icon name="auto" /> {{ t('app.auto') }}
        </button>
        <button class="btn" :disabled="!store.loaded || aiRunning" :title="t('app.aiGradeTip')" @click="onAutoAI">
          <Icon name="ai" /> {{ aiRunning ? '…' : t('app.aiGrade') }}
        </button>
        <button class="btn" :class="{ on: store.aiStudioOpen }" :disabled="!store.loaded" :title="t('app.aiStudioTip')" @click="openAiStudio">
          <Icon name="layers" /> {{ t('app.aiStudio') }}
        </button>
        <button class="btn" :disabled="!store.loaded" :title="t('aiChat.title')" @click="openAiChat">
          <Icon name="chat" /> {{ tx('对话', 'Chat') }}
        </button>

        <span class="sep" />

        <button class="btn" :class="{ on: cropActive }" :disabled="!store.loaded" :title="t('app.cropTip')" @click="toggleCrop">
          <Icon name="crop" /> {{ t('app.crop') }}
        </button>
        <button class="btn" :class="{ on: stampActive }" :disabled="!store.loaded" :title="t('app.stampTip')" @click="toggleStamp">
          <Icon name="stamp" /> {{ t('app.stamp') }}
        </button>
        <button class="btn" :class="{ on: tab === 'mask' }" :disabled="!store.loaded" :title="t('tab.mask')" @click="openMaskTab">
          <Icon name="mask" /> {{ t('tab.mask') }}
        </button>
        <NDropdown :options="rotateOptions" trigger="click" @select="onRotate">
          <button class="btn icon" :disabled="!store.loaded" :title="t('crop.rotateR')">
            <Icon name="rotateR" />
          </button>
        </NDropdown>

        <span class="grow" />

        <div class="seg" role="group">
          <button :class="{ on: !blackwhite }" @click="setBlackwhite(false)">{{ tx('彩色', 'Color') }}</button>
          <button :class="{ on: blackwhite }" :disabled="!store.loaded" @click="setBlackwhite(true)">{{ t('bw.enable') }}</button>
        </div>
        <button class="btn" :class="{ on: comparing }" :disabled="!store.loaded || !store.originalBmp"
                :title="t('app.compareTip')" @click="toggleCompare">
          <Icon name="compare" /> {{ tx('对比', 'Before') }}
        </button>
        <button class="btn" :title="t('app.prefsTip')" @click="openPrefs">
          <Icon name="palette" /> {{ t('app.prefs') }}
        </button>
        <button class="btn icon" :title="t('app.aiSettingsTip')" @click="openAiSettings">
          <Icon name="settings" />
          <span v-if="!aiConfigured()" class="alert-dot" />
        </button>
      </header>

      <!-- 裁剪条 -->
      <div v-if="cropActive" class="subbar">
        <NDropdown :options="aspectOptions" trigger="click"
                   @select="(k: string) => store.cropAspect = k">
          <button class="btn outline">{{ t('crop.ratio') }} · {{ aspectLabel }} <Icon name="chevron" :size="12" /></button>
        </NDropdown>
        <div class="sub-slider">
          <MiniSlider :label="t('crop.angle')" :hint="HINT.cropAngle" :value="store.cropAngle"
                      :min="-45" :max="45" :step="0.1"
                      :decimals="1" suffix="°" :label-width="52"
                      @update:value="(v: number) => { store.cropAngle = v; notifyChanged() }" />
        </div>
        <span class="grow" />
        <span class="sub-hint"><span class="kbd">Enter</span> {{ t('crop.apply') }} · <span class="kbd">Esc</span> {{ t('crop.done') }}</span>
        <button class="btn solid" :disabled="store.cropPending" @click="applyCrop">{{ t('crop.apply') }}</button>
        <button class="btn" @click="cancelCrop">{{ t('crop.done') }}</button>
      </div>

      <!-- Photoshop 式仿制图章工具条 -->
      <div v-if="stampActive" class="subbar stamp-bar">
        <div class="sub-slider sm">
          <MiniSlider :label="t('stamp.size')" :hint="t('stamp.sizeTip')" :value="store.stampSize"
                      :min="2" :max="100" :label-width="38"
                      @update:value="(v: number) => (store.stampSize = v)" />
        </div>
        <div class="sub-slider sm">
          <MiniSlider :label="t('stamp.feather')" :hint="t('stamp.featherTip')" :value="store.stampFeather"
                      :min="0" :max="100" :label-width="46"
                      @update:value="(v: number) => (store.stampFeather = v)" />
        </div>
        <div class="sub-slider sm">
          <MiniSlider :label="t('stamp.opacity')" :hint="t('stamp.opacityTip')" :value="store.stampOpacity"
                      :min="10" :max="100" unit="%" :label-width="52"
                      @update:value="(v: number) => (store.stampOpacity = v)" />
        </div>
        <button
          class="btn outline"
          :class="{ on: store.stampAligned }"
          :title="t('stamp.alignedTip')"
          @click="store.stampAligned = !store.stampAligned"
        >
          {{ t('stamp.aligned') }}
        </button>
        <button
          class="btn outline"
          :class="{ on: store.stampPickingSource }"
          :title="t('stamp.pickTip')"
          @click="toggleStampPicking"
        >
          <Icon name="crosshair" :size="13" />
          {{ store.stampPickingSource ? t('stamp.pickActive') : t('stamp.pick') }}
        </button>
        <button class="btn outline" @click="stampUndo()">{{ t('stamp.undo') }}</button>
        <button class="btn outline" @click="stampClear()">{{ t('stamp.clear') }}</button>
        <span class="grow" />
        <span class="sub-hint"><span class="kbd">Alt</span>+Click · <span class="kbd">[</span><span class="kbd">]</span></span>
        <button class="btn solid" @click="toggleStamp">{{ t('stamp.done') }}</button>
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
              <span class="banner-text">{{ maskName }} · {{ tx('局部调整中', 'Local Mask Edit') }}</span>
              <button class="mini square" :class="{ on: maskOverlayVisible() }"
                      :title="`${t('mask.showOverlay')} (O)`" @click="toggleMaskOverlay()">
                <Icon name="eye" :size="11" />
              </button>
              <button class="mini" @click="exitMaskEdit()">{{ t('crop.done') }}</button>
            </div>

            <template v-if="!store.loaded">
              <div class="pane-empty">
                <Icon name="image" :size="26" :stroke="1.2" />
                <p>{{ t('app.noImage') }}</p>
                <span>{{ t('app.dropTitle') }}</span>
              </div>
            </template>

            <template v-else>
              <div v-if="tab === 'light'" class="stack">
                <SliderRow :label="t('param.exposure')" :hint="HINT.exposure"
                           :min="-5" :max="5" :step="0.01" :decimals="2" suffix=" EV"
                           v-model="target.exposure" @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.contrast')" :hint="HINT.contrast" :min="-100" :max="100" v-model="target.contrast"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.highlights')" :hint="HINT.highlights" :min="-100" :max="100" v-model="target.highlights"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.shadows')" :hint="HINT.shadows" :min="-100" :max="100" v-model="target.shadows"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.whites')" :hint="HINT.whites" :min="-100" :max="100" v-model="target.whites"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.blacks')" :hint="HINT.blacks" :min="-100" :max="100" v-model="target.blacks"
                           @input="notifyChanged" @release="notifyCommitted" />
              </div>

              <div v-else-if="tab === 'color'" class="stack">
                <SliderRow :label="t('param.temp')" :hint="HINT.temperature" :min="-100" :max="100" v-model="target.temperature"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.tint')" :hint="HINT.tint" :min="-100" :max="100" v-model="target.tint"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.saturation')" :hint="HINT.saturation" :min="-100" :max="100" v-model="target.saturation"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow v-if="!isMaskTarget" :label="t('param.vibrance')" :hint="HINT.vibrance"
                           :min="-100" :max="100"
                           v-model="store.adjustments.vibrance"
                           @input="notifyChanged" @release="notifyCommitted" />
              </div>

              <div v-else-if="tab === 'effects'" class="stack">
                <SliderRow :label="t('param.clarity')" :hint="HINT.clarity" :min="-100" :max="100" v-model="target.clarity"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.denoise')" :hint="HINT.denoise" :min="0" :max="100" v-model="target.denoise"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.sharpness')" :hint="HINT.sharpen" :min="0" :max="100" v-model="target.sharpen"
                           @input="notifyChanged" @release="notifyCommitted" />
                <SliderRow :label="t('param.dehaze')" :hint="HINT.dehaze" :min="-100" :max="100" v-model="target.dehaze"
                           @input="notifyChanged" @release="notifyCommitted" />
                <p class="pane-tip">{{ tx('降噪分亮/色两路：色度重滤去色斑，亮度轻滤保纹理。', 'Split chroma & luma noise reduction preserves fine textures while removing color blotches.') }}</p>
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
        <span v-if="store.meta" class="dim">{{ store.meta.gpu ? tx('GPU 加速', 'GPU Accelerated') : tx('CPU 渲染', 'CPU Render') }}</span>
        <span v-if="cropActive" class="dim">{{ t('crop.title') }}</span>
        <span v-else-if="stampActive" class="dim">{{ t('stamp.title') }}</span>
        <span v-else-if="isMaskTarget" class="dim">{{ tx('蒙版局部编辑', 'Local Mask Edit') }}</span>
        <span class="grow" />
        <span class="msg">{{ store.status }}</span>
      </footer>

      <!-- 拖放遮罩 -->
      <div v-if="dragOver" class="dropzone">
        <div class="dropzone-inner">
          <Icon name="image" :size="30" :stroke="1.2" />
          <span>{{ tx('松开以打开照片', 'Drop to open photos') }}</span>
        </div>
      </div>
    </div>
    <AiSettings />
    <AiStudio />
    <AiChat />
    <PreferencesModal />
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

/* Soft ambient blue layer behind the frosted-glass topbar & sidebar in light/dark themes */
:global(:root[data-theme="light"]) .app {
  background:
    radial-gradient(650px 180px at 88% 4%, rgba(56, 189, 248, 0.24), transparent 72%),
    radial-gradient(480px 360px at 94% 62%, rgba(96, 165, 250, 0.22), transparent 74%),
    radial-gradient(520px 140px at 20% 0%, rgba(147, 197, 253, 0.24), transparent 70%),
    #ffffff;
}
:global(:root[data-theme="dark"]) .app {
  background:
    radial-gradient(650px 180px at 86% 4%, rgba(30, 64, 175, 0.34), transparent 72%),
    radial-gradient(500px 380px at 94% 65%, rgba(29, 78, 216, 0.26), transparent 74%),
    radial-gradient(520px 140px at 20% 0%, rgba(37, 99, 235, 0.22), transparent 70%),
    #000000;
}

/* ---------- 顶栏 ---------- */
.topbar {
  flex: none;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 3px;
  min-height: 48px;
  padding: 5px 12px 5px 14px;
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-bottom: 1px solid var(--line);
  z-index: 4;
  box-shadow: var(--glass-shadow);
}
.topbar :deep(.btn) {
  height: 28px;
  padding: 0 9px;
  font-size: 11.5px;
}
.topbar :deep(.btn.icon) {
  width: 28px;
  padding: 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-right: 12px;
  user-select: none;
  flex: none;
}
.mark {
  width: 21px;
  height: 21px;
  color: var(--accent);
  opacity: 0.96;
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
  animation: pulse 1.5s var(--ease) infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 var(--accent-soft); }
  70% { box-shadow: 0 0 0 7px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}
.sep {
  width: 1px;
  height: 20px;
  margin: 0 6px;
  background: linear-gradient(180deg, transparent, var(--line-2), transparent);
  flex: none;
}
.grow { flex: 1; min-width: 8px; }
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
  flex-wrap: wrap;
  gap: 10px;
  min-height: 42px;
  padding: 4px 16px;
  background: var(--glass-sub);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-bottom: 1px solid var(--line);
  z-index: 3;
  animation: slide-down 0.16s var(--ease);
}
@keyframes slide-down {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: none; }
}
.sub-slider { width: 200px; }
.sub-slider.sm { width: 158px; }
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
  width: 348px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-left: 1px solid var(--line);
  box-shadow: var(--glass-shadow);
  z-index: 2;
}
.side-head {
  flex: none;
  padding: 12px 16px 10px;
  background: radial-gradient(120% 100% at 50% 0%, var(--accent-soft), transparent 72%);
}
.tabs {
  flex: none;
  display: flex;
  align-items: center;
  gap: 1px;
  padding: 0 6px;
  border-bottom: 1px solid var(--line);
  overflow-x: auto;
  scrollbar-width: none;
}
.tabs::-webkit-scrollbar { display: none; }
.tab {
  position: relative;
  flex: 1 0 auto;
  height: 32px;
  padding: 0 3px;
  border: 0;
  background: transparent;
  color: var(--text-3);
  font-size: 10px;
  letter-spacing: 0;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.14s var(--ease);
}
.tab:hover { color: var(--text-2); }
.tab.on { color: var(--text); font-weight: 600; }
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
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-top: 1px solid var(--line);
  font-size: 10.5px;
  letter-spacing: 0.03em;
  color: var(--text-2);
  z-index: 2;
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
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
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
  background: var(--glass-modal);
  color: var(--accent-text);
  font-size: 13px;
  letter-spacing: 0.04em;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
}
</style>
