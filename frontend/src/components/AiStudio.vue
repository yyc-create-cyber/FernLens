<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { NButton, NSwitch } from 'naive-ui'
import {
  useEditor, closeAiStudio, runDepthRelight, runColorMatch, jumpToHistory,
  applyDof, clearDof, setDofPicking, refreshLocalAI,
} from '@/stores/editor'
import type { DepthRelightPreset, ColorMatchResult } from '@/api/client'
import MiniSlider from '@/components/MiniSlider.vue'
import Icon from '@/components/Icon'
import { t, tx } from '@/lib/i18n'

const store = useEditor()

const tab = ref<'dof' | 'depth' | 'match'>('dof')

// ---- Tab 0: AI Depth of Field (Optical Bokeh) ----
const dofBusy = ref(false)
const dofErr = ref('')

const depthModelReady = computed(() =>
  Boolean(store.localAI?.runtime && store.localAI?.models?.depth_anything_v2?.ready),
)

watch(() => store.aiStudioOpen, (open) => {
  if (open && !store.localAI) void refreshLocalAI()
})

async function runApplyDof(): Promise<void> {
  if (!store.loaded || dofBusy.value) return
  dofBusy.value = true
  dofErr.value = ''
  try {
    await applyDof()
  } catch (err) {
    dofErr.value = err instanceof Error ? err.message : String(err)
  } finally {
    dofBusy.value = false
  }
}

async function runClearDof(): Promise<void> {
  if (!store.loaded || dofBusy.value) return
  dofBusy.value = true
  dofErr.value = ''
  try {
    await clearDof()
  } catch (err) {
    dofErr.value = err instanceof Error ? err.message : String(err)
  } finally {
    dofBusy.value = false
  }
}

function onDofSliderRelease(): void {
  if (store.dofActive && store.loaded && !dofBusy.value) {
    void runApplyDof()
  }
}

// ---- Tab 1: AI Depth Relight ----
const depthPreset = ref<DepthRelightPreset>('studio')
const depthStrength = ref(85)     // 10..150 (%)
const depthBias = ref(0)          // -40..40 (%)
const depthBusy = ref(false)
const depthErr = ref('')
const depthPreviewPng = ref('')

const PRESETS: { id: DepthRelightPreset; nameKey: any; descKey: any }[] = [
  { id: 'studio',     nameKey: 'studio.preset.studio',     descKey: 'studio.preset.studioDesc' },
  { id: 'atmosphere', nameKey: 'studio.preset.atmosphere', descKey: 'studio.preset.atmosphereDesc' },
  { id: 'warm_cool',  nameKey: 'studio.preset.warm_cool',  descKey: 'studio.preset.warm_coolDesc' },
  { id: 'cinematic',  nameKey: 'studio.preset.cinematic',  descKey: 'studio.preset.cinematicDesc' },
]

async function applyDepth(masksOnly = false): Promise<void> {
  if (!store.loaded || depthBusy.value) return
  depthBusy.value = true
  depthErr.value = ''
  try {
    const res = await runDepthRelight({
      preset: depthPreset.value,
      strength: depthStrength.value / 100,
      nearBias: depthBias.value / 100,
      masksOnly,
    })
    if (res?.depth_preview_png) {
      depthPreviewPng.value = `data:image/png;base64,${res.depth_preview_png}`
    }
  } catch (err) {
    depthErr.value = err instanceof Error ? err.message : String(err)
  } finally {
    depthBusy.value = false
  }
}

// ---- Tab 2: AI Reference Color Match ----
const matchFile = ref<File | null>(null)
const matchFileUrl = ref('')
const matchLibId = ref<number | null>(null)
const matchStrength = ref(75)     // 10..100 (%)
const matchProtect = ref(false)
const matchBusy = ref(false)
const matchErr = ref('')
const matchRes = ref<ColorMatchResult | null>(null)
const matchPrevHistoryIdx = ref<number | null>(null)

const otherImages = computed(() =>
  store.library.filter((im) => im.id !== store.currentImageId),
)

watch(() => store.currentImageId, () => {
  depthPreviewPng.value = ''
  matchRes.value = null
  matchPrevHistoryIdx.value = null
  if (matchLibId.value === store.currentImageId) matchLibId.value = null
})

function onPickRefFile(ev: Event): void {
  const input = ev.target as HTMLInputElement
  const f = input.files?.[0]
  input.value = ''
  if (!f) return
  if (matchFileUrl.value) URL.revokeObjectURL(matchFileUrl.value)
  matchFile.value = f
  matchFileUrl.value = URL.createObjectURL(f)
  matchLibId.value = null
  matchRes.value = null
}

function selectLibRef(id: number): void {
  if (matchFileUrl.value) {
    URL.revokeObjectURL(matchFileUrl.value)
    matchFileUrl.value = ''
  }
  matchFile.value = null
  matchLibId.value = id
  matchRes.value = null
}

async function applyMatch(): Promise<void> {
  if (!store.loaded || matchBusy.value) return
  if (!matchFile.value && matchLibId.value == null) return
  matchBusy.value = true
  matchErr.value = ''
  const beforeIdx = store.historyIndex
  try {
    const res = await runColorMatch({
      file: matchFile.value ?? undefined,
      refId: matchLibId.value ?? undefined,
      strength: matchStrength.value / 100,
      protectSubject: matchProtect.value,
    })
    if (res) {
      matchRes.value = res
      matchPrevHistoryIdx.value = beforeIdx
    }
  } catch (err) {
    matchErr.value = err instanceof Error ? err.message : String(err)
  } finally {
    matchBusy.value = false
  }
}

function undoMatch(): void {
  if (matchPrevHistoryIdx.value !== null && matchPrevHistoryIdx.value >= 0) {
    jumpToHistory(matchPrevHistoryIdx.value)
    matchPrevHistoryIdx.value = null
    matchRes.value = null
  }
}
</script>

<template>
  <aside v-if="store.aiStudioOpen" class="studio-drawer glass-surface" role="dialog">
    <header class="studio-head">
      <div class="studio-title-wrap">
        <Icon name="layers" :size="16" />
        <div>
          <div class="studio-title">{{ t('studio.title') }}</div>
          <div class="studio-sub">{{ t('studio.subtitle') }}</div>
        </div>
      </div>
      <button type="button" class="btn icon" @click="closeAiStudio">
        <Icon name="close" :size="14" />
      </button>
    </header>

    <div class="studio-tabs">
      <button type="button" :class="{ on: tab === 'dof' }" @click="tab = 'dof'">
        {{ t('studio.tabDof') }}
      </button>
      <button type="button" :class="{ on: tab === 'depth' }" @click="tab = 'depth'">
        {{ t('studio.tabDepth') }}
      </button>
      <button type="button" :class="{ on: tab === 'match' }" @click="tab = 'match'">
        {{ t('studio.tabMatch') }}
      </button>
    </div>

    <div class="studio-body">
      <!-- Tab 0: AI Depth of Field (Depth Anything V2 Bokeh) -->
      <template v-if="tab === 'dof'">
        <div class="dof-badges">
          <span class="model-pill" :class="{ ready: depthModelReady }">
            Depth Anything V2 (HF)
          </span>
          <span v-if="store.dofActive" class="active-pill">
            {{ t('studio.dofActiveBadge') }}
          </span>
        </div>

        <p class="desc">{{ t('studio.dofDesc') }}</p>

        <div class="dof-pick-row">
          <NButton
            size="small"
            :type="store.dofPicking ? 'primary' : 'default'"
            :disabled="!store.loaded || dofBusy"
            @click="setDofPicking(!store.dofPicking)"
          >
            {{ store.dofPicking ? t('studio.dofPicking') : t('studio.dofPickBtn') }}
          </NButton>
        </div>

        <MiniSlider
          :label="t('studio.dofFocus')"
          :label-width="104"
          :value="Math.round(store.dofFocus * 100)"
          :min="0"
          :max="100"
          :step="1"
          suffix="%"
          @update:value="(v: number) => (store.dofFocus = v / 100)"
          @release="onDofSliderRelease"
        />
        <MiniSlider
          :label="t('studio.dofAperture')"
          :label-width="104"
          :value="store.dofAperture"
          :min="0"
          :max="100"
          :step="1"
          suffix="%"
          @update:value="(v: number) => (store.dofAperture = v)"
          @release="onDofSliderRelease"
        />
        <MiniSlider
          :label="t('studio.dofRange')"
          :label-width="104"
          :value="store.dofRange"
          :min="0"
          :max="60"
          :step="1"
          suffix="%"
          @update:value="(v: number) => (store.dofRange = v)"
          @release="onDofSliderRelease"
        />
        <MiniSlider
          :label="t('studio.dofBokeh')"
          :label-width="104"
          :value="store.dofBokeh"
          :min="0"
          :max="100"
          :step="2"
          suffix="%"
          @update:value="(v: number) => (store.dofBokeh = v)"
          @release="onDofSliderRelease"
        />

        <div class="action-row">
          <NButton
            type="primary"
            size="small"
            :loading="dofBusy"
            :disabled="!store.loaded"
            @click="runApplyDof"
          >
            {{ dofBusy ? t('studio.working') : t('studio.dofApply') }}
          </NButton>
          <NButton
            size="small"
            :disabled="!store.loaded || (!store.dofActive && !store.dofFocusPt)"
            @click="runClearDof"
          >
            {{ t('studio.dofClear') }}
          </NButton>
        </div>

        <p v-if="dofErr" class="err">{{ dofErr }}</p>

        <div v-if="store.dofPreviewPng" class="depth-preview-box">
          <div class="block-title">{{ t('studio.dofFocusPreview') }}</div>
          <img
            :src="`data:image/png;base64,${store.dofPreviewPng}`"
            class="depth-img"
            alt="Focal plane depth preview"
          />
        </div>
      </template>

      <!-- Tab 1: AI Depth Relight -->
      <template v-else-if="tab === 'depth'">
        <p class="desc">{{ t('studio.depthDesc') }}</p>

        <div class="block-title">{{ t('studio.depthPreset') }}</div>
        <div class="preset-grid">
          <button
            v-for="p in PRESETS"
            :key="p.id"
            type="button"
            class="preset-card"
            :class="{ on: depthPreset === p.id }"
            @click="depthPreset = p.id"
          >
            <div class="preset-name">{{ t(p.nameKey) }}</div>
            <div class="preset-desc">{{ t(p.descKey) }}</div>
          </button>
        </div>

        <MiniSlider
          :label="t('studio.strength')"
          :value="depthStrength"
          :min="10"
          :max="150"
          :step="5"
          :def="85"
          unit="%"
          @update:value="(v: number) => (depthStrength = v)"
        />
        <MiniSlider
          :label="t('studio.nearBias')"
          :value="depthBias"
          :min="-40"
          :max="40"
          :step="2"
          :def="0"
          bipolar
          unit="%"
          @update:value="(v: number) => (depthBias = v)"
        />

        <div class="action-row">
          <NButton
            type="primary"
            size="small"
            :loading="depthBusy"
            :disabled="!store.loaded"
            @click="applyDepth(false)"
          >
            {{ depthBusy ? t('studio.working') : t('studio.applyDepth') }}
          </NButton>
          <NButton
            size="small"
            :loading="depthBusy"
            :disabled="!store.loaded"
            @click="applyDepth(true)"
          >
            {{ t('studio.masksOnlyBtn') }}
          </NButton>
        </div>

        <p v-if="depthErr" class="err">{{ depthErr }}</p>

        <div v-if="depthPreviewPng" class="depth-preview-box">
          <div class="block-title">{{ t('studio.depthPreview') }}</div>
          <img :src="depthPreviewPng" class="depth-img" alt="Depth map preview" />
        </div>
        <p class="hint-note">{{ t('studio.replaceHint') }}</p>
      </template>

      <!-- Tab 2: AI Reference Color Match -->
      <template v-else>
        <p class="desc">{{ t('studio.matchDesc') }}</p>

        <label class="upload-box">
          <input type="file" accept="image/*" hidden @change="onPickRefFile" />
          <Icon name="upload" :size="15" />
          <span>{{ matchFile ? matchFile.name : t('studio.matchUpload') }}</span>
        </label>

        <img v-if="matchFileUrl" :src="matchFileUrl" class="ref-thumb" alt="Reference" />

        <div class="lib-section">
          <div class="block-title">{{ t('studio.matchLibrary') }}</div>
          <div v-if="otherImages.length" class="lib-chips">
            <button
              v-for="im in otherImages"
              :key="im.id"
              type="button"
              class="mini"
              :class="{ on: matchLibId === im.id }"
              @click="selectLibRef(im.id)"
            >
              {{ im.name }}
            </button>
          </div>
          <p v-else class="hint-note">{{ t('studio.matchNoneLib') }}</p>
        </div>

        <MiniSlider
          :label="t('studio.strength')"
          :value="matchStrength"
          :min="10"
          :max="100"
          :step="5"
          :def="75"
          unit="%"
          @update:value="(v: number) => (matchStrength = v)"
        />

        <label class="protect-row">
          <span>{{ t('studio.matchProtect') }}</span>
          <NSwitch v-model:value="matchProtect" size="small" />
        </label>

        <div class="action-row">
          <NButton
            type="primary"
            size="small"
            :loading="matchBusy"
            :disabled="!store.loaded || (!matchFile && matchLibId === null)"
            @click="applyMatch"
          >
            {{ matchBusy ? t('studio.working') : t('studio.matchApply') }}
          </NButton>
          <NButton
            v-if="matchPrevHistoryIdx !== null"
            size="small"
            @click="undoMatch"
          >
            {{ t('studio.matchUndo') }}
          </NButton>
        </div>

        <p v-if="matchErr" class="err">{{ matchErr }}</p>

        <div v-if="matchRes" class="palette-card">
          <div class="palette-group">
            <span class="block-title">{{ t('studio.refPalette') }}</span>
            <div class="swatches">
              <span
                v-for="(hex, i) in matchRes.ref_palette"
                :key="'r' + i"
                class="swatch"
                :style="{ background: hex }"
                :title="hex"
              />
            </div>
          </div>
          <div class="palette-group">
            <span class="block-title">{{ t('studio.srcPalette') }}</span>
            <div class="swatches">
              <span
                v-for="(hex, i) in matchRes.src_palette"
                :key="'s' + i"
                class="swatch"
                :style="{ background: hex }"
                :title="hex"
              />
            </div>
          </div>
          <p class="hint-note">{{ tx('已提取色彩并融合为可手动微调的影调/色轮参数', matchRes.note) }}</p>
        </div>
      </template>
    </div>
  </aside>
</template>

<style scoped>
.studio-drawer {
  position: fixed;
  right: 332px;
  bottom: 34px;
  z-index: 42;
  width: min(390px, calc(100vw - 360px));
  max-height: calc(100vh - 110px);
  display: flex;
  flex-direction: column;
  border-radius: 14px;
  border: 1px solid var(--line-2);
  box-shadow: var(--glass-shadow);
  overflow: hidden;
}
.studio-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px 14px 10px;
  border-bottom: 1px solid var(--line);
}
.studio-title-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--accent);
}
.studio-title {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--text);
}
.studio-sub {
  font-size: 10.5px;
  color: var(--text-3);
}
.studio-tabs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  padding: 8px 14px;
  border-bottom: 1px solid var(--line);
}
.dof-badges {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.model-pill,
.active-pill {
  font-size: 10.5px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: var(--surface-1);
  color: var(--text-3);
}
.model-pill.ready {
  color: var(--accent-text);
  border-color: var(--accent-line);
  background: var(--accent-soft);
}
.active-pill {
  color: #10b981;
  border-color: rgba(16, 185, 129, 0.45);
  background: rgba(16, 185, 129, 0.12);
}
.dof-pick-row {
  display: flex;
  gap: 8px;
}
.studio-tabs button {
  height: 28px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface-1);
  color: var(--text-2);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.16s var(--ease);
}
.studio-tabs button.on {
  background: var(--accent-soft);
  border-color: var(--accent-line);
  color: var(--accent-text);
}
.studio-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px 14px 14px;
  display: flex;
  flex-direction: column;
  gap: 11px;
}
.desc {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-2);
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px;
}
.preset-card {
  padding: 8px 9px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--surface-1);
  text-align: left;
  cursor: pointer;
  transition: all 0.15s var(--ease);
}
.preset-card:hover {
  border-color: var(--accent-line);
  background: var(--surface-2);
}
.preset-card.on {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.preset-name {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text);
}
.preset-desc {
  margin-top: 2px;
  font-size: 10px;
  line-height: 1.35;
  color: var(--text-3);
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 2px;
}
.err {
  margin: 0;
  font-size: 11.5px;
  color: var(--danger);
}
.hint-note {
  margin: 0;
  font-size: 10.5px;
  color: var(--text-3);
  line-height: 1.45;
}
.depth-preview-box {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.depth-img {
  width: 100%;
  max-height: 136px;
  object-fit: contain;
  border-radius: 7px;
  border: 1px solid var(--line);
  background: var(--well);
}
.upload-box {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px;
  border-radius: 9px;
  border: 1.5px dashed var(--line-3);
  background: var(--surface-1);
  color: var(--text-2);
  font-size: 12px;
  cursor: pointer;
}
.upload-box:hover {
  border-color: var(--accent);
  color: var(--text);
}
.ref-thumb {
  width: 100%;
  max-height: 120px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--line);
}
.lib-section {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.lib-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.protect-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11.5px;
  color: var(--text-2);
}
.palette-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 9px 10px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--surface-1);
}
.palette-group {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.swatches {
  display: flex;
  gap: 5px;
}
.swatch {
  width: 18px;
  height: 18px;
  border-radius: 4px;
  border: 1px solid var(--line-2);
}
</style>
