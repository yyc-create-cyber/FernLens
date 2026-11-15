<script setup lang="ts">
/** 滤镜面板：基础预设 + 自定义滤镜。
 *  套用是独占的：每次只保留一个滤镜，换滤镜会先复位全局影调/曲线/分级再套新的，
 *  所以不会和上一个滤镜叠加。不动蒙版/取色点；可撤销（历史里记为「滤镜 · 名字」）。
 *  自定义滤镜存 localStorage，可导出/导入 JSON。 */
import { computed, nextTick, ref } from 'vue'
import Icon from '@/components/Icon'
import { BUILTIN_FILTERS, filterSwatch } from '@/lib/filters'
import type { FilterPreset } from '@/lib/filters'
import { t, tx } from '@/lib/i18n'
import {
  useEditor, applyFilter, saveCurrentAsFilter, deleteCustomFilter,
  exportFilters, importFiltersFromFile,
} from '@/stores/editor'

const store = useEditor()

const custom = computed(() => store.customFilters)
const saving = ref(false)
const nameInput = ref<HTMLInputElement | null>(null)
const nameValue = ref('')
const fileInput = ref<HTMLInputElement | null>(null)

function beginSave(): void {
  saving.value = true
  nameValue.value = `我的滤镜 ${custom.value.length + 1}`
  void nextTick(() => {
    nameInput.value?.focus()
    nameInput.value?.select()
  })
}

function confirmSave(): void {
  const p = saveCurrentAsFilter(nameValue.value)
  if (!p) return
  saving.value = false
  nameValue.value = ''
}

function cancelSave(): void {
  saving.value = false
  nameValue.value = ''
}

function onDelete(p: FilterPreset): void {
  deleteCustomFilter(p.id)
}

function onExport(p: FilterPreset): void {
  exportFilters([p.id])
}

async function onFileChosen(e: Event): Promise<void> {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  try { await importFiltersFromFile(file) } catch { /* 状态栏已给出原因 */ }
}
</script>

<template>
  <div class="filter-panel">
    <div class="fp-tools">
      <button class="btn outline fp-tool" :title="t('filters.saveCurrent')" @click="beginSave">
        <Icon name="plus" :size="12" /> {{ tx('保存当前', 'Save Current') }}
      </button>
      <button class="btn outline fp-tool" :disabled="!custom.length" :title="tx('导出全部自定义滤镜为 JSON', 'Export presets as JSON')"
              @click="exportFilters()">
        <Icon name="download" :size="12" /> {{ tx('导出', 'Export') }}
      </button>
      <button class="btn outline fp-tool" :title="tx('从 JSON 文件导入滤镜', 'Import JSON preset')" @click="fileInput?.click()">
        <Icon name="upload" :size="12" /> {{ tx('导入', 'Import') }}
      </button>
      <input ref="fileInput" type="file" accept="application/json,.json" hidden @change="onFileChosen" />
    </div>

    <div v-if="saving" class="save-box">
      <input
        ref="nameInput"
        v-model="nameValue"
        class="pl-input"
        maxlength="40"
        :placeholder="t('filters.saveName')"
        @keydown.enter.prevent="confirmSave"
        @keydown.esc.prevent="cancelSave"
      />
      <button class="mini" @click="confirmSave">{{ tx('保存', 'Save') }}</button>
      <button class="mini" @click="cancelSave">{{ tx('取消', 'Cancel') }}</button>
    </div>

    <div class="fp-note">
      <div class="fp-note-head"><span class="fp-dot" />{{ tx('滤镜只包含全局影调与色轮', 'Global look presets') }}</div>
      <p>
        {{ tx(
          '包含全局影调、质感、曲线与分级色轮。点选预设会先复位再套用，不会与上一个预设叠加。',
          'Applies global tone, clarity, curves and 3-way color grading cleanly without stacking masks or crops.'
        ) }}
      </p>
    </div>

    <div class="sec-title">{{ t('filters.builtin') }}</div>
    <div class="fgrid">
      <button
        v-for="p in BUILTIN_FILTERS"
        :key="p.id"
        class="fcard"
        :class="{ on: store.activeFilter === p.id }"
        :title="p.note || p.name"
        @click="applyFilter(p)"
      >
        <span class="fthumb" :style="{ background: filterSwatch(p.look) }" />
        <span class="fname">{{ p.name }}</span>
      </button>
    </div>

    <div class="sec-title">
      {{ t('filters.custom') }} <span class="sec-count">{{ custom.length }}</span>
    </div>
    <div v-if="!custom.length" class="empty">
      {{ t('filters.emptyCustom') }}
    </div>
    <div
      v-for="p in custom"
      :key="p.id"
      class="frow"
      :class="{ on: store.activeFilter === p.id }"
      @click="applyFilter(p)"
    >
      <span class="fthumb sm" :style="{ background: filterSwatch(p.look) }" />
      <span class="fname grow">{{ p.name }}</span>
      <button class="mini square" title="导出此滤镜为 JSON" @click.stop="onExport(p)">
        <Icon name="download" :size="11" />
      </button>
      <button class="mini square danger" title="删除此滤镜" @click.stop="onDelete(p)">
        <Icon name="close" :size="11" />
      </button>
    </div>

    <p class="pane-tip">
      蒙版与取色点跟具体画面绑定，跨图复用会错位，所以不写进滤镜。
      套用后被覆盖的字段可以在「历史」里撤销。
    </p>
  </div>
</template>

<style scoped>
/* 顶部说明：明确滤镜的能力边界，避免误以为能带走蒙版/裁剪 */
.fp-note {
  margin: 10px 0 2px;
  padding: 8px 9px 9px;
  border-radius: var(--r-2);
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.fp-note-head {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-text);
}
.fp-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent);
  flex: none;
}
.fp-note p {
  margin: 5px 0 0;
  font-size: 10.5px;
  line-height: 1.65;
  color: var(--text-2);
}
.fp-note p.fp-note-warn { color: var(--text-3); }

.fp-tools {
  display: flex;
  gap: 6px;
  margin-bottom: 4px;
}
.fp-tool {
  flex: 1;
  justify-content: center;
  font-size: 11px;
}
.save-box {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
  padding: 8px;
  border-radius: var(--r-2);
  background: var(--panel-2);
  box-shadow: inset 0 0 0 1px var(--line);
}
.pl-input {
  flex: 1;
  min-width: 0;
  height: 26px;
  padding: 0 8px;
  border: 0;
  border-radius: 7px;
  background: var(--bg);
  color: var(--text);
  font-size: 12px;
  outline: none;
  box-shadow: inset 0 0 0 1px var(--line);
}
.pl-input:focus { box-shadow: inset 0 0 0 1px var(--accent-line); }

.sec-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 14px 0 8px;
  font-size: 10.5px;
  letter-spacing: 0.6px;
  color: var(--text-3);
  text-transform: uppercase;
}
.sec-count {
  padding: 0 5px;
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-3);
  font-size: 10px;
  letter-spacing: 0;
}

.fgrid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 7px;
}
.fcard {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 5px;
  border: 0;
  border-radius: var(--r-2);
  background: var(--surface-1);
  color: var(--text-2);
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
  box-shadow: inset 0 0 0 1px transparent;
}
.fcard:hover { background: var(--surface-hover); color: var(--text); }
.fcard.on {
  background: var(--accent-soft);
  color: var(--accent-text);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.fthumb {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 10;
  border-radius: 6px;
  box-shadow: inset 0 0 0 1px var(--line-2);
}
.fthumb.sm {
  flex: none;
  width: 26px;
  height: 26px;
  aspect-ratio: auto;
  border-radius: 5px;
}
.fname {
  font-size: 11.5px;
  line-height: 1.2;
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.frow {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 6px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.14s var(--ease);
}
.frow:hover { background: var(--surface-hover); }
.frow.on { background: var(--accent-soft); }
.frow .fname { text-align: left; }
.grow { flex: 1; }
.empty {
  padding: 10px 2px;
  font-size: 11px;
  line-height: 1.7;
  color: var(--text-3);
}
</style>
