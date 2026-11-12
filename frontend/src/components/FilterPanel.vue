<script setup lang="ts">
/** 滤镜面板：基础预设 + 自定义滤镜。
 *  套用是独占的：每次只保留一个滤镜，换滤镜会先复位全局影调/曲线/分级再套新的，
 *  所以不会和上一个滤镜叠加。不动蒙版/取色点；可撤销（历史里记为「滤镜 · 名字」）。
 *  自定义滤镜存 localStorage，可导出/导入 JSON。 */
import { computed, nextTick, ref } from 'vue'
import Icon from '@/components/Icon'
import { BUILTIN_FILTERS, filterSwatch } from '@/lib/filters'
import type { FilterPreset } from '@/lib/filters'
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
      <button class="btn outline fp-tool" title="把当前全局参数保存为新滤镜" @click="beginSave">
        <Icon name="plus" :size="12" /> 保存当前
      </button>
      <button class="btn outline fp-tool" :disabled="!custom.length" title="导出全部自定义滤镜为 JSON"
              @click="exportFilters()">
        <Icon name="download" :size="12" /> 导出
      </button>
      <button class="btn outline fp-tool" title="从 JSON 文件导入滤镜" @click="fileInput?.click()">
        <Icon name="upload" :size="12" /> 导入
      </button>
      <input ref="fileInput" type="file" accept="application/json,.json" hidden @change="onFileChosen" />
    </div>

    <div v-if="saving" class="save-box">
      <input
        ref="nameInput"
        v-model="nameValue"
        class="pl-input"
        maxlength="40"
        placeholder="滤镜名称"
        @keydown.enter.prevent="confirmSave"
        @keydown.esc.prevent="cancelSave"
      />
      <button class="mini" @click="confirmSave">保存</button>
      <button class="mini" @click="cancelSave">取消</button>
    </div>

    <div class="fp-note">
      <div class="fp-note-head"><span class="fp-dot" />滤镜只包含基础内容</div>
      <p>
        只有全局影调（曝光 / 对比 / 高光 / 阴影 / 白平衡 / 饱和度等）、效果（清晰度 / 降噪 / 锐度 / 去朦胧）、
        曲线与分级色轮。
      </p>
      <p>
        <strong>一次只套用一个</strong>：点另一个滤镜会先复位再套用，不会和上一个叠加；
        想叠加请直接手动调参数。
      </p>
      <p class="fp-note-warn">
        不含蒙版（含 AI 识别蒙版）、取色点、仿制图章、裁剪/旋转，也不含局部调整与历史记录。
      </p>
    </div>

    <div class="sec-title">基础滤镜</div>
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
      我的滤镜 <span class="sec-count">{{ custom.length }}</span>
    </div>
    <div v-if="!custom.length" class="empty">
      调好一张照片后点「保存当前」，就能把自己的参数变成滤镜，随时复用或导出成 JSON。
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
  background: rgba(224, 161, 99, 0.07);
  box-shadow: inset 0 0 0 1px rgba(224, 161, 99, 0.16);
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
  background: rgba(255, 255, 255, 0.07);
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
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-2);
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
  box-shadow: inset 0 0 0 1px transparent;
}
.fcard:hover { background: rgba(255, 255, 255, 0.07); color: var(--text); }
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
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.35);
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
.frow:hover { background: rgba(255, 255, 255, 0.05); }
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
