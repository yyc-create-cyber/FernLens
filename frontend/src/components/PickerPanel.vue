<script setup lang="ts">
/** 取色器面板：从图像取色 / 8 个预设色相（可再微调）/ 取色点列表。
 *  对照旧 PickerPanel + _PickerRow 行为；滑块统一用 MiniSlider。 */
import { computed } from 'vue'
import type { PickerState } from '@/api/types'
import MiniSlider from '@/components/MiniSlider.vue'
import Icon from '@/components/Icon'
import { HS_PRESETS, presetColor } from '@/lib/presets'
import { useEditor, notifyChanged, notifyCommitted,
         armNewPick, armRepick, cancelPick, newPresetPicker, deletePicker } from '@/stores/editor'

const props = defineProps<{ pickers: PickerState[] }>()
const store = useEditor()

const pickers = computed(() => props.pickers)

function swatchCss(color: [number, number, number]): string {
  const [r, g, b] = color.map(c => Math.round(c * 255))
  return `rgb(${r},${g},${b})`
}
function dotCss(hue: number): string {
  const [r, g, b] = presetColor(hue).map(c => Math.round(c * 255))
  return `background: rgb(${r},${g},${b})`
}
function setField(i: number, field: 'span' | 'hue_shift' | 'saturation' | 'luminance', v: number): void {
  const p = pickers.value[i]
  if (!p) return
  ;(p as unknown as Record<string, number>)[field] = v
  notifyChanged()
}

const hint = computed(() => {
  if (store.pickTarget === 'new') return '在图像上单击拾取颜色…'
  if (typeof store.pickTarget === 'number') return `在图像上单击以更新取色点 ${store.pickTarget + 1}…`
  return ''
})
</script>

<template>
  <div class="picker-panel">
    <button
      class="pick-btn"
      :class="{ armed: store.pickTarget === 'new' }"
      @click="store.pickTarget === 'new' ? cancelPick() : armNewPick()"
    >
      <Icon name="plus" :size="12" /> 从图像取色
    </button>
    <div v-if="hint" class="hint">{{ hint }}</div>

    <div class="presets">
      <button
        v-for="p in HS_PRESETS"
        :key="p.name"
        class="dot"
        :style="dotCss(p.hue)"
        :title="`新建「${p.name}」取色点（色相 ${p.hue}°）`"
        @click="newPresetPicker(p.hue)"
      />
    </div>

    <div v-if="!pickers.length" class="empty">
      暂无取色点。点上方按钮从画面取色，或直接点一排色点快速建立。
    </div>

    <div
      v-for="(p, i) in pickers"
      :key="i"
      class="row"
      :class="{ selected: store.pickSelected === i }"
      @click="store.pickSelected = i"
    >
      <div class="row-head">
        <span class="swatch" :style="{ background: swatchCss(p.color) }" />
        <span class="info">
          <span class="num">色相 {{ p.hue.toFixed(0) }}°</span>
          <span class="range num">±{{ p.span.toFixed(0) }}°</span>
        </span>
        <button class="mini square" title="在图像上重新拾取颜色" @click.stop="armRepick(i)">
          <Icon name="eye" :size="11" />
        </button>
        <button class="mini square danger" title="删除此取色点" @click.stop="deletePicker(i)">
          <Icon name="close" :size="11" />
        </button>
      </div>
      <div class="row-sliders">
        <MiniSlider label="影响范围" :value="p.span" :min="5" :max="180" :label-width="52" suffix="°"
                    @update:value="(v: number) => setField(i, 'span', v)"
                    @release="notifyCommitted" />
        <MiniSlider label="色相偏移" :value="p.hue_shift" :min="-60" :max="60" :label-width="52" suffix="°"
                    @update:value="(v: number) => setField(i, 'hue_shift', v)"
                    @release="notifyCommitted" />
        <MiniSlider label="饱和度" :value="p.saturation" :min="-100" :max="100" :label-width="52"
                    @update:value="(v: number) => setField(i, 'saturation', v)"
                    @release="notifyCommitted" />
        <MiniSlider label="明亮度" :value="p.luminance" :min="-100" :max="100" :label-width="52"
                    @update:value="(v: number) => setField(i, 'luminance', v)"
                    @release="notifyCommitted" />
      </div>
    </div>

    <p v-if="pickers.length" class="tip">
      单击色块选中，<span class="kbd">Delete</span> 删除选中取色点。
    </p>
  </div>
</template>

<style scoped>
.picker-panel { display: flex; flex-direction: column; gap: 9px; }
.pick-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  height: 28px;
  border: 0;
  border-radius: var(--r-1);
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-2);
  font-size: 11.5px;
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
}
.pick-btn:hover { background: rgba(255, 255, 255, 0.10); color: var(--text); }
.pick-btn.armed {
  background: var(--accent-soft);
  color: var(--accent-text);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.hint { font-size: 10.5px; color: var(--accent-text); }
.presets {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 6px;
}
.dot {
  width: 100%;
  aspect-ratio: 1;
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.16);
  transition: transform 0.14s var(--ease), box-shadow 0.14s var(--ease);
}
.dot:hover {
  transform: scale(1.14);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.34), 0 3px 10px rgba(0, 0, 0, 0.5);
}
.empty {
  padding: 10px 12px;
  border-radius: var(--r-2);
  background: rgba(255, 255, 255, 0.025);
  font-size: 11px;
  line-height: 1.65;
  color: var(--text-3);
}
.row {
  border-radius: var(--r-2);
  background: var(--panel-2);
  box-shadow: inset 0 0 0 1px var(--line);
  padding: 7px 9px;
  cursor: pointer;
  transition: background 0.14s var(--ease), box-shadow 0.14s var(--ease);
}
.row:hover { background: #15181e; }
.row.selected {
  background: rgba(224, 161, 99, 0.06);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.row-head { display: flex; align-items: center; gap: 7px; margin-bottom: 4px; }
.swatch {
  width: 18px;
  height: 18px;
  border-radius: 5px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
  flex: none;
}
.info { display: flex; align-items: baseline; gap: 8px; flex: 1; min-width: 0; }
.info .num { font-size: 11px; color: var(--text-2); }
.info .range { font-size: 10px; color: var(--text-3); }
.row-sliders { display: flex; flex-direction: column; }
.tip { margin: 2px 0 0; font-size: 10.5px; color: var(--text-3); }
</style>