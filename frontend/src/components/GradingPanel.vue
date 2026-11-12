<script setup lang="ts">
/** 颜色分级：阴影/中间调/高光/全局 四个色轮（2×2）+ 选中色轮的饱和度/亮度 + 混合。
 *  对照旧 GradingPanel；轮与滑块的双向语义见 ColorWheel 注释。
 *  adj 由父级传入（全局参数或选中蒙版的局部参数）。
 *  布局：四个轮先选“色阶”，参数在下方面板统一给足宽度——避免每个轮挤两条短滑轨。 */
import { computed, ref } from 'vue'
import ColorWheel from '@/components/ColorWheel.vue'
import SliderRow from '@/components/SliderRow.vue'
import { notifyChanged, notifyCommitted } from '@/stores/editor'
import { RANGE_LABELS } from '@/lib/presets'
import type { AdjustmentsLike, GradingRange } from '@/api/types'
import { GRADING_RANGES } from '@/api/types'

const props = defineProps<{ adj: AdjustmentsLike }>()

const active = ref<GradingRange>('shadows')
const activeWheel = computed(() => props.adj.wheels[active.value])

function onWheelChange(range: GradingRange, displayHue: number, displaySat: number): void {
  const w = props.adj.wheels[range]
  w.hue = displayHue
  w.saturation = displaySat * 100
  notifyChanged()
}
</script>

<template>
  <div class="grading">
    <div class="wheels">
      <div
        v-for="range in GRADING_RANGES"
        :key="range"
        class="cell"
        :class="{ active: active === range, engaged: Math.abs(adj.wheels[range].saturation) > 0.01 }"
        @pointerdown="active = range"
      >
        <ColorWheel
          :hue="adj.wheels[range].hue"
          :saturation="adj.wheels[range].saturation"
          :disc-r="33"
          @wheel-change="(h: number, s: number) => onWheelChange(range, h, s)"
          @released="notifyCommitted"
        />
        <span class="caption">{{ RANGE_LABELS[range] }}</span>
      </div>
    </div>

    <div class="params">
      <div class="params-head">
        <span class="p-dot" />
        <span class="p-title">{{ RANGE_LABELS[active] }}</span>
        <span class="p-hue num">H {{ activeWheel.hue.toFixed(0) }}°</span>
      </div>
      <SliderRow label="饱和度" :min="-100" :max="100"
                 v-model="activeWheel.saturation"
                 @input="notifyChanged" @release="notifyCommitted" />
      <SliderRow label="亮度" :min="-100" :max="100"
                 v-model="activeWheel.luminance"
                 @input="notifyChanged" @release="notifyCommitted" />
      <SliderRow label="混合" :min="0" :max="100" :default-value="50"
                 v-model="adj.blend"
                 @input="notifyChanged" @release="notifyCommitted" />
    </div>
  </div>
</template>

<style scoped>
.grading { display: flex; flex-direction: column; }
.wheels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 6px 4px 5px;
  border-radius: var(--r-2);
  cursor: pointer;
  transition: background 0.16s var(--ease), box-shadow 0.16s var(--ease);
}
.cell:hover { background: rgba(255, 255, 255, 0.03); }
.cell.active {
  background: rgba(255, 255, 255, 0.05);
  box-shadow: inset 0 0 0 1px var(--line-2);
}
.cell :deep(canvas) { display: block; }
.caption {
  font-size: 10.5px;
  letter-spacing: 0.3px;
  color: var(--text-3);
  user-select: none;
  transition: color 0.14s var(--ease);
}
.cell.engaged .caption { color: var(--text-2); }
.cell.active .caption { color: var(--text); }

.params {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.params-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 3px;
}
.p-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent);
}
.p-title { font-size: 11px; color: var(--text-2); letter-spacing: 0.3px; }
.p-hue { margin-left: auto; font-size: 10px; color: var(--text-3); }
</style>