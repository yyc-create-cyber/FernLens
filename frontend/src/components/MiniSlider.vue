<script setup lang="ts">
/** 紧凑滑块（面板内用）：标签 · 轨 · 读数，拖动即回调 update:value。
 *  与 SliderRow 共用全局 .pl-range 轨道样式，只做“受控输入”这一件事。 */
import { computed, onBeforeUnmount, ref } from 'vue'
import { vTip } from '@/lib/tip'

const props = withDefaults(defineProps<{
  label?: string
  value: number
  min: number
  max: number
  step?: number
  decimals?: number
  suffix?: string
  labelWidth?: number
  hint?: string
}>(), {
  label: '',
  step: 1,
  decimals: 0,
  suffix: '',
  labelWidth: 56,
  hint: '',
})

const emit = defineEmits<{
  (e: 'update:value', v: number): void
  (e: 'release'): void
}>()

const trackEl = ref<HTMLElement | null>(null)
const dragging = ref(false)

const bipolar = computed(() => props.min < 0 && props.max > 0)
const pct = computed(() => {
  const span = props.max - props.min || 1
  return ((props.value - props.min) / span) * 100
})
const fillStyle = computed(() => {
  if (bipolar.value) {
    const zeroPct = ((0 - props.min) / (props.max - props.min || 1)) * 100
    const left = Math.min(zeroPct, pct.value)
    return { left: left + '%', width: Math.abs(pct.value - zeroPct) + '%' }
  }
  return { left: '0%', width: pct.value + '%' }
})
const display = computed(() => props.value.toFixed(props.decimals) + props.suffix)

function snap(v: number): number {
  const stepped = Math.round((v - props.min) / props.step) * props.step + props.min
  const clamped = Math.min(props.max, Math.max(props.min, stepped))
  return Number(clamped.toFixed(props.decimals))
}
function valueFromX(clientX: number): number {
  const el = trackEl.value
  if (!el) return props.value
  const rect = el.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / (rect.width || 1)))
  return snap(props.min + ratio * (props.max - props.min))
}
function onPointerDown(e: PointerEvent): void {
  dragging.value = true
  ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
  emit('update:value', valueFromX(e.clientX))
}
function onPointerMove(e: PointerEvent): void {
  if (!dragging.value) return
  emit('update:value', valueFromX(e.clientX))
}
function onPointerUp(): void {
  if (!dragging.value) return
  dragging.value = false
  emit('release')
}
onBeforeUnmount(() => { dragging.value = false })
</script>

<template>
  <div class="mrow">
    <span v-if="label" v-tip="hint" class="mrow-label" :class="{ 'has-hint': hint }"
          :style="{ width: labelWidth + 'px' }">{{ label }}</span>
    <div
      ref="trackEl"
      class="pl-range"
      :class="{ dragging }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
    >
      <div class="pl-fill" :style="fillStyle" />
      <div v-if="bipolar" class="pl-zero" />
      <div class="pl-thumb" :style="{ left: pct + '%' }" />
    </div>
    <span class="mrow-val num">{{ display }}</span>
  </div>
</template>

<style scoped>
.mrow {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 26px;
  user-select: none;
}
.mrow-label {
  flex: none;
  font-size: 11px;
  color: var(--text-2);
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.mrow-label.has-hint {
  cursor: help;
  text-decoration: underline dotted rgba(255, 243, 228, 0.2);
  text-underline-offset: 3px;
}
.mrow-label.has-hint:hover { text-decoration-color: var(--accent-line); color: var(--text); }
.mrow-val {
  width: 44px;
  flex: none;
  text-align: right;
  font-size: 11px;
  color: var(--text-2);
}
</style>
