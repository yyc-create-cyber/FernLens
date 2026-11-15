<script setup lang="ts">
/** 滑块行（紧凑单行）：标签 · 轨 · 读数。拖动实时预览、松手提交；
 *  双击标签/读数复位；悬停时读数让位给复位按钮（无布局跳动）；
 *  偏离默认值时读数转为高亮色，方便一眼扫出“改过哪些”。 */
import { computed, onBeforeUnmount, ref } from 'vue'
import Icon from '@/components/Icon'
import { vTip } from '@/lib/tip'

const props = withDefaults(defineProps<{
  label: string
  min: number
  max: number
  modelValue: number
  defaultValue?: number
  step?: number
  decimals?: number
  suffix?: string
  /** 悬浮说明（鼠标停在标签上 0.26s 后弹出） */
  hint?: string
}>(), {
  defaultValue: 0,
  step: 1,
  decimals: 0,
  suffix: '',
  hint: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', v: number): void
  (e: 'input'): void        // 拖动中
  (e: 'release'): void      // 松手
}>()

const trackEl = ref<HTMLElement | null>(null)
const dragging = ref(false)

// 是否双向（有正有负）——决定填充从中心还是从左端起
const bipolar = computed(() => props.min < 0 && props.max > 0)
const modified = computed(() => props.modelValue !== props.defaultValue)

const display = computed(() => props.modelValue.toFixed(props.decimals) + props.suffix)

const pct = computed(() => {
  const span = props.max - props.min || 1
  return ((props.modelValue - props.min) / span) * 100
})
const fillStyle = computed(() => {
  if (bipolar.value) {
    const zeroPct = ((0 - props.min) / (props.max - props.min || 1)) * 100
    const left = Math.min(zeroPct, pct.value)
    const width = Math.abs(pct.value - zeroPct)
    return { left: left + '%', width: width + '%' }
  }
  return { left: '0%', width: pct.value + '%' }
})

function snap(v: number): number {
  const stepped = Math.round((v - props.min) / props.step) * props.step + props.min
  const clamped = Math.min(props.max, Math.max(props.min, stepped))
  return Number(clamped.toFixed(props.decimals))
}

function valueFromX(clientX: number): number {
  const el = trackEl.value
  if (!el) return props.modelValue
  const rect = el.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / (rect.width || 1)))
  return snap(props.min + ratio * (props.max - props.min))
}

function onInput(v: number): void {
  emit('update:modelValue', v)
  emit('input')
}

function onPointerDown(e: PointerEvent): void {
  dragging.value = true
  ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
  onInput(valueFromX(e.clientX))
}
function onPointerMove(e: PointerEvent): void {
  if (!dragging.value) return
  onInput(valueFromX(e.clientX))
}
function onPointerUp(): void {
  if (!dragging.value) return
  dragging.value = false
  emit('release')
}
onBeforeUnmount(() => { dragging.value = false })

function reset(): void {
  emit('update:modelValue', props.defaultValue)
  emit('release')
}
</script>

<template>
  <div class="srow" :class="{ modified }">
    <span v-tip="hint" class="srow-label" :class="{ 'has-hint': hint }"
          title="双击复位" @dblclick="reset">{{ label }}</span>
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
    <span class="srow-val num" title="双击复位" @dblclick="reset">{{ display }}</span>
    <button class="srow-reset" title="复位" @click="reset">
      <Icon name="reset" :size="12" />
    </button>
  </div>
</template>

<style scoped>
.srow {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  height: 29px;
  padding: 0 6px 0 0;
  margin: 0 -6px;
  border-radius: var(--r-1);
  user-select: none;
  transition: background 0.14s var(--ease);
}
.srow:hover { background: var(--surface-1); }
/* 改过的参数：左侧一道琥珀刻线，扫一眼就知道哪里动过 */
.srow.modified::before {
  content: "";
  position: absolute;
  left: -8px;
  top: 50%;
  width: 2px;
  height: 12px;
  margin-top: -6px;
  border-radius: 2px;
  background: var(--accent);
  opacity: 0.75;
}
.srow-label {
  width: 58px;
  flex: none;
  padding-left: 8px;
  font-size: 11.5px;
  color: var(--text-2);
  letter-spacing: 0.02em;
  cursor: default;
  transition: color 0.14s var(--ease);
  white-space: nowrap;
}
/* 有说明的标签：虚线下划线暗示「可以停一下看看」 */
.srow-label.has-hint {
  cursor: help;
  text-decoration: underline dotted var(--line-2);
  text-underline-offset: 3px;
}
.srow-label.has-hint:hover { text-decoration-color: var(--accent-line); }
.srow:hover .srow-label { color: var(--text); }
.srow-val {
  width: 46px;
  flex: none;
  text-align: right;
  font-size: 11.5px;
  color: var(--text-3);
  cursor: default;
  transition: color 0.14s var(--ease), opacity 0.12s var(--ease);
}
.srow.modified .srow-val { color: var(--accent-text); }
.srow:hover .srow-val { opacity: 0; }

/* 复位按钮叠在读数位置上：悬停时不产生任何位移 */
.srow-reset {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 20px;
  height: 20px;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: var(--text-3);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s var(--ease), color 0.12s var(--ease);
}
.srow:hover .srow-reset { opacity: 1; pointer-events: auto; }
.srow-reset:hover { color: var(--accent); background: var(--accent-soft); }
</style>
