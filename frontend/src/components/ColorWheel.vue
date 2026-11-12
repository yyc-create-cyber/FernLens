<script setup lang="ts">
/** 颜色分级色轮：外环色相（0° 在 3 点钟方向、逆时针）、内盘拖动设饱和度。
 *  对照旧 ui_widgets.ColorWheel + GradingPanel 的正负语义：
 *  负饱和度显示为 hue+180、|sat|；拖轮写回正饱和度与显示角。 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = withDefaults(defineProps<{
  hue: number              // WheelState.hue（度）
  saturation: number       // WheelState.saturation（-100..100）
  discR?: number           // 内盘半径（px），面板按需缩放
}>(), { discR: 40 })

const emit = defineEmits<{
  (e: 'wheelChange', displayHue: number, displaySat: number): void
  (e: 'released'): void
}>()

const DISC_R = props.discR
const RING_W = 8
const SIZE = (DISC_R + RING_W) * 2 + 6
const THUMB_R = DISC_R > 30 ? 5 : 4

const canvas = ref<HTMLCanvasElement | null>(null)
let dragging = false

// 显示语义（GradingPanel._refresh_wheel）：负饱和度 -> hue+180、|sat|
function displayState(): { hue: number; sat: number } {
  const negative = props.saturation < 0
  return {
    hue: ((props.hue + (negative ? 180 : 0)) % 360 + 360) % 360,
    sat: Math.min(1, Math.abs(props.saturation) / 100),
  }
}

function draw(): void {
  const cv = canvas.value
  if (!cv) return
  const dpr = window.devicePixelRatio || 1
  cv.width = Math.round(SIZE * dpr)
  cv.height = Math.round(SIZE * dpr)
  const ctx = cv.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  const c = SIZE / 2
  ctx.clearRect(0, 0, SIZE, SIZE)

  // 外环：逐段画弧，色相 h 位于屏幕角 -h（逆时针，与 Qt ConicalGradient 一致）
  // canvas arc 默认顺时针，故从 -(h+2) 画到 -h 只扫 2°
  const ringR = DISC_R + RING_W / 2
  ctx.lineWidth = RING_W
  for (let h = 0; h < 360; h += 2) {
    const a0 = (-h * Math.PI) / 180
    const a1 = (-(h + 2) * Math.PI) / 180
    ctx.beginPath()
    ctx.arc(c, c, ringR, a1 - 0.005, a0 + 0.005)
    ctx.strokeStyle = `hsl(${h + 1}, 100%, 50%)`
    ctx.stroke()
  }

  // 内盘
  ctx.beginPath()
  ctx.arc(c, c, DISC_R, 0, Math.PI * 2)
  ctx.fillStyle = '#171a20'
  ctx.fill()
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(255,255,255,0.10)'
  ctx.stroke()

  // 拖动点（y 向上：屏幕 y 取 -sin）
  const { hue, sat } = displayState()
  const hRad = (hue * Math.PI) / 180
  const r = sat * (DISC_R - 3)
  const dx = c + r * Math.cos(hRad)
  const dy = c - r * Math.sin(hRad)
  const [sr, sg, sb] = hsvToRgb255(hue / 360, sat, 1)
  ctx.beginPath()
  ctx.arc(dx, dy, THUMB_R, 0, Math.PI * 2)
  ctx.fillStyle = `rgb(${sr},${sg},${sb})`
  ctx.fill()
  ctx.lineWidth = 1.5
  ctx.strokeStyle = '#f1f2f5'
  ctx.stroke()
}

function hsvToRgb255(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  const arr = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6]
  return [Math.round(arr[0] * 255), Math.round(arr[1] * 255), Math.round(arr[2] * 255)]
}

function updateFromEvent(e: PointerEvent): void {
  const rect = canvas.value!.getBoundingClientRect()
  const dx = e.clientX - rect.left - SIZE / 2
  const dy = e.clientY - rect.top - SIZE / 2
  const hue = ((Math.atan2(-dy, dx) * 180) / Math.PI % 360 + 360) % 360
  const sat = Math.min(Math.hypot(dx, dy) / (DISC_R - 3), 1)
  emit('wheelChange', hue, sat)
  draw()
}

function onPointerDown(e: PointerEvent): void {
  dragging = true
  canvas.value!.setPointerCapture(e.pointerId)
  updateFromEvent(e)
}
function onPointerMove(e: PointerEvent): void {
  if (dragging) updateFromEvent(e)
}
function onPointerUp(e: PointerEvent): void {
  if (dragging) {
    dragging = false
    canvas.value?.releasePointerCapture(e.pointerId)
    emit('released')
  }
}

onMounted(draw)
onBeforeUnmount(() => { dragging = false })
watch(() => [props.hue, props.saturation], () => draw())
</script>

<template>
  <canvas
    ref="canvas"
    class="color-wheel"
    :style="{ width: SIZE + 'px', height: SIZE + 'px' }"
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerUp"
  />
</template>

<style scoped>
.color-wheel {
  display: block;
  cursor: crosshair;
  touch-action: none;
}
</style>
