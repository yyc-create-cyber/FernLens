<script setup lang="ts">
/** 直方图（64 bins）：master 亮度 + RGB 通道叠画（screen 混合）。
 *  暗底、极细网格、底部渐隐——安静地待在面板顶部，不抢视线。 */
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import type { HistData } from '@/api/client'

const props = defineProps<{ hist: HistData | null }>()
const canvas = ref<HTMLCanvasElement | null>(null)
let ro: ResizeObserver | null = null

function drawPath(ctx: CanvasRenderingContext2D, bins: number[], w: number, h: number): void {
  const n = bins.length
  if (!n) return
  ctx.beginPath()
  ctx.moveTo(0, h)
  for (let i = 0; i < n; i++) {
    // 轻微平滑：相邻三点加权，避免锯齿状尖刺
    const a = bins[Math.max(0, i - 1)]
    const b = bins[i]
    const c = bins[Math.min(n - 1, i + 1)]
    const v = (a + 2 * b + c) / 4
    ctx.lineTo((i / (n - 1)) * w, h - Math.min(1, v) * (h - 3))
  }
  ctx.lineTo(w, h)
  ctx.closePath()
  ctx.fill()
}

function redraw(): void {
  const cv = canvas.value
  if (!cv) return
  const dpr = window.devicePixelRatio || 1
  const w = cv.clientWidth
  const h = cv.clientHeight
  if (!w || !h) return
  cv.width = Math.round(w * dpr)
  cv.height = Math.round(h * dpr)
  const ctx = cv.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  // 25/50/75% 竖向参考线
  ctx.strokeStyle = 'rgba(255,255,255,0.045)'
  ctx.lineWidth = 1
  for (const f of [0.25, 0.5, 0.75]) {
    ctx.beginPath()
    ctx.moveTo(Math.round(w * f) + 0.5, 0)
    ctx.lineTo(Math.round(w * f) + 0.5, h)
    ctx.stroke()
  }

  const hist = props.hist
  if (!hist) return
  ctx.globalCompositeOperation = 'screen'      // RGB 叠加处趋白，与主流直方图一致
  ctx.fillStyle = 'rgba(140,146,158,0.42)'
  drawPath(ctx, hist.master, w, h)
  ctx.fillStyle = 'rgba(255,84,84,0.60)'
  drawPath(ctx, hist.red, w, h)
  ctx.fillStyle = 'rgba(78,226,128,0.55)'
  drawPath(ctx, hist.green, w, h)
  ctx.fillStyle = 'rgba(92,132,255,0.62)'
  drawPath(ctx, hist.blue, w, h)
  ctx.globalCompositeOperation = 'source-over'
}

onMounted(() => {
  redraw()
  ro = new ResizeObserver(() => redraw())
  if (canvas.value) ro.observe(canvas.value)
})
onBeforeUnmount(() => ro?.disconnect())
watch(() => props.hist, redraw, { deep: false })
</script>

<template>
  <canvas ref="canvas" class="hist" />
</template>

<style scoped>
.hist {
  display: block;
  width: 100%;
  height: 86px;
  border-radius: var(--r-2);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.025), rgba(255, 255, 255, 0) 60%),
    var(--well);
  box-shadow: inset 0 0 0 1px var(--line);
}
</style>