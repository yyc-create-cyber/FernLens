/** 悬浮提示（v-tip 指令）。
 *
 *  为什么不用原生 title：浏览器要等约 1 秒、样式不可控、长文本会截断。
 *  这里挂一个单例气泡到 body，position: fixed —— 不受面板 overflow:auto 裁剪，
 *  也不会和滚动容器打架；每次显示前重新测量位置，窗口缩放/滚动即自动躲开边缘。
 *
 *  用法：
 *    <span v-tip="'说明文字'">曝光</span>
 *    <span v-tip="{ text: '说明', delay: 0 }">…</span>
 *  指令会在鼠标移入 260ms 后弹出（避免扫过一串滑块时闪个不停），移出立刻收起。 */

type TipBinding = string | { text: string; delay?: number }

const SHOW_DELAY = 260
const GAP = 10
const MAX_W = 280

let bubble: HTMLDivElement | null = null
let arrow: HTMLDivElement | null = null
let timer: number | undefined

function ensureEl(): HTMLDivElement {
  if (bubble) return bubble
  bubble = document.createElement('div')
  bubble.className = 'pl-tip'
  bubble.setAttribute('role', 'tooltip')
  arrow = document.createElement('div')
  arrow.className = 'pl-tip-arrow'
  bubble.appendChild(arrow)
  bubble.addEventListener('pointerdown', hide)
  document.body.appendChild(bubble)
  return bubble
}

function place(target: HTMLElement): void {
  const el = ensureEl()
  const rect = target.getBoundingClientRect()
  const elRect = el.getBoundingClientRect()

  // 默认放上方；顶部空间不够就翻到下方
  let top = rect.top - elRect.height - GAP
  let below = false
  if (top < 8) {
    top = rect.bottom + GAP
    below = true
  }
  // 水平以目标中心对齐，再夹到视口内
  let left = rect.left + rect.width / 2 - elRect.width / 2
  const maxLeft = window.innerWidth - elRect.width - 10
  left = Math.min(Math.max(10, left), Math.max(10, maxLeft))

  el.style.top = `${Math.round(top)}px`
  el.style.left = `${Math.round(left)}px`
  el.classList.toggle('below', below)

  // 箭头指向目标中心，但不出气泡圆角
  if (arrow) {
    const cx = Math.min(
      Math.max(rect.left + rect.width / 2 - left, 14),
      elRect.width - 14,
    )
    arrow.style.left = `${Math.round(cx)}px`
  }
}

let current: HTMLElement | null = null

export function show(target: HTMLElement, text: string): void {
  if (!text) return
  const el = ensureEl()
  // 先写文本再量尺寸：否则首帧按旧内容算高度，位置会抖一下
  el.textContent = ''
  el.appendChild(document.createTextNode(text))
  if (arrow) el.appendChild(arrow)
  el.style.maxWidth = `${MAX_W}px`
  el.classList.add('on')
  current = target
  place(target)
}

export function hide(): void {
  window.clearTimeout(timer)
  timer = undefined
  current = null
  bubble?.classList.remove('on')
}

const onScrollOrResize = (): void => {
  if (current) place(current)
}

function textOf(b: unknown): string {
  if (typeof b === 'string') return b
  const o = b as { text?: string } | null
  return o?.text || ''
}

export const vTip = {
  mounted(el: HTMLElement, binding: { value: TipBinding }) {
    const delay = typeof binding.value === 'object' && binding.value?.delay != null
      ? binding.value.delay
      : SHOW_DELAY
    const enter = (): void => {
      const text = textOf(binding.value)
      if (!text) return
      window.clearTimeout(timer)
      timer = window.setTimeout(() => show(el, text), delay)
    }
    const leave = (): void => hide()
    el.addEventListener('pointerenter', enter)
    el.addEventListener('pointerleave', leave)
    el.addEventListener('pointerdown', leave)
    const store = el as HTMLElement & { __tipHandlers?: [() => void, () => void] }
    store.__tipHandlers = [enter, leave]
  },
  updated(el: HTMLElement, binding: { value: TipBinding }) {
    // 文案随状态变化（例如蒙版类型切换）时无需重建，读取时已是最新
    void el
    void binding
  },
  unmounted(el: HTMLElement) {
    const store = el as HTMLElement & { __tipHandlers?: [() => void, () => void] }
    const h = store.__tipHandlers
    if (h) {
      el.removeEventListener('pointerenter', h[0])
      el.removeEventListener('pointerleave', h[1])
      el.removeEventListener('pointerdown', h[1])
      delete store.__tipHandlers
    }
    hide()
  },
}

if (typeof window !== 'undefined') {
  window.addEventListener('scroll', onScrollOrResize, true)
  window.addEventListener('resize', onScrollOrResize)
}
