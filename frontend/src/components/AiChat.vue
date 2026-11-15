<script setup lang="ts">
/** AI 对话框：三种模式共用一套前端直连接口（不新增后端路由）。
 *  1. 讨论 —— 视觉模型围绕当前画面聊天，不改参数。
 *  2. 参数 —— 视觉模型返回调色 JSON，前端解析后直接应用，可一键撤销。
 *  3. 生图 —— 把原图 + 提示词发给图像编辑模型，结果作为新图片加入图库。
 */
import { computed, nextTick, onBeforeUnmount, reactive, ref, watch } from 'vue'
import Icon from '@/components/Icon'
import { t, tx } from '@/lib/i18n'
import {
  useEditor, applyAdjustmentDict, aiConfigured, aiImageConfigured, closeAiChat, notifyCommitted,
  openAiSettings, openFiles,
} from '@/stores/editor'
import { aiChat, aiImageEdit, bitmapToDataUrl, bitmapToJpegBlob } from '@/api/ai'
import type { AiChatMessage, AiContentPart } from '@/api/ai'
import { TONE_FIELDS, scaleAiAdjustments } from '@/api/types'
import { PARAM_SENSITIVITY } from '@/lib/aiPrompt'
import type { Adjustments } from '@/api/types'

type ChatMode = 'chat' | 'adjust' | 'image'

interface UiMessage {
  id: number
  role: 'user' | 'assistant' | 'system'
  text: string
  imageUrl?: string
  json?: Record<string, unknown>
  snapshot?: Adjustments
  error?: boolean
}

const store = useEditor()
const mode = ref<ChatMode>('chat')
const input = ref('')
const sending = ref(false)
const listEl = ref<HTMLDivElement | null>(null)
const messages = reactive<Record<ChatMode, UiMessage[]>>({
  chat: [],
  adjust: [],
  image: [],
})
const histories: Record<ChatMode, AiChatMessage[]> = {
  chat: [],
  adjust: [],
  image: [],
}
let seq = 0
let controller: AbortController | null = null

const modes = computed<{ key: ChatMode; label: string }[]>(() => [
  { key: 'chat', label: tx('讨论', 'Discuss') },
  { key: 'adjust', label: tx('参数', 'Adjust') },
  { key: 'image', label: tx('生图', 'Generate') },
])

const activeMessages = computed(() => messages[mode.value])
/** 1 讨论 / 2 参数 走对话模型，3 生图 走图生图模型——两套端点各自判断。 */
const configReady = computed(() =>
  mode.value === 'image' ? aiImageConfigured() : aiConfigured())
const canSend = computed(() =>
  !!input.value.trim() && !sending.value && store.loaded && configReady.value)

const modeDesc = computed(() => {
  if (mode.value === 'chat') return '围绕当前画面讨论构图、光影与调色思路，不会改动参数。'
  if (mode.value === 'adjust') return '描述想要的效果，AI 返回调色 JSON 并直接应用，可一键撤销。'
  return '把原图发给生图模型，按描述重绘，并作为新图片加入图库。'
})

const placeholder = computed(() => {
  if (mode.value === 'chat') return '例如：这张照片的光影问题在哪？怎么调更有氛围？'
  if (mode.value === 'adjust') return '例如：调成暖调黄昏电影感，压高光，阴影偏青'
  return '例如：把背景换成雪山日落，保留人物和构图'
})

const emptyText = computed(() => {
  if (mode.value === 'chat') return '问点什么，比如「这张照片的问题在哪」'
  if (mode.value === 'adjust') return '描述想要的效果，AI 会直接改参数'
  return '描述要改成的画面，AI 会重绘原图并加入图库'
})

const hint = computed(() => {
  if (!configReady.value) {
    return mode.value === 'image' ? '未配置图生图端点' : '未配置对话模型端点'
  }
  return 'Enter 发送 · Shift+Enter 换行'
})

function scrollToBottom(): void {
  const el = listEl.value
  if (el) el.scrollTop = el.scrollHeight
}

function push(kind: ChatMode, msg: Omit<UiMessage, 'id'>): UiMessage {
  const item: UiMessage = { id: ++seq, ...msg }
  messages[kind].push(item)
  void nextTick(scrollToBottom)
  return item
}

function currentBitmap(original = false): ImageBitmap | null {
  if (original && store.originalBmp) return store.originalBmp
  return store.fullFrame?.bmp ?? store.previewFrame?.bmp ?? store.originalBmp
}

function systemPrompt(kind: 'chat' | 'adjust'): string {
  if (kind === 'chat') {
    return [
      '你是 FernLens 的照片讨论助手。根据随消息附上的当前渲染图和用户文字，',
      '进行专业、具体、可执行的摄影与调色讨论。回答中文，简洁直接，不要客套。',
      '用户问怎么调时，给出明确的方向和参数建议；不要输出 JSON，不要改图片。',
    ].join('')
  }
  const current = JSON.stringify({ ...store.adjustments, masks: undefined })
  return [
    '你是 FernLens 的调色助手。根据图片和用户要求，输出一个严格的 JSON 对象。',
    '格式：{"note":"一句话说明","adjustments":{...}}。只输出 JSON，不要 markdown，不要解释。',
    `adjustments 可用字段：${TONE_FIELDS.join(', ')}；exposure 范围 -5..5，其余 -100..100，blackwhite 0..100。`,
    PARAM_SENSITIVITY,
    'curves: {"master"|"red"|"green"|"blue": [[x,y], ...]}，x/y 取 0..1。',
    'wheels: {"shadows"|"midtones"|"highlights"|"global": {"hue":0..360,"saturation":-100..100,"luminance":-100..100}}。',
    'pickers: [{"hue":0..360,"span":5..180,"hue_shift":-60..60,"saturation":-100..100,"luminance":-100..100}]。',
    '克制优先：只做用户要求且确实需要的调整，常规项 5~20，确有问题的关键项 25~35，极少超过 40；',
    '曝光通常不超过 ±0.5 EV；曲线用轻微 S（中点位移 ≤0.12），不要四个通道全给强曲线；',
    '颜色分级只做点睛（单轮 saturation/luminance 多在 5~20）；同一影调区间不要叠加三种以上手段；',
    '没有要求就留 0，宁可少一分也不要过度后期。',
    `AI 调整强度当前为 ${store.aiConfig.strength.toFixed(2)}（你返回的所有数值会先乘这个系数再应用）。`,
    `当前参数：${current}`,
  ].join('\n')
}

function userContent(text: string, dataUrl: string): AiContentPart[] {
  const parts: AiContentPart[] = [{ type: 'text', text }]
  if (dataUrl) parts.push({ type: 'image_url', image_url: { url: dataUrl } })
  return parts
}

async function runChat(kind: 'chat' | 'adjust', text: string): Promise<void> {
  const bitmap = currentBitmap()
  const dataUrl = bitmap ? await bitmapToDataUrl(bitmap, 1280, 0.86) : ''
  const history = histories[kind]
  const requestMessages: AiChatMessage[] = [
    { role: 'system', content: systemPrompt(kind) },
    ...history,
    { role: 'user', content: userContent(text, dataUrl) },
  ]
  const result = await aiChat(store.aiConfig, requestMessages, {
    signal: controller?.signal,
    temperature: kind === 'adjust' ? 0.2 : 0.7,
  })
  history.push({ role: 'user', content: text })
  history.push({ role: 'assistant', content: result.text })
  if (history.length > 24) history.splice(0, history.length - 24)

  if (kind === 'chat') {
    push('chat', { role: 'assistant', text: result.text.trim() })
    return
  }
  applyJson(result.text)
}

function extractJson(text: string): Record<string, unknown> {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  const body = fenced ? fenced[1] : text
  const start = body.indexOf('{')
  const end = body.lastIndexOf('}')
  if (start < 0 || end <= start) throw new Error('没有找到 JSON')
  const parsed: unknown = JSON.parse(body.slice(start, end + 1))
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('JSON 不是对象')
  }
  return parsed as Record<string, unknown>
}

function applyJson(rawText: string): void {
  let parsed: Record<string, unknown>
  try {
    parsed = extractJson(rawText)
  } catch {
    push('adjust', { role: 'assistant', text: rawText.trim(), error: true })
    throw new Error('AI 返回的不是合法 JSON，未应用任何调整')
  }

  const candidate = parsed.adjustments
  const adj = (candidate && typeof candidate === 'object' && !Array.isArray(candidate))
    ? candidate as Record<string, unknown>
    : parsed
  const hasFields = TONE_FIELDS.some((key) => typeof adj[key] === 'number')
    || (!!adj.curves && typeof adj.curves === 'object')
    || (!!adj.wheels && typeof adj.wheels === 'object')
    || Array.isArray(adj.pickers)
  if (!hasFields) throw new Error('AI 返回的 JSON 没有可应用的调整字段')

  // 按「AI 调整强度」统一缩放（默认 0.6），模型给得再猛也不会一键过度后期
  const scaled = scaleAiAdjustments(adj, store.aiConfig.strength)
  const snapshot = JSON.parse(JSON.stringify(store.adjustments)) as Adjustments
  applyAdjustmentDict(scaled)
  const note = typeof parsed.note === 'string' && parsed.note.trim()
    ? parsed.note.trim()
    : '已应用 AI 参数'
  // 只保留最近一次 AI 调整的撤销入口，避免旧快照覆盖后续修改。
  for (const old of messages.adjust) old.snapshot = undefined
  push('adjust', { role: 'assistant', text: note, json: scaled, snapshot })
  store.status = note
}

async function runImageEdit(prompt: string): Promise<void> {
  const bitmap = currentBitmap(true)
  if (!bitmap) throw new Error('没有可用的原图')
  const source = await bitmapToJpegBlob(bitmap, 1536, 0.92)
  const result = await aiImageEdit(store.aiConfig, prompt, source, {
    signal: controller?.signal,
  })
  const ext = result.blob.type.includes('jpeg') ? 'jpg'
    : result.blob.type.includes('webp') ? 'webp' : 'png'
  const stamp = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19)
  const name = `AI-${stamp}.${ext}`
  const file = new File([result.blob], name, { type: result.blob.type || 'image/png' })
  const messageId = push('image', {
    role: 'assistant',
    text: `已生成 ${name}，正在加入图库…`,
    imageUrl: result.objectUrl,
  }).id
  const beforeId = store.currentImageId
  await openFiles([file])
  if (store.currentImageId === beforeId) {
    throw new Error(store.status || '生成图加入图库失败')
  }
  const live = messages.image.find((m) => m.id === messageId)
  if (live) live.text = `已生成 ${name}，已加入图库`
  store.status = `AI 生图完成：${name}`
}

async function send(): Promise<void> {
  const text = input.value.trim()
  if (!text || sending.value) return
  if (!store.loaded) {
    store.status = '请先打开照片'
    return
  }
  if (!configReady.value) {
    store.status = '请先在 AI 设置中配置端点'
    openAiSettings()
    return
  }

  const kind = mode.value
  input.value = ''
  push(kind, { role: 'user', text })
  sending.value = true
  controller = new AbortController()
  try {
    if (kind === 'image') await runImageEdit(text)
    else await runChat(kind, text)
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      push(kind, { role: 'system', text: '已停止本次请求' })
    } else {
      push(kind, { role: 'system', text: friendlyError(err), error: true })
    }
  } finally {
    sending.value = false
    controller = null
    await nextTick()
    scrollToBottom()
  }
}

function cancel(): void {
  controller?.abort()
}

function undo(msg: UiMessage): void {
  if (!msg.snapshot) return
  store.adjustments = msg.snapshot
  msg.snapshot = undefined
  notifyCommitted()
  store.status = '已撤销本次 AI 调整'
}

function clearMode(): void {
  const kind = mode.value
  for (const msg of messages[kind]) {
    if (msg.imageUrl?.startsWith('blob:')) URL.revokeObjectURL(msg.imageUrl)
  }
  messages[kind].splice(0)
  histories[kind].splice(0)
}

function friendlyError(err: unknown): string {
  const text = err instanceof Error ? err.message : String(err)
  if (/Failed to fetch|NetworkError|Load failed/i.test(text)) {
    return '请求失败：网络不可达，或该接口不允许浏览器跨域（CORS）。'
  }
  return text
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

watch(mode, () => { void nextTick(scrollToBottom) })
watch(() => store.aiChatOpen, (open) => {
  if (open) void nextTick(scrollToBottom)
})
onBeforeUnmount(() => {
  for (const list of Object.values(messages)) {
    for (const msg of list) {
      if (msg.imageUrl?.startsWith('blob:')) URL.revokeObjectURL(msg.imageUrl)
    }
  }
})
</script>

<template>
  <div v-show="store.aiChatOpen" class="ai-chat">
    <header class="ac-head">
      <div class="ac-title"><Icon name="chat" :size="14" /> {{ t('aiChat.title') }}</div>
      <div class="ac-actions">
        <button class="mini" :title="t('history.clear')" @click="clearMode">{{ t('history.clear') }}</button>
        <button class="mini square" :title="t('aiSet.close')" @click="closeAiChat">
          <Icon name="close" :size="13" />
        </button>
      </div>
    </header>

    <div class="ac-modes">
      <button
        v-for="m in modes"
        :key="m.key"
        class="ac-mode"
        :class="{ on: mode === m.key }"
        @click="mode = m.key"
      >{{ m.label }}</button>
    </div>

    <p class="ac-desc">{{ modeDesc }}</p>

    <div ref="listEl" class="ac-list">
      <div v-if="!activeMessages.length" class="ac-empty">
        <Icon name="ai" :size="24" :stroke="1.3" />
        <p>{{ emptyText }}</p>
      </div>

      <div
        v-for="m in activeMessages"
        :key="m.id"
        class="ac-row"
        :class="m.role"
      >
        <div class="ac-bubble" :class="{ error: m.error }">
          <img v-if="m.imageUrl" :src="m.imageUrl" class="ac-image" alt="AI 生成图" />
          <div v-if="m.text" class="ac-text">{{ m.text }}</div>
          <pre v-if="m.json" class="ac-json">{{ prettyJson(m.json) }}</pre>
          <div v-if="m.snapshot" class="ac-actions">
            <button class="mini" @click="undo(m)">撤销本次调整</button>
          </div>
        </div>
      </div>

      <div v-if="sending" class="ac-row assistant">
        <div class="ac-bubble ac-thinking"><span /><span /><span /></div>
      </div>
    </div>

    <div class="ac-input">
      <textarea
        v-model="input"
        rows="2"
        :placeholder="placeholder"
        :disabled="sending"
        @keydown.enter.exact.prevent="send"
      />
      <div class="ac-foot">
        <span class="ac-hint">{{ hint }}</span>
        <button v-if="!configReady" class="mini" @click="openAiSettings">{{ t('app.aiSettings') }}</button>
        <button v-if="sending" class="mini" @click="cancel">{{ tx('停止', 'Stop') }}</button>
        <button class="btn solid" :disabled="!canSend" @click="send">{{ tx('发送', 'Send') }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ai-chat {
  position: fixed;
  right: 12px;
  top: 56px;
  bottom: 34px;
  z-index: 25;
  width: 400px;
  max-width: calc(100vw - 24px);
  display: flex;
  flex-direction: column;
  min-height: 0;
  background: var(--glass-modal);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--line-2);
  border-radius: var(--r-3);
  box-shadow: 0 18px 55px rgba(0, 0, 0, 0.35);
  overflow: hidden;
}

.ac-head {
  flex: none;
  display: flex;
  align-items: center;
  height: 42px;
  padding: 0 12px;
  border-bottom: 1px solid var(--line);
}
.ac-title {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12.5px;
  font-weight: 600;
  letter-spacing: 0.3px;
  color: var(--text);
}
.ac-actions { display: flex; align-items: center; gap: 5px; }

.ac-modes {
  flex: none;
  display: flex;
  gap: 2px;
  padding: 8px 12px 0;
}
.ac-mode {
  flex: 1;
  height: 26px;
  border: 0;
  border-radius: var(--r-1);
  background: transparent;
  color: var(--text-3);
  font-size: 11.5px;
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
}
.ac-mode:hover { color: var(--text-2); background: var(--surface-1); }
.ac-mode.on {
  background: var(--accent-soft);
  color: var(--accent-text);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}

.ac-desc {
  flex: none;
  margin: 8px 12px 6px;
  font-size: 10.5px;
  line-height: 1.55;
  color: var(--text-3);
}

.ac-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 4px 12px 10px;
}
.ac-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 38px 12px;
  color: var(--text-3);
  text-align: center;
}
.ac-empty p { margin: 0; font-size: 11px; line-height: 1.6; }

.ac-row { display: flex; margin-top: 8px; }
.ac-row.user { justify-content: flex-end; }
.ac-row.system { justify-content: center; }
.ac-row.assistant { justify-content: flex-start; }
.ac-bubble {
  max-width: 88%;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--panel-2);
  color: var(--text);
  font-size: 11.5px;
  line-height: 1.68;
  white-space: pre-wrap;
  word-break: break-word;
}
.ac-row.user .ac-bubble {
  background: var(--accent-soft);
  color: var(--accent-text);
  box-shadow: inset 0 0 0 1px var(--accent-line);
}
.ac-row.system .ac-bubble {
  max-width: 100%;
  padding: 4px 8px;
  background: transparent;
  color: var(--text-3);
  font-size: 10.5px;
  text-align: center;
}
.ac-bubble.error { color: var(--danger); }
.ac-text { white-space: pre-wrap; }
.ac-image {
  display: block;
  max-width: 100%;
  margin-bottom: 7px;
  border-radius: var(--r-1);
  background: var(--well);
}
.ac-json {
  max-height: 180px;
  margin: 7px 0 0;
  padding: 7px 8px;
  overflow: auto;
  border-radius: var(--r-1);
  background: var(--well);
  color: var(--text-2);
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 10px;
  line-height: 1.55;
  white-space: pre;
}
.ac-actions { display: flex; gap: 5px; margin-top: 7px; }

.ac-thinking { display: inline-flex; gap: 4px; padding: 10px 12px; }
.ac-thinking span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--text-3);
  animation: ac-bounce 1s infinite ease-in-out;
}
.ac-thinking span:nth-child(2) { animation-delay: 0.12s; }
.ac-thinking span:nth-child(3) { animation-delay: 0.24s; }
@keyframes ac-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
  30% { transform: translateY(-3px); opacity: 1; }
}

.ac-input {
  flex: none;
  padding: 9px 10px 10px;
  border-top: 1px solid var(--line);
  background: var(--panel-2);
}
.ac-input textarea {
  display: block;
  width: 100%;
  resize: none;
  padding: 8px 9px;
  border: 1px solid var(--line);
  border-radius: var(--r-2);
  outline: none;
  background: var(--well);
  color: var(--text);
  font: inherit;
  font-size: 11.5px;
  line-height: 1.55;
}
.ac-input textarea::placeholder { color: var(--text-3); }
.ac-input textarea:focus { border-color: var(--accent-line); }
.ac-foot {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 7px;
}
.ac-hint {
  flex: 1;
  font-size: 10px;
  color: var(--text-3);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
