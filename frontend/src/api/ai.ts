/** 前端直连 OpenAI 兼容接口的两个 API（不新增后端路由）：
 *  1) aiChat        —— 多模态对话；讨论模式返回文本，参数模式返回 JSON。
 *  2) aiImageEdit   —— 原图 + 提示词交给生图/图像编辑模型。
 *  密钥与请求体都只在本机前端使用，和「AI 设置」里保存的配置一致。
 */
import type { AiConfig, AiEndpoint } from './client'

export type AiRole = 'system' | 'user' | 'assistant'

export interface AiTextPart {
  type: 'text'
  text: string
}

export interface AiImagePart {
  type: 'image_url'
  image_url: { url: string }
}

export type AiContentPart = AiTextPart | AiImagePart

export interface AiChatMessage {
  role: AiRole
  content: string | AiContentPart[]
}

export interface AiChatOptions {
  temperature?: number
  maxTokens?: number
  signal?: AbortSignal
}

export interface AiChatResult {
  text: string
  model: string
  raw: unknown
}

export interface AiImageOptions {
  signal?: AbortSignal
  size?: string
}

export interface AiImageResult {
  blob: Blob
  objectUrl: string
  model: string
  raw: unknown
}

function endpoint(cfg: AiEndpoint, path: string): string {
  const base = cfg.baseUrl.trim().replace(/\/+$/, '')
  if (!base) throw new Error('未配置 AI 端点 URL')
  return `${base}${path}`
}

function authHeaders(cfg: AiEndpoint): Record<string, string> {
  const key = cfg.apiKey.trim()
  if (!key) throw new Error('未配置 API Key')
  return { Authorization: `Bearer ${key}` }
}

function networkError(err: unknown): Error {
  if (err instanceof Error && err.name === 'AbortError') return err
  const message = err instanceof Error ? err.message : String(err)
  return new Error(
    `请求失败：${message}。请检查网络，或确认该接口允许浏览器跨域（CORS）。`,
  )
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    const detail = body?.error?.message ?? body?.detail ?? body?.message
    if (typeof detail === 'string' && detail) return detail
    return JSON.stringify(body).slice(0, 400)
  } catch {
    return `${res.status} ${res.statusText}`
  }
}

/** 多模态对话。messages 直接按 OpenAI /chat/completions 格式传入。 */
export async function aiChat(
  cfg: AiConfig,
  messages: AiChatMessage[],
  opts: AiChatOptions = {},
): Promise<AiChatResult> {
  const llm = cfg.llm                       // 1 讨论 / 2 参数 → LLM 端点
  const model = llm.model.trim()
  if (!model) throw new Error('未配置对话模型 ID')

  const body: Record<string, unknown> = {
    model,
    messages,
    temperature: opts.temperature ?? 0.7,
    stream: false,
  }
  if (opts.maxTokens) body.max_tokens = opts.maxTokens

  let res: Response
  try {
    res = await fetch(endpoint(llm, '/chat/completions'), {
      method: 'POST',
      headers: { ...authHeaders(llm), 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: opts.signal,
    })
  } catch (err) {
    throw networkError(err)
  }
  if (!res.ok) throw new Error(await readError(res))

  const raw = (await res.json()) as {
    model?: string
    choices?: { message?: { content?: unknown } }[]
  }
  const content = raw?.choices?.[0]?.message?.content
  let text = ''
  if (typeof content === 'string') text = content
  else if (Array.isArray(content)) {
    text = content
      .map((part) => (part && typeof part === 'object' && 'text' in part
        ? String((part as { text?: unknown }).text ?? '')
        : ''))
      .join('')
  }
  if (!text.trim()) throw new Error('AI 没有返回文本内容')
  return { text, model: raw?.model || model, raw }
}

async function postImageEdit(
  url: string,
  form: FormData,
  cfg: AiEndpoint,
  signal?: AbortSignal,
): Promise<Response> {
  try {
    return await fetch(url, {
      method: 'POST',
      headers: authHeaders(cfg),
      body: form,
      signal,
    })
  } catch (err) {
    throw networkError(err)
  }
}

function imageForm(
  model: string,
  prompt: string,
  image: Blob,
  opts: AiImageOptions,
  withFormat: boolean,
): FormData {
  const form = new FormData()
  form.append('model', model)
  form.append('prompt', prompt)
  form.append('image', image, 'source.png')
  form.append('n', '1')
  if (opts.size) form.append('size', opts.size)
  // 大多数 OpenAI 兼容端点支持 b64_json；不支持时会自动重试去掉该字段。
  if (withFormat) form.append('response_format', 'b64_json')
  return form
}

function base64ToBlob(b64: string): Blob {
  const clean = b64.includes(',') ? (b64.split(',').pop() ?? '') : b64
  const bin = atob(clean)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type: 'image/png' })
}

/** 图像编辑：把原图与提示词发给 /images/edits，返回生成图 Blob。
 *  兼容 b64_json 与 url 两种返回；url 会在前端下载成 Blob。 */
export async function aiImageEdit(
  cfg: AiConfig,
  prompt: string,
  image: Blob,
  opts: AiImageOptions = {},
): Promise<AiImageResult> {
  const ep = cfg.image                       // 3 生图 → 图生图端点
  const model = ep.model.trim()
  if (!model) throw new Error('未配置生图模型 ID')
  const url = endpoint(ep, '/images/edits')

  let res = await postImageEdit(url, imageForm(model, prompt, image, opts, true), ep, opts.signal)
  if (!res.ok && (res.status === 400 || res.status === 422)) {
    // 某些端点不接受 response_format，去掉后重试一次。
    res = await postImageEdit(url, imageForm(model, prompt, image, opts, false), ep, opts.signal)
  }
  if (!res.ok) throw new Error(await readError(res))

  const raw = (await res.json()) as {
    model?: string
    data?: { b64_json?: string; url?: string }[]
    error?: { message?: string }
  }
  const item = raw?.data?.[0]
  if (!item) throw new Error(raw?.error?.message || '生图接口没有返回图像')

  let blob: Blob
  if (typeof item.b64_json === 'string' && item.b64_json) {
    blob = base64ToBlob(item.b64_json)
  } else if (typeof item.url === 'string' && item.url) {
    const imgRes = await fetch(item.url)
    if (!imgRes.ok) throw new Error(`下载生成图失败：${imgRes.status}`)
    blob = await imgRes.blob()
  } else {
    throw new Error('生图接口返回格式不支持（缺少 b64_json / url）')
  }
  return { blob, objectUrl: URL.createObjectURL(blob), model: raw?.model || model, raw }
}

/** ImageBitmap -> 压缩 JPEG Blob（用于上传/生图）。 */
export async function bitmapToJpegBlob(
  bitmap: ImageBitmap,
  maxEdge = 1280,
  quality = 0.86,
): Promise<Blob> {
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height))
  const w = Math.max(1, Math.round(bitmap.width * scale))
  const h = Math.max(1, Math.round(bitmap.height * scale))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('无法创建图像画布')
  ctx.drawImage(bitmap, 0, 0, w, h)
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error('图像编码失败'))
    }, 'image/jpeg', quality)
  })
}

/** Blob -> data URL（多模态消息里用）。 */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('读取图像失败'))
    reader.readAsDataURL(blob)
  })
}

/** ImageBitmap -> 压缩 JPEG data URL（对话/参数模式附图）。 */
export async function bitmapToDataUrl(
  bitmap: ImageBitmap,
  maxEdge = 1280,
  quality = 0.86,
): Promise<string> {
  return await blobToDataUrl(await bitmapToJpegBlob(bitmap, maxEdge, quality))
}
