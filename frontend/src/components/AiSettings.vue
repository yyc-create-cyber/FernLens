<script setup lang="ts">
/** AI 设置弹窗：两套 OpenAI 兼容端点（LLM / 图生图）+ 本地像素级引擎。
 *  - 对话模型：讨论(1)、参数(2)、一键精修、AI 蒙版
 *  - 图生图模型：生图(3)
 *  两套可以指向不同厂商，各自独立测试连接。本地引擎（SAM2 + U²-Net）
 *  就绪后 AI 蒙版默认走本地（免费、图像不出本机），云端作后备。 */
import { computed, ref, watch } from 'vue'
import { NAutoComplete, NButton, NInput, NModal, NSlider } from 'naive-ui'
import { useEditor, saveAiConfig, saveAiIntent, refreshLocalAI } from '@/stores/editor'
import {
  testAiEndpoint, downloadLocalModels, localModelsProgress,
  aiEndpointComplete, emptyAiConfig, AI_STRENGTH_MIN, AI_STRENGTH_MAX,
} from '@/api/client'
import type { AiConfig, AiEndpoint } from '@/api/client'

const store = useEditor()

// 编辑副本：打开时从 store 拷贝，保存时才写回（localStorage）
const draft = ref<AiConfig>(emptyAiConfig())
// 精修提示词：可选，随每次 AI 请求发送（空 = 模型自选风格）
const intentDraft = ref('')
// 每套端点各自维护「测试连接」的状态与模型列表
type EpKey = 'llm' | 'image'
const models = ref<Record<EpKey, string[]>>({ llm: [], image: [] })
const testing = ref<Record<EpKey, boolean>>({ llm: false, image: false })
const testMessage = ref<Record<EpKey, string>>({ llm: '', image: '' })
const testOk = ref<Record<EpKey, boolean | null>>({ llm: null, image: null })
const openSection = ref<EpKey>('llm')

// ---- 本地引擎 ----
const local = computed(() => store.localAI)
const localReady = computed(() => !!local.value?.ready)
const downloading = ref(false)
const dlCurrent = ref('')
const dlPercent = ref(0)
const dlError = ref('')
let pollTimer: ReturnType<typeof setInterval> | null = null

function stopPoll(): void {
  if (pollTimer !== null) { clearInterval(pollTimer); pollTimer = null }
}

async function pollProgress(): Promise<void> {
  try {
    const p = await localModelsProgress()
    if (p.total > 0) dlPercent.value = Math.round(p.received / p.total * 100)
    dlCurrent.value = p.current
    dlError.value = p.error
    if (!p.running) {
      stopPoll()
      downloading.value = false
      await refreshLocalAI()
    }
  } catch { /* 轮询失败静默重试 */ }
}

async function onDownload(): Promise<void> {
  if (downloading.value) return
  try {
    await downloadLocalModels()
    downloading.value = true
    dlError.value = ''
    pollTimer = setInterval(() => void pollProgress(), 1000)
  } catch (err) {
    dlError.value = err instanceof Error ? err.message : String(err)
  }
}

watch(() => store.aiSettingsOpen, (open) => {
  if (open) {
    draft.value = {
      llm: { ...store.aiConfig.llm }, image: { ...store.aiConfig.image },
      strength: store.aiConfig.strength,
    }
    intentDraft.value = store.aiIntent
    models.value = { llm: [], image: [] }
    testMessage.value = { llm: '', image: '' }
    testOk.value = { llm: null, image: null }
    openSection.value = 'llm'
    void refreshLocalAI()
  } else {
    stopPoll()
    downloading.value = false
  }
})

const complete = computed(() => aiEndpointComplete(draft.value.llm))
const imageComplete = computed(() => aiEndpointComplete(draft.value.image))

function modelOptions(key: EpKey) {
  return models.value[key].map((m) => ({ label: m, value: m }))
}

async function onTest(key: EpKey): Promise<void> {
  if (testing.value[key]) return
  const ep: AiEndpoint = draft.value[key]
  if (!ep.baseUrl.trim() || !ep.apiKey.trim()) {
    testOk.value[key] = false
    testMessage.value[key] = '请先填写 URL 与 API Key'
    return
  }
  testing.value[key] = true
  testOk.value[key] = null
  testMessage.value[key] = '连接中…'
  try {
    const r = await testAiEndpoint(ep)
    models.value[key] = r.models
    testOk.value[key] = true
    testMessage.value[key] = r.models.length
      ? `连接成功，共 ${r.models.length} 个模型（模型 ID 可从下拉选择）`
      : '连接成功，但端点未返回模型列表，请手动填写模型 ID'
  } catch (err) {
    testOk.value[key] = false
    testMessage.value[key] = err instanceof Error ? err.message : String(err)
  } finally {
    testing.value[key] = false
  }
}

function onSave(): void {
  saveAiConfig(draft.value)
  saveAiIntent(intentDraft.value)
  store.aiSettingsOpen = false
  store.status = complete.value || imageComplete.value ? 'AI 设置已保存' : 'AI 设置已清空'
}
</script>

<template>
  <NModal :show="store.aiSettingsOpen" @update:show="(v: boolean) => store.aiSettingsOpen = v">
    <div class="ai-settings">
      <h3>AI 设置</h3>
      <p class="desc">
        两套 OpenAI 兼容端点，可以指向不同厂商：<b>对话模型</b>用于讨论(1)、参数(2)与一键精修；
        <b>图生图模型</b>只用于生图(3)。配置仅保存在本机，图像会上传到对应端点。
        <b>蒙版识别不走云端</b>，只用下面的本地引擎。
        也可用环境变量 PHOTOLAB_AI_BASE_URL / PHOTOLAB_AI_KEY / PHOTOLAB_AI_MODEL 作为对话模型的后备。
      </p>

      <!-- 对话模型（1、2 号功能） -->
      <div class="ep-block">
        <div class="ep-head">
          <span class="ep-title">对话模型 <i>讨论 / 参数 / 精修</i></span>
          <span class="ep-state" :class="{ ok: complete }">{{ complete ? '已配置' : '未配置' }}</span>
        </div>
        <label class="field">
          <span class="f-label">URL</span>
          <NInput v-model:value="draft.llm.baseUrl" placeholder="https://api.example.com/v1"
                  size="small"
                  @input="() => { models.llm = []; testOk.llm = null; testMessage.llm = '' }" />
        </label>
        <label class="field">
          <span class="f-label">API Key</span>
          <NInput v-model:value="draft.llm.apiKey" type="password" show-password-on="click"
                  placeholder="sk-…" size="small" />
        </label>
        <label class="field">
          <span class="f-label">模型 ID</span>
          <NAutoComplete v-model:value="draft.llm.model" :options="modelOptions('llm')"
                         placeholder="例如 gpt-4o / qwen-vl-max（可测试后从列表选择）"
                         size="small" blur-input-on-select clearable />
        </label>
        <div class="ep-foot">
          <NButton size="tiny" :loading="testing.llm" @click="onTest('llm')">测试连接</NButton>
          <span class="test-msg" :class="{ ok: testOk.llm === true, bad: testOk.llm === false }">
            {{ testMessage.llm }}
          </span>
        </div>
      </div>

      <!-- 图生图模型（3 号功能） -->
      <div class="ep-block">
        <div class="ep-head">
          <span class="ep-title">图生图模型 <i>生图</i></span>
          <span class="ep-state" :class="{ ok: imageComplete }">{{ imageComplete ? '已配置' : '未配置' }}</span>
        </div>
        <label class="field">
          <span class="f-label">URL</span>
          <NInput v-model:value="draft.image.baseUrl" placeholder="https://api.example.com/v1"
                  size="small"
                  @input="() => { models.image = []; testOk.image = null; testMessage.image = '' }" />
        </label>
        <label class="field">
          <span class="f-label">API Key</span>
          <NInput v-model:value="draft.image.apiKey" type="password" show-password-on="click"
                  placeholder="sk-…" size="small" />
        </label>
        <label class="field">
          <span class="f-label">模型 ID</span>
          <NAutoComplete v-model:value="draft.image.model" :options="modelOptions('image')"
                         placeholder="例如 gpt-image-1 / flux-kontext / seedream"
                         size="small" blur-input-on-select clearable />
        </label>
        <div class="ep-foot">
          <NButton size="tiny" :loading="testing.image" @click="onTest('image')">测试连接</NButton>
          <span class="test-msg" :class="{ ok: testOk.image === true, bad: testOk.image === false }">
            {{ testMessage.image }}
          </span>
        </div>
        <p class="intent-hint">按 OpenAI 兼容 /images/edits 调用（原图 + 提示词 → 新图）。</p>
      </div>

      <label class="field intent-field">
        <span class="f-label">精修要求</span>
        <NInput v-model:value="intentDraft" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }"
                placeholder="可选，例如：调成暖调黄昏电影感 / 日系胶片、低饱和高光柔 / 高反差黑白"
                size="small" maxlength="600" show-count />
      </label>
      <p class="intent-hint">
        每次点「AI」都会按此要求做精修；留空则由模型自行判断风格。
      </p>

      <label class="field strength-field">
        <span class="f-label">
          AI 调整强度
          <b class="strength-val">{{ Math.round(draft.strength * 100) }}%</b>
        </span>
        <NSlider v-model:value="draft.strength" :min="AI_STRENGTH_MIN" :max="AI_STRENGTH_MAX"
                 :step="0.05" :tooltip="false" />
      </label>
      <p class="intent-hint">
        统一缩放 AI 返回的幅度（参数 / 曲线 / 色轮 / 取色器）。默认 60% 偏克制；
        觉得 AI 调得太猛就往下调，想要更强风格化再往上调。
      </p>

      <!-- 本地像素级引擎 -->
      <div class="local-engine">
        <div class="le-title">本地像素级引擎 <span v-if="localReady" class="le-ok">已就绪 · AI 蒙版默认使用</span></div>
        <p class="le-desc">SAM 2 + U²-Net 本地推理：像素级掩码、免费、图像不出本机（CPU 约 2~10 秒/次）。</p>
        <template v-if="local">
          <div v-for="(st, key) in local.models" :key="key" class="le-model">
            <span :class="st.ready ? 'le-dot ok' : 'le-dot'" />
            <span class="le-name">{{ st.file }}</span>
            <span class="le-size">{{ (st.size / 1024 / 1024).toFixed(0) }} MB</span>
          </div>
        </template>
        <p v-if="local && !local.runtime" class="le-warn">
          未安装 onnxruntime：pip install onnxruntime
        </p>
        <div class="le-actions">
          <NButton size="small" :loading="downloading" :disabled="localReady || !local?.runtime"
                   @click="onDownload">
            {{ localReady ? '模型已就绪' : '下载模型（约 575 MB，hf-mirror）' }}
          </NButton>
        </div>
        <div v-if="downloading" class="le-progress">
          <div class="bar"><div class="fill" :style="{ width: dlPercent + '%' }" /></div>
          <span class="pct">{{ dlPercent }}%</span>
          <span class="cur">{{ dlCurrent }}</span>
        </div>
        <p v-if="dlError" class="le-warn">下载失败：{{ dlError }}</p>
      </div>

      <div class="btns">
        <span class="spacer" />
        <NButton size="small" @click="store.aiSettingsOpen = false">取消</NButton>
        <NButton size="small" type="primary" :disabled="!complete" @click="onSave">保存</NButton>
      </div>
    </div>
  </NModal>
</template>

<style scoped>
.ai-settings {
  width: min(500px, 92vw);
  max-height: 88vh;
  overflow-y: auto;
  background: var(--panel-2);
  border-radius: var(--r-3);
  padding: 18px 20px 16px;
  box-shadow: 0 0 0 1px var(--line-2), 0 24px 60px rgba(0, 0, 0, 0.6);
}
h3 {
  margin: 0 0 8px;
  font-size: 13.5px;
  font-weight: 600;
  letter-spacing: 0.4px;
  color: var(--text);
}
.desc { margin: 0 0 14px; font-size: 11px; color: var(--text-3); line-height: 1.65; }
.field { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.f-label { width: 56px; flex: none; font-size: 11.5px; color: var(--text-2); }
.field :deep(.n-input), .field :deep(.n-auto-complete) { flex: 1; }
.intent-field { align-items: flex-start; }
.intent-field .f-label { padding-top: 5px; }
.strength-field { margin-top: 2px; }
.strength-field :deep(.n-slider) { flex: 1; }
.strength-val { color: var(--text-1); font-weight: 600; }
.intent-hint { margin: -3px 0 10px 66px; font-size: 10.5px; color: var(--text-3); line-height: 1.55; }
.test-msg { min-height: 15px; margin: 2px 0 10px; font-size: 10.5px; color: var(--text-3); }
.test-msg.ok { color: var(--success); }
.test-msg.bad { color: var(--danger); }

.ep-block {
  border-radius: var(--r-2);
  padding: 11px 12px 8px;
  margin-bottom: 10px;
  background: rgba(255, 255, 255, 0.02);
  box-shadow: inset 0 0 0 1px var(--line);
}
.ep-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 9px; }
.ep-title { font-size: 11.5px; color: var(--text); font-weight: 600; }
.ep-title i { font-style: normal; font-weight: 400; color: var(--text-3); margin-left: 4px; }
.ep-state { margin-left: auto; font-size: 10px; color: var(--text-3); }
.ep-state.ok { color: var(--success); }
.ep-foot { display: flex; align-items: center; gap: 9px; margin-top: 2px; }
.ep-foot .test-msg { margin: 0; flex: 1; }

.local-engine {
  border-radius: var(--r-2);
  padding: 10px 12px;
  margin-bottom: 12px;
  background: rgba(255, 255, 255, 0.025);
  box-shadow: inset 0 0 0 1px var(--line);
}
.le-title { font-size: 11.5px; color: var(--text-2); margin-bottom: 5px; }
.le-ok { color: var(--success); font-size: 10.5px; margin-left: 6px; }
.le-desc { margin: 0 0 7px; font-size: 10.5px; color: var(--text-3); line-height: 1.55; }
.le-model { display: flex; align-items: center; gap: 7px; font-size: 10.5px; color: var(--text-3); padding: 1.5px 0; }
.le-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-3); flex: none; }
.le-dot.ok { background: var(--success); box-shadow: 0 0 6px rgba(78, 214, 166, 0.6); }
.le-name { flex: 1; color: var(--text-2); }
.le-size { color: var(--text-3); }
.le-warn { font-size: 10.5px; color: var(--warn); margin: 5px 0 0; }
.le-actions { margin-top: 8px; }
.le-progress { display: flex; align-items: center; gap: 8px; margin-top: 7px; }
.le-progress .bar {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.09);
  overflow: hidden;
}
.le-progress .fill {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-hi));
  transition: width 0.3s var(--ease);
}
.le-progress .pct { font-size: 10.5px; color: var(--text-2); width: 34px; text-align: right; }
.le-progress .cur {
  font-size: 10px;
  color: var(--text-3);
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.btns {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
}
.spacer { flex: 1; }
</style>
