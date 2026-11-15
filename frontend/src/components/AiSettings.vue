<script setup lang="ts">
/** AI 设置弹窗：两套 OpenAI 兼容端点（LLM / 图生图）+ 本地像素级引擎。
 *  - 对话模型：讨论(1)、参数(2)、一键精修、AI 蒙版
 *  - 图生图模型：生图(3)
 *  两套可以指向不同厂商，各自独立测试连接。本地引擎（SAM2 + U²-Net）
 *  就绪后 AI 蒙版默认走本地（免费、图像不出本机），云端作后备。 */
import { computed, ref, watch } from 'vue'
import { NAutoComplete, NButton, NInput, NModal, NSlider } from 'naive-ui'
import { useEditor, saveAiConfig, saveAiIntent, refreshLocalAI } from '@/stores/editor'
import { t, tx } from '@/lib/i18n'
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
  store.status = complete.value || imageComplete.value
    ? tx('AI 设置已保存', 'AI settings saved')
    : tx('AI 设置已清空', 'AI settings cleared')
}
</script>

<template>
  <NModal :show="store.aiSettingsOpen" @update:show="(v: boolean) => store.aiSettingsOpen = v">
    <div class="ai-settings">
      <h3>{{ t('aiSet.title') }}</h3>
      <p class="desc">
        {{ tx(
          '两套 OpenAI 兼容端点，可以指向不同厂商：对话模型用于讨论、参数与一键精修；图生图模型只用于生图。AI 蒙版与景深模型完全在本地 HuggingFace ONNX 引擎运行，无需云端 API。',
          'Configure two independent OpenAI-compatible endpoints (LLM for chat & AI grading, Image for /images/edits). Local segmentation & depth models run offline via HuggingFace ONNX.'
        ) }}
      </p>

      <!-- 对话模型（1、2 号功能） -->
      <div class="ep-block">
        <div class="ep-head">
          <span class="ep-title">{{ tx('对话模型', 'Chat / Grading LLM') }} <i>{{ tx('讨论 / 参数 / 精修', 'Chat / Params / Auto') }}</i></span>
          <span class="ep-state" :class="{ ok: complete }">{{ complete ? tx('已配置', 'Ready') : tx('未配置', 'Not set') }}</span>
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
          <span class="f-label">{{ tx('模型 ID', 'Model ID') }}</span>
          <NAutoComplete v-model:value="draft.llm.model" :options="modelOptions('llm')"
                         placeholder="gpt-4o / qwen-vl-max"
                         size="small" blur-input-on-select clearable />
        </label>
        <div class="ep-foot">
          <NButton size="tiny" :loading="testing.llm" @click="onTest('llm')">{{ tx('测试连接', 'Test') }}</NButton>
          <span class="test-msg" :class="{ ok: testOk.llm === true, bad: testOk.llm === false }">
            {{ testMessage.llm }}
          </span>
        </div>
      </div>

      <!-- 图生图模型（3 号功能） -->
      <div class="ep-block">
        <div class="ep-head">
          <span class="ep-title">{{ tx('图生图模型', 'Image Edit API') }} <i>{{ tx('生图', 'Gen') }}</i></span>
          <span class="ep-state" :class="{ ok: imageComplete }">{{ imageComplete ? tx('已配置', 'Ready') : tx('未配置', 'Not set') }}</span>
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
          <span class="f-label">{{ tx('模型 ID', 'Model ID') }}</span>
          <NAutoComplete v-model:value="draft.image.model" :options="modelOptions('image')"
                         placeholder="gpt-image-1 / flux-kontext"
                         size="small" blur-input-on-select clearable />
        </label>
        <div class="ep-foot">
          <NButton size="tiny" :loading="testing.image" @click="onTest('image')">{{ tx('测试连接', 'Test') }}</NButton>
          <span class="test-msg" :class="{ ok: testOk.image === true, bad: testOk.image === false }">
            {{ testMessage.image }}
          </span>
        </div>
        <p class="intent-hint">{{ tx('按 OpenAI 兼容 /images/edits 调用（原图 + 提示词 → 新图）。', 'Uses OpenAI-compatible /images/edits endpoint.') }}</p>
      </div>

      <label class="field intent-field">
        <span class="f-label">{{ tx('精修要求', 'Prompt') }}</span>
        <NInput v-model:value="intentDraft" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }"
                :placeholder="tx('可选，例如：暖调黄昏电影感 / 日系通透胶片 / 高反差黑白', 'Optional style prompt for AI Grade')"
                size="small" maxlength="600" show-count />
      </label>

      <label class="field strength-field">
        <span class="f-label">
          {{ tx('AI 强度', 'Strength') }}
          <b class="strength-val">{{ Math.round(draft.strength * 100) }}%</b>
        </span>
        <NSlider v-model:value="draft.strength" :min="AI_STRENGTH_MIN" :max="AI_STRENGTH_MAX"
                 :step="0.05" :tooltip="false" />
      </label>

      <!-- 本地像素级引擎 -->
      <div class="local-engine">
        <div class="le-title">
          {{ t('aiSet.tabLocal') }}
          <span v-if="localReady" class="le-ok">{{ tx('已就绪 · 默认本地运行', 'Ready · Offline Local') }}</span>
        </div>
        <p class="le-desc">{{ t('aiSet.localDesc') }}</p>
        <template v-if="local">
          <div v-for="(st, key) in local.models" :key="key" class="le-model">
            <span :class="st.ready ? 'le-dot ok' : 'le-dot'" />
            <span class="le-name">{{ st.file }}</span>
            <span class="le-size">{{ (st.size / 1024 / 1024).toFixed(1) }} MB</span>
          </div>
        </template>
        <p v-if="local && !local.runtime" class="le-warn">
          onnxruntime not installed: pip install onnxruntime
        </p>
        <div class="le-actions">
          <NButton size="small" :loading="downloading" :disabled="localReady || !local?.runtime"
                   @click="onDownload">
            {{ localReady ? tx('全部模型已就绪', 'All Models Ready') : t('aiSet.downloadAll') }}
          </NButton>
        </div>
        <div v-if="downloading" class="le-progress">
          <div class="bar"><div class="fill" :style="{ width: dlPercent + '%' }" /></div>
          <span class="pct">{{ dlPercent }}%</span>
          <span class="cur">{{ dlCurrent }}</span>
        </div>
        <p v-if="dlError" class="le-warn">{{ dlError }}</p>
      </div>

      <div class="btns">
        <span class="spacer" />
        <NButton size="small" @click="store.aiSettingsOpen = false">{{ tx('取消', 'Cancel') }}</NButton>
        <NButton size="small" type="primary" :disabled="!complete" @click="onSave">{{ tx('保存', 'Save') }}</NButton>
      </div>
    </div>
  </NModal>
</template>

<style scoped>
.ai-settings {
  width: min(500px, 92vw);
  max-height: 88vh;
  overflow-y: auto;
  background: var(--glass-modal);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-radius: var(--r-3);
  padding: 18px 20px 16px;
  box-shadow: 0 0 0 1px var(--line-2), 0 24px 60px rgba(0, 0, 0, 0.35);
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
  background: var(--surface-1);
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
  background: var(--surface-1);
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
  background: var(--track);
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
