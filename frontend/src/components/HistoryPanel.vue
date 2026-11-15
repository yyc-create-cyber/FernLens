<script setup lang="ts">
/** 历史记录面板：每图独立，一步 = 一份完整参数快照。
 *  点任意一步整体回到该状态（滑块/曲线/分级/蒙版/取色一起回退），
 *  回退后再调整会从该步开新分支（丢弃其后的步骤，与 LR 一致）。 */
import { nextTick, ref, watch } from 'vue'
import Icon from '@/components/Icon'
import { t, tx } from '@/lib/i18n'
import {
  useEditor, jumpToHistory, undoHistory, redoHistory, clearHistory,
} from '@/stores/editor'

const store = useEditor()
const listEl = ref<HTMLDivElement | null>(null)

const canUndo = () => store.historyIndex > 0
const canRedo = () => store.historyIndex >= 0 && store.historyIndex < store.history.length - 1

function hhmmss(at: number): string {
  const d = new Date(at)
  const p = (n: number): string => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** 当前步滚动到可视区（新加一步或跳转后）。 */
async function scrollToCurrent(): Promise<void> {
  await nextTick()
  const box = listEl.value
  if (!box) return
  const row = box.querySelector<HTMLElement>(`[data-step="${store.historyIndex}"]`)
  if (!row) return
  const rb = row.getBoundingClientRect()
  const bb = box.getBoundingClientRect()
  const delta = (rb.top - bb.top) - box.clientHeight / 2 + rb.height / 2
  box.scrollTo({ top: Math.max(0, box.scrollTop + delta), behavior: 'smooth' })
}

watch(() => store.historyIndex, () => { void scrollToCurrent() })
watch(() => store.currentImageId, () => { void scrollToCurrent() })
</script>

<template>
  <div class="history-panel">
    <div class="bar">
      <span class="pos">
        <b>{{ store.historyIndex + 1 }}</b> / {{ store.history.length }}
      </span>
      <button class="mini" :disabled="!canUndo()" :title="t('app.undoTip')" @click="undoHistory()">
        <Icon name="rotateL" :size="11" /> {{ t('history.undo') }}
      </button>
      <button class="mini" :disabled="!canRedo()" :title="t('app.redoTip')" @click="redoHistory()">
        <Icon name="rotateR" :size="11" /> {{ t('history.redo') }}
      </button>
      <button class="mini" :disabled="store.history.length <= 1" :title="t('history.clear')"
              @click="clearHistory()">{{ t('history.clear') }}</button>
    </div>

    <div v-if="!store.history.length" class="empty">
      {{ t('history.empty') }}
    </div>

    <div v-else ref="listEl" class="list">
      <button
        v-for="(h, i) in store.history"
        :key="i"
        class="row"
        :data-step="i"
        :class="{ on: i === store.historyIndex, future: i > store.historyIndex }"
        :title="`#${i + 1}: ${h.label}`"
        @click="jumpToHistory(i)"
      >
        <span class="idx">{{ i + 1 }}</span>
        <span class="label">{{ h.label }}</span>
        <span class="time">{{ hhmmss(h.at) }}</span>
      </button>
    </div>

    <p class="tip">
      {{ tx('点任意一步可整体回到该状态；回退后再调整会从该步继续。', 'Click any step to revert all adjustments; further edits branch from there.') }}
    </p>
  </div>
</template>

<style scoped>
.history-panel { display: flex; flex-direction: column; gap: 8px; min-height: 0; }
.bar { display: flex; align-items: center; gap: 6px; }
.pos { flex: 1; font-size: 11px; color: var(--text-3); }
.pos b { color: var(--text); font-weight: 600; }
.mini:disabled { opacity: 0.35; cursor: default; }
.empty {
  padding: 12px;
  border-radius: var(--r-2);
  background: var(--surface-1);
  font-size: 11px;
  line-height: 1.65;
  color: var(--text-3);
}
.list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 360px;
  overflow-y: auto;
  padding-right: 2px;
  border-radius: var(--r-2);
  background: var(--surface-1);
  box-shadow: inset 0 0 0 1px var(--line);
}
.row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-height: 26px;
  padding: 4px 9px;
  border: 0;
  background: transparent;
  color: var(--text-2);
  font-size: 11.5px;
  text-align: left;
  cursor: pointer;
  transition: background 0.12s var(--ease), color 0.12s var(--ease);
}
.row:hover { background: var(--surface-hover); color: var(--text); }
.row.on { background: var(--accent-soft); color: var(--text); }
.row.on .idx { color: var(--accent-text); }
.row.future { opacity: 0.42; }
.idx {
  flex: none;
  width: 22px;
  font-size: 10px;
  color: var(--text-3);
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.time {
  flex: none;
  font-size: 10px;
  color: var(--text-3);
  font-variant-numeric: tabular-nums;
}
.tip { margin: 2px 0 0; font-size: 10.5px; line-height: 1.6; color: var(--text-3); }
</style>
