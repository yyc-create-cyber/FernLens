<script setup lang="ts">
/** 蒙版面板：AI 识别 / 三种新建 / 列表（选中即进入编辑）/ 羽化 / 反向 / 画笔与径向参数。
 *  列表项用眼睛图标控制启用，选中项展开局部参数。 */
import { NSwitch } from 'naive-ui'
import MiniSlider from '@/components/MiniSlider.vue'
import Icon from '@/components/Icon'
import { HINT } from '@/lib/hints'
import { vTip } from '@/lib/tip'
import { useEditor, notifyChanged, notifyCommitted,
         addMask, selectMask, deleteMask, undoBrushStroke, clearBrushStrokes,
         exitMaskEdit, toggleMaskOverlay, revealMaskOverlay, setMaskHover,
         addAiMask, AI_TARGET_NAMES,
         linearMaskAngle, setLinearMaskAngle } from '@/stores/editor'
import type { AiMaskTarget } from '@/api/types'

const store = useEditor()

const KIND_ICON = { linear: 'linear', radial: 'radial', brush: 'brush', ai: 'ai' } as const
const AI_TARGETS: AiMaskTarget[] = ['subject', 'person', 'background', 'sky']

function setGeom(i: number, field: 'feather' | 'size' | 'radius' | 'aspect' | 'angle', v: number): void {
  const m = store.adjustments.masks[i]
  if (!m) return
  ;(m as unknown as Record<string, number>)[field] = v
  notifyChanged()
}
</script>

<template>
  <div class="mask-panel">
    <div class="block">
      <div class="block-title">AI 识别</div>
      <div class="grid2">
        <button v-for="t in AI_TARGETS" :key="t" class="chip"
                :disabled="!store.loaded || store.aiBusy !== null"
                :title="store.aiBusy === t ? '识别中…' : `由 AI 识别画面中的${AI_TARGET_NAMES[t]}并创建蒙版`"
                @click="addAiMask(t)">
          <Icon name="ai" :size="12" />
          {{ store.aiBusy === t ? '识别中' : AI_TARGET_NAMES[t] }}
        </button>
      </div>
    </div>

    <div class="block">
      <div class="block-title">手动新建</div>
      <div class="grid3">
        <button v-tip="HINT.maskLinear" class="chip" @click="addMask('linear')">
          <Icon name="linear" :size="13" />线性</button>
        <button v-tip="HINT.maskRadial" class="chip" @click="addMask('radial')">
          <Icon name="radial" :size="13" />径向</button>
        <button v-tip="HINT.maskBrushKind" class="chip" @click="addMask('brush')">
          <Icon name="brush" :size="13" />画笔</button>
      </div>
    </div>

    <div v-if="!store.adjustments.masks.length" class="empty">
      还没有蒙版。蒙版可对画面局部做全套调整——基础、效果、曲线、分级、取色。
    </div>

    <div class="list">
      <div
        v-for="(m, i) in store.adjustments.masks"
        :key="i"
        class="mask-row"
        :class="{ selected: store.maskSelected === i, off: !m.enabled }"
        @click="selectMask(i)"
      >
        <div class="mask-head" @mouseenter="setMaskHover(true)" @mouseleave="setMaskHover(false)">
          <Icon :name="KIND_ICON[m.kind]" :size="13" class="kind" />
          <span class="name">{{ m.name || m.kind }}</span>
          <button class="mini square" :class="{ on: m.enabled }" :title="m.enabled ? '停用蒙版' : '启用蒙版'"
                  @click.stop="m.enabled = !m.enabled; notifyCommitted()">
            <Icon name="eye" :size="12" />
          </button>
          <button class="mini square danger" title="删除蒙版" @click.stop="deleteMask(i)">
            <Icon name="close" :size="11" />
          </button>
        </div>

        <div v-if="store.maskSelected === i" class="mask-body" @click.stop>
          <div class="edit-row">
            <span class="edit-hint">正在编辑局部调整</span>
            <button class="mini" title="查看蒙版范围（也可悬停蒙版名称预览）"
                    @click="revealMaskOverlay(1800)">
              <Icon name="eye" :size="11" /> 预览
            </button>
            <button class="mini" :class="{ on: store.maskOverlay }" title="叠加总开关（O 键）"
                    @click="toggleMaskOverlay()">
              叠加
            </button>
            <button class="mini accent" title="退出局部编辑（Esc）" @click="exitMaskEdit()">完成</button>
          </div>

          <MiniSlider v-if="m.kind !== 'ai'" label="羽化" :hint="HINT.maskFeather"
                      :value="m.feather" :min="0" :max="100"
                      :label-width="52"
                      @update:value="(v: number) => setGeom(i, 'feather', v)" />
          <div v-else class="hint-row">
            <span class="h-label">羽化</span>
            <span class="h-text">AI 蒙版按识别边界作用，不羽化</span>
          </div>

          <div class="hint-row">
            <span class="h-label">反向</span>
            <span class="h-text">作用于蒙版之外</span>
            <NSwitch :value="m.invert" size="small"
                     @update:value="(v: boolean) => { m.invert = v; notifyCommitted() }" />
          </div>

          <div v-if="m.kind === 'ai'" class="hint-row">
            <span class="h-label">来源</span>
            <span class="h-text">
              {{ m.mask_png ? '本地像素级' : '旧版网格' }}识别{{ AI_TARGET_NAMES[m.ai_target as AiMaskTarget] || '' }}{{ m.mask_png ? '' : ` · ${m.grid_w}×${m.grid_h}` }}
            </span>
          </div>

          <template v-if="m.kind === 'linear'">
            <MiniSlider label="角度" :hint="HINT.maskAngle" :value="linearMaskAngle(m)"
                        :min="0" :max="360" :step="0.5"
                        :decimals="1" suffix="°" :label-width="52"
                        @update:value="(v: number) => setLinearMaskAngle(i, v)" />
            <div class="btn-row">
              <button class="mini" title="逆时针 15°（Q 微调 1°）"
                      @click="setLinearMaskAngle(i, linearMaskAngle(m) - 15)">
                <Icon name="rotateL" :size="11" /> 15°
              </button>
              <button class="mini" title="顺时针 15°（E 微调 1°）"
                      @click="setLinearMaskAngle(i, linearMaskAngle(m) + 15)">
                <Icon name="rotateR" :size="11" /> 15°
              </button>
              <span class="h-text">拖橙色手柄或 Q/E 旋转</span>
            </div>
          </template>

          <template v-if="m.kind === 'radial'">
            <MiniSlider label="半径" :hint="HINT.maskRadius" :value="m.radius"
                        :min="0.02" :max="1.5" :step="0.005"
                        :decimals="2" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'radius', v)" />
            <MiniSlider label="长宽比" :hint="HINT.maskAspect" :value="m.aspect"
                        :min="0.1" :max="5" :step="0.02"
                        :decimals="2" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'aspect', v)" />
            <MiniSlider label="旋转" :hint="HINT.maskRotate" :value="m.angle"
                        :min="0" :max="360" :step="0.5"
                        :decimals="0" suffix="°" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'angle', v)" />
          </template>

          <template v-if="m.kind === 'brush'">
            <MiniSlider label="笔刷" :hint="HINT.maskBrush" :value="m.size"
                        :min="1" :max="100" :step="1"
                        :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'size', v)" />
            <div class="btn-row">
              <button class="mini" @click="undoBrushStroke(i)">撤销一笔</button>
              <button class="mini" @click="clearBrushStrokes(i)">清除笔迹</button>
            </div>
          </template>
        </div>
      </div>
    </div>

    <p v-if="store.maskSelected !== null" class="tip">
      在画布上拖动编辑；再点一次蒙版或按「完成」退出（Esc）。悬停蒙版名称或点「预览」查看红色范围，平时不遮挡画面。
    </p>
  </div>
</template>

<style scoped>
.mask-panel { display: flex; flex-direction: column; gap: 14px; }
.block { display: flex; flex-direction: column; gap: 7px; }
.block-title {
  font-size: 10px;
  letter-spacing: 1.1px;
  color: var(--text-3);
  text-transform: uppercase;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
.chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  height: 26px;
  border: 0;
  border-radius: var(--r-1);
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-2);
  font-size: 11.5px;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
}
.chip:hover:not(:disabled) { background: rgba(255, 255, 255, 0.10); color: var(--text); }
.chip:disabled { opacity: 0.35; cursor: default; }

.empty {
  padding: 12px;
  border-radius: var(--r-2);
  background: rgba(255, 255, 255, 0.025);
  font-size: 11px;
  line-height: 1.65;
  color: var(--text-3);
}

.list { display: flex; flex-direction: column; gap: 6px; }
.mask-row {
  border-radius: var(--r-2);
  background: var(--panel-2);
  box-shadow: inset 0 0 0 1px var(--line);
  padding: 7px 9px;
  cursor: pointer;
  transition: box-shadow 0.14s var(--ease), background 0.14s var(--ease);
}
.mask-row:hover { background: #15181e; }
.mask-row.selected { box-shadow: inset 0 0 0 1px var(--accent-line); background: rgba(224, 161, 99, 0.06); }
.mask-row.off .name, .mask-row.off .kind { opacity: 0.45; }
.mask-head { display: flex; align-items: center; gap: 7px; }
.kind { color: var(--text-2); flex: none; }
.name {
  flex: 1;
  font-size: 12px;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mask-body {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: 2px;
  cursor: default;
}
.edit-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.edit-hint { flex: 1; font-size: 10.5px; color: var(--text-3); }
.mini.accent { background: var(--accent-soft); color: var(--accent-text); }
.mini.accent:hover { background: rgba(224, 161, 99, 0.22); color: #d6daff; }
.hint-row { display: flex; align-items: center; gap: 8px; height: 24px; }
.h-label { width: 52px; flex: none; font-size: 11px; color: var(--text-2); }
.h-text { flex: 1; font-size: 10.5px; color: var(--text-3); }
.btn-row { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
.tip { margin: 4px 0 0; font-size: 10.5px; line-height: 1.65; color: var(--text-3); }
</style>
