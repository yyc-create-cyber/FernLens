<script setup lang="ts">
/** 蒙版面板：AI 识别 / 三种新建 / 列表（选中即进入编辑）/ 羽化 / 反向 / 画笔与径向参数。
 *  列表项用眼睛图标控制启用，选中项展开局部参数。 */
import { NSwitch } from 'naive-ui'
import MiniSlider from '@/components/MiniSlider.vue'
import Icon from '@/components/Icon'
import { HINT } from '@/lib/hints'
import { vTip } from '@/lib/tip'
import { t, tx } from '@/lib/i18n'
import { useEditor, notifyChanged, notifyCommitted,
         addMask, selectMask, deleteMask, undoBrushStroke, clearBrushStrokes,
         exitMaskEdit, toggleMaskOverlay, revealMaskOverlay, setMaskHover,
         undoPolygonPoint, clearPolygonPoints, confirmPolygonMask,
         cancelPolygonDrawing, redrawPolygonMask,
         addAiMask, AI_TARGET_NAMES,
         linearMaskAngle, setLinearMaskAngle } from '@/stores/editor'
import type { AiMaskTarget } from '@/api/types'

const store = useEditor()

const KIND_ICON = { linear: 'linear', radial: 'radial', brush: 'brush', polygon: 'polygon', ai: 'ai' } as const
const AI_TARGETS: AiMaskTarget[] = ['subject', 'person', 'background', 'sky', 'depth_near', 'depth_far']

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
      <div class="block-title">{{ t('mask.aiTitle') }}</div>
      <div class="grid2">
        <button v-for="tgt in AI_TARGETS" :key="tgt" class="chip"
                :disabled="!store.loaded || store.aiBusy !== null"
                :title="store.aiBusy === tgt ? tx('识别中…', 'Detecting…') : t(`mask.target.${tgt}` as any)"
                @click="addAiMask(tgt)">
          <Icon name="ai" :size="12" />
          {{ store.aiBusy === tgt ? tx('识别中', 'Busy…') : t(`mask.target.${tgt}` as any) }}
        </button>
      </div>
    </div>

    <div class="block">
      <div class="block-title">{{ t('mask.manualTitle') }}</div>
      <div class="grid2">
        <button v-tip="HINT.maskLinear" class="chip" @click="addMask('linear')">
          <Icon name="linear" :size="13" />{{ t('mask.linear') }}</button>
        <button v-tip="HINT.maskRadial" class="chip" @click="addMask('radial')">
          <Icon name="radial" :size="13" />{{ t('mask.radial') }}</button>
        <button v-tip="HINT.maskBrushKind" class="chip" @click="addMask('brush')">
          <Icon name="brush" :size="13" />{{ t('mask.brush') }}</button>
        <button v-tip="HINT.maskPolygon" class="chip" @click="addMask('polygon')">
          <Icon name="polygon" :size="13" />{{ t('mask.polygon') }}</button>
      </div>
    </div>

    <div v-if="!store.adjustments.masks.length" class="empty">
      {{ t('mask.empty') }}
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
          <button class="mini square" :class="{ on: m.enabled }" :title="m.enabled ? tx('停用蒙版', 'Disable') : tx('启用蒙版', 'Enable')"
                  @click.stop="m.enabled = !m.enabled; notifyCommitted()">
            <Icon name="eye" :size="12" />
          </button>
          <button class="mini square danger" :title="t('mask.delete')" @click.stop="deleteMask(i)">
            <Icon name="close" :size="11" />
          </button>
        </div>

        <div v-if="store.maskSelected === i" class="mask-body" @click.stop>
          <div class="edit-row">
            <span class="edit-hint">{{ tx('正在编辑局部调整', 'Editing local adjustments') }}</span>
            <button class="mini" :title="tx('查看蒙版范围', 'Preview overlay')"
                    @click="revealMaskOverlay(1800)">
              <Icon name="eye" :size="11" /> {{ tx('预览', 'Peek') }}
            </button>
            <button class="mini" :class="{ on: store.maskOverlay }" title="O"
                    @click="toggleMaskOverlay()">
              {{ t('mask.showOverlay') }}
            </button>
            <button class="mini accent" title="Esc" @click="exitMaskEdit()">{{ t('crop.done') }}</button>
          </div>

          <MiniSlider v-if="m.kind !== 'ai'" :label="t('mask.feather')" :hint="HINT.maskFeather"
                      :value="m.feather" :min="0" :max="100"
                      :label-width="52"
                      @update:value="(v: number) => setGeom(i, 'feather', v)" />
          <div v-else class="hint-row">
            <span class="h-label">{{ t('mask.feather') }}</span>
            <span class="h-text">{{ tx('AI 像素掩码已含柔和边界', 'AI pixel mask includes soft edges') }}</span>
          </div>

          <div class="hint-row">
            <span class="h-label">{{ t('mask.invert') }}</span>
            <span class="h-text">{{ tx('作用于蒙版之外', 'Affect outside mask') }}</span>
            <NSwitch :value="m.invert" size="small"
                     @update:value="(v: boolean) => { m.invert = v; notifyCommitted() }" />
          </div>

          <div v-if="m.kind === 'ai'" class="hint-row">
            <span class="h-label">{{ tx('来源', 'Engine') }}</span>
            <span class="h-text">
              {{ m.mask_png ? tx('本地 HuggingFace 模型', 'Local HuggingFace ONNX') : tx('云端网格', 'Cloud grid') }} · {{ AI_TARGET_NAMES[m.ai_target as AiMaskTarget] || m.ai_target }}
            </span>
          </div>

          <template v-if="m.kind === 'linear'">
            <MiniSlider :label="tx('角度', 'Angle')" :hint="HINT.maskAngle" :value="linearMaskAngle(m)"
                        :min="0" :max="360" :step="0.5"
                        :decimals="1" suffix="°" :label-width="52"
                        @update:value="(v: number) => setLinearMaskAngle(i, v)" />
            <div class="btn-row">
              <button class="mini" title="Q" @click="setLinearMaskAngle(i, linearMaskAngle(m) - 15)">
                <Icon name="rotateL" :size="11" /> 15°
              </button>
              <button class="mini" title="E" @click="setLinearMaskAngle(i, linearMaskAngle(m) + 15)">
                <Icon name="rotateR" :size="11" /> 15°
              </button>
              <span class="h-text">{{ tx('拖橙色手柄或 Q/E 旋转', 'Drag handle or Q/E') }}</span>
            </div>
          </template>

          <template v-if="m.kind === 'radial'">
            <MiniSlider :label="tx('半径', 'Radius')" :hint="HINT.maskRadius" :value="m.radius"
                        :min="0.02" :max="1.5" :step="0.005"
                        :decimals="2" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'radius', v)" />
            <MiniSlider :label="tx('长宽比', 'Aspect')" :hint="HINT.maskAspect" :value="m.aspect"
                        :min="0.1" :max="5" :step="0.02"
                        :decimals="2" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'aspect', v)" />
            <MiniSlider :label="tx('旋转', 'Rotate')" :hint="HINT.maskRotate" :value="m.angle"
                        :min="0" :max="360" :step="0.5"
                        :decimals="0" suffix="°" :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'angle', v)" />
          </template>

          <template v-if="m.kind === 'brush'">
            <MiniSlider :label="t('mask.brushSize')" :hint="HINT.maskBrush" :value="m.size"
                        :min="1" :max="100" :step="1"
                        :label-width="52"
                        @update:value="(v: number) => setGeom(i, 'size', v)" />
            <div class="btn-row">
              <button class="mini" @click="undoBrushStroke(i)">{{ t('stamp.undo') }}</button>
              <button class="mini" @click="clearBrushStrokes(i)">{{ t('mask.clearBrush') }}</button>
            </div>
          </template>

          <template v-if="m.kind === 'polygon'">
            <div class="hint-row">
              <span class="h-label">{{ tx('顶点', 'Points') }}</span>
              <span class="h-text">{{ t('mask.polyStatus', { n: m.points.length }) }}</span>
            </div>
            <div v-if="store.polyDrawing" class="btn-row wrap">
              <button class="mini accent" :disabled="m.points.length < 3" @click="confirmPolygonMask()">
                {{ t('mask.polyConfirm') }}
              </button>
              <button class="mini" :disabled="!m.points.length" @click="undoPolygonPoint()">
                {{ t('mask.polyUndo') }}
              </button>
              <button class="mini" :disabled="!m.points.length" @click="clearPolygonPoints()">
                {{ t('mask.polyClear') }}
              </button>
              <button class="mini" @click="cancelPolygonDrawing()">
                {{ t('mask.polyCancel') }}
              </button>
            </div>
            <div v-else class="btn-row">
              <button class="mini" @click="redrawPolygonMask(i)">{{ t('mask.polyRedraw') }}</button>
              <span class="h-text">{{ tx('拖拽画布顶点可微调形状', 'Drag vertices on canvas to refine') }}</span>
            </div>
            <p v-if="store.polyDrawing" class="poly-tip">{{ t('mask.polyPickTip') }}</p>
          </template>
        </div>
      </div>
    </div>

    <p v-if="store.maskSelected !== null" class="tip">
      {{ tx('在画布上拖动编辑；再点一次蒙版或按「完成」退出（Esc）。', 'Drag on canvas to edit; click mask again or press Done (Esc) to exit.') }}
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
  background: var(--surface-1);
  color: var(--text-2);
  font-size: 11px;
  white-space: nowrap;
  cursor: pointer;
  transition: background 0.14s var(--ease), color 0.14s var(--ease);
}
.chip:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.chip:disabled { opacity: 0.35; cursor: default; }

.empty {
  padding: 12px;
  border-radius: var(--r-2);
  background: var(--surface-1);
  font-size: 11px;
  line-height: 1.65;
  color: var(--text-3);
}

.list { display: flex; flex-direction: column; gap: 6px; }
.mask-row {
  border-radius: var(--r-2);
  background: var(--surface-1);
  box-shadow: inset 0 0 0 1px var(--line);
  padding: 7px 9px;
  cursor: pointer;
  transition: box-shadow 0.14s var(--ease), background 0.14s var(--ease);
}
.mask-row:hover { background: var(--surface-hover); }
.mask-row.selected { box-shadow: inset 0 0 0 1px var(--accent-line); background: var(--accent-soft); }
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
.mini.accent:hover { background: var(--surface-3); color: var(--text); }
.hint-row { display: flex; align-items: center; gap: 8px; height: 24px; }
.h-label { width: 52px; flex: none; font-size: 11px; color: var(--text-2); }
.h-text { flex: 1; font-size: 10.5px; color: var(--text-3); }
.btn-row { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
.btn-row.wrap { flex-wrap: wrap; }
.poly-tip { margin: 4px 0 0; font-size: 10px; line-height: 1.5; color: var(--accent-text); }
.tip { margin: 4px 0 0; font-size: 10.5px; line-height: 1.65; color: var(--text-3); }
</style>
