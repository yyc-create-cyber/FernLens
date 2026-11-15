<script setup lang="ts">
import { NModal, NButton } from 'naive-ui'
import { useEditor, closePrefs } from '@/stores/editor'
import { applyTheme, themeId, type ThemeId } from '@/lib/theme'
import { lang, setLang, t, type Lang } from '@/lib/i18n'
import Icon from '@/components/Icon'

const store = useEditor()

const THEMES: { id: ThemeId; canvas: string; bar: string; textColor: string; blur: boolean }[] = [
  { id: 'light', canvas: '#ffffff', bar: 'rgba(195, 228, 255, 0.85)', textColor: '#0a0f1c', blur: true },
  { id: 'dark',  canvas: '#000000', bar: 'rgba(12, 34, 78, 0.88)',   textColor: '#ffffff', blur: true },
  { id: 'art',   canvas: '#ebd5b3', bar: '#e2c596',                  textColor: '#16110b', blur: false },
]

const LANGS: { id: Lang; labelKey: 'prefs.lang.zh' | 'prefs.lang.en'; badge: string }[] = [
  { id: 'zh', labelKey: 'prefs.lang.zh', badge: '中' },
  { id: 'en', labelKey: 'prefs.lang.en', badge: 'EN' },
]
</script>

<template>
  <NModal
    v-model:show="store.prefsOpen"
    preset="card"
    :title="t('prefs.title')"
    style="width: min(520px, 92vw)"
    :bordered="true"
  >
    <p class="prefs-sub">{{ t('prefs.subtitle') }}</p>

    <div class="prefs-section">
      <div class="block-title">{{ t('prefs.themeLabel') }}</div>
      <div class="theme-grid">
        <button
          v-for="item in THEMES"
          :key="item.id"
          type="button"
          class="theme-card"
          :class="{ active: themeId === item.id }"
          @click="applyTheme(item.id)"
        >
          <div class="theme-preview" :style="{ background: item.canvas, color: item.textColor }">
            <div
              class="pv-top"
              :style="{ background: item.bar, backdropFilter: item.blur ? 'blur(8px)' : 'none' }"
            >
              <span class="pv-dot" :style="{ background: item.textColor }" />
              <span class="pv-pill" :style="{ background: item.textColor }" />
            </div>
            <div class="pv-body">
              <div class="pv-canvas" />
              <div
                class="pv-side"
                :style="{ background: item.bar, backdropFilter: item.blur ? 'blur(8px)' : 'none' }"
              >
                <span class="pv-line" :style="{ background: item.textColor }" />
                <span class="pv-line short" :style="{ background: item.textColor }" />
              </div>
            </div>
          </div>
          <div class="theme-meta">
            <div class="theme-name">
              <Icon v-if="themeId === item.id" name="ai" :size="12" />
              <span>{{ t(`prefs.theme.${item.id}.name` as any) }}</span>
            </div>
            <div class="theme-desc">{{ t(`prefs.theme.${item.id}.desc` as any) }}</div>
          </div>
        </button>
      </div>
    </div>

    <div class="prefs-section">
      <div class="block-title">{{ t('prefs.langLabel') }}</div>
      <div class="lang-row">
        <button
          v-for="item in LANGS"
          :key="item.id"
          type="button"
          class="lang-btn"
          :class="{ active: lang === item.id }"
          @click="setLang(item.id)"
        >
          <span class="lang-badge num">{{ item.badge }}</span>
          <span>{{ t(item.labelKey) }}</span>
        </button>
      </div>
    </div>

    <template #footer>
      <div class="prefs-footer">
        <NButton type="primary" size="small" @click="closePrefs">{{ t('prefs.close') }}</NButton>
      </div>
    </template>
  </NModal>
</template>

<style scoped>
.prefs-sub {
  margin: 0 0 16px;
  font-size: 12px;
  color: var(--text-3);
}
.prefs-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 18px;
}
.theme-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.theme-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  border-radius: 10px;
  border: 1.5px solid var(--line-2);
  background: var(--surface-1);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.16s var(--ease), transform 0.16s var(--ease), background 0.16s var(--ease);
}
.theme-card:hover {
  border-color: var(--accent-line);
  background: var(--surface-2);
}
.theme-card.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  box-shadow: 0 0 0 1px var(--accent-line);
}
.theme-preview {
  height: 68px;
  border-radius: 7px;
  overflow: hidden;
  border: 1px solid var(--line-2);
  display: flex;
  flex-direction: column;
}
.pv-top {
  height: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 0 6px;
  border-bottom: 1px solid rgba(128, 128, 128, 0.22);
}
.pv-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  opacity: 0.7;
}
.pv-pill {
  width: 24px;
  height: 3px;
  border-radius: 99px;
  opacity: 0.45;
}
.pv-body {
  flex: 1;
  display: flex;
}
.pv-canvas {
  flex: 1;
}
.pv-side {
  width: 34px;
  padding: 7px 5px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-left: 1px solid rgba(128, 128, 128, 0.22);
}
.pv-line {
  height: 3px;
  border-radius: 99px;
  opacity: 0.55;
}
.pv-line.short {
  width: 65%;
  opacity: 0.35;
}
.theme-meta {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.theme-name {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
}
.theme-desc {
  font-size: 10.5px;
  line-height: 1.35;
  color: var(--text-3);
}
.lang-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.lang-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 9px;
  border: 1.5px solid var(--line-2);
  background: var(--surface-1);
  color: var(--text-2);
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.16s var(--ease);
}
.lang-btn:hover {
  border-color: var(--accent-line);
  color: var(--text);
}
.lang-btn.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--text);
}
.lang-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 22px;
  border-radius: 5px;
  background: var(--surface-3);
  color: var(--text);
  font-size: 11px;
  font-weight: 700;
}
.prefs-footer {
  display: flex;
  justify-content: flex-end;
}
</style>
