import { computed, ref, watch } from 'vue'
import { darkTheme, type GlobalThemeOverrides } from 'naive-ui'

export type ThemeId = 'light' | 'dark' | 'art'

const STORAGE_KEY = 'fernlens.theme'

function readInitialTheme(): ThemeId {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark' || saved === 'art') return saved
  } catch { /* ignore */ }
  return 'dark'
}

export const themeId = ref<ThemeId>(readInitialTheme())

export function applyTheme(id: ThemeId): void {
  themeId.value = id
  try {
    localStorage.setItem(STORAGE_KEY, id)
  } catch { /* ignore */ }
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = id
  }
}

if (typeof document !== 'undefined') {
  document.documentElement.dataset.theme = themeId.value
}

watch(themeId, (id) => {
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = id
  }
})

export const naiveTheme = computed(() => (themeId.value === 'dark' ? darkTheme : null))

export const naiveThemeOverrides = computed<GlobalThemeOverrides>(() => {
  if (themeId.value === 'light') {
    return {
      common: {
        primaryColor: '#2563eb',
        primaryColorHover: '#1d4ed8',
        primaryColorPressed: '#1e40af',
        bodyColor: '#ffffff',
        cardColor: '#ffffff',
        modalColor: '#f5faff',
        popoverColor: '#ffffff',
        inputColor: '#f1f7ff',
        textColorBase: '#0b1220',
        textColor1: '#0b1220',
        textColor2: '#1f2937',
        textColor3: '#475569',
        borderColor: 'rgba(37, 99, 235, 0.22)',
        dividerColor: 'rgba(37, 99, 235, 0.14)',
        borderRadius: '8px',
        fontFamily: 'Inter, "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif',
      },
      Tabs: {
        tabTextColorLine: '#475569',
        tabTextColorActiveLine: '#1d4ed8',
        tabTextColorHoverLine: '#0b1220',
        barColor: '#2563eb',
      },
      Input: {
        color: '#ffffff',
        colorFocus: '#ffffff',
        border: '1px solid rgba(37, 99, 235, 0.24)',
        borderFocus: '1px solid #2563eb',
        textColor: '#0b1220',
      },
      Select: {
        peers: {
          InternalSelection: {
            color: '#ffffff',
            border: '1px solid rgba(37, 99, 235, 0.24)',
            textColor: '#0b1220',
          },
        },
      },
    }
  }

  if (themeId.value === 'art') {
    return {
      common: {
        primaryColor: '#8b4513',
        primaryColorHover: '#a0522d',
        primaryColorPressed: '#6e340b',
        bodyColor: '#ecd7b4',
        cardColor: '#f5e7ce',
        modalColor: '#f7ead3',
        popoverColor: '#f7ead3',
        inputColor: '#fbf1df',
        textColorBase: '#16110b',
        textColor1: '#16110b',
        textColor2: '#2c2217',
        textColor3: '#5c4b37',
        borderColor: 'rgba(78, 52, 24, 0.26)',
        dividerColor: 'rgba(78, 52, 24, 0.16)',
        borderRadius: '6px',
        fontFamily: 'Georgia, "Noto Serif SC", "Songti SC", "Microsoft YaHei UI", serif',
      },
      Tabs: {
        tabTextColorLine: '#5c4b37',
        tabTextColorActiveLine: '#7c3410',
        tabTextColorHoverLine: '#16110b',
        barColor: '#8b4513',
      },
      Input: {
        color: '#fbf1df',
        colorFocus: '#fff8ec',
        border: '1px solid rgba(78, 52, 24, 0.28)',
        borderFocus: '1px solid #8b4513',
        textColor: '#16110b',
      },
    }
  }

  // 'dark': 黑色背景，白色文字，深蓝色毛玻璃侧栏与顶栏
  return {
    common: {
      primaryColor: '#60a5fa',
      primaryColorHover: '#93c5fd',
      primaryColorPressed: '#3b82f6',
      bodyColor: '#000000',
      cardColor: '#081226',
      modalColor: '#09152c',
      popoverColor: '#0b1936',
      inputColor: '#071224',
      textColorBase: '#ffffff',
      textColor1: '#ffffff',
      textColor2: '#e2e8f0',
      textColor3: '#94a3b8',
      borderColor: 'rgba(96, 165, 250, 0.22)',
      dividerColor: 'rgba(96, 165, 250, 0.14)',
      borderRadius: '8px',
      fontFamily: 'Inter, "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif',
    },
    Tabs: {
      tabTextColorLine: '#94a3b8',
      tabTextColorActiveLine: '#93c5fd',
      tabTextColorHoverLine: '#ffffff',
      barColor: '#60a5fa',
    },
    Input: {
      color: '#071224',
      colorFocus: '#0a1830',
      border: '1px solid rgba(96, 165, 250, 0.24)',
      borderFocus: '1px solid #60a5fa',
      textColor: '#ffffff',
    },
  }
})

export interface CanvasThemePalette {
  viewportBg: string
  cardBg: string
  wellBg: string
  gridLine: string
  axisLine: string
  curveBorder: string
  text: string
  muted: string
  accent: string
  wheelBg: string
  wheelBorder: string
}

export function getCanvasPalette(id: ThemeId = themeId.value): CanvasThemePalette {
  if (id === 'light') {
    return {
      viewportBg: '#ffffff',
      cardBg: '#f4f9ff',
      wellBg: '#f0f7ff',
      gridLine: 'rgba(15, 23, 42, 0.08)',
      axisLine: 'rgba(15, 23, 42, 0.24)',
      curveBorder: 'rgba(37, 99, 235, 0.22)',
      text: '#0f172a',
      muted: '#475569',
      accent: '#2563eb',
      wheelBg: '#eef5ff',
      wheelBorder: 'rgba(37, 99, 235, 0.30)',
    }
  }
  if (id === 'art') {
    return {
      viewportBg: '#ecd7b4',
      cardBg: '#f5e7ce',
      wellBg: '#e4caa0',
      gridLine: 'rgba(44, 34, 23, 0.11)',
      axisLine: 'rgba(44, 34, 23, 0.28)',
      curveBorder: 'rgba(78, 52, 24, 0.26)',
      text: '#16110b',
      muted: '#5c4b37',
      accent: '#8b4513',
      wheelBg: '#e2c598',
      wheelBorder: 'rgba(78, 52, 24, 0.34)',
    }
  }
  return {
    viewportBg: '#000000',
    cardBg: '#081226',
    wellBg: '#050b16',
    gridLine: 'rgba(148, 163, 184, 0.10)',
    axisLine: 'rgba(148, 163, 184, 0.24)',
    curveBorder: 'rgba(96, 165, 250, 0.24)',
    text: '#ffffff',
    muted: '#94a3b8',
    accent: '#60a5fa',
    wheelBg: '#0b1830',
    wheelBorder: 'rgba(96, 165, 250, 0.32)',
  }
}
