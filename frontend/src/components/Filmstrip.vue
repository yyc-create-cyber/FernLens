<script setup lang="ts">
/** 底部胶片条：多图缩略图，点击切换（每图独立参数），✕ 从图库移除。
 *  无图时不占空间；悬停浮起、当前项用强调色细环标记。 */
import { useEditor, selectImage, removeImage } from '@/stores/editor'
import { thumbUrl } from '@/api/client'
import Icon from '@/components/Icon'

const store = useEditor()
</script>

<template>
  <div v-if="store.library.length" class="filmstrip">
    <div class="rail">
      <div
        v-for="img in store.library"
        :key="img.id"
        class="cell"
        :class="{ current: img.current }"
        :title="`${img.name}（${img.width}×${img.height}）点击切换`"
        @click="selectImage(img.id)"
      >
        <img :src="thumbUrl(img.id, img.serial)" :alt="img.name" draggable="false" />
        <span class="label">{{ img.name }}</span>
        <button class="del" title="从图库移除" @click.stop="removeImage(img.id)">
          <Icon name="close" :size="10" />
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.filmstrip {
  flex: none;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--panel);
  border-top: 1px solid var(--line);
}
.rail {
  display: flex;
  gap: 9px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 3px 0;
  min-width: 0;
}
.cell {
  position: relative;
  flex: none;
  width: 88px;
  height: 62px;
  border-radius: var(--r-1);
  overflow: hidden;
  background: var(--well);
  box-shadow: inset 0 0 0 1px var(--line);
  cursor: pointer;
  transition: box-shadow 0.16s var(--ease), transform 0.16s var(--ease);
}
.cell:hover {
  transform: translateY(-2px);
  box-shadow: inset 0 0 0 1px var(--line-2), 0 8px 18px rgba(0, 0, 0, 0.5);
}
.cell.current {
  box-shadow: inset 0 0 0 1px var(--accent), 0 0 0 2px rgba(224, 161, 99, 0.16),
              0 8px 18px rgba(0, 0, 0, 0.55);
}
.cell.current::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  box-shadow: inset 0 0 22px rgba(224, 161, 99, 0.12);
}
.cell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  user-select: none;
}
.label {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 8px 5px 3px;
  font-size: 9.5px;
  color: #dfe2e8;
  background: linear-gradient(180deg, rgba(6, 7, 10, 0), rgba(6, 7, 10, 0.85));
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  opacity: 0;
  transition: opacity 0.14s var(--ease);
  pointer-events: none;
}
.cell:hover .label,
.cell.current .label { opacity: 1; }
.del {
  position: absolute;
  top: 3px;
  right: 3px;
  width: 16px;
  height: 16px;
  border: 0;
  border-radius: 5px;
  background: rgba(6, 7, 10, 0.7);
  color: #b9bec8;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.14s var(--ease), color 0.14s var(--ease);
}
.cell:hover .del { opacity: 1; }
.del:hover { color: var(--danger); }
</style>
