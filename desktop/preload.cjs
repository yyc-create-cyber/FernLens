const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('fernlensNative', {
  openImageDialog: () => ipcRenderer.invoke('open-image-dialog'),
  saveImageDialog: (defaultName) => ipcRenderer.invoke('save-image-dialog', { defaultName }),
  writeFile: (filePath, bytes) => ipcRenderer.invoke('write-file', { filePath, bytes }),
})
