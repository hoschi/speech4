import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    basicSsl(),
  ],
  server: {
    proxy: {
      '/upload': 'http://localhost:8888',
      '/train': 'http://localhost:8888',
      '/ollama/stream': 'http://localhost:8888',
      '/ws/stream': 'http://localhost:8888',
    },
    https: true
  }
})
