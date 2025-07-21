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
      '/upload': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
    },
    https: true
  }
})