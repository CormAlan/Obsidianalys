import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Under utveckling: kör `obsidianalys --dev` (port 7331) och `npm run dev`.
// Proxyn skickar dev-token till C++-servern.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7331',
        changeOrigin: true,
        headers: { 'X-Obsidianalys-Token': 'dev' },
      },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
