// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  integrations: [react()],
  vite: {
    plugins: [tailwind()],
  },
  build: {
    assets: '_astro',
  },
  output: 'static',
  site: 'https://iimblsm-translator-frontend.onrender.com', 
});
