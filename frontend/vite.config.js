import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0', // Allow external connections (needed for Docker)
		port: 5174,
		strictPort: true,
		proxy: {
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true,
				secure: false,
				configure: (proxy, options) => {
					proxy.on('error', (err, req, res) => {
						console.log('Proxy error:', err);
					});
					proxy.on('proxyReq', (proxyReq, req, res) => {
						console.log('Proxying request to:', options.target + req.url);
					});
				}
			}
		}
	},
	preview: {
		host: '0.0.0.0',
		port: 3000,
		strictPort: true,
		proxy: {
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true,
				secure: false,
			}
		}
	}
}); 