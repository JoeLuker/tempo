import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0', // Allow external connections (needed for Docker)
		port: 5173,
		strictPort: true,
		open: !process.env.DOCKER_ENV, // Don't open browser in Docker
		proxy: {
			'/api': {
				// Use host.docker.internal to reach host machine from Docker
				target: process.env.DOCKER_ENV 
					? 'http://host.docker.internal:8000'
					: 'http://localhost:8000',
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
				target: process.env.DOCKER_ENV 
					? 'http://host.docker.internal:8000'
					: 'http://localhost:8000',
				changeOrigin: true,
				secure: false,
			}
		}
	}
}); 