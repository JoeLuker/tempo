<script lang="ts">
	import ElkFlow, { type Token } from '$lib/components/ElkFlow.svelte';
	import '../app.css';

	let tokens = $state<Token[]>([]);
	let prompt = $state('Once upon a time');
	let isGenerating = $state(false);
	let errorMessage = $state<string | null>(null);
	let statusMessage = $state<string>('');

	// Debug indicator lights
	let backendConnected = $state<boolean>(false);
	let apiResponse = $state<string>('waiting');
	let elkRendered = $state<boolean>(false);

	// Check backend connection on mount
	$effect(() => {
		fetch('/api/generate', { method: 'OPTIONS' })
			.then(() => { backendConnected = true; })
			.catch(() => { backendConnected = false; });
	});

	async function generateTokens() {
		isGenerating = true;
		errorMessage = null;
		tokens = [];
		elkRendered = false;
		apiResponse = 'requesting';
		statusMessage = 'Connecting to TEMPO...';

		try {
			statusMessage = 'Generating tokens...';
			const response = await fetch('/api/generate', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					prompt: prompt,
					selection_threshold: 0.25,
					max_tokens: 20,
					isolate: false,
					seed: 42
				})
			});

			if (!response.ok) {
				apiResponse = 'error';
				throw new Error(`API error: ${response.status} ${response.statusText}`);
			}

			apiResponse = 'success';
			const data = await response.json();
			statusMessage = 'Processing results...';

			// Convert API nodes to Token format
			const convertedTokens = data.nodes.map((node: any) => ({
				id: node.id,
				text: node.text.trim(),
				type: node.logical_step === 0 ? 'prompt' : (node.is_parallel ? 'parallel' : 'single'),
				probability: node.probability,
				parent_ids: node.parent_ids,  // Keep all parents for convergence support
				step: node.logical_step
			}));

			// Set all tokens at once (streaming causes Elk to be called too many times)
			tokens = convertedTokens;

			// Wait a moment for Elk to render
			await new Promise(r => setTimeout(r, 1000));

			elkRendered = true;
			statusMessage = `Generated ${tokens.length} tokens in ${data.generation_time.toFixed(2)}s`;
		} catch (error) {
			console.error('Generation failed:', error);
			apiResponse = 'error';
			errorMessage = error instanceof Error ? error.message : 'Failed to generate tokens';
			statusMessage = '';
		} finally {
			isGenerating = false;
		}
	}
</script>

<main>
	<header>
		<h1>TEMPO Token Visualizer</h1>
		<p>Interactive D3.js tree showing parallel token generation</p>
	</header>

	<!-- Debug Indicator Lights -->
	<div class="debug-indicators">
		<div class="indicator">
			<span class="indicator-light" class:green={backendConnected} class:red={!backendConnected}></span>
			<span class="indicator-label">Backend: {backendConnected ? 'Connected' : 'Disconnected'}</span>
		</div>
		<div class="indicator">
			<span class="indicator-light"
				class:green={apiResponse === 'success'}
				class:yellow={apiResponse === 'requesting'}
				class:red={apiResponse === 'error'}
				class:gray={apiResponse === 'waiting'}></span>
			<span class="indicator-label">API: {apiResponse}</span>
		</div>
		<div class="indicator">
			<span class="indicator-light" class:green={elkRendered} class:gray={!elkRendered}></span>
			<span class="indicator-label">Elk Render: {elkRendered ? 'Complete' : 'Waiting'}</span>
		</div>
	</div>

	<div class="controls-panel">
		<input
			type="text"
			bind:value={prompt}
			placeholder="Enter prompt..."
			disabled={isGenerating}
		/>
		<button onclick={generateTokens} disabled={isGenerating}>
			{#if isGenerating}
				<span class="spinner"></span>
				Generating...
			{:else}
				Generate
			{/if}
		</button>
	</div>

	{#if errorMessage}
		<div class="error-message">
			⚠️ {errorMessage}
		</div>
	{/if}

	{#if statusMessage}
		<div class="status-message">
			{statusMessage}
		</div>
	{/if}

	<div class="visualization">
		{#if tokens.length === 0 && !isGenerating && !errorMessage}
			<div class="empty-state">
				<div class="empty-icon">🌳</div>
				<h3>Ready to visualize</h3>
				<p>Enter a prompt above and click Generate to see TEMPO's parallel token generation in action</p>
			</div>
		{:else}
			<ElkFlow {tokens} />
		{/if}
	</div>
</main>

<style>
	main {
		padding: var(--spacing-base);
		max-width: 1400px;
		margin: 0 auto;
		min-height: 100vh;
	}

	header {
		text-align: center;
		margin-bottom: 20px;
	}

	header h1 {
		font-size: clamp(24px, 5vw, 32px);
		color: #1e293b;
		margin: 0 0 10px 0;
	}

	header p {
		color: #64748b;
		font-size: clamp(14px, 3vw, 16px);
	}

	.debug-indicators {
		display: flex;
		gap: 16px;
		margin-bottom: 20px;
		padding: 12px;
		background: #f8fafc;
		border-radius: 8px;
		border: 1px solid #e2e8f0;
		flex-wrap: wrap;
		justify-content: center;
	}

	.indicator {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 12px;
		background: white;
		border-radius: 6px;
		border: 1px solid #e2e8f0;
	}

	.indicator-light {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		border: 1px solid rgba(0, 0, 0, 0.1);
		transition: all 0.3s ease;
		box-shadow: 0 0 4px rgba(0, 0, 0, 0.1);
	}

	.indicator-light.green {
		background: #10b981;
		box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
	}

	.indicator-light.yellow {
		background: #f59e0b;
		box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
		animation: pulse 1.5s ease-in-out infinite;
	}

	.indicator-light.red {
		background: #ef4444;
		box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
	}

	.indicator-light.gray {
		background: #94a3b8;
	}

	.indicator-label {
		font-size: 13px;
		color: #475569;
		font-weight: 500;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}

	.controls-panel {
		display: flex;
		gap: 12px;
		margin-bottom: 20px;
		flex-wrap: wrap;
	}

	.controls-panel input {
		flex: 1;
		min-width: 200px;
		padding: 14px 18px;
		border: 2px solid #e2e8f0;
		border-radius: 12px;
		font-size: 16px;
		/* Prevent zoom on iOS */
		font-size: max(16px, 1em);
		min-height: var(--tap-target-min);
	}

	.controls-panel input:focus {
		outline: none;
		border-color: #9333ea;
		box-shadow: 0 0 0 3px rgba(147, 51, 234, 0.1);
	}

	.controls-panel button {
		padding: 14px 28px;
		background: #9333ea;
		color: white;
		border: none;
		border-radius: 12px;
		font-size: 16px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
		min-height: var(--tap-target-min);
		min-width: 120px;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		/* Better touch feedback */
		-webkit-tap-highlight-color: rgba(147, 51, 234, 0.3);
	}

	.controls-panel button:active:not(:disabled) {
		transform: scale(0.98);
	}

	.controls-panel button:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.spinner {
		width: 16px;
		height: 16px;
		border: 2px solid rgba(255, 255, 255, 0.3);
		border-top-color: white;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.error-message {
		background: #fee;
		border: 2px solid #fcc;
		color: #c33;
		padding: 12px 16px;
		border-radius: 8px;
		margin-bottom: 12px;
		font-size: 14px;
	}

	.status-message {
		background: #e0f2fe;
		border: 2px solid #7dd3fc;
		color: #0369a1;
		padding: 12px 16px;
		border-radius: 8px;
		margin-bottom: 12px;
		font-size: 14px;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: calc(100vh - 300px);
		min-height: 300px;
		color: #64748b;
		text-align: center;
		padding: 40px 20px;
	}

	.empty-icon {
		font-size: 80px;
		margin-bottom: 20px;
		opacity: 0.5;
	}

	.empty-state h3 {
		font-size: 24px;
		color: #334155;
		margin: 0 0 12px 0;
	}

	.empty-state p {
		font-size: 16px;
		max-width: 400px;
		line-height: 1.6;
		margin: 0;
	}

	.visualization {
		height: calc(100vh - 250px);
		min-height: 400px;
		background: #fafafa;
		border-radius: 12px;
		overflow: hidden;
		/* Prevent pull-to-refresh on mobile */
		overscroll-behavior: contain;
	}

	/* Tablet landscape optimization */
	@media (min-width: 768px) and (max-width: 1024px) {
		main {
			padding: 24px;
		}

		.controls-panel {
			gap: 16px;
		}

		.visualization {
			height: calc(100vh - 220px);
		}
	}

	/* Tablet portrait optimization */
	@media (max-width: 767px) {
		main {
			padding: 12px;
		}

		header {
			margin-bottom: 20px;
		}

		.controls-panel {
			flex-direction: column;
			gap: 12px;
		}

		.controls-panel input,
		.controls-panel button {
			width: 100%;
		}

		.visualization {
			height: calc(100vh - 300px);
		}
	}
</style>
