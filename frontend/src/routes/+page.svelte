<script lang="ts">
	import ElkFlow, { type Token } from '$lib/components/ElkFlow.svelte';
	import '../app.css';

	let tokens = $state<Token[]>([]);
	let attentionMatrix = $state<number[][] | undefined>(undefined);
	let isGenerating = $state(false);
	let errorMessage = $state<string | null>(null);
	let statusMessage = $state<string>('');
	let showAttention = $state(false);
	let showSettings = $state(false);

	// Basic parameters
	let prompt = $state('Once upon a time');
	let maxTokens = $state(20);
	let selectionThreshold = $state(0.25);
	let seed = $state(42);

	// Advanced parameters
	let isolate = $state(false);
	let useRetroactivePruning = $state(false);
	let attentionThreshold = $state(0.01);
	let dynamicThreshold = $state(false);
	let finalThreshold = $state(1.0);
	let bezierP1 = $state(0.2);
	let bezierP2 = $state(0.8);

	// Debug indicator lights
	let backendConnected = $state<boolean>(false);

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
		statusMessage = 'Generating...';

		try {
			const response = await fetch('/api/generate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					prompt,
					max_tokens: maxTokens,
					selection_threshold: selectionThreshold,
					seed,
					isolate,
					use_retroactive_removal: useRetroactivePruning,
					attention_threshold: attentionThreshold,
					dynamic_threshold: dynamicThreshold,
					final_threshold: finalThreshold,
					bezier_p1: bezierP1,
					bezier_p2: bezierP2
				})
			});

			if (!response.ok) {
				throw new Error(`API error: ${response.status}`);
			}

			const data = await response.json();

			// Convert API nodes to Token format
			const convertedTokens = data.nodes.map((node: any) => ({
				id: node.id,
				text: node.text.trim(),
				type: node.logical_step === 0 ? 'prompt' : (node.is_parallel ? 'parallel' : 'single'),
				probability: node.probability,
				parent_ids: node.parent_ids,
				step: node.logical_step
			}));

			tokens = convertedTokens;
			attentionMatrix = data.attention_matrix;
			statusMessage = `Generated ${tokens.length} tokens in ${data.generation_time.toFixed(2)}s`;
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Failed to generate';
			statusMessage = '';
		} finally {
			isGenerating = false;
		}
	}

	function resetToDefaults() {
		maxTokens = 20;
		selectionThreshold = 0.25;
		seed = 42;
		isolate = false;
		useRetroactivePruning = false;
		attentionThreshold = 0.01;
		dynamicThreshold = false;
		finalThreshold = 1.0;
		bezierP1 = 0.2;
		bezierP2 = 0.8;
	}
</script>

<main>
	<!-- Mobile-optimized header -->
	<header>
		<h1>🌳 TEMPO</h1>
		<p>Parallel Token Generation</p>
		<div class="status-indicator" class:connected={backendConnected}>
			<span class="dot"></span>
			{backendConnected ? 'Connected' : 'Offline'}
		</div>
	</header>

	<!-- Mobile-first controls -->
	<div class="controls">
		<div class="input-group">
			<label for="prompt">Prompt</label>
			<input
				id="prompt"
				type="text"
				bind:value={prompt}
				placeholder="Enter your prompt..."
				disabled={isGenerating}
			/>
		</div>

		<div class="action-row">
			<button class="btn-primary" onclick={generateTokens} disabled={isGenerating}>
				{#if isGenerating}
					<span class="spinner"></span>
					Generating...
				{:else}
					✨ Generate
				{/if}
			</button>

			<button class="btn-secondary" onclick={() => showSettings = !showSettings}>
				⚙️ Settings
			</button>
		</div>

		<!-- Collapsible Settings Panel -->
		{#if showSettings}
			<div class="settings-panel">
				<div class="settings-header">
					<h3>Generation Settings</h3>
					<button class="btn-text" onclick={resetToDefaults}>Reset</button>
				</div>

				<div class="setting-group">
					<h4>Basic</h4>

					<div class="setting">
						<label>
							<span>Max Tokens</span>
							<span class="value">{maxTokens}</span>
						</label>
						<input type="range" bind:value={maxTokens} min="5" max="100" step="5" />
						<p class="hint">Maximum tokens to generate</p>
					</div>

					<div class="setting">
						<label>
							<span>Selection Threshold</span>
							<span class="value">{selectionThreshold.toFixed(2)}</span>
						</label>
						<input type="range" bind:value={selectionThreshold} min="0.05" max="0.95" step="0.05" />
						<p class="hint">Probability threshold for parallel tokens (lower = more parallel)</p>
					</div>

					<div class="setting">
						<label>
							<span>Random Seed</span>
							<span class="value">{seed}</span>
						</label>
						<input type="number" bind:value={seed} min="0" max="99999" />
						<p class="hint">Seed for reproducible generation</p>
					</div>
				</div>

				<div class="setting-group">
					<h4>Advanced</h4>

					<div class="setting">
						<label class="checkbox-label">
							<input type="checkbox" bind:checked={isolate} />
							<span>Isolate Parallel Tokens</span>
						</label>
						<p class="hint">Prevent parallel tokens from attending to each other</p>
					</div>

					<div class="setting">
						<label class="checkbox-label">
							<input type="checkbox" bind:checked={useRetroactivePruning} />
							<span>Retroactive Pruning</span>
						</label>
						<p class="hint">Prune tokens based on attention patterns</p>
					</div>

					{#if useRetroactivePruning}
						<div class="setting nested">
							<label>
								<span>Attention Threshold</span>
								<span class="value">{attentionThreshold.toFixed(3)}</span>
							</label>
							<input type="range" bind:value={attentionThreshold} min="0.001" max="0.1" step="0.001" />
							<p class="hint">Minimum attention to keep tokens</p>
						</div>
					{/if}

					<div class="setting">
						<label class="checkbox-label">
							<input type="checkbox" bind:checked={dynamicThreshold} />
							<span>Dynamic Threshold</span>
						</label>
						<p class="hint">Gradually increase threshold over time</p>
					</div>

					{#if dynamicThreshold}
						<div class="setting nested">
							<label>
								<span>Final Threshold</span>
								<span class="value">{finalThreshold.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={finalThreshold} min="0.1" max="1.0" step="0.05" />
						</div>

						<div class="setting nested">
							<label>
								<span>Bezier P1</span>
								<span class="value">{bezierP1.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={bezierP1} min="0" max="1" step="0.1" />
						</div>

						<div class="setting nested">
							<label>
								<span>Bezier P2</span>
								<span class="value">{bezierP2.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={bezierP2} min="0" max="1" step="0.1" />
						</div>
					{/if}
				</div>
			</div>
		{/if}

		{#if attentionMatrix && tokens.length > 0}
			<div class="attention-toggle">
				<label>
					<input type="checkbox" bind:checked={showAttention} />
					<span>Show Attention Arches</span>
				</label>
			</div>
		{/if}
	</div>

	<!-- Status messages -->
	{#if errorMessage}
		<div class="message error">⚠️ {errorMessage}</div>
	{/if}

	{#if statusMessage}
		<div class="message success">✓ {statusMessage}</div>
	{/if}

	<!-- Visualization -->
	<div class="visualization">
		{#if tokens.length === 0 && !isGenerating}
			<div class="empty-state">
				<div class="empty-icon">🌳</div>
				<h3>Ready to Generate</h3>
				<p>Enter a prompt and tap Generate to visualize TEMPO's parallel token generation</p>
			</div>
		{:else}
			<ElkFlow {tokens} {attentionMatrix} {showAttention} />
		{/if}
	</div>
</main>

<style>
	main {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		overflow: hidden;
	}

	header {
		padding: 16px 20px;
		background: rgba(255, 255, 255, 0.95);
		backdrop-filter: blur(10px);
		border-bottom: 1px solid rgba(0, 0, 0, 0.1);
		text-align: center;
		position: relative;
	}

	header h1 {
		font-size: 28px;
		font-weight: 800;
		margin: 0;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}

	header p {
		font-size: 13px;
		color: #64748b;
		margin: 4px 0 0 0;
		font-weight: 500;
	}

	.status-indicator {
		position: absolute;
		top: 16px;
		right: 20px;
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: #ef4444;
		font-weight: 600;
	}

	.status-indicator.connected {
		color: #10b981;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: currentColor;
		animation: pulse 2s ease-in-out infinite;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}

	.controls {
		padding: 16px 20px;
		background: rgba(255, 255, 255, 0.95);
		backdrop-filter: blur(10px);
		border-bottom: 1px solid rgba(0, 0, 0, 0.1);
		max-height: 60vh;
		overflow-y: auto;
		-webkit-overflow-scrolling: touch;
	}

	.input-group {
		margin-bottom: 12px;
	}

	.input-group label {
		display: block;
		font-size: 14px;
		font-weight: 600;
		color: #334155;
		margin-bottom: 6px;
	}

	.input-group input {
		width: 100%;
		padding: 14px 16px;
		border: 2px solid #e2e8f0;
		border-radius: 12px;
		font-size: 16px;
		background: white;
		transition: all 0.2s;
	}

	.input-group input:focus {
		outline: none;
		border-color: #667eea;
		box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
	}

	.action-row {
		display: flex;
		gap: 12px;
		margin-bottom: 12px;
	}

	.btn-primary {
		flex: 1;
		padding: 16px 24px;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		border: none;
		border-radius: 12px;
		font-size: 16px;
		font-weight: 700;
		cursor: pointer;
		transition: all 0.2s;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		min-height: 54px;
		-webkit-tap-highlight-color: rgba(102, 126, 234, 0.3);
	}

	.btn-primary:active:not(:disabled) {
		transform: scale(0.98);
	}

	.btn-primary:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.btn-secondary {
		padding: 16px 24px;
		background: white;
		color: #667eea;
		border: 2px solid #667eea;
		border-radius: 12px;
		font-size: 16px;
		font-weight: 700;
		cursor: pointer;
		transition: all 0.2s;
		min-height: 54px;
		-webkit-tap-highlight-color: rgba(102, 126, 234, 0.1);
	}

	.btn-secondary:active {
		transform: scale(0.98);
		background: #f8f9ff;
	}

	.btn-text {
		background: none;
		border: none;
		color: #667eea;
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		padding: 8px 12px;
		border-radius: 6px;
		transition: background 0.2s;
	}

	.btn-text:active {
		background: rgba(102, 126, 234, 0.1);
	}

	.settings-panel {
		margin-top: 12px;
		padding: 16px;
		background: white;
		border-radius: 12px;
		border: 2px solid #e2e8f0;
		animation: slideDown 0.2s ease-out;
	}

	@keyframes slideDown {
		from {
			opacity: 0;
			transform: translateY(-10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.settings-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 16px;
	}

	.settings-header h3 {
		font-size: 18px;
		font-weight: 700;
		color: #1e293b;
		margin: 0;
	}

	.setting-group {
		margin-bottom: 20px;
	}

	.setting-group:last-child {
		margin-bottom: 0;
	}

	.setting-group h4 {
		font-size: 14px;
		font-weight: 700;
		color: #64748b;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		margin: 0 0 12px 0;
		padding-top: 12px;
		border-top: 1px solid #e2e8f0;
	}

	.setting-group:first-child h4 {
		padding-top: 0;
		border-top: none;
	}

	.setting {
		margin-bottom: 16px;
	}

	.setting.nested {
		margin-left: 20px;
		padding-left: 16px;
		border-left: 3px solid #e2e8f0;
	}

	.setting label {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 15px;
		font-weight: 600;
		color: #334155;
		margin-bottom: 8px;
	}

	.setting label .value {
		font-size: 14px;
		color: #667eea;
		font-weight: 700;
		padding: 4px 10px;
		background: rgba(102, 126, 234, 0.1);
		border-radius: 6px;
	}

	.checkbox-label {
		cursor: pointer;
		display: flex !important;
		align-items: center;
		gap: 10px;
	}

	.checkbox-label input[type="checkbox"] {
		width: 24px;
		height: 24px;
		cursor: pointer;
		accent-color: #667eea;
	}

	.setting input[type="range"] {
		width: 100%;
		height: 8px;
		border-radius: 4px;
		background: #e2e8f0;
		outline: none;
		-webkit-appearance: none;
		cursor: pointer;
	}

	.setting input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 24px;
		height: 24px;
		border-radius: 50%;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		cursor: pointer;
		box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
		transition: transform 0.1s;
	}

	.setting input[type="range"]::-webkit-slider-thumb:active {
		transform: scale(1.2);
	}

	.setting input[type="number"] {
		width: 100%;
		padding: 12px;
		border: 2px solid #e2e8f0;
		border-radius: 8px;
		font-size: 16px;
	}

	.hint {
		font-size: 13px;
		color: #94a3b8;
		margin: 4px 0 0 0;
		line-height: 1.4;
	}

	.attention-toggle {
		margin-top: 12px;
		padding: 12px 16px;
		background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
		border: 2px solid #f59e0b;
		border-radius: 12px;
		cursor: pointer;
		-webkit-tap-highlight-color: rgba(245, 158, 11, 0.2);
	}

	.attention-toggle label {
		display: flex;
		align-items: center;
		gap: 10px;
		cursor: pointer;
	}

	.attention-toggle input[type="checkbox"] {
		width: 24px;
		height: 24px;
		cursor: pointer;
		accent-color: #f59e0b;
	}

	.attention-toggle span {
		font-size: 15px;
		font-weight: 600;
		color: #92400e;
	}

	.spinner {
		width: 18px;
		height: 18px;
		border: 3px solid rgba(255, 255, 255, 0.3);
		border-top-color: white;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.message {
		margin: 12px 20px;
		padding: 14px 16px;
		border-radius: 12px;
		font-size: 14px;
		font-weight: 500;
		animation: slideDown 0.2s ease-out;
	}

	.message.error {
		background: #fee;
		color: #dc2626;
		border: 2px solid #fca5a5;
	}

	.message.success {
		background: #f0fdf4;
		color: #15803d;
		border: 2px solid #86efac;
	}

	.visualization {
		flex: 1;
		background: #0f172a;
		overflow: hidden;
		position: relative;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		padding: 40px 20px;
		text-align: center;
	}

	.empty-icon {
		font-size: 64px;
		margin-bottom: 20px;
		opacity: 0.7;
	}

	.empty-state h3 {
		font-size: 22px;
		font-weight: 700;
		color: #e2e8f0;
		margin: 0 0 12px 0;
	}

	.empty-state p {
		font-size: 15px;
		color: #94a3b8;
		line-height: 1.6;
		max-width: 400px;
		margin: 0;
	}

	/* Desktop adjustments */
	@media (min-width: 768px) {
		main {
			flex-direction: row;
		}

		.controls {
			width: 400px;
			max-height: none;
			overflow-y: auto;
			border-right: 1px solid rgba(0, 0, 0, 0.1);
			border-bottom: none;
		}

		.visualization {
			flex: 1;
		}
	}
</style>
