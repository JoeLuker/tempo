<script lang="ts">
	import ElkFlow, { type Token } from '$lib/components/ElkFlow.svelte';
	import '../app.css';

	let tokens = $state<Token[]>([]);
	let attentionMatrix = $state<number[][] | undefined>(undefined);
	let isGenerating = $state(false);
	let errorMessage = $state<string | null>(null);
	let showAttention = $state(false);

	// Generation parameters
	let prompt = $state('Once upon a time');
	let maxTokens = $state(20);
	let selectionThreshold = $state(0.25);
	let seed = $state(42);
	let isolate = $state(false);
	let useRetroactivePruning = $state(false);
	let attentionThreshold = $state(0.01);
	let dynamicThreshold = $state(false);
	let finalThreshold = $state(1.0);
	let bezierP1 = $state(0.2);
	let bezierP2 = $state(0.8);

	// Stats
	let backendConnected = $state<boolean>(false);
	let generationTime = $state<number>(0);
	let tokenCount = $state<number>(0);
	let tokensPerSecond = $state<number>(0);

	$effect(() => {
		fetch('/api/generate', { method: 'OPTIONS' })
			.then(() => { backendConnected = true; })
			.catch(() => { backendConnected = false; });

		// Keyboard shortcuts
		const handleKeyboard = (e: KeyboardEvent) => {
			if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !isGenerating) {
				e.preventDefault();
				generateTokens();
			}
			if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
				e.preventDefault();
				document.getElementById('prompt')?.focus();
			}
		};

		window.addEventListener('keydown', handleKeyboard);
		return () => window.removeEventListener('keydown', handleKeyboard);
	});

	async function generateTokens() {
		isGenerating = true;
		errorMessage = null;
		tokens = [];

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

			if (!response.ok) throw new Error(`API error: ${response.status}`);

			const data = await response.json();

			tokens = data.nodes.map((node: any) => ({
				id: node.id,
				text: node.text.trim(),
				type: node.logical_step === 0 ? 'prompt' : (node.is_parallel ? 'parallel' : 'single'),
				probability: node.probability,
				parent_ids: node.parent_ids,
				step: node.logical_step
			}));

			attentionMatrix = data.attention_matrix;
			generationTime = data.generation_time;
			tokenCount = tokens.length;
			tokensPerSecond = tokenCount / generationTime;
		} catch (error) {
			errorMessage = error instanceof Error ? error.message : 'Generation failed';
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

<div class="dashboard">
	<!-- Top Navigation Bar -->
	<nav class="navbar">
		<div class="nav-left">
			<h1>🌳 TEMPO</h1>
			<span class="subtitle">Parallel Token Generation Playground</span>
		</div>
		<div class="nav-center">
			<div class="status-badge" class:connected={backendConnected}>
				<span class="dot"></span>
				{backendConnected ? 'Backend Connected' : 'Backend Offline'}
			</div>
		</div>
		<div class="nav-right">
			<div class="shortcuts">
				<kbd>⌘ Enter</kbd> Generate
				<span class="sep">•</span>
				<kbd>⌘ K</kbd> Focus
			</div>
		</div>
	</nav>

	<!-- Main Dashboard Grid -->
	<div class="dashboard-grid">
		<!-- Left Sidebar: Controls & Settings -->
		<aside class="sidebar">
			<section class="control-section">
				<h2>Generation</h2>

				<div class="form-group">
					<label for="prompt">Prompt</label>
					<textarea
						id="prompt"
						bind:value={prompt}
						placeholder="Enter your prompt..."
						rows="3"
						disabled={isGenerating}
					></textarea>
				</div>

				<button class="btn-generate" onclick={generateTokens} disabled={isGenerating}>
					{#if isGenerating}
						<span class="spinner"></span>
						Generating...
					{:else}
						✨ Generate Tokens
					{/if}
				</button>
			</section>

			<section class="control-section">
				<div class="section-header">
					<h2>Parameters</h2>
					<button class="btn-link" onclick={resetToDefaults}>Reset</button>
				</div>

				<div class="param-group">
					<div class="param">
						<label>
							<span>Max Tokens</span>
							<span class="param-value">{maxTokens}</span>
						</label>
						<input type="range" bind:value={maxTokens} min="5" max="100" step="5" />
					</div>

					<div class="param">
						<label>
							<span>Selection Threshold</span>
							<span class="param-value">{selectionThreshold.toFixed(2)}</span>
						</label>
						<input type="range" bind:value={selectionThreshold} min="0.05" max="0.95" step="0.05" />
						<p class="param-hint">Lower = more parallel paths</p>
					</div>

					<div class="param">
						<label>
							<span>Random Seed</span>
							<span class="param-value">{seed}</span>
						</label>
						<input type="number" bind:value={seed} min="0" max="99999" />
					</div>
				</div>
			</section>

			<section class="control-section">
				<h2>Advanced</h2>

				<div class="param-group">
					<label class="checkbox">
						<input type="checkbox" bind:checked={isolate} />
						<span>Isolate Parallel Tokens</span>
					</label>

					<label class="checkbox">
						<input type="checkbox" bind:checked={useRetroactivePruning} />
						<span>Retroactive Pruning</span>
					</label>

					{#if useRetroactivePruning}
						<div class="param nested">
							<label>
								<span>Attention Threshold</span>
								<span class="param-value">{attentionThreshold.toFixed(3)}</span>
							</label>
							<input type="range" bind:value={attentionThreshold} min="0.001" max="0.1" step="0.001" />
						</div>
					{/if}

					<label class="checkbox">
						<input type="checkbox" bind:checked={dynamicThreshold} />
						<span>Dynamic Threshold</span>
					</label>

					{#if dynamicThreshold}
						<div class="param nested">
							<label>
								<span>Final Threshold</span>
								<span class="param-value">{finalThreshold.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={finalThreshold} min="0.1" max="1.0" step="0.05" />
						</div>

						<div class="param nested">
							<label>
								<span>Bezier P1</span>
								<span class="param-value">{bezierP1.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={bezierP1} min="0" max="1" step="0.1" />
						</div>

						<div class="param nested">
							<label>
								<span>Bezier P2</span>
								<span class="param-value">{bezierP2.toFixed(2)}</span>
							</label>
							<input type="range" bind:value={bezierP2} min="0" max="1" step="0.1" />
						</div>
					{/if}
				</div>
			</section>

			{#if attentionMatrix && tokens.length > 0}
				<section class="control-section">
					<label class="checkbox attention-checkbox">
						<input type="checkbox" bind:checked={showAttention} />
						<span>Show Attention Arches</span>
					</label>
				</section>
			{/if}
		</aside>

		<!-- Center: Visualization -->
		<main class="viz-container">
			{#if errorMessage}
				<div class="alert alert-error">
					<strong>Error:</strong> {errorMessage}
				</div>
			{/if}

			<div class="visualization">
				{#if tokens.length === 0 && !isGenerating}
					<div class="empty-viz">
						<div class="empty-icon">🌳</div>
						<h3>Ready to Generate</h3>
						<p>Configure your parameters and press <kbd>⌘ Enter</kbd> or click Generate</p>
					</div>
				{:else}
					<ElkFlow {tokens} {attentionMatrix} {showAttention} />
				{/if}
			</div>
		</main>

		<!-- Right Sidebar: Stats & Info -->
		<aside class="stats-panel">
			<section class="stat-card">
				<h3>Generation Stats</h3>

				{#if tokenCount > 0}
					<div class="stat">
						<span class="stat-label">Tokens Generated</span>
						<span class="stat-value">{tokenCount}</span>
					</div>
					<div class="stat">
						<span class="stat-label">Generation Time</span>
						<span class="stat-value">{generationTime.toFixed(2)}s</span>
					</div>
					<div class="stat">
						<span class="stat-label">Tokens/Second</span>
						<span class="stat-value">{tokensPerSecond.toFixed(1)}</span>
					</div>
					<div class="stat">
						<span class="stat-label">Threshold Used</span>
						<span class="stat-value">{selectionThreshold.toFixed(2)}</span>
					</div>
				{:else}
					<p class="stat-empty">No generation yet</p>
				{/if}
			</section>

			<section class="stat-card">
				<h3>About TEMPO</h3>
				<p class="info-text">
					TEMPO explores parallel token generation by processing multiple possibilities
					simultaneously at each step using modified RoPE embeddings.
				</p>
				<div class="legend">
					<div class="legend-item">
						<span class="legend-color prompt"></span>
						<span>Prompt Token</span>
					</div>
					<div class="legend-item">
						<span class="legend-color single"></span>
						<span>Single Path</span>
					</div>
					<div class="legend-item">
						<span class="legend-color parallel"></span>
						<span>Parallel Paths</span>
					</div>
				</div>
			</section>
		</aside>
	</div>
</div>

<style>
	:global(body) {
		margin: 0;
		overflow: hidden;
	}

	.dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: #f8fafc;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	}

	/* Top Navigation */
	.navbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 16px 32px;
		background: white;
		border-bottom: 1px solid #e2e8f0;
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
		z-index: 100;
	}

	.nav-left {
		display: flex;
		align-items: baseline;
		gap: 16px;
	}

	.nav-left h1 {
		margin: 0;
		font-size: 24px;
		font-weight: 800;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}

	.subtitle {
		font-size: 14px;
		color: #64748b;
		font-weight: 500;
	}

	.nav-center {
		flex: 1;
		display: flex;
		justify-content: center;
	}

	.status-badge {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 8px 16px;
		background: #fee;
		border: 1px solid #fca5a5;
		border-radius: 8px;
		font-size: 13px;
		font-weight: 600;
		color: #dc2626;
	}

	.status-badge.connected {
		background: #f0fdf4;
		border-color: #86efac;
		color: #15803d;
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
		50% { opacity: 0.4; }
	}

	.shortcuts {
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: 13px;
		color: #64748b;
	}

	.shortcuts kbd {
		background: #f1f5f9;
		border: 1px solid #cbd5e1;
		border-radius: 4px;
		padding: 4px 8px;
		font-size: 11px;
		font-family: 'SF Mono', monospace;
		color: #475569;
	}

	.sep {
		color: #cbd5e1;
	}

	/* Dashboard Grid */
	.dashboard-grid {
		display: grid;
		grid-template-columns: 340px 1fr 280px;
		gap: 0;
		flex: 1;
		overflow: hidden;
	}

	/* Sidebar */
	.sidebar {
		background: white;
		border-right: 1px solid #e2e8f0;
		overflow-y: auto;
		padding: 24px;
	}

	.control-section {
		margin-bottom: 32px;
	}

	.control-section:last-child {
		margin-bottom: 0;
	}

	.section-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 16px;
	}

	.control-section h2 {
		font-size: 16px;
		font-weight: 700;
		color: #1e293b;
		margin: 0 0 16px 0;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.form-group {
		margin-bottom: 16px;
	}

	.form-group label {
		display: block;
		font-size: 13px;
		font-weight: 600;
		color: #475569;
		margin-bottom: 8px;
	}

	.form-group textarea {
		width: 100%;
		padding: 12px;
		border: 2px solid #e2e8f0;
		border-radius: 8px;
		font-size: 14px;
		font-family: inherit;
		resize: vertical;
		transition: border-color 0.2s;
	}

	.form-group textarea:focus {
		outline: none;
		border-color: #667eea;
	}

	.btn-generate {
		width: 100%;
		padding: 16px;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		border: none;
		border-radius: 8px;
		font-size: 16px;
		font-weight: 700;
		cursor: pointer;
		transition: transform 0.1s;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
	}

	.btn-generate:hover:not(:disabled) {
		transform: translateY(-1px);
		box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
	}

	.btn-generate:active:not(:disabled) {
		transform: translateY(0);
	}

	.btn-generate:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.btn-link {
		background: none;
		border: none;
		color: #667eea;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		padding: 4px 8px;
		border-radius: 4px;
	}

	.btn-link:hover {
		background: #f1f5f9;
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

	/* Parameters */
	.param-group {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.param {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.param.nested {
		margin-left: 24px;
		padding-left: 16px;
		border-left: 3px solid #e2e8f0;
	}

	.param label {
		display: flex;
		justify-content: space-between;
		align-items: center;
		font-size: 13px;
		font-weight: 600;
		color: #475569;
	}

	.param-value {
		background: #f1f5f9;
		padding: 4px 10px;
		border-radius: 6px;
		color: #667eea;
		font-weight: 700;
		font-size: 12px;
	}

	.param input[type="range"] {
		width: 100%;
		height: 6px;
		border-radius: 3px;
		background: #e2e8f0;
		outline: none;
		-webkit-appearance: none;
		cursor: pointer;
	}

	.param input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		cursor: pointer;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
	}

	.param input[type="number"] {
		padding: 8px 12px;
		border: 2px solid #e2e8f0;
		border-radius: 6px;
		font-size: 14px;
	}

	.param-hint {
		font-size: 12px;
		color: #94a3b8;
		margin: 0;
	}

	.checkbox {
		display: flex;
		align-items: center;
		gap: 10px;
		cursor: pointer;
		padding: 10px;
		border-radius: 6px;
		transition: background 0.2s;
	}

	.checkbox:hover {
		background: #f8fafc;
	}

	.checkbox input[type="checkbox"] {
		width: 18px;
		height: 18px;
		cursor: pointer;
		accent-color: #667eea;
	}

	.checkbox span {
		font-size: 14px;
		font-weight: 500;
		color: #334155;
	}

	.attention-checkbox {
		background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
		border: 2px solid #f59e0b;
		padding: 12px;
	}

	/* Visualization */
	.viz-container {
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: #f8fafc;
	}

	.alert {
		margin: 16px;
		padding: 12px 16px;
		border-radius: 8px;
		font-size: 14px;
	}

	.alert-error {
		background: #fee;
		border: 2px solid #fca5a5;
		color: #dc2626;
	}

	.visualization {
		flex: 1;
		overflow: hidden;
		background: #0f172a;
		margin: 16px;
		border-radius: 12px;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
	}

	.empty-viz {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: #e2e8f0;
		text-align: center;
	}

	.empty-icon {
		font-size: 80px;
		margin-bottom: 24px;
		opacity: 0.6;
	}

	.empty-viz h3 {
		font-size: 24px;
		margin: 0 0 12px 0;
		font-weight: 700;
	}

	.empty-viz p {
		font-size: 15px;
		color: #94a3b8;
		margin: 0;
	}

	.empty-viz kbd {
		background: #1e293b;
		border: 1px solid #334155;
		padding: 4px 8px;
		border-radius: 4px;
		font-family: 'SF Mono', monospace;
		font-size: 13px;
	}

	/* Stats Panel */
	.stats-panel {
		background: white;
		border-left: 1px solid #e2e8f0;
		overflow-y: auto;
		padding: 24px;
	}

	.stat-card {
		background: #f8fafc;
		border: 1px solid #e2e8f0;
		border-radius: 12px;
		padding: 20px;
		margin-bottom: 20px;
	}

	.stat-card h3 {
		font-size: 14px;
		font-weight: 700;
		color: #1e293b;
		margin: 0 0 16px 0;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.stat {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 12px 0;
		border-bottom: 1px solid #e2e8f0;
	}

	.stat:last-child {
		border-bottom: none;
	}

	.stat-label {
		font-size: 13px;
		color: #64748b;
		font-weight: 500;
	}

	.stat-value {
		font-size: 16px;
		font-weight: 700;
		color: #667eea;
	}

	.stat-empty {
		font-size: 13px;
		color: #94a3b8;
		text-align: center;
		padding: 20px 0;
		margin: 0;
	}

	.info-text {
		font-size: 13px;
		line-height: 1.6;
		color: #64748b;
		margin: 0 0 16px 0;
	}

	.legend {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 13px;
		color: #475569;
	}

	.legend-color {
		width: 16px;
		height: 16px;
		border-radius: 4px;
		border: 2px solid #1e293b;
	}

	.legend-color.prompt {
		background: #7c3aed;
	}

	.legend-color.single {
		background: #06b6d4;
	}

	.legend-color.parallel {
		background: #f97316;
	}
</style>
