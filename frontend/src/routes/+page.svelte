<script lang="ts">
	import ElkFlow from '$lib/components/ElkFlow.svelte';
	import '../app.css';

	interface Token {
		id: number;
		text: string;
		type: 'prompt' | 'single' | 'parallel';
		probability: number;
		parent_id: number | null;
		step: number;
	}

	let tokens = $state<Token[]>([]);
	let prompt = $state('Once upon a time');
	let isGenerating = $state(false);
	let errorMessage = $state<string | null>(null);
	let statusMessage = $state<string>('');

	async function generateTokens() {
		isGenerating = true;
		errorMessage = null;
		tokens = [];
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
				throw new Error(`API error: ${response.status} ${response.statusText}`);
			}

			const data = await response.json();
			statusMessage = 'Processing results...';

			// Convert API response to token tree format
			const convertedTokens = convertStepsToTokens(data.steps);

			// Stream tokens in one by one for visual effect
			for (let i = 0; i < convertedTokens.length; i++) {
				await new Promise(r => setTimeout(r, 200));
				tokens = convertedTokens.slice(0, i + 1);
			}

			statusMessage = `Generated ${tokens.length} tokens in ${data.generation_time.toFixed(2)}s`;
		} catch (error) {
			console.error('Generation failed:', error);
			errorMessage = error instanceof Error ? error.message : 'Failed to generate tokens';
			statusMessage = '';
		} finally {
			isGenerating = false;
		}
	}

	function convertStepsToTokens(steps: any[]): Token[] {
		const tokens: Token[] = [];
		let tokenId = 0;
		let lastParentIds: number[] = [];  // Track ALL current parallel paths

		for (const step of steps) {
			if (step.type === 'prompt') {
				const id = tokenId++;
				tokens.push({
					id,
					text: step.tokens[0],
					type: 'prompt',
					probability: 1.0,
					parent_id: null,
					step: step.step_number
				});
				lastParentIds = [id];
			} else if (step.type === 'single') {
				const id = tokenId++;
				tokens.push({
					id,
					text: step.tokens[0],
					type: 'single',
					probability: step.probabilities?.[0] ?? 0.9,
					parent_id: lastParentIds[0],  // Primary parent (first path)
					step: step.step_number,
					additionalParents: lastParentIds.slice(1)  // Store other parents for custom rendering
				} as any);
				lastParentIds = [id];  // Converge back to single path
			} else if (step.type === 'parallel') {
				const parentIds: number[] = [];
				for (let i = 0; i < step.tokens.length; i++) {
					const id = tokenId++;
					tokens.push({
						id,
						text: step.tokens[i],
						type: 'parallel',
						probability: step.probabilities?.[i] ?? 0.5,
						parent_id: lastParentIds[0],
						step: step.step_number
					});
					parentIds.push(id);
				}
				lastParentIds = parentIds;  // Diverge into multiple parallel paths
			}
		}

		return tokens;
	}
</script>

<main>
	<header>
		<h1>TEMPO Token Visualizer</h1>
		<p>Interactive D3.js tree showing parallel token generation</p>
	</header>

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
		margin-bottom: 30px;
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
