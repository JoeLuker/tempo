<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import ELK from 'elkjs/lib/elk.bundled.js';

	export interface Token {
		id: string;
		text: string;
		type: 'prompt' | 'single' | 'parallel';
		probability: number;
		parent_ids: string[];  // Changed to array to support convergence
		step: number;
	}

	interface Props {
		tokens: Token[];
		attentionMatrix?: number[][];  // [seq_len, seq_len] averaged across layers/heads
		showAttention?: boolean;
	}

	let { tokens, attentionMatrix, showAttention = false }: Props = $props();

	let containerRef: HTMLDivElement | null = $state(null);
	let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = $state(null);
	let g: d3.Selection<SVGGElement, unknown, null, undefined> | null = $state(null);
	let width = $state(800);
	let height = $state(600);
	let mounted = $state(false);
	let selectedTokenId = $state<string | null>(null);  // Track clicked token for attention filtering

	const elk = new ELK();

	// Best practice: Use $derived for computed values
	const hasTokens = $derived(tokens.length > 0);
	const hasAttention = $derived(attentionMatrix && attentionMatrix.length > 0);
	const shouldShowAttention = $derived(showAttention && hasAttention);

	// Reactive effect for visualization updates
	// Best practice: Explicitly track dependencies synchronously
	$effect(() => {
		// Use derived values for cleaner dependency tracking
		if (mounted && hasTokens) {
			updateVisualization();
		}
		// Track these to trigger re-renders when they change
		void shouldShowAttention;
		void selectedTokenId;
	});

	onMount(() => {
		if (!containerRef) return;

		// Create SVG
		svg = d3
			.select(containerRef)
			.append('svg')
			.attr('width', '100%')
			.attr('height', '100%')
			.style('background', '#0f172a');

		// Create main group for zoom/pan
		g = svg.append('g');

		// Setup zoom behavior with mobile-friendly settings
		const zoom = d3
			.zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.1, 4])
			// Mobile-optimized: smoother pinch-zoom
			.wheelDelta((event) => {
				// Reduce zoom sensitivity for better control
				return -event.deltaY * (event.deltaMode === 1 ? 0.05 : event.deltaMode ? 1 : 0.002);
			})
			.on('zoom', (event) => {
				g?.attr('transform', event.transform);
			});

		svg.call(zoom);

		// Update dimensions
		updateDimensions();
		window.addEventListener('resize', updateDimensions);

		mounted = true;

		// Best practice: Cleanup function removes event listeners
		return () => {
			window.removeEventListener('resize', updateDimensions);
			// Clear SVG on unmount to prevent memory leaks
			if (svg) {
				svg.selectAll('*').remove();
				svg.remove();
			}
		};
	});

	function updateDimensions() {
		if (!containerRef) return;
		const rect = containerRef.getBoundingClientRect();
		width = Math.max(300, rect.width);
		height = Math.max(300, rect.height);
	}

	async function updateVisualization() {
		if (!g || !svg || tokens.length === 0) {
			console.log('[ElkFlow] updateVisualization skipped:', { g: !!g, svg: !!svg, tokensLength: tokens.length });
			return;
		}

		console.log('[ElkFlow] Starting visualization update with', tokens.length, 'tokens');

		// Clear previous visualization
		g.selectAll('*').remove();

		// Convert tokens to ELK graph
		const { graph: elkGraph, tokenMap } = await buildElkGraph(tokens);
		console.log('[ElkFlow] Built Elk graph:', JSON.stringify(elkGraph, null, 2));

		// Layout with ELK
		try {
			const layout = await elk.layout(elkGraph);
			console.log('[ElkFlow] Elk layout complete:', layout);

			// Render the layout
			renderLayout(layout, tokenMap);
			console.log('[ElkFlow] Rendering complete');

			// Reset zoom to fit content
			resetZoom();
			console.log('[ElkFlow] Zoom reset complete');
		} catch (error) {
			console.error('[ElkFlow] Elk layout error:', error);
			throw error;
		}
	}

	async function buildElkGraph(tokens: Token[]) {
		// Group tokens by step
		const stepMap = new Map<number, Token[]>();
		tokens.forEach((token) => {
			if (!stepMap.has(token.step)) {
				stepMap.set(token.step, []);
			}
			stepMap.get(token.step)!.push(token);
		});

		const sortedSteps = Array.from(stepMap.keys()).sort((a, b) => a - b);

		// Build nodes - store token lookup map
		const tokenMap = new Map<string, Token>();
		tokens.forEach(token => tokenMap.set(token.id, token));

		// Mobile-optimized: larger touch targets
		const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
		const nodeWidth = isMobile ? 100 : 80;
		const nodeHeight = isMobile ? 50 : 40;

		const nodes = tokens.map((token) => ({
			id: token.id,
			width: nodeWidth,
			height: nodeHeight
		}));

		// Build edges - handle multiple parents for convergence (many-to-one)
		const edges: any[] = [];
		tokens.forEach((token) => {
			token.parent_ids.forEach((parent_id) => {
				edges.push({
					id: `${parent_id}_to_${token.id}`,
					sources: [parent_id],
					targets: [token.id]
				});
			});
		});

		return {
			graph: {
				id: 'root',
				layoutOptions: {
					'elk.algorithm': 'layered',
					'elk.direction': 'RIGHT'
				},
				children: nodes,
				edges: edges
			},
			tokenMap
		};
	}

	function renderLayout(layout: any, tokenMap: Map<string, Token>) {
		if (!g) return;

		// Add arrowhead marker
		svg!
			.append('defs')
			.append('marker')
			.attr('id', 'arrowhead')
			.attr('viewBox', '0 0 10 10')
			.attr('refX', 9)
			.attr('refY', 5)
			.attr('markerWidth', 6)
			.attr('markerHeight', 6)
			.attr('orient', 'auto')
			.append('path')
			.attr('d', 'M 0 0 L 10 5 L 0 10 z')
			.attr('fill', '#94a3b8');

		// Render edges
		layout.edges?.forEach((edge: any) => {
			const sections = edge.sections || [];
			sections.forEach((section: any) => {
				const points = [
					{ x: section.startPoint.x, y: section.startPoint.y },
					...(section.bendPoints || []),
					{ x: section.endPoint.x, y: section.endPoint.y }
				];

				const line = d3
					.line<{ x: number; y: number }>()
					.x((d) => d.x)
					.y((d) => d.y)
					.curve(d3.curveBasis);

				g!.append('path')
					.attr('d', line(points))
					.attr('fill', 'none')
					.attr('stroke', '#475569')
					.attr('stroke-width', 2)
					.attr('marker-end', 'url(#arrowhead)')
					.attr('opacity', 0.6);
			});
		});

		// Render nodes
		layout.children?.forEach((node: any) => {
			const token = tokenMap.get(node.id)!;

			const nodeGroup = g!.append('g').attr('class', 'node');

			// Node rectangle
			const nodeColor =
				token.type === 'prompt'
					? '#7c3aed'
					: token.type === 'parallel'
						? '#f97316'
						: '#06b6d4';

			// Check if this token is selected
			const isSelected = selectedTokenId === token.id;

			nodeGroup
				.append('rect')
				.attr('x', node.x)
				.attr('y', node.y)
				.attr('width', node.width)
				.attr('height', node.height)
				.attr('rx', 6)
				.attr('fill', nodeColor)
				.attr('stroke', isSelected ? '#fbbf24' : '#1e293b')
				.attr('stroke-width', isSelected ? 4 : 2)
				.style('cursor', 'pointer')
				.style('transition', 'stroke 0.15s ease, stroke-width 0.15s ease')
				.on('click', () => {
					// Toggle selection
					selectedTokenId = selectedTokenId === token.id ? null : token.id;
				})
				.on('mouseenter', function () {
					if (!isSelected) {
						// Use transition for smooth hover effect
						d3.select(this)
							.transition()
							.duration(150)
							.ease(d3.easeCubicOut)
							.attr('stroke', '#f59e0b')
							.attr('stroke-width', 3);
					}
				})
				.on('mouseleave', function () {
					if (!isSelected) {
						// Use transition for smooth hover exit
						d3.select(this)
							.transition()
							.duration(150)
							.ease(d3.easeCubicOut)
							.attr('stroke', '#1e293b')
							.attr('stroke-width', 2);
					}
				});

			// Node text
			nodeGroup
				.append('text')
				.attr('x', node.x + node.width / 2)
				.attr('y', node.y + node.height / 2 + 5)
				.attr('text-anchor', 'middle')
				.attr('fill', '#fff')
				.attr('font-size', 12)
				.attr('font-weight', 600)
				.text(token.text);

			// Probability badge for parallel tokens
			if (token.type === 'parallel' && token.probability) {
				nodeGroup
					.append('text')
					.attr('x', node.x + node.width / 2)
					.attr('y', node.y + node.height / 2 - 8)
					.attr('text-anchor', 'middle')
					.attr('fill', '#fff')
					.attr('font-size', 10)
					.attr('opacity', 0.9)
					.text(`${(token.probability * 100).toFixed(0)}%`);
			}
		});

		// Render attention arches if enabled (use derived value)
		if (shouldShowAttention && attentionMatrix) {
			renderAttentionArches(layout, tokenMap, attentionMatrix, selectedTokenId);
		}

		// Center the graph
		const graphWidth = layout.width || 0;
		const graphHeight = layout.height || 0;
		const offsetX = (width - graphWidth) / 2;
		const offsetY = (height - graphHeight) / 2;
		g.attr('transform', `translate(${Math.max(20, offsetX)}, ${Math.max(20, offsetY)})`);
	}

	function renderAttentionArches(layout: any, tokenMap: Map<string, Token>, attentionMatrix: number[][], selectedTokenId: string | null) {
		if (!g) return;

		console.log('[ElkFlow] Rendering attention arches for matrix:', attentionMatrix.length, 'selected:', selectedTokenId);

		// Create a map from token IDs to their index in the sequence
		const tokenIdToIndex = new Map<string, number>();
		const sortedTokens = Array.from(tokenMap.values()).sort((a, b) => a.step - b.step);
		sortedTokens.forEach((token, idx) => {
			tokenIdToIndex.set(token.id, idx);
		});

		// Get selected token index if any
		const selectedIdx = selectedTokenId ? tokenIdToIndex.get(selectedTokenId) : undefined;

		// Create a map from layout node IDs to their positions
		const nodePositions = new Map<string, { x: number; y: number; width: number; height: number }>();
		layout.children?.forEach((node: any) => {
			nodePositions.set(node.id, {
				x: node.x + node.width / 2,  // Center X
				y: node.y + node.height / 2,  // Center Y
				width: node.width,
				height: node.height
			});
		});

		// Filter attention weights by threshold (only show significant attention)
		const attentionThreshold = 0.05;  // Only show attention > 5%

		// Draw attention arches
		const archGroup = g.append('g').attr('class', 'attention-arches').lower(); // Draw behind nodes

		sortedTokens.forEach((targetToken, targetIdx) => {
			const targetPos = nodePositions.get(targetToken.id);
			if (!targetPos || targetIdx >= attentionMatrix.length) return;

			const attentionWeights = attentionMatrix[targetIdx];

			sortedTokens.forEach((sourceToken, sourceIdx) => {
				if (sourceIdx >= attentionWeights.length) return;

				const weight = attentionWeights[sourceIdx];
				if (weight < attentionThreshold) return;  // Skip weak attention
				if (sourceIdx === targetIdx) return;  // Skip self-attention

				// Filter by selected token if any
				if (selectedIdx !== undefined) {
					// Only show arches connected to the selected token
					if (sourceIdx !== selectedIdx && targetIdx !== selectedIdx) return;
				}

				const sourcePos = nodePositions.get(sourceToken.id);
				if (!sourcePos) return;

				// Draw arched path from source to target
				const dx = targetPos.x - sourcePos.x;
				const dy = targetPos.y - sourcePos.y;
				const distance = Math.sqrt(dx * dx + dy * dy);

				// Create control points for cubic Bézier arch (smoother than quadratic)
				const midX = (sourcePos.x + targetPos.x) / 2;
				const midY = (sourcePos.y + targetPos.y) / 2;

				// Scale arch height AGGRESSIVELY to prevent overlaps
				const horizontalDist = Math.abs(dx);
				const stepDiff = Math.abs(targetIdx - sourceIdx);
				const baseHeight = Math.max(80, horizontalDist * 1.2 + stepDiff * 50);
				const archHeight = Math.min(600, baseHeight);

				// Calculate perpendicular offset for arch
				const perpX = -(dy / distance) * archHeight;
				const perpY = (dx / distance) * archHeight;

				// Two control points for cubic Bézier (smoother curve)
				const cp1X = sourcePos.x + dx * 0.25 + perpX;
				const cp1Y = sourcePos.y + dy * 0.25 + perpY;
				const cp2X = sourcePos.x + dx * 0.75 + perpX;
				const cp2Y = sourcePos.y + dy * 0.75 + perpY;

				// Create cubic Bézier curve path for smoother arches
				const pathData = `M ${sourcePos.x},${sourcePos.y} C ${cp1X},${cp1Y} ${cp2X},${cp2Y} ${targetPos.x},${targetPos.y}`;

				// Determine color based on selection
				const isConnectedToSelection = selectedIdx !== undefined && (sourceIdx === selectedIdx || targetIdx === selectedIdx);
				const archColor = isConnectedToSelection ? '#22c55e' : '#fbbf24';  // Green if selected, amber otherwise

				// Draw the arch with opacity based on attention weight
				archGroup
					.append('path')
					.attr('d', pathData)
					.attr('fill', 'none')
					.attr('stroke', archColor)
					.attr('stroke-width', Math.max(1, weight * 5))  // Thicker lines
					.attr('opacity', Math.min(0.9, weight * 2))  // More visible
					.style('pointer-events', 'none');  // Don't interfere with node interactions

				// Optional: Add weight label for very strong attention only
				if (weight > 0.3) {
					// Place label at midpoint of cubic Bézier curve
					const labelX = (sourcePos.x + cp1X + cp2X + targetPos.x) / 4;
					const labelY = (sourcePos.y + cp1Y + cp2Y + targetPos.y) / 4;

					archGroup
						.append('text')
						.attr('x', labelX)
						.attr('y', labelY)
						.attr('text-anchor', 'middle')
						.attr('fill', archColor)
						.attr('font-size', 10)
						.attr('font-weight', 600)
						.attr('opacity', 0.8)
						.style('pointer-events', 'none')
						.text(`${(weight * 100).toFixed(0)}%`);
				}
			});
		});

		console.log('[ElkFlow] Attention arches rendered');
	}

	function resetZoom() {
		if (!svg || !g) return;

		const bounds = (g.node() as SVGGraphicsElement).getBBox();
		const fullWidth = bounds.width;
		const fullHeight = bounds.height;
		const midX = bounds.x + fullWidth / 2;
		const midY = bounds.y + fullHeight / 2;

		if (fullWidth === 0 || fullHeight === 0) return;

		const scale = Math.min(
			0.95,
			Math.min(width / fullWidth, height / fullHeight)
		);
		const translate = [width / 2 - scale * midX, height / 2 - scale * midY];

		// Best practice: Use css() for better performance (runs off main thread)
		svg
			.transition()
			.duration(750)
			.ease(d3.easeCubicInOut)
			.call(
				d3.zoom<SVGSVGElement, unknown>().transform as any,
				d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
			);
	}

	export function zoomIn() {
		if (!svg) return;
		svg
			.transition()
			.duration(300)
			.ease(d3.easeCubicOut)
			.call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.3);
	}

	export function zoomOut() {
		if (!svg) return;
		svg
			.transition()
			.duration(300)
			.ease(d3.easeCubicOut)
			.call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 0.7);
	}
</script>

<div bind:this={containerRef} class="elk-flow-container">
	<!-- Desktop zoom controls -->
	<div class="zoom-controls">
		<button onclick={zoomIn} title="Zoom In (or scroll)">+</button>
		<button onclick={resetZoom} title="Reset View">⟲</button>
		<button onclick={zoomOut} title="Zoom Out (or scroll)">−</button>
	</div>
</div>

<style>
	.elk-flow-container {
		width: 100%;
		height: 100%;
		overflow: hidden;
		border-radius: 8px;
		position: relative;
		/* Mobile touch optimizations */
		touch-action: none;
		-webkit-user-select: none;
		user-select: none;
	}

	.zoom-controls {
		position: absolute;
		bottom: 20px;
		right: 20px;
		display: none;
		flex-direction: column;
		gap: 8px;
		z-index: 10;
	}

	.zoom-controls button {
		width: 44px;
		height: 44px;
		background: rgba(255, 255, 255, 0.95);
		backdrop-filter: blur(10px);
		border: 2px solid rgba(102, 126, 234, 0.2);
		border-radius: 8px;
		font-size: 20px;
		font-weight: 700;
		color: #667eea;
		cursor: pointer;
		transition: all 0.2s;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
	}

	.zoom-controls button:hover {
		background: #667eea;
		color: white;
		border-color: #667eea;
		transform: scale(1.05);
	}

	.zoom-controls button:active {
		transform: scale(0.95);
	}

	:global(.node) {
		transition: all 0.2s ease;
	}

	/* Mobile-specific node sizing */
	@media (max-width: 767px) {
		:global(.node rect) {
			/* Larger touch targets on mobile */
			rx: 8px !important;
		}

		:global(.node text) {
			/* Slightly larger text on mobile */
			font-size: 14px !important;
		}
	}

	/* Desktop zoom controls */
	@media (min-width: 768px) {
		.zoom-controls {
			display: flex;
		}

		.zoom-controls button {
			width: 48px;
			height: 48px;
			font-size: 22px;
		}
	}
</style>
