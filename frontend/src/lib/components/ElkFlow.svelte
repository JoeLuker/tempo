<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import ELK from 'elkjs/lib/elk.bundled.js';

	export interface Token {
		id: string;
		text: string;
		type: 'prompt' | 'single' | 'parallel';
		probability: number;
		parent_id: string | null;
		step: number;
	}

	interface Props {
		tokens: Token[];
	}

	let { tokens }: Props = $props();

	let containerRef: HTMLDivElement | null = $state(null);
	let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = $state(null);
	let g: d3.Selection<SVGGElement, unknown, null, undefined> | null = $state(null);
	let width = $state(800);
	let height = $state(600);
	let mounted = $state(false);

	const elk = new ELK();

	// Reactive effect for visualization updates
	$effect(() => {
		if (mounted && tokens.length > 0) {
			updateVisualization();
		}
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

		// Setup zoom behavior
		const zoom = d3
			.zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.1, 4])
			.on('zoom', (event) => {
				g?.attr('transform', event.transform);
			});

		svg.call(zoom);

		// Update dimensions
		updateDimensions();
		window.addEventListener('resize', updateDimensions);

		mounted = true;

		return () => {
			window.removeEventListener('resize', updateDimensions);
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

		const nodes = tokens.map((token) => ({
			id: token.id,
			width: 80,
			height: 40
		}));

		// Build edges - simply use parent relationships directly
		const edges: any[] = [];
		tokens.forEach((token) => {
			if (token.parent_id) {
				edges.push({
					id: `${token.parent_id}_to_${token.id}`,
					sources: [token.parent_id],
					targets: [token.id]
				});
			}
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

			nodeGroup
				.append('rect')
				.attr('x', node.x)
				.attr('y', node.y)
				.attr('width', node.width)
				.attr('height', node.height)
				.attr('rx', 6)
				.attr('fill', nodeColor)
				.attr('stroke', '#1e293b')
				.attr('stroke-width', 2)
				.style('cursor', 'pointer')
				.on('mouseenter', function () {
					d3.select(this).attr('stroke', '#f59e0b').attr('stroke-width', 3);
				})
				.on('mouseleave', function () {
					d3.select(this).attr('stroke', '#1e293b').attr('stroke-width', 2);
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

		// Center the graph
		const graphWidth = layout.width || 0;
		const graphHeight = layout.height || 0;
		const offsetX = (width - graphWidth) / 2;
		const offsetY = (height - graphHeight) / 2;
		g.attr('transform', `translate(${Math.max(20, offsetX)}, ${Math.max(20, offsetY)})`);
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

		svg
			.transition()
			.duration(750)
			.call(
				d3.zoom<SVGSVGElement, unknown>().transform as any,
				d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
			);
	}

	export function zoomIn() {
		if (!svg) return;
		svg.transition().duration(300).call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.3);
	}

	export function zoomOut() {
		if (!svg) return;
		svg.transition().duration(300).call(d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 0.7);
	}
</script>

<div bind:this={containerRef} class="elk-flow-container"></div>

<style>
	.elk-flow-container {
		width: 100%;
		height: 100%;
		overflow: hidden;
		border-radius: 8px;
	}

	:global(.node) {
		transition: all 0.2s ease;
	}
</style>
