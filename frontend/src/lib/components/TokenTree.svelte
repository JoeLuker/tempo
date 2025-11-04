<script lang="ts">
	import * as d3 from 'd3';
	import { onMount } from 'svelte';

	interface Token {
		id: number;
		text: string;
		type: 'prompt' | 'single' | 'parallel';
		probability: number;
		parent_id: number | null;
		step: number;
	}

	interface Props {
		tokens: Token[];
	}

	let { tokens }: Props = $props();

	let containerRef = $state<HTMLDivElement>();
	let svgRef = $state<SVGSVGElement>();
	let width = $state(800);
	let height = $state(600);
	let selectedNode = $state<Token | null>(null);
	let hoveredNode = $state<Token | null>(null);
	let mounted = $state(false);

	let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | undefined;
	let g: d3.Selection<SVGGElement, unknown, null, undefined> | undefined;
	let zoom: d3.ZoomBehavior<SVGSVGElement, unknown> | undefined;

	let hierarchyData = $derived.by(() => {
		if (!tokens.length) return null;
		return buildHierarchy(tokens);
	});

	function updateDimensions() {
		if (!containerRef) return;
		const rect = containerRef.getBoundingClientRect();
		width = Math.max(300, rect.width);
		height = Math.max(300, rect.height);
	}

	onMount(() => {
		if (!svgRef) return;

		updateDimensions();
		window.addEventListener('resize', updateDimensions);

		svg = d3.select(svgRef);
		g = svg.append('g').attr('class', 'main-group');
		zoom = d3.zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.1, 4])
			.on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
				if (g) g.attr('transform', event.transform.toString());
			});
		svg.call(zoom);

		mounted = true;

		return () => {
			window.removeEventListener('resize', updateDimensions);
		};
	});

	$effect(() => {
		if (!mounted || !g || !hierarchyData) return;
		updateVisualization(hierarchyData);
	});
	
	function buildHierarchy(flatTokens: Token[]) {
		const map = new Map<number, any>();
		const root = { id: -1, text: 'START', type: 'prompt' as const, children: [] };
		map.set(-1, root);
		flatTokens.forEach(token => { map.set(token.id, { ...token, children: [] }); });
		flatTokens.forEach(token => {
			const parentId = token.parent_id ?? -1;
			const parent = map.get(parentId);
			const child = map.get(token.id);
			if (parent && child) parent.children.push(child);
		});
		return d3.hierarchy(root);
	}
	
	function updateVisualization(hierarchy: d3.HierarchyNode<any>) {
		if (!g) return;

		// Use responsive dimensions with padding
		const padding = 50;
		const treeLayout = d3.tree<any>()
			.size([width - padding * 2, height - padding * 2])
			.separation((a, b) => (a.parent === b.parent ? 1 : 1.5));

		const treeData = treeLayout(hierarchy);

		// Center the tree
		g.attr('transform', `translate(${padding}, ${padding})`);

		// Links
		const links = g.selectAll<SVGPathElement, d3.HierarchyLink<any>>('.link')
			.data(treeData.links(), (d) => `${d.source.data.id}-${d.target.data.id}`);

		links.exit().remove();

		links.enter()
			.append('path')
			.attr('class', 'link')
			.merge(links)
			.attr('fill', 'none')
			.attr('stroke', (d) => d.target.data.type === 'parallel' ? '#f97316' : '#94a3b8')
			.attr('stroke-width', 2)
			.attr('d', (d) => {
				const sourceX = d.source.x;
				const sourceY = d.source.y;
				const targetX = d.target.x;
				const targetY = d.target.y;
				const midY = (sourceY + targetY) / 2;
				return `M${sourceX},${sourceY}C${sourceX},${midY} ${targetX},${midY} ${targetX},${targetY}`;
			});

		// Nodes
		const nodes = g.selectAll<SVGGElement, d3.HierarchyNode<any>>('.node')
			.data(treeData.descendants(), (d) => d.data.id);

		nodes.exit().remove();

		const nodeEnter = nodes.enter()
			.append('g')
			.attr('class', 'node')
			.on('click', (_, d) => {
				selectedNode = selectedNode?.id === d.data.id ? null : d.data;
			});

		nodeEnter.append('circle');
		nodeEnter.append('text');

		const nodeUpdate = nodeEnter.merge(nodes);

		nodeUpdate
			.attr('transform', (d) => `translate(${d.x},${d.y})`);

		nodeUpdate.select('circle')
			.attr('r', (d) => d.data.type === 'parallel' ? 12 : 8)
			.attr('fill', (d) => {
				if (d.data.type === 'prompt') return '#9333ea';
				if (d.data.type === 'single') return '#22c55e';
				return '#f97316';
			})
			.attr('stroke', '#fff')
			.attr('stroke-width', 3);

		nodeUpdate.select('text')
			.attr('dy', -20)
			.attr('text-anchor', 'middle')
			.attr('font-size', 14)
			.attr('fill', '#1e293b')
			.text((d) => d.data.text);
	}

	function resetZoom() {
		if (!svg || !zoom) return;
		svg.transition().duration(300).call(zoom.transform as any, d3.zoomIdentity);
	}

	function zoomIn() {
		if (!svg || !zoom) return;
		svg.transition().duration(300).call(zoom.scaleBy as any, 1.3);
	}

	function zoomOut() {
		if (!svg || !zoom) return;
		svg.transition().duration(300).call(zoom.scaleBy as any, 0.7);
	}
</script>

<div class="token-tree" bind:this={containerRef}>
	<div class="controls">
		<button onclick={zoomIn}>Zoom In</button>
		<button onclick={zoomOut}>Zoom Out</button>
		<button onclick={resetZoom}>Reset</button>
	</div>
	<svg bind:this={svgRef} {width} {height} viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" class="tree-svg"></svg>
	{#if selectedNode}
		<div class="info-panel">
			<h3>Selected Token</h3>
			<p><strong>Text:</strong> {selectedNode.text}</p>
			<p><strong>Type:</strong> {selectedNode.type}</p>
			<p><strong>Probability:</strong> {(selectedNode.probability * 100).toFixed(1)}%</p>
			<button onclick={() => selectedNode = null}>Close</button>
		</div>
	{/if}
</div>

<style>
	.token-tree {
		position: relative;
		width: 100%;
		height: 100%;
		background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
		border-radius: 12px;
		overflow: hidden;
	}

	.controls {
		position: absolute;
		top: 12px;
		left: 12px;
		z-index: 10;
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}

	.controls button {
		padding: 10px 16px;
		background: white;
		border: 2px solid #e2e8f0;
		border-radius: 8px;
		cursor: pointer;
		font-size: 14px;
		font-weight: 500;
		min-height: 44px;
		transition: all 0.2s;
		-webkit-tap-highlight-color: rgba(147, 51, 234, 0.2);
	}

	.controls button:active {
		transform: scale(0.96);
		background: #f8fafc;
	}

	.tree-svg {
		cursor: grab;
		display: block;
		width: 100%;
		height: 100%;
		touch-action: none;
	}

	.tree-svg:active {
		cursor: grabbing;
	}

	.info-panel {
		position: absolute;
		bottom: 20px;
		left: 50%;
		transform: translateX(-50%);
		background: white;
		padding: 16px;
		border-radius: 12px;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
		z-index: 10;
		max-width: 90%;
		min-width: 200px;
	}

	.info-panel h3 {
		margin: 0 0 12px 0;
		font-size: 16px;
		color: #1e293b;
	}

	.info-panel p {
		margin: 6px 0;
		font-size: 14px;
		color: #64748b;
	}

	.info-panel button {
		margin-top: 12px;
		padding: 8px 16px;
		background: #9333ea;
		color: white;
		border: none;
		border-radius: 6px;
		cursor: pointer;
		width: 100%;
		font-size: 14px;
		font-weight: 500;
		min-height: 44px;
	}

	@media (max-width: 767px) {
		.controls {
			top: 8px;
			left: 8px;
			right: 8px;
			gap: 6px;
		}

		.controls button {
			flex: 1;
			min-width: 80px;
			padding: 8px 12px;
			font-size: 13px;
		}

		.info-panel {
			bottom: 12px;
			left: 12px;
			right: 12px;
			transform: none;
			max-width: none;
		}
	}
</style>
