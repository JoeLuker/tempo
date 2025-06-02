<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  
  export let tokenData: any = null;
  export let height = 600;
  
  let container: HTMLElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  
  interface TreeNode {
    id: string;
    text: string;
    probability: number;
    position: number;
    pruned: boolean;
    children: TreeNode[];
  }
  
  function buildTree(data: any): TreeNode {
    if (!data.raw_token_data || data.raw_token_data.length === 0) {
      return { id: 'root', text: 'START', probability: 1, position: -1, pruned: false, children: [] };
    }
    
    // Create root node
    const root: TreeNode = {
      id: 'root',
      text: 'START',
      probability: 1,
      position: -1,
      pruned: false,
      children: []
    };
    
    // Build tree structure from token sets
    let currentLevel = [root];
    
    data.raw_token_data.forEach((tokenSet: any, idx: number) => {
      const nextLevel: TreeNode[] = [];
      
      // Create a map of pruned tokens for quick lookup
      const prunedMap = new Map(tokenSet.pruned_tokens.map((t: any) => [t.id, t]));
      
      // Process all original tokens
      tokenSet.original_tokens.forEach((token: any) => {
        const isPruned = !prunedMap.has(token.id);
        const node: TreeNode = {
          id: `${idx}-${token.id}`,
          text: token.text.trim(),
          probability: token.probability,
          position: tokenSet.position,
          pruned: isPruned,
          children: []
        };
        
        // Connect to all non-pruned nodes in current level
        currentLevel.forEach(parent => {
          if (!parent.pruned) {
            parent.children.push(node);
          }
        });
        
        nextLevel.push(node);
      });
      
      // Only non-pruned nodes continue to next level
      currentLevel = nextLevel.filter(n => !n.pruned);
    });
    
    return root;
  }
  
  function renderTree(root: TreeNode) {
    if (!container) return;
    
    // Clear previous content
    d3.select(container).selectAll('*').remove();
    
    const width = container.clientWidth;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'token-tree');
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create tree layout
    const treeLayout = d3.tree<TreeNode>()
      .size([innerHeight, innerWidth])
      .separation((a, b) => a.parent === b.parent ? 1 : 1.5);
    
    // Convert to hierarchy
    const hierarchy = d3.hierarchy(root);
    const treeData = treeLayout(hierarchy);
    
    // Create links
    const link = g.selectAll('.link')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal<any, any>()
        .x(d => d.y)
        .y(d => d.x))
      .style('fill', 'none')
      .style('stroke', d => d.target.data.pruned ? '#ef4444' : '#10b981')
      .style('stroke-width', d => Math.max(1, d.target.data.probability * 3))
      .style('opacity', d => d.target.data.pruned ? 0.3 : 0.7)
      .style('stroke-dasharray', d => d.target.data.pruned ? '5,5' : 'none');
    
    // Create nodes
    const node = g.selectAll('.node')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.y},${d.x})`);
    
    // Add circles for nodes
    node.append('circle')
      .attr('r', d => d.data.position === -1 ? 8 : Math.max(4, d.data.probability * 10))
      .style('fill', d => {
        if (d.data.position === -1) return '#6366f1'; // Root
        return d.data.pruned ? '#fbbf24' : '#10b981'; // Pruned vs kept
      })
      .style('stroke', d => d.data.pruned ? '#f59e0b' : '#059669')
      .style('stroke-width', 2)
      .style('opacity', d => d.data.pruned ? 0.5 : 1);
    
    // Add text background for better readability
    node.append('rect')
      .attr('x', d => {
        const textLength = Math.min(d.data.text.length * 7, 80);
        return -textLength / 2 - 4;
      })
      .attr('y', -20)
      .attr('width', d => {
        const textLength = Math.min(d.data.text.length * 7, 80);
        return textLength + 8;
      })
      .attr('height', 20)
      .attr('rx', 3)
      .style('fill', 'white')
      .style('fill-opacity', 0.9)
      .style('stroke', d => d.data.pruned ? '#fbbf24' : '#10b981')
      .style('stroke-width', 1);
    
    // Add text labels
    node.append('text')
      .attr('dy', -6)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', d => d.data.position === -1 ? 'bold' : 'normal')
      .style('fill', d => d.data.position === -1 ? '#6366f1' : '#1f2937')
      .text(d => {
        if (d.data.text.length > 12) {
          return d.data.text.substring(0, 12) + '...';
        }
        return d.data.text;
      });
    
    // Add probability labels
    node.append('text')
      .attr('dy', 20)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#6b7280')
      .style('font-weight', '500')
      .text(d => d.data.position === -1 ? '' : `${(d.data.probability * 100).toFixed(1)}%`);
    
    // Add tooltips
    node.append('title')
      .text(d => `Token: ${d.data.text}\nProbability: ${(d.data.probability * 100).toFixed(2)}%\nPosition: ${d.data.position}\nStatus: ${d.data.pruned ? 'Pruned' : 'Kept'}`);
  }
  
  $: if (tokenData && container) {
    const tree = buildTree(tokenData);
    renderTree(tree);
  }
  
  onMount(() => {
    return () => {
      if (svg) {
        svg.remove();
      }
    };
  });
</script>

<div bind:this={container} class="w-full h-full">
  {#if !tokenData}
    <div class="flex items-center justify-center h-full text-gray-500">
      <p>No token data to visualize</p>
    </div>
  {/if}
</div>

<style>
  :global(.token-tree) {
    background: transparent;
  }
  
  :global(.token-tree .link) {
    transition: all 0.3s ease;
  }
  
  :global(.token-tree .node circle) {
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  :global(.token-tree .node:hover circle) {
    transform: scale(1.2);
  }
</style>