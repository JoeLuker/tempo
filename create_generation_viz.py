#!/usr/bin/env python3
"""
Create interactive HTML visualization of TEMPO generation process.
Generates a beautiful D3.js tree diagram showing parallel token paths.
"""

import subprocess
import json
import yaml
from pathlib import Path
import webbrowser


def run_tempo_with_capture(prompt, threshold=0.25, max_tokens=15):
    """Run TEMPO and capture generation data."""
    config = {
        'prompt': prompt,
        'max_tokens': max_tokens,
        'selection_threshold': threshold,
        'seed': 42,
        'output_dir': './viz_data',
        'debug_mode': False,
    }

    Path('./viz_data').mkdir(exist_ok=True)
    config_path = './viz_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Running TEMPO with prompt: '{prompt}'...")
    subprocess.run(
        ['python3', 'run_tempo.py', '--config', config_path],
        capture_output=True,
        check=True
    )

    with open('./viz_data/results.json') as f:
        return json.load(f)


def parse_generation_tree(generated_text, prompt):
    """Parse formatted text into tree structure."""
    import re

    # Remove ANSI codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', generated_text)

    # Remove prompt
    if prompt in clean_text:
        clean_text = clean_text.split(prompt, 1)[1]

    # Parse parallel tokens [token1/token2]
    nodes = []
    node_id = 0

    # Root node
    nodes.append({
        'id': node_id,
        'text': prompt,
        'type': 'prompt',
        'parent': None
    })
    current_parent = node_id
    node_id += 1

    # Find parallel sets
    parallel_pattern = r'\[([^\]]+)\]'
    current_pos = 0

    for match in re.finditer(parallel_pattern, clean_text):
        # Text before parallel set
        before = clean_text[current_pos:match.start()].strip()
        if before:
            nodes.append({
                'id': node_id,
                'text': before,
                'type': 'single',
                'parent': current_parent
            })
            current_parent = node_id
            node_id += 1

        # Parallel tokens
        parallel_text = match.group(1)
        tokens = [t.strip() for t in parallel_text.split('/')]

        # Create parallel set node
        parallel_parent = current_parent
        for token in tokens:
            nodes.append({
                'id': node_id,
                'text': token,
                'type': 'parallel',
                'parent': parallel_parent
            })
            node_id += 1

        # Continue from first parallel token
        current_parent = parallel_parent + 1

        current_pos = match.end()

    # Remaining text
    remaining = clean_text[current_pos:].strip()
    if remaining:
        nodes.append({
            'id': node_id,
            'text': remaining,
            'type': 'single',
            'parent': current_parent
        })

    return nodes


def create_html_visualization(results, prompt, threshold, output_file='generation_viz.html'):
    """Create interactive D3.js visualization."""

    nodes = parse_generation_tree(results['generated_text'], prompt)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TEMPO Generation Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            color: #2d3748;
            margin-top: 0;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}

        .info {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }}

        .info-item {{
            margin: 5px 0;
            color: #4a5568;
        }}

        .info-label {{
            font-weight: bold;
            color: #2d3748;
        }}

        svg {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #fafafa;
        }}

        .node circle {{
            cursor: pointer;
            stroke-width: 2px;
        }}

        .node.prompt circle {{
            fill: #667eea;
            stroke: #5568d3;
        }}

        .node.single circle {{
            fill: #48bb78;
            stroke: #38a169;
        }}

        .node.parallel circle {{
            fill: #ed8936;
            stroke: #dd6b20;
        }}

        .node text {{
            font-size: 14px;
            font-family: 'Courier New', monospace;
            pointer-events: none;
        }}

        .link {{
            fill: none;
            stroke: #cbd5e0;
            stroke-width: 2px;
        }}

        .node:hover circle {{
            stroke-width: 4px;
            filter: brightness(1.1);
        }}

        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
        }}

        .legend-item {{
            display: inline-block;
            margin-right: 25px;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 14px;
        }}

        .legend-prompt {{
            background: #667eea;
            color: white;
        }}

        .legend-single {{
            background: #48bb78;
            color: white;
        }}

        .legend-parallel {{
            background: #ed8936;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌲 TEMPO Generation Tree Visualization</h1>

        <div class="info">
            <div class="info-item">
                <span class="info-label">Prompt:</span> "{prompt}"
            </div>
            <div class="info-item">
                <span class="info-label">Selection Threshold:</span> {threshold}
            </div>
            <div class="info-item">
                <span class="info-label">Generated Text:</span> {results['raw_generated_text']}
            </div>
            <div class="info-item">
                <span class="info-label">Generation Time:</span> {results['generation_time']:.2f}s
            </div>
        </div>

        <div id="tree"></div>

        <div class="legend">
            <span class="legend-item legend-prompt">● Prompt</span>
            <span class="legend-item legend-single">● Single Token</span>
            <span class="legend-item legend-parallel">● Parallel Token (Multiple Paths)</span>
        </div>
    </div>

    <script>
        const nodes = {json.dumps(nodes)};

        // Build hierarchy
        const root = d3.stratify()
            .id(d => d.id)
            .parentId(d => d.parent)
            (nodes);

        const width = 1200;
        const height = 600;

        const tree = d3.tree()
            .size([height - 100, width - 200])
            .separation((a, b) => a.parent === b.parent ? 1.5 : 2);

        const treeData = tree(root);

        const svg = d3.select('#tree')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', 'translate(80, 50)');

        // Links
        svg.selectAll('.link')
            .data(treeData.links())
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x));

        // Nodes
        const node = svg.selectAll('.node')
            .data(treeData.descendants())
            .enter()
            .append('g')
            .attr('class', d => `node ${{d.data.type}}`)
            .attr('transform', d => `translate(${{d.y}},${{d.x}})`);

        node.append('circle')
            .attr('r', 8);

        node.append('text')
            .attr('dy', -15)
            .attr('text-anchor', 'middle')
            .style('font-weight', d => d.data.type === 'prompt' ? 'bold' : 'normal')
            .text(d => d.data.text);

        // Add interactivity
        node.on('mouseover', function(event, d) {{
            d3.select(this).select('circle')
                .transition()
                .duration(200)
                .attr('r', 12);
        }})
        .on('mouseout', function(event, d) {{
            d3.select(this).select('circle')
                .transition()
                .duration(200)
                .attr('r', 8);
        }});
    </script>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\n✅ Visualization created: {output_file}")
    return output_file


if __name__ == '__main__':
    import sys
    import os

    prompt = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else "Once upon a time"
    threshold = 0.25

    print("="*80)
    print("Creating TEMPO Generation Visualization")
    print("="*80)

    # Run TEMPO
    results = run_tempo_with_capture(prompt, threshold)

    # Create visualization
    output_file = create_html_visualization(results, prompt, threshold)

    # Open in browser
    print(f"\n🌐 Opening visualization in browser...")
    webbrowser.open('file://' + os.path.abspath(output_file))

    # Cleanup
    os.remove('./viz_config.yaml')

    print("\n✨ Done! The visualization is now open in your browser.")
    print("="*80)
