#!/usr/bin/env python3
"""
Create interactive HTML visualization of TEMPO generation process.
Shows parallel tokens as simultaneous alternatives, not branching paths.
"""

import subprocess
import json
import yaml
from pathlib import Path
import webbrowser
import re


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


def parse_generation_sequence(generated_text, prompt):
    """Parse formatted text into sequence of steps."""
    # Remove ANSI codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', generated_text)

    # Remove prompt
    if prompt in clean_text:
        clean_text = clean_text.split(prompt, 1)[1]

    steps = []

    # Add prompt as first step
    steps.append({
        'type': 'prompt',
        'tokens': [prompt]
    })

    # Parse parallel sets [token1/token2]
    parallel_pattern = r'\[([^\]]+)\]'
    current_pos = 0

    for match in re.finditer(parallel_pattern, clean_text):
        # Text before parallel set (single tokens)
        before = clean_text[current_pos:match.start()].strip()
        if before:
            steps.append({
                'type': 'single',
                'tokens': [before]
            })

        # Parallel tokens
        parallel_text = match.group(1)
        tokens = [t.strip() for t in parallel_text.split('/')]
        steps.append({
            'type': 'parallel',
            'tokens': tokens
        })

        current_pos = match.end()

    # Remaining text
    remaining = clean_text[current_pos:].strip()
    if remaining:
        steps.append({
            'type': 'single',
            'tokens': [remaining]
        })

    return steps


def create_html_visualization(results, prompt, threshold, output_file='generation_viz.html'):
    """Create interactive visualization showing parallel tokens correctly."""

    steps = parse_generation_sequence(results['generated_text'], prompt)

    # Convert steps to JSON-safe format
    steps_json = json.dumps(steps)

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

        #sequence {{
            margin: 30px 0;
            min-height: 400px;
        }}

        .step {{
            margin: 20px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .step-number {{
            background: #667eea;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}

        .step-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .token-box {{
            padding: 12px 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            transition: all 0.2s;
            cursor: pointer;
        }}

        .token-box:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .token-prompt {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}

        .token-single {{
            background: #48bb78;
            color: white;
        }}

        .token-parallel {{
            background: #ed8936;
            color: white;
            border: 2px solid #dd6b20;
        }}

        .parallel-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-left: 20px;
            border-left: 3px solid #ed8936;
        }}

        .parallel-label {{
            font-size: 12px;
            color: #718096;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .connector {{
            width: 2px;
            height: 20px;
            background: #cbd5e0;
            margin: 0 auto;
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
        <h1>🔄 TEMPO Parallel Generation Sequence</h1>

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
            <div class="info-item">
                <span class="info-label">Key Concept:</span> Parallel tokens exist <strong>simultaneously</strong> at the same logical position (not separate branches!)
            </div>
        </div>

        <div id="sequence"></div>

        <div class="legend">
            <span class="legend-item legend-prompt">● Prompt (Starting Point)</span>
            <span class="legend-item legend-single">● Single Token (Only One Choice)</span>
            <span class="legend-item legend-parallel">● Parallel Tokens (Multiple Choices at Same Position)</span>
        </div>
    </div>

    <script>
        const steps = {steps_json};

        const sequenceDiv = d3.select('#sequence');

        steps.forEach((step, i) => {{
            const stepDiv = sequenceDiv.append('div')
                .attr('class', 'step');

            stepDiv.append('div')
                .attr('class', 'step-number')
                .text(i);

            const contentDiv = stepDiv.append('div')
                .attr('class', 'step-content');

            if (step.type === 'parallel') {{
                contentDiv.append('div')
                    .attr('class', 'parallel-label')
                    .text(`Parallel Set (all ${{step.tokens.length}} exist simultaneously)`);

                const parallelGroup = contentDiv.append('div')
                    .attr('class', 'parallel-group');

                step.tokens.forEach(token => {{
                    parallelGroup.append('div')
                        .attr('class', 'token-box token-parallel')
                        .text(token);
                }});
            }} else {{
                step.tokens.forEach(token => {{
                    contentDiv.append('div')
                        .attr('class', `token-box token-${{step.type}}`)
                        .text(token);
                }});
            }}

            // Add connector except for last step
            if (i < steps.length - 1) {{
                sequenceDiv.append('div')
                    .attr('class', 'connector');
            }}
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

    print("\n✨ Done! The visualization shows parallel tokens as simultaneous alternatives.")
    print("="*80)
