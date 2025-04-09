#!/usr/bin/env python3
"""
TEMPO: Threshold-Enabled Multiple Parallel Outputs Generator

This script runs the TEMPO generator with various options.
"""

import os
import sys
import subprocess

def main():
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the command to run the generate.py script
    generate_script = os.path.join(script_dir, "src", "generate.py")
    
    # Pass all command line arguments to the script
    cmd = [sys.executable, generate_script] + sys.argv[1:]
    
    # Set PYTHONPATH to include the project root directory
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir + ":" + env.get("PYTHONPATH", "")
    
    # Run the command
    subprocess.run(cmd, env=env)
    
    # Print usage examples
    print("\nUsage examples:")
    print("-" * 50)
    print("# Standard generation:")
    print("python run_generate.py --standard-generation --prompt \"Your prompt here\"")
    print("\n# Parallel generation without custom attention:")
    print("python run_generate.py --prompt \"Your prompt here\" --threshold 0.05")
    print("\n# Parallel generation with custom attention:")
    print("python run_generate.py --prompt \"Your prompt here\" --threshold 0.05 --custom-attention")
    print("\n# Parallel generation with pruning:")
    print("python run_generate.py --prompt \"Your prompt here\" --threshold 0.05 --custom-attention --use-pruning")
    print("-" * 50)
    
if __name__ == "__main__":
    main() 