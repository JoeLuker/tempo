#!/usr/bin/env python3
"""Simple test: Does position gap affect temporal perception?

This uses a simpler approach - just manually inject position offsets
into a standard generation run to see if the model's output changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_tempo import main as run_tempo_main
import argparse


# Test prompts that query temporal perception
TEST_PROMPTS = [
    "We've been talking for",
    "This conversation started",
    "How long have we been chatting?",
]


def test_hypothesis():
    """Quick test of position-as-time hypothesis."""

    print("="*70)
    print("POSITION-AS-TIME PERCEPTION TEST")
    print("="*70)
    print()
    print("Hypothesis: LLMs perceive elapsed time through position indices")
    print()
    print("Testing with prompt: 'We've been talking for'")
    print()

    prompt = "We've been talking for"

    print("CONTROL: Normal generation (positions 0, 1, 2, ...)")
    print("-"*70)

    # For now, let's just document the experimental design
    # Actual implementation requires modifying the generation loop

    print("""
Experimental Design:

Control Condition:
    Prompt tokens at positions: 0, 1, 2, 3, 4
    Generated tokens at positions: 5, 6, 7, 8, 9, ...
    Expected: "a few seconds" / "a moment" / short duration

Treatment 1 (Gap to 1000):
    Prompt tokens at positions: 0, 1, 2, 3, 4
    Generated tokens at positions: 1000, 1001, 1002, ...
    Position gap: 995
    Expected IF hypothesis true: "a while" / "some time" / medium duration

Treatment 2 (Gap to 10000):
    Prompt tokens at positions: 0, 1, 2, 3, 4
    Generated tokens at positions: 10000, 10001, 10002, ...
    Position gap: 9995
    Expected IF hypothesis true: "hours" / "a long time" / long duration

To implement this, we need to modify the token generation loop to:
1. Process prompt with positions 0...N-1
2. For generated tokens, use positions starting at OFFSET instead of N
3. This requires patching the position_ids tensor before each forward pass

The key insight: RoPE uses position indices to encode relative distances.
If the model uses these as temporal markers, large gaps should affect
its perception of elapsed conversation time.

This is testable with TEMPO's RoPE modification infrastructure!
""")

    print("="*70)
    print("STATUS: Experimental framework created")
    print("NEXT: Implement position injection in generation loop")
    print("="*70)


if __name__ == "__main__":
    test_hypothesis()
