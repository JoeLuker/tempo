#!/usr/bin/env python3
"""Expanded test suite with 50+ diverse prompts for better statistical power.

This extends beyond the initial 13 prompts to cover:
- More categories
- More examples per category
- Edge cases and controlled scenarios
"""

from dataclasses import dataclass
from typing import List

@dataclass
class PromptTest:
    """Test prompt configuration."""
    name: str
    prompt: str
    category: str
    expected_tokens: int = 5  # Expected min tokens for validity


# Original 13 prompts
ORIGINAL_PROMPTS = [
    # Narrative (3)
    PromptTest("narrative_1", "Once upon a time in a distant galaxy", "narrative", 5),
    PromptTest("narrative_2", "The old wizard slowly climbed the mountain", "narrative", 5),
    PromptTest("narrative_3", "Deep in the forest, a mysterious creature", "narrative", 5),

    # Factual (3)
    PromptTest("factual_1", "The capital of France is", "factual", 2),
    PromptTest("factual_2", "Photosynthesis is the process by which", "factual", 4),
    PromptTest("factual_3", "The largest planet in our solar system", "factual", 4),

    # Technical (2)
    PromptTest("technical_1", "Machine learning algorithms can be classified into", "technical", 4),
    PromptTest("technical_2", "The algorithm complexity of quicksort is", "technical", 4),

    # Conversational (2)
    PromptTest("conversational_1", "How are you doing today? I'm", "conversational", 3),
    PromptTest("conversational_2", "Hey, did you see that movie? It was", "conversational", 4),

    # Simple (2)
    PromptTest("simple_1", "The cat sat on the", "simple", 3),
    PromptTest("simple_2", "I went to the store and bought", "simple", 4),

    # Complex (1)
    PromptTest("complex_1", "Despite the significant challenges facing modern society", "complex", 6),
]


# Expanded prompts for better coverage
EXPANDED_PROMPTS = [
    # More Narrative
    PromptTest("narrative_4", "The dragon flew over the mountains carrying", "narrative", 5),
    PromptTest("narrative_5", "In a world where magic was forbidden", "narrative", 5),
    PromptTest("narrative_6", "She opened the ancient book and began to", "narrative", 5),
    PromptTest("narrative_7", "The spaceship landed on the alien planet as", "narrative", 5),
    PromptTest("narrative_8", "Long ago, in a kingdom by the sea", "narrative", 5),

    # More Factual
    PromptTest("factual_4", "Water boils at 100 degrees Celsius at", "factual", 4),
    PromptTest("factual_5", "The human body contains approximately", "factual", 3),
    PromptTest("factual_6", "Gravity is the force that", "factual", 4),
    PromptTest("factual_7", "DNA stands for", "factual", 2),
    PromptTest("factual_8", "The speed of light in vacuum is", "factual", 4),

    # More Technical
    PromptTest("technical_3", "Neural networks consist of layers of", "technical", 4),
    PromptTest("technical_4", "The time complexity of binary search is", "technical", 4),
    PromptTest("technical_5", "Object-oriented programming is based on", "technical", 4),
    PromptTest("technical_6", "A hash table provides O(1) average", "technical", 4),
    PromptTest("technical_7", "REST APIs typically use HTTP methods such as", "technical", 5),

    # More Conversational
    PromptTest("conversational_3", "That sounds great! When should we", "conversational", 3),
    PromptTest("conversational_4", "I can't believe it's already Friday! This week", "conversational", 4),
    PromptTest("conversational_5", "Thanks for helping me out. I really appreciate", "conversational", 4),
    PromptTest("conversational_6", "What do you think about the new", "conversational", 4),
    PromptTest("conversational_7", "Sorry I'm late, traffic was terrible and", "conversational", 4),

    # More Simple
    PromptTest("simple_3", "The dog ran to the", "simple", 3),
    PromptTest("simple_4", "She likes to eat", "simple", 3),
    PromptTest("simple_5", "The sun is very", "simple", 3),
    PromptTest("simple_6", "We went to see the", "simple", 4),
    PromptTest("simple_7", "He walked down the street and", "simple", 4),

    # More Complex
    PromptTest("complex_2", "Notwithstanding the considerable evidence to the contrary", "complex", 6),
    PromptTest("complex_3", "The philosophical implications of quantum mechanics suggest that", "complex", 6),
    PromptTest("complex_4", "Throughout history, civilizations have risen and fallen due to", "complex", 6),
    PromptTest("complex_5", "The intersection of artificial intelligence and human consciousness", "complex", 6),

    # Question prompts (new category)
    PromptTest("question_1", "What is the meaning of life?", "question", 4),
    PromptTest("question_2", "How does the internet work?", "question", 3),
    PromptTest("question_3", "Why is the sky blue?", "question", 3),
    PromptTest("question_4", "When did World War II end?", "question", 3),
    PromptTest("question_5", "Where is the Eiffel Tower located?", "question", 4),

    # Incomplete sentences (new category)
    PromptTest("incomplete_1", "If I could go anywhere, I would", "incomplete", 4),
    PromptTest("incomplete_2", "The most important thing in life is", "incomplete", 4),
    PromptTest("incomplete_3", "When I grow up I want to", "incomplete", 5),
    PromptTest("incomplete_4", "My favorite part of the day is when", "incomplete", 5),
    PromptTest("incomplete_5", "The best way to learn something is", "incomplete", 4),

    # List/enumeration prompts (new category)
    PromptTest("list_1", "The three primary colors are", "list", 3),
    PromptTest("list_2", "Common programming languages include Python, Java, and", "list", 5),
    PromptTest("list_3", "Steps to bake a cake: first,", "list", 3),
    PromptTest("list_4", "My top five favorite movies are", "list", 4),
    PromptTest("list_5", "Ingredients for pancakes include flour, eggs, milk, and", "list", 5),

    # Code/technical fragments (new category)
    PromptTest("code_1", "def hello_world():", "code", 2),
    PromptTest("code_2", "import numpy as np\nimport", "code", 3),
    PromptTest("code_3", "for i in range(10):", "code", 3),
    PromptTest("code_4", "class Animal:\n    def __init__(self", "code", 4),
    PromptTest("code_5", "SELECT * FROM users WHERE", "code", 4),
]


def get_all_prompts() -> List[PromptTest]:
    """Get all prompts (original + expanded)."""
    return ORIGINAL_PROMPTS + EXPANDED_PROMPTS


def get_prompts_by_category(category: str) -> List[PromptTest]:
    """Get prompts for a specific category."""
    all_prompts = get_all_prompts()
    return [p for p in all_prompts if p.category == category]


def get_categories() -> List[str]:
    """Get all unique categories."""
    all_prompts = get_all_prompts()
    return sorted(list(set(p.category for p in all_prompts)))


if __name__ == "__main__":
    all_prompts = get_all_prompts()
    categories = get_categories()

    print(f"Total prompts: {len(all_prompts)}")
    print(f"\nBy category:")
    for cat in categories:
        cat_prompts = get_prompts_by_category(cat)
        print(f"  {cat:15s}: {len(cat_prompts):2d} prompts")

    print(f"\nSample prompts per category:")
    for cat in categories:
        cat_prompts = get_prompts_by_category(cat)
        print(f"\n{cat}:")
        for p in cat_prompts[:3]:
            print(f"  - {p.name}: \"{p.prompt}\"")
