# Parallel Threshold Output Generation

This project implements and evaluates a non-autoregressive text generation mechanism called "Parallel Threshold Output" using Mistral-7B on Apple Silicon with MPS.

## Overview

Standard autoregressive text generation may constrain a model's ability to express potentially concurrent internal states. This experiment tests generating multiple tokens simultaneously based on a probability threshold.

## Core Mechanism

The Parallel Threshold Output mechanism works as follows:
1. Perform a forward pass to get logit probabilities for the next token position
2. Apply softmax and identify all tokens whose probabilities exceed a threshold
3. Output this set of tokens as the generation for the current step
4. Update the context with this set of tokens
5. Repeat the process

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the generator
python src/generate.py
```

## Project Structure

- `src/parallel_generator.py`: Implementation of the Parallel Threshold Output mechanism
- `src/model_loader.py`: Utilities for loading the Mistral-7B model on MPS
- `src/generate.py`: CLI for running generation experiments 