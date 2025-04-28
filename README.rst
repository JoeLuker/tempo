TEMPO: Threshold-based Exploration with Multipath Parallel Output
================================================================

|license| |stars| |issues|

.. |license| image:: https://img.shields.io/github/license/JoeLuker/tempo
   :target: https://github.com/JoeLuker/tempo/blob/main/LICENSE
   :alt: GitHub

.. |stars| image:: https://img.shields.io/github/stars/JoeLuker/tempo
   :target: https://github.com/JoeLuker/tempo/stargazers
   :alt: GitHub stars

.. |issues| image:: https://img.shields.io/github/issues/JoeLuker/tempo
   :target: https://github.com/JoeLuker/tempo/issues
   :alt: GitHub issues

TEMPO implements and evaluates a non-autoregressive text generation mechanism using transformers with capabilities like parallel generation and early exits.

Features
--------

- Multiple parallel token generation strategies
- Early-exit capability for faster inference
- Monte Carlo Tree Search (MCTS) for path optimization
- Advanced pruning strategies
- Visualization and analytical tools
- Optimized for Apple Silicon with MPS

Installation
-----------

.. code-block:: bash

    pip install tempo-generation

Quick Start
----------

.. code-block:: python

    from tempo.modeling import EarlyExitTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")

    # Create early exit transformer
    early_exit_model = EarlyExitTransformer(model=model)

    # Generate with early exits
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    output_ids = early_exit_model.generate_with_early_exits(inputs.input_ids)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

Command Line Usage
-----------------

.. code-block:: bash

    # Basic usage
    tempo --prompt "Your prompt here" --threshold 0.1

    # With early exit
    tempo --prompt "Your prompt here" --early-exit

    # Run the demo
    tempo-demo --model "JackFram/llama-68m" --prompt "The capital of France is" --compare

Documentation
------------

For full documentation, visit: https://github.com/JoeLuker/tempo

License
-------

MIT License 