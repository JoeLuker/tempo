import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Add this for handling imports when run as a script
if __name__ == "__main__":
    # Add the parent directory to the Python path so we can import from src
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)
    from src.custom_transformer_model import CustomParallelAttentionModel
else:
    # Regular imports when imported as a module
    from src.custom_transformer_model import CustomParallelAttentionModel

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_mps=True, use_custom_attention=False):
    """
    Load the Mistral-7B model and tokenizer optimized for MPS on Apple Silicon.
    
    Args:
        model_name (str): HuggingFace model identifier
        use_mps (bool): Whether to use MPS for acceleration
        use_custom_attention (bool): Whether to wrap the model with custom attention support
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Check if MPS is available
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"MPS not available, using: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimizations for MPS
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for better performance
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    
    # Move model to device after loading
    model = model.to(device)
    
    # Optionally wrap with custom attention model
    if use_custom_attention:
        print(f"Wrapping model with custom parallel attention support")
        model = CustomParallelAttentionModel(model)
    
    return model, tokenizer


def prepare_input(text, tokenizer, device="mps"):
    """
    Tokenize input text and prepare it for the model.
    
    Args:
        text (str): Input text to tokenize
        tokenizer: HuggingFace tokenizer
        device (str): Device to place tensors on
    
    Returns:
        dict: Input tensors for the model
    """
    encoded = tokenizer(text, return_tensors="pt")
    
    # Move to device
    for key in encoded:
        if isinstance(encoded[key], torch.Tensor):
            encoded[key] = encoded[key].to(device)
    
    return encoded 