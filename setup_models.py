#!/usr/bin/env python3
"""
TEMPO Model Setup Script
Downloads and caches required models for TEMPO
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import json

# Check for required dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download, HfFolder
    import torch
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


# Default model configuration
DEFAULT_MODEL = "deepcogito/cogito-v1-preview-llama-3B"
MODEL_CONFIGS = {
    "deepcogito/cogito-v1-preview-llama-3B": {
        "revision": None,
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "description": "Cogito v1 Preview - 3B parameter model optimized for TEMPO"
    },
    "meta-llama/Llama-2-7b-hf": {
        "revision": None,
        "trust_remote_code": False,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "description": "Llama 2 7B - Requires HuggingFace authentication"
    },
    "mistralai/Mistral-7B-v0.1": {
        "revision": None,
        "trust_remote_code": False,
        "torch_dtype": "auto", 
        "low_cpu_mem_usage": True,
        "description": "Mistral 7B v0.1 - Open-source 7B model"
    }
}


class ModelSetup:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.logger = self._setup_logger()
        self.cache_dir = self._get_cache_dir()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("TEMPO-ModelSetup")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _get_cache_dir(self) -> Path:
        """Get the HuggingFace cache directory"""
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return Path(hf_home) / "hub"
        
    def check_model_exists(self) -> bool:
        """Check if model is already downloaded"""
        model_path = self.cache_dir / f"models--{self.model_id.replace('/', '--')}"
        
        if model_path.exists() and any(model_path.iterdir()):
            # Try to load tokenizer as a quick check
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                self.logger.info(f"Model '{self.model_id}' found in cache")
                return True
            except Exception:
                self.logger.warning(f"Model files found but may be incomplete")
                return False
        return False
        
    def get_model_size(self) -> str:
        """Estimate model size for download"""
        size_estimates = {
            "3B": "~6-8 GB",
            "7B": "~13-15 GB",
            "13B": "~25-30 GB",
            "70B": "~130-140 GB"
        }
        
        for size_key in size_estimates:
            if size_key.lower() in self.model_id.lower():
                return size_estimates[size_key]
        return "Unknown size"
        
    def check_disk_space(self) -> bool:
        """Check if there's enough disk space"""
        import shutil
        
        stat = shutil.disk_usage(self.cache_dir.parent)
        free_gb = stat.free / (1024**3)
        
        # Rough estimates
        required_gb = 10  # Default
        if "3b" in self.model_id.lower():
            required_gb = 10
        elif "7b" in self.model_id.lower():
            required_gb = 20
        elif "13b" in self.model_id.lower():
            required_gb = 35
        elif "70b" in self.model_id.lower():
            required_gb = 150
            
        if free_gb < required_gb:
            self.logger.warning(
                f"Low disk space: {free_gb:.1f}GB free, "
                f"~{required_gb}GB recommended for {self.model_id}"
            )
            return False
        return True
        
    def download_model(self, force: bool = False) -> bool:
        """Download the model"""
        if not force and self.check_model_exists():
            self.logger.info("Model already downloaded. Use --force to re-download.")
            return True
            
        if not self.check_disk_space():
            response = input("Continue with low disk space? (y/N): ")
            if response.lower() != 'y':
                return False
                
        self.logger.info(f"Downloading model: {self.model_id}")
        self.logger.info(f"Estimated size: {self.get_model_size()}")
        self.logger.info("This may take several minutes...")
        
        try:
            # Get model config if available
            config = MODEL_CONFIGS.get(self.model_id, {})
            
            # Download model files
            self.logger.info("Downloading model files...")
            snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
                revision=config.get("revision"),
                resume_download=True
            )
            
            # Try loading tokenizer to verify
            self.logger.info("Verifying download...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=config.get("trust_remote_code", False)
            )
            
            # Try loading model config (not full model to save memory)
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=config.get("trust_remote_code", False)
            )
            
            self.logger.info(f"✓ Model '{self.model_id}' downloaded successfully!")
            self.logger.info(f"  Model type: {model_config.model_type}")
            self.logger.info(f"  Cache location: {self.cache_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model: {e}")
            if "token" in str(e).lower():
                self.logger.info("\nThis model may require authentication.")
                self.logger.info("Please run: huggingface-cli login")
            return False
            
    def verify_model(self) -> bool:
        """Verify model can be loaded"""
        self.logger.info(f"Verifying model: {self.model_id}")
        
        try:
            # Get device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
            self.logger.info(f"Using device: {device}")
            
            # Load tokenizer
            config = MODEL_CONFIGS.get(self.model_id, {})
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=config.get("trust_remote_code", False)
            )
            
            # Quick tokenization test
            test_text = "Hello, TEMPO!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            self.logger.info("✓ Model verification successful!")
            self.logger.info(f"  Tokenizer test: '{test_text}' -> {len(tokens)} tokens")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup and download models for TEMPO"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model ID to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if model exists, don't download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model after download"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable model configurations:")
        print("-" * 60)
        for model_id, config in MODEL_CONFIGS.items():
            print(f"\n{model_id}:")
            print(f"  {config['description']}")
            if model_id == DEFAULT_MODEL:
                print("  (DEFAULT)")
        print()
        return
        
    # Setup model
    setup = ModelSetup(args.model)
    
    if args.check_only:
        if setup.check_model_exists():
            print(f"✓ Model '{args.model}' is already downloaded")
            sys.exit(0)
        else:
            print(f"✗ Model '{args.model}' not found")
            sys.exit(1)
            
    # Download model
    success = setup.download_model(force=args.force)
    
    if success and (args.verify or not setup.check_model_exists()):
        success = setup.verify_model()
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()