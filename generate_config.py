#!/usr/bin/env python3
"""
TEMPO Interactive Configuration Generator
Helps users create a customized config.json file
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigGenerator:
    def __init__(self):
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from config.example.json"""
        config_example = Path("config.example.json")
        
        if config_example.exists():
            with open(config_example, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Fallback defaults if config.example.json doesn't exist
            return {
                "logging": {
                    "enable_file_logging": True,
                    "log_dir": "logs",
                    "log_level": "INFO",
                    "console_logging": True
                },
                "model": {
                    "model_id": "deepcogito/cogito-v1-preview-llama-3B",
                    "device": None,
                    "quantization": None,
                    "trust_remote_code": True,
                    "use_fast_tokenizer": True,
                    "revision": None,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": "auto"
                },
                "generation": {
                    "max_length": 200,
                    "top_k": 50,
                    "top_p": 0.95,
                    "temperature": 0.8,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                    "beam_width": 1,
                    "selection_threshold": 0.1,
                    "use_dynamic_thresholding": False,
                    "use_retroactive_pruning": False,
                    "attention_threshold": 0.01,
                    "use_parallel_generation": True,
                    "max_parallel_tokens": 5
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "cors_origins": ["http://localhost:5173", "http://localhost:5174"],
                    "debug": False,
                    "enable_docs": True,
                    "api_version": "v1",
                    "max_concurrent_requests": 100
                },
                "debug": {
                    "global_debug": False,
                    "module_debug": {
                        "model_wrapper": False,
                        "token_generator": False,
                        "parallel_generator": False,
                        "rope_modifier": False,
                        "attention_manager": False,
                        "pruner": False,
                        "selector": False,
                        "formatter": False
                    }
                }
            }
    
    def print_section(self, title: str):
        """Print a section header"""
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}\n")
        
    def get_bool_input(self, prompt: str, default: bool) -> bool:
        """Get boolean input from user"""
        default_str = "Y" if default else "N"
        other_str = "n" if default else "y"
        
        while True:
            response = input(f"{prompt} [{default_str}/{other_str}]: ").strip().lower()
            if not response:
                return default
            if response in ['y', 'yes', 'true', '1']:
                return True
            if response in ['n', 'no', 'false', '0']:
                return False
            print("Please enter Y/N")
            
    def get_string_input(self, prompt: str, default: str, allow_empty: bool = False) -> Optional[str]:
        """Get string input from user"""
        response = input(f"{prompt} [{default}]: ").strip()
        if not response:
            return default if not allow_empty else None
        return response
        
    def get_int_input(self, prompt: str, default: int, min_val: Optional[int] = None, 
                      max_val: Optional[int] = None) -> int:
        """Get integer input from user"""
        while True:
            response = input(f"{prompt} [{default}]: ").strip()
            if not response:
                return default
            try:
                value = int(response)
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid integer")
                
    def get_float_input(self, prompt: str, default: float, min_val: float = 0.0, 
                        max_val: float = 1.0) -> float:
        """Get float input from user"""
        while True:
            response = input(f"{prompt} [{default}]: ").strip()
            if not response:
                return default
            try:
                value = float(response)
                if value < min_val or value > max_val:
                    print(f"Value must be between {min_val} and {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number")
                
    def configure_basic_settings(self):
        """Configure basic settings"""
        self.print_section("Basic Settings")
        
        print("What type of setup would you like?")
        print("1. Quick setup (recommended defaults)")
        print("2. Custom setup (configure everything)")
        print("3. Research setup (optimized for experimentation)")
        
        setup_type = input("\nChoice [1]: ").strip() or "1"
        
        if setup_type == "1":
            # Quick setup - just ask for essentials
            print("\nGreat! Let's do a quick setup.")
            
            model = self.get_string_input(
                "\nModel to use",
                self.config["model"]["model_id"]
            )
            self.config["model"]["model_id"] = model
            
            if "7b" in model.lower():
                self.config["generation"]["max_length"] = 150
                self.config["model"]["quantization"] = "4bit"
                print("Note: Enabled 4-bit quantization for 7B model")
                
        elif setup_type == "3":
            # Research setup
            print("\nConfiguring for research...")
            self.config["debug"]["global_debug"] = True
            self.config["logging"]["log_level"] = "DEBUG"
            self.config["generation"]["use_retroactive_pruning"] = True
            self.config["generation"]["use_dynamic_thresholding"] = True
            print("Enabled debug mode and advanced features")
            
        else:
            # Custom setup
            self._configure_model()
            self._configure_generation()
            self._configure_api()
            self._configure_logging()
            self._configure_debug()
            
    def _configure_model(self):
        """Configure model settings"""
        self.print_section("Model Configuration")
        
        # Model selection
        print("Available models:")
        print("1. deepcogito/cogito-v1-preview-llama-3B (default, 3B params)")
        print("2. mistralai/Mistral-7B-v0.1 (7B params)")
        print("3. meta-llama/Llama-2-7b-hf (7B params, requires auth)")
        print("4. Custom model")
        
        choice = input("\nSelect model [1]: ").strip() or "1"
        
        if choice == "1":
            self.config["model"]["model_id"] = "deepcogito/cogito-v1-preview-llama-3B"
        elif choice == "2":
            self.config["model"]["model_id"] = "mistralai/Mistral-7B-v0.1"
        elif choice == "3":
            self.config["model"]["model_id"] = "meta-llama/Llama-2-7b-hf"
            print("Note: This model requires HuggingFace authentication")
        else:
            custom = input("Enter model ID: ").strip()
            if custom:
                self.config["model"]["model_id"] = custom
                
        # Device configuration
        print("\nDevice configuration:")
        print("1. Auto-detect (recommended)")
        print("2. CUDA (NVIDIA GPU)")
        print("3. MPS (Apple Silicon)")
        print("4. CPU only")
        
        device_choice = input("Select device [1]: ").strip() or "1"
        
        device_map = {"1": None, "2": "cuda", "3": "mps", "4": "cpu"}
        self.config["model"]["device"] = device_map.get(device_choice, None)
        
        # Quantization
        if "7b" in self.config["model"]["model_id"].lower():
            use_quant = self.get_bool_input(
                "\nUse quantization for memory efficiency?",
                True
            )
            if use_quant:
                print("1. 4-bit (recommended for 7B models)")
                print("2. 8-bit")
                quant = input("Select quantization [1]: ").strip() or "1"
                self.config["model"]["quantization"] = "4bit" if quant == "1" else "8bit"
                
    def _configure_generation(self):
        """Configure generation settings"""
        self.print_section("Generation Configuration")
        
        # Basic generation params
        self.config["generation"]["max_length"] = self.get_int_input(
            "Maximum generation length",
            self.config["generation"]["max_length"],
            50, 1000
        )
        
        self.config["generation"]["temperature"] = self.get_float_input(
            "Temperature (0.1-2.0, higher=more creative)",
            self.config["generation"]["temperature"],
            0.1, 2.0
        )
        
        self.config["generation"]["selection_threshold"] = self.get_float_input(
            "Selection threshold for parallel tokens (0.01-0.5)",
            self.config["generation"]["selection_threshold"],
            0.01, 0.5
        )
        
        # Advanced features
        use_advanced = self.get_bool_input(
            "\nConfigure advanced pruning features?",
            False
        )
        
        if use_advanced:
            self.config["generation"]["use_retroactive_pruning"] = self.get_bool_input(
                "Enable retroactive pruning?",
                self.config["generation"]["use_retroactive_pruning"]
            )
            
            if self.config["generation"]["use_retroactive_pruning"]:
                self.config["generation"]["attention_threshold"] = self.get_float_input(
                    "Attention threshold (0.001-0.1)",
                    self.config["generation"]["attention_threshold"],
                    0.001, 0.1
                )
                
                self.config["generation"]["use_dynamic_thresholding"] = self.get_bool_input(
                    "Use dynamic thresholding?",
                    self.config["generation"]["use_dynamic_thresholding"]
                )
                
    def _configure_api(self):
        """Configure API settings"""
        self.print_section("API Configuration")
        
        configure_api = self.get_bool_input(
            "Configure API settings?",
            False
        )
        
        if configure_api:
            self.config["api"]["host"] = self.get_string_input(
                "API host",
                self.config["api"]["host"]
            )
            
            self.config["api"]["port"] = self.get_int_input(
                "API port",
                self.config["api"]["port"],
                1024, 65535
            )
            
            self.config["api"]["enable_docs"] = self.get_bool_input(
                "Enable API documentation?",
                self.config["api"]["enable_docs"]
            )
            
    def _configure_logging(self):
        """Configure logging settings"""
        self.print_section("Logging Configuration")
        
        configure_logging = self.get_bool_input(
            "Configure logging settings?",
            False
        )
        
        if configure_logging:
            self.config["logging"]["enable_file_logging"] = self.get_bool_input(
                "Enable file logging?",
                self.config["logging"]["enable_file_logging"]
            )
            
            if self.config["logging"]["enable_file_logging"]:
                self.config["logging"]["log_dir"] = self.get_string_input(
                    "Log directory",
                    self.config["logging"]["log_dir"]
                )
                
            print("\nLog level:")
            print("1. DEBUG (most verbose)")
            print("2. INFO (default)")
            print("3. WARNING")
            print("4. ERROR (least verbose)")
            
            level_choice = input("Select level [2]: ").strip() or "2"
            level_map = {"1": "DEBUG", "2": "INFO", "3": "WARNING", "4": "ERROR"}
            self.config["logging"]["log_level"] = level_map.get(level_choice, "INFO")
            
    def _configure_debug(self):
        """Configure debug settings"""
        self.print_section("Debug Configuration")
        
        self.config["debug"]["global_debug"] = self.get_bool_input(
            "Enable global debug mode?",
            self.config["debug"]["global_debug"]
        )
        
        if self.config["debug"]["global_debug"]:
            configure_modules = self.get_bool_input(
                "Configure individual module debugging?",
                False
            )
            
            if configure_modules:
                for module in self.config["debug"]["module_debug"]:
                    self.config["debug"]["module_debug"][module] = self.get_bool_input(
                        f"Debug {module}?",
                        self.config["debug"]["module_debug"][module]
                    )
                    
    def save_config(self):
        """Save configuration to file"""
        output_file = "config.json"
        
        # Check if config.json already exists
        if Path(output_file).exists():
            overwrite = self.get_bool_input(
                f"\n{output_file} already exists. Overwrite?",
                False
            )
            if not overwrite:
                output_file = input("Enter alternative filename [config.custom.json]: ").strip()
                if not output_file:
                    output_file = "config.custom.json"
                    
        # Save configuration
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
            
        print(f"\n✓ Configuration saved to {output_file}")
        
        # Show how to use it
        print("\nTo use this configuration:")
        print(f"1. With environment variable: export TEMPO_CONFIG_FILE={output_file}")
        print(f"2. In your code: config = TempoConfig.from_file('{output_file}')")
        print(f"3. The API and CLI will automatically load {output_file} if it exists")
        
    def run(self):
        """Run the configuration generator"""
        print("TEMPO Configuration Generator")
        print("=============================")
        print("This tool will help you create a custom configuration file.\n")
        
        try:
            self.configure_basic_settings()
            self.save_config()
            
            print("\nConfiguration complete!")
            
        except KeyboardInterrupt:
            print("\n\nConfiguration cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)


if __name__ == "__main__":
    generator = ConfigGenerator()
    generator.run()