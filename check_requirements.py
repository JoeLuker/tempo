#!/usr/bin/env python3
"""
TEMPO System Requirements Checker
Verifies that the system meets all requirements for running TEMPO
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class RequirementsChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        
    def print_header(self):
        """Print header"""
        print("\nTEMPO System Requirements Checker")
        print("=" * 50)
        print()
        
    def print_check(self, name: str, passed: bool, message: str = "", warning: bool = False):
        """Print a check result"""
        if passed:
            symbol = "✓" if not warning else "⚠"
            color = "\033[92m" if not warning else "\033[93m"
            self.checks_passed += 1
            if warning:
                self.warnings.append(message)
        else:
            symbol = "✗"
            color = "\033[91m"
            self.checks_failed += 1
            
        reset = "\033[0m"
        status = "PASS" if passed else "FAIL"
        if warning:
            status = "WARN"
            
        print(f"{color}{symbol}{reset} {name:<30} [{status}] {message}")
        
    def check_python_version(self) -> Tuple[bool, str]:
        """Check Python version"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version_str}"
        else:
            return False, f"Python {version_str} (3.8+ required)"
            
    def check_memory(self) -> Tuple[bool, str, bool]:
        """Check available memory"""
        try:
            if platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True)
                total_bytes = int(result.stdout.strip())
                total_gb = total_bytes / (1024**3)
            elif platform.system() == "Linux":
                with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if line.startswith('MemTotal:'):
                            total_kb = int(line.split()[1])
                            total_gb = total_kb / (1024**2)
                            break
            elif platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong),
                    ]
                    
                memoryStatus = MEMORYSTATUS()
                memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
                total_gb = memoryStatus.dwTotalPhys / (1024**3)
            else:
                return True, "Unable to detect", True
                
            if total_gb >= 16:
                return True, f"{total_gb:.1f}GB", False
            elif total_gb >= 8:
                return True, f"{total_gb:.1f}GB (16GB+ recommended)", True
            else:
                return False, f"{total_gb:.1f}GB (16GB+ required)", False
                
        except Exception:
            return True, "Unable to detect", True
            
    def check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            stat = shutil.disk_usage(os.getcwd())
            free_gb = stat.free / (1024**3)
            
            if free_gb >= 20:
                return True, f"{free_gb:.1f}GB free"
            else:
                return False, f"{free_gb:.1f}GB free (20GB+ required)"
        except Exception:
            return True, "Unable to detect"
            
    def check_gpu(self) -> Tuple[bool, str]:
        """Check for GPU availability"""
        gpu_info = []
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info.append(f"CUDA GPU {i}: {gpu_name}")
                return True, ", ".join(gpu_info)
            elif torch.backends.mps.is_available():
                return True, "Apple Silicon GPU (MPS)"
            else:
                return True, "No GPU detected (CPU mode available)"
                
        except ImportError:
            return True, "PyTorch not installed yet"
            
    def check_package_installed(self, package: str) -> bool:
        """Check if a Python package is installed"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
            
    def check_dependencies(self) -> Dict[str, bool]:
        """Check Python dependencies"""
        required_packages = {
            "transformers": "transformers",
            "torch": "torch",
            "numpy": "numpy",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "pydantic": "pydantic",
            "huggingface_hub": "huggingface-hub",
            "matplotlib": "matplotlib",
            "tqdm": "tqdm",
            "pyyaml": "yaml"
        }
        
        results = {}
        for display_name, import_name in required_packages.items():
            results[display_name] = self.check_package_installed(import_name)
            
        return results
        
    def check_node_npm(self) -> Tuple[bool, str]:
        """Check Node.js and npm for frontend"""
        try:
            node_result = subprocess.run(['node', '--version'], 
                                       capture_output=True, text=True)
            node_version = node_result.stdout.strip()
            
            npm_result = subprocess.run(['npm', '--version'], 
                                      capture_output=True, text=True)
            npm_version = npm_result.stdout.strip()
            
            # Extract major version
            major_version = int(node_version.split('.')[0].replace('v', ''))
            
            if major_version >= 16:
                return True, f"Node {node_version}, npm {npm_version}"
            else:
                return False, f"Node {node_version} (16+ required)"
                
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return True, "Not installed (optional for CLI-only usage)"
            
    def check_ports(self) -> List[Tuple[int, bool]]:
        """Check if required ports are available"""
        import socket
        
        ports = [8000, 5173, 5174]  # API, Frontend old, Frontend new
        results = []
        
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
                sock.close()
                results.append((port, True))
            except OSError:
                results.append((port, False))
                
        return results
        
    def check_model_cache(self) -> Tuple[bool, str]:
        """Check if default model is cached"""
        try:
            from transformers import AutoTokenizer
            
            model_id = "deepcogito/cogito-v1-preview-llama-3B"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache = cache_dir / f"models--{model_id.replace('/', '--')}"
            
            if model_cache.exists():
                # Try loading tokenizer to verify
                try:
                    AutoTokenizer.from_pretrained(
                        model_id,
                        local_files_only=True,
                        cache_dir=cache_dir
                    )
                    return True, "Model cached"
                except Exception:
                    return True, "Model cache found (may need verification)"
            else:
                return True, "Model will be downloaded on first use"
                
        except ImportError:
            return True, "Check after installing dependencies"
            
    def check_file_structure(self) -> Tuple[bool, List[str]]:
        """Check if required files exist"""
        required_files = [
            "run_tempo.py",
            "api.py",
            "requirements.txt",
            "src/__init__.py",
            "frontend/package.json"
        ]
        
        missing = []
        for file in required_files:
            if not Path(file).exists():
                missing.append(file)
                
        return len(missing) == 0, missing
        
    def run_checks(self):
        """Run all system checks"""
        self.print_header()
        
        # Python version
        passed, message = self.check_python_version()
        self.print_check("Python Version", passed, message)
        
        # Memory
        passed, message, warning = self.check_memory()
        self.print_check("System Memory", passed, message, warning)
        
        # Disk space
        passed, message = self.check_disk_space()
        self.print_check("Disk Space", passed, message)
        
        # GPU
        passed, message = self.check_gpu()
        self.print_check("GPU Availability", passed, message)
        
        # File structure
        passed, missing = self.check_file_structure()
        if not passed:
            message = f"Missing: {', '.join(missing[:3])}"
            if len(missing) > 3:
                message += f" and {len(missing)-3} more"
        else:
            message = "All required files present"
        self.print_check("Project Structure", passed, message)
        
        print("\nDependency Status:")
        print("-" * 50)
        
        # Python dependencies
        deps = self.check_dependencies()
        all_installed = all(deps.values())
        
        if all_installed:
            self.print_check("Python Dependencies", True, "All installed")
        else:
            missing_deps = [k for k, v in deps.items() if not v]
            self.print_check("Python Dependencies", False, 
                           f"Missing: {', '.join(missing_deps[:3])}")
            if len(missing_deps) > 3:
                print(f"         ... and {len(missing_deps)-3} more")
                
        # Node.js/npm
        passed, message = self.check_node_npm()
        self.print_check("Node.js/npm", passed, message)
        
        # Ports
        port_results = self.check_ports()
        blocked_ports = [p for p, available in port_results if not available]
        if blocked_ports:
            self.print_check("Port Availability", True, 
                           f"Ports {', '.join(map(str, blocked_ports))} in use", 
                           warning=True)
        else:
            self.print_check("Port Availability", True, "All ports available")
            
        # Model cache
        passed, message = self.check_model_cache()
        self.print_check("Model Cache", passed, message)
        
        # Summary
        print("\n" + "=" * 50)
        print(f"Checks passed: {self.checks_passed}")
        print(f"Checks failed: {self.checks_failed}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
                
        if self.checks_failed == 0:
            print("\n✅ System is ready for TEMPO!")
            print("\nNext steps:")
            print("1. Run: pip install -r requirements.txt")
            print("2. Run: python3 run_tempo.py --prompt 'Hello world'")
        else:
            print("\n❌ Some requirements are not met.")
            print("\nPlease address the failed checks before proceeding.")
            
        return self.checks_failed == 0
        
    def export_report(self, filename: str = "system_report.json"):
        """Export system report to JSON"""
        report = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "checks": {
                "passed": self.checks_passed,
                "failed": self.checks_failed,
                "warnings": self.warnings
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nSystem report saved to {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check system requirements for TEMPO"
    )
    parser.add_argument(
        "--export",
        help="Export system report to JSON file",
        metavar="FILENAME"
    )
    
    args = parser.parse_args()
    
    checker = RequirementsChecker()
    success = checker.run_checks()
    
    if args.export:
        checker.export_report(args.export)
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()