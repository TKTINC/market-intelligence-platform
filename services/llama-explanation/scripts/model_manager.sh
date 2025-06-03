# services/llama-explanation/scripts/model_manager.py
"""
Model management utilities for Llama 2-7B service
Handles model download, quantization, and validation
"""

import os
import hashlib
import requests
from pathlib import Path
import subprocess
import argparse
from typing import Optional

class LlamaModelManager:
    """Manager for Llama model operations"""
    
    def __init__(self, models_dir: str = "/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            "llama-2-7b-chat": {
                "url": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
                "quantized_name": "llama-2-7b-explanations.Q4_K_M.gguf",
                "expected_size_gb": 3.8,
                "sha256": None  # Add when available
            }
        }
    
    def download_base_model(self, model_name: str = "llama-2-7b-chat") -> bool:
        """Download base Llama model"""
        if model_name not in self.models:
            print(f"Unknown model: {model_name}")
            return False
        
        model_config = self.models[model_name]
        model_dir = self.models_dir / model_name
        
        print(f"Downloading {model_name} to {model_dir}...")
        
        try:
            # Use git clone for Hugging Face models
            cmd = [
                "git", "clone", model_config["url"], str(model_dir)
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✅ Downloaded {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def quantize_model(
        self, 
        model_name: str = "llama-2-7b-chat",
        quantization: str = "Q4_K_M"
    ) -> bool:
        """Quantize model to GGUF format"""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            print(f"Base model not found: {model_dir}")
            return False
        
        quantized_name = self.models[model_name]["quantized_name"]
        quantized_path = self.models_dir / quantized_name
        
        print(f"Quantizing {model_name} to {quantization}...")
        
        try:
            # Find the original model file
            model_file = None
            for ext in ["pytorch_model.bin", "model.safetensors"]:
                candidate = model_dir / ext
                if candidate.exists():
                    model_file = candidate
                    break
            
            if not model_file:
                print("❌ Could not find model file to quantize")
                return False
            
            # Run quantization
            cmd = [
                "python", "-m", "llama_cpp.quantize",
                str(model_file),
                str(quantized_path),
                quantization
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✅ Quantized model saved to {quantized_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization failed: {e}")
            return False
    
    def validate_model(self, model_path: Optional[str] = None) -> bool:
        """Validate quantized model"""
        if model_path is None:
            model_path = self.models_dir / "llama-2-7b-explanations.Q4_K_M.gguf"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"Validating model: {model_path}")
        
        # Check file size
        file_size_gb = model_path.stat().st_size / (1024**3)
        expected_size = 3.8  # Expected size for Q4_K_M quantized Llama-2-7B
        
        if file_size_gb < expected_size * 0.8 or file_size_gb > expected_size * 1.2:
            print(f"⚠️  Unexpected file size: {file_size_gb:.1f}GB (expected ~{expected_size}GB)")
        else:
            print(f"✅ File size OK: {file_size_gb:.1f}GB")
        
        # Try to load with llama-cpp-python
        try:
            from llama_cpp import Llama
            
            print("Loading model for validation...")
            model = Llama(
                model_path=str(model_path),
                n_ctx=512,  # Small context for testing
                n_gpu_layers=0,  # CPU only for validation
                verbose=False
            )
            
            # Test generation
            prompt = "The capital of France is"
            response = model(prompt, max_tokens=10, echo=False)
            
            if response and "choices" in response:
                generated_text = response["choices"][0]["text"]
                print(f"✅ Model validation successful")
                print(f"   Test prompt: '{prompt}'")
                print(f"   Generated: '{generated_text.strip()}'")
                return True
            else:
                print("❌ Model generated invalid response")
                return False
                
        except ImportError:
            print("⚠️  llama-cpp-python not available, skipping load test")
            return True
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def create_mock_model(self) -> bool:
        """Create mock model for development/testing"""
        mock_path = self.models_dir / "llama-2-7b-explanations.Q4_K_M.gguf"
        
        print(f"Creating mock model: {mock_path}")
        
        try:
            with open(mock_path, 'w') as f:
                f.write("MOCK_LLAMA_MODEL_FOR_DEVELOPMENT")
            
            print(f"✅ Mock model created: {mock_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create mock model: {e}")
            return False
    
    def list_models(self):
        """List available models"""
        print("Available models in", self.models_dir)
        print("-" * 50)
        
        for item in self.models_dir.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024**2)
                print(f"📄 {item.name} ({size_mb:.1f} MB)")
            elif item.is_dir():
                print(f"📁 {item.name}/")
    
    def cleanup_models(self, keep_quantized: bool = True):
        """Cleanup downloaded models"""
        print("Cleaning up models...")
        
        for item in self.models_dir.iterdir():
            if item.is_dir():
                # Remove base model directories
                print(f"Removing directory: {item}")
                subprocess.run(["rm", "-rf", str(item)])
            elif item.is_file() and not keep_quantized:
                # Remove quantized models if requested
                print(f"Removing file: {item}")
                item.unlink()
        
        print("✅ Cleanup complete")

def main():
    """Main function for model management"""
    parser = argparse.ArgumentParser(description="Llama Model Manager")
    parser.add_argument("--models-dir", default="/models", 
                       help="Models directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download base model")
    download_parser.add_argument("--model", default="llama-2-7b-chat",
                               help="Model to download")
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model")
    quantize_parser.add_argument("--model", default="llama-2-7b-chat",
                               help="Model to quantize")
    quantize_parser.add_argument("--format", default="Q4_K_M",
                               help="Quantization format")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument("--model-path", help="Path to model file")
    
    # Mock command
    subparsers.add_parser("mock", help="Create mock model for development")
    
    # List command
    subparsers.add_parser("list", help="List available models")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup models")
    cleanup_parser.add_argument("--keep-quantized", action="store_true",
                               help="Keep quantized models")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = LlamaModelManager(args.models_dir)
    
    if args.command == "download":
        success = manager.download_base_model(args.model)
    elif args.command == "quantize":
        success = manager.quantize_model(args.model, args.format)
    elif args.command == "validate":
        success = manager.validate_model(args.model_path)
    elif args.command == "mock":
        success = manager.create_mock_model()
    elif args.command == "list":
        manager.list_models()
        success = True
    elif args.command == "cleanup":
        manager.cleanup_models(args.keep_quantized)
        success = True
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
