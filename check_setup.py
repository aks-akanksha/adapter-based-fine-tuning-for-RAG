#!/usr/bin/env python3
"""
Quick setup checker script to verify GPU, dependencies, and configuration.
"""
import sys
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required. Current:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'torch', 'transformers', 'datasets', 'peft', 'sentence_transformers',
        'faiss', 'evaluate', 'tensorboard'
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - MISSING")
            missing.append(pkg)
    return len(missing) == 0

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        from src.gpu_utils import print_gpu_info
        return print_gpu_info()
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False

def check_config():
    """Check if config.yaml exists and is valid."""
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml not found")
        return False
    
    try:
        from src.config_validator import validate_config
        is_valid, warnings = validate_config('config.yaml')
        if is_valid:
            print("✅ config.yaml is valid")
        else:
            print("⚠️  config.yaml has warnings:")
            for w in warnings:
                print(f"   {w}")
        return True
    except Exception as e:
        print(f"⚠️  Could not validate config: {e}")
        return False

def main():
    print("="*60)
    print("SETUP CHECKER")
    print("="*60)
    print()
    
    print("1. Python Version:")
    py_ok = check_python_version()
    print()
    
    print("2. Dependencies:")
    deps_ok = check_dependencies()
    print()
    
    print("3. GPU:")
    gpu_ok = check_gpu()
    print()
    
    print("4. Configuration:")
    config_ok = check_config()
    print()
    
    print("="*60)
    if all([py_ok, deps_ok, gpu_ok, config_ok]):
        print("✅ All checks passed! You're ready to train.")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    print("="*60)

if __name__ == "__main__":
    main()

