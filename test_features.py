#!/usr/bin/env python3
"""
Quick test script to verify all new features work correctly.
"""
import sys

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.enhanced_evaluator import evaluate_model, analyze_errors, print_error_analysis
        print("✅ Enhanced evaluator")
    except Exception as e:
        print(f"❌ Enhanced evaluator: {e}")
        return False
    
    try:
        from src.advanced_rag import build_advanced_rag_pipeline, AdvancedRAGRetriever
        print("✅ Advanced RAG")
    except Exception as e:
        print(f"❌ Advanced RAG: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit")
    except Exception as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly")
    except Exception as e:
        print(f"❌ Plotly: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU utilities."""
    print("\nTesting GPU utilities...")
    try:
        from src.gpu_utils import print_gpu_info, check_gpu_availability
        gpu_info = check_gpu_availability()
        if gpu_info.get('cuda_available', False):
            print(f"✅ GPU detected: {gpu_info['device_name']}")
            return True
        else:
            print("⚠️  No GPU detected")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_config():
    """Test config validation."""
    print("\nTesting config validation...")
    try:
        from src.config_validator import validate_config
        is_valid, warnings = validate_config('config.yaml')
        if is_valid:
            print("✅ Config is valid")
        else:
            print(f"⚠️  Config has warnings: {len([w for w in warnings if not w.startswith('💡')])}")
        return True
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def main():
    print("="*60)
    print("FEATURE VERIFICATION TEST")
    print("="*60)
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_gpu()
    all_passed &= test_config()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All features verified!")
        print("\nYou can now:")
        print("  1. Run training: python main.py")
        print("  2. Launch web interface: streamlit run web_app.py")
        print("  3. Analyze results: python analyze_results.py")
    else:
        print("⚠️  Some features need attention")
    print("="*60)

if __name__ == "__main__":
    main()

