"""
Configuration validator and optimizer for GPU-specific settings.
"""
import yaml
from typing import Dict, List, Tuple
from src.gpu_utils import check_gpu_availability, suggest_batch_size


def validate_config(config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate configuration file and return validation results.
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load config: {str(e)}"]
    
    # Check required sections
    required_sections = ['global_settings', 'experiments']
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing required section: {section}")
            return False, warnings
    
    # Check global_settings
    gs = config['global_settings']
    required_gs = ['data', 'rag', 'training']
    for req in required_gs:
        if req not in gs:
            warnings.append(f"Missing required global_settings section: {req}")
    
    # Check experiments
    if not isinstance(config['experiments'], list) or len(config['experiments']) == 0:
        warnings.append("No experiments defined in config")
        return False, warnings
    
    # Validate each experiment
    for i, exp in enumerate(config['experiments']):
        if 'experiment_name' not in exp:
            warnings.append(f"Experiment {i+1} missing 'experiment_name'")
        if 'model' not in exp:
            warnings.append(f"Experiment {i+1} missing 'model' section")
        else:
            if 'base_model' not in exp['model']:
                warnings.append(f"Experiment {i+1} missing 'base_model'")
    
    # Check GPU-specific optimizations
    gpu_info = check_gpu_availability()
    if gpu_info['available']:
        training_config = gs.get('training', {})
        batch_size = training_config.get('per_device_train_batch_size', 2)
        gradient_accum = training_config.get('gradient_accumulation_steps', 1)
        
        # Suggest optimizations
        for exp in config['experiments']:
            adapter_type = exp.get('model', {}).get('adapter_type', 'none')
            suggestions = suggest_batch_size(gpu_info['total_memory_gb'], adapter_type)
            
            if batch_size < suggestions['per_device_batch_size']:
                warnings.append(
                    f"💡 For {adapter_type}, consider increasing batch_size to {suggestions['per_device_batch_size']} "
                    f"(current: {batch_size})"
                )
            if gradient_accum < suggestions['gradient_accumulation_steps']:
                warnings.append(
                    f"💡 Consider using gradient_accumulation_steps={suggestions['gradient_accumulation_steps']} "
                    f"(current: {gradient_accum})"
                )
    
    is_valid = len([w for w in warnings if not w.startswith('💡')]) == 0
    return is_valid, warnings


def optimize_config_for_gpu(config_path: str, output_path: str = None) -> Dict:
    """
    Optimize configuration based on available GPU.
    
    Args:
        config_path: Path to input config.yaml
        output_path: Path to save optimized config (if None, returns dict only)
    
    Returns:
        Optimized configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gpu_info = check_gpu_availability()
    
    if not gpu_info['available']:
        print("⚠️  No GPU detected. Using default CPU settings.")
        return config
    
    print(f"✅ Optimizing config for GPU: {gpu_info['device_name']} ({gpu_info['total_memory_gb']:.2f} GB)")
    
    # Optimize training settings
    training = config['global_settings']['training']
    
    # Get suggestions for PEFT methods (most common)
    suggestions = suggest_batch_size(gpu_info['total_memory_gb'], 'lora')
    
    # Update if not explicitly set
    if 'per_device_train_batch_size' not in training or training['per_device_train_batch_size'] < suggestions['per_device_batch_size']:
        training['per_device_train_batch_size'] = suggestions['per_device_batch_size']
        print(f"   Set batch_size to {suggestions['per_device_batch_size']}")
    
    if 'gradient_accumulation_steps' not in training:
        training['gradient_accumulation_steps'] = suggestions['gradient_accumulation_steps']
        print(f"   Set gradient_accumulation_steps to {suggestions['gradient_accumulation_steps']}")
    
    # Enable optimizations
    if 'use_tensorboard' not in training:
        training['use_tensorboard'] = True
    if 'enable_checkpointing' not in training:
        training['enable_checkpointing'] = True
    if 'save_total_limit' not in training:
        training['save_total_limit'] = 2
    
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"💾 Optimized config saved to: {output_path}")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and optimize config.yaml")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--optimize', action='store_true', help='Optimize config for current GPU')
    parser.add_argument('--output', type=str, help='Output path for optimized config')
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_config_for_gpu(args.config, args.output)
    else:
        is_valid, warnings = validate_config(args.config)
        if is_valid:
            print("✅ Config is valid!")
        else:
            print("❌ Config has issues:")
        for warning in warnings:
            print(f"   {warning}")

