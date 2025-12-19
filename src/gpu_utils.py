"""
GPU utilities for monitoring and optimization.
Provides real-time GPU memory tracking and optimization suggestions.
"""
import torch
import psutil
from typing import Dict, Optional
import subprocess
import sys


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        Dictionary with GPU information including availability, name, memory, etc.
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'total_memory_gb': None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
    }
    
    if info['available']:
        device = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info['total_memory_gb'] = props.total_memory / 1e9
        info['compute_capability'] = f"{props.major}.{props.minor}"
    
    return info


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory usage statistics in GB.
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total,
        'utilization_percent': (reserved / total) * 100 if total > 0 else 0
    }


def get_gpu_stats_nvidia_smi() -> Optional[Dict[str, any]]:
    """
    Get GPU stats using nvidia-smi (more detailed than PyTorch).
    
    Returns:
        Dictionary with GPU stats or None if nvidia-smi is not available.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu', 
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(values[0]),
                'memory_total_mb': int(values[1]),
                'utilization_percent': int(values[2]),
                'temperature_c': int(values[3])
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    
    return None


def print_gpu_info():
    """Print comprehensive GPU information."""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    gpu_info = check_gpu_availability()
    
    if not gpu_info['available']:
        print("❌ CUDA is not available. Training will use CPU (very slow).")
        return False
    
    print(f"✅ CUDA Available: {gpu_info['available']}")
    print(f"   Device: {gpu_info['device_name']}")
    print(f"   Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"   CUDA Version: {gpu_info['cuda_version']}")
    print(f"   PyTorch Version: {gpu_info['pytorch_version']}")
    print(f"   Compute Capability: {gpu_info['compute_capability']}")
    
    # Get current memory usage
    mem_info = get_gpu_memory_info()
    print(f"\n   Current Memory Usage:")
    print(f"   - Allocated: {mem_info['allocated']:.2f} GB")
    print(f"   - Reserved: {mem_info['reserved']:.2f} GB")
    print(f"   - Free: {mem_info['free']:.2f} GB")
    print(f"   - Utilization: {mem_info['utilization_percent']:.1f}%")
    
    # Try to get nvidia-smi stats
    nvidia_stats = get_gpu_stats_nvidia_smi()
    if nvidia_stats:
        print(f"\n   GPU Stats (nvidia-smi):")
        print(f"   - Memory Used: {nvidia_stats['memory_used_mb']} MB")
        print(f"   - GPU Utilization: {nvidia_stats['utilization_percent']}%")
        print(f"   - Temperature: {nvidia_stats['temperature_c']}°C")
    
    print("="*60 + "\n")
    return True


def suggest_batch_size(total_memory_gb: float, adapter_type: str, model_size: str = "small") -> Dict[str, int]:
    """
    Suggest optimal batch size based on GPU memory.
    
    Args:
        total_memory_gb: Total GPU memory in GB
        adapter_type: Type of adapter ('lora', 'ia3', 'adalora', 'none')
        model_size: Model size category ('small', 'medium', 'large')
    
    Returns:
        Dictionary with suggested batch sizes and gradient accumulation steps.
    """
    suggestions = {
        'per_device_batch_size': 2,
        'gradient_accumulation_steps': 4,
        'max_length': 512
    }
    
    # For PEFT methods, we can use larger batches
    if adapter_type != 'none':
        if total_memory_gb >= 8:
            suggestions['per_device_batch_size'] = 4
            suggestions['gradient_accumulation_steps'] = 2
        elif total_memory_gb >= 6:
            suggestions['per_device_batch_size'] = 3
            suggestions['gradient_accumulation_steps'] = 3
    else:
        # Full fine-tuning needs smaller batches
        if total_memory_gb >= 8:
            suggestions['per_device_batch_size'] = 2
            suggestions['gradient_accumulation_steps'] = 4
        else:
            suggestions['per_device_batch_size'] = 1
            suggestions['gradient_accumulation_steps'] = 8
    
    return suggestions


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def monitor_gpu_memory(interval: float = 1.0, duration: float = 5.0):
    """
    Monitor GPU memory usage for a specified duration.
    
    Args:
        interval: Time between measurements in seconds
        duration: Total monitoring duration in seconds
    """
    import time
    
    if not torch.cuda.is_available():
        print("GPU not available for monitoring.")
        return
    
    print(f"Monitoring GPU memory for {duration} seconds...")
    max_allocated = 0
    max_reserved = 0
    
    start_time = time.time()
    while time.time() - start_time < duration:
        mem_info = get_gpu_memory_info()
        max_allocated = max(max_allocated, mem_info['allocated'])
        max_reserved = max(max_reserved, mem_info['reserved'])
        
        print(f"  Allocated: {mem_info['allocated']:.2f} GB | "
              f"Reserved: {mem_info['reserved']:.2f} GB | "
              f"Free: {mem_info['free']:.2f} GB")
        
        time.sleep(interval)
    
    print(f"\nPeak Memory Usage:")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Max Reserved: {max_reserved:.2f} GB")

