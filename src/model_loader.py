from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, IA3Config, AdaLoraConfig, TaskType
import torch

def load_model_with_adapters(base_model_name, adapter_type, adapter_config, total_training_steps=None):
    """
    Loads a base LLM and applies the specified PEFT adapter.
    Returns the model, tokenizer, and a dictionary of parameter information.
    """
    print(f"Loading base model: {base_model_name}")

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True
    }

    # Conditionally apply quantization only for adapter-based tuning
    if adapter_type != 'none':
        print("Applying 4-bit quantization for PEFT.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        print("Loading model in half-precision (fp16) for Full Fine-Tuning.")
        model_kwargs["torch_dtype"] = torch.float16


    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_type != 'none':
        print(f"Applying {adapter_type} adapter...")
        if adapter_type == 'lora':
            config = adapter_config['lora_config']
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **config)
        elif adapter_type == 'ia3':
            config = adapter_config['ia3_config']
            peft_config = IA3Config(task_type=TaskType.CAUSAL_LM, **config)
        elif adapter_type == 'adalora':
            config = adapter_config['adalora_config']
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                total_step=total_training_steps, # Pass total_step here
                **config
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
            
        model = get_peft_model(model, peft_config)
        
    # Get parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    params_info = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": (trainable_params / total_params) * 100
    }
    print(f"Trainable params: {params_info['trainable_params']:,} || All params: {params_info['total_params']:,} || Trainable %: {params_info['trainable_percent']:.4f}")

    return model, tokenizer, params_info