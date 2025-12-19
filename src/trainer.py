from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
import torch
import os
from src.gpu_utils import get_gpu_memory_info, clear_gpu_cache
from src.advanced_rag import AdvancedRAGRetriever
import time

def format_prompt(question, context):
    """Formats the input for the LLM."""
    return f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


class CustomTrainer(Trainer):
    """Custom Trainer with GPU memory monitoring."""
    
    def __init__(self, *args, monitor_gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor_gpu = monitor_gpu
        self.gpu_memory_logs = []
    
    def log(self, logs, start_time=None):
        """Override log to include GPU memory info."""
        if self.monitor_gpu and torch.cuda.is_available():
            mem_info = get_gpu_memory_info()
            logs['gpu_memory_allocated_gb'] = mem_info['allocated']
            logs['gpu_memory_reserved_gb'] = mem_info['reserved']
            logs['gpu_memory_free_gb'] = mem_info['free']
            self.gpu_memory_logs.append({
                'step': self.state.global_step,
                'allocated': mem_info['allocated'],
                'reserved': mem_info['reserved']
            })
        super().log(logs, start_time)


def train_model(model, tokenizer, train_dataset, retriever, training_args_config, rag_config, adapter_type, 
                experiment_name=None, use_tensorboard=True, enable_checkpointing=True):
    """
    Sets up the Trainer, fine-tunes the model, and returns the training time.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        retriever: RAG retriever
        training_args_config: Training configuration dictionary
        rag_config: RAG configuration dictionary
        adapter_type: Type of adapter ('lora', 'ia3', 'adalora', 'none')
        experiment_name: Name of the experiment (for logging)
        use_tensorboard: Whether to use TensorBoard logging
        enable_checkpointing: Whether to enable gradient checkpointing
    """
    def tokenize_and_format(examples):
        # Support advanced RAG strategies
        strategy = rag_config.get('retrieval_strategy', 'dense')
        if isinstance(retriever, AdvancedRAGRetriever):
            # Advanced retriever with strategy support
            retrieved_contexts = [
                "\n".join(retriever.retrieve(q, top_k=rag_config['top_k'], strategy=strategy)) 
                for q in examples['question']
            ]
        else:
            # Basic retriever
            retrieved_contexts = [
                "\n".join(retriever.retrieve(q, top_k=rag_config['top_k'])) 
                for q in examples['question']
            ]
        prompts = [format_prompt(q, c) + " " + a for q, c, a in zip(examples['question'], retrieved_contexts, examples['answer'])]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

    print("Tokenizing dataset...")
    # Map the dataset and keep only the tokenizer output columns
    processed_dataset = train_dataset.map(
        tokenize_and_format, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    # Ensure the dataset has the required columns
    if 'input_ids' not in processed_dataset.column_names:
        raise ValueError("Tokenization failed: 'input_ids' not found in processed dataset")

    # Calculate training steps
    num_samples = len(processed_dataset)
    batch_size = training_args_config.get('per_device_train_batch_size', 2)
    gradient_accumulation = training_args_config.get('gradient_accumulation_steps', 1)
    epochs = training_args_config.get('num_train_epochs', 1)
    
    effective_batch_size = batch_size * gradient_accumulation
    steps_per_epoch = (num_samples + effective_batch_size - 1) // effective_batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * training_args_config.get('warmup_ratio', 0.1))

    print(f"Training Configuration:")
    print(f"  - Samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulation}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Warmup steps: {warmup_steps}")

    # Conditionally set fp16 based on whether we are doing full fine-tuning
    # Full fine-tuning should not use fp16 to avoid gradient scaling issues
    if adapter_type == 'none':
        print("Disabling fp16 for Full Fine-Tuning to prevent gradient scaling issues.")
        use_fp16 = False
        use_bf16 = False
    else:
        use_fp16 = training_args_config.get('fp16', True)
        use_bf16 = training_args_config.get('bf16', False)

    # Enable gradient checkpointing for memory efficiency
    # Note: Gradient checkpointing with 4-bit quantized models can be tricky
    # We'll only enable it for full fine-tuning or if explicitly requested
    if enable_checkpointing:
        if adapter_type == 'none':
            # Full fine-tuning can use gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                try:
                    model.gradient_checkpointing_enable()
                    print("✅ Gradient checkpointing enabled for memory efficiency.")
                except Exception as e:
                    print(f"⚠️  Could not enable gradient checkpointing: {e}")
        else:
            # For PEFT with quantization, gradient checkpointing is handled differently
            # The model should already be optimized by PEFT
            print("ℹ️  Gradient checkpointing skipped for PEFT (already memory-efficient)")

    # Set up logging
    log_dir = f"{training_args_config['output_dir']}/logs"
    if experiment_name:
        log_dir = f"{log_dir}/{experiment_name}"
    
    report_to = []
    if use_tensorboard:
        report_to.append("tensorboard")
        print(f"📊 TensorBoard logs will be saved to: {log_dir}")
        print(f"   View with: tensorboard --logdir {training_args_config['output_dir']}/logs")

    training_args = TrainingArguments(
        output_dir=training_args_config['output_dir'],
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=training_args_config.get('learning_rate', 0.0003),
        warmup_steps=warmup_steps,
        logging_dir=log_dir,
        logging_steps=training_args_config.get('logging_steps', 10),
        save_steps=training_args_config.get('save_steps', 500),
        save_total_limit=training_args_config.get('save_total_limit', 2),  # Keep only last 2 checkpoints
        report_to=report_to if report_to else ["none"],
        fp16=use_fp16,
        bf16=use_bf16,  # For newer GPUs
        dataloader_pin_memory=True,
        dataloader_num_workers=training_args_config.get('dataloader_num_workers', 0),
        load_best_model_at_end=training_args_config.get('load_best_model_at_end', False),
        metric_for_best_model=training_args_config.get('metric_for_best_model', None),
        greater_is_better=training_args_config.get('greater_is_better', True),
        save_strategy=training_args_config.get('save_strategy', 'steps'),
        eval_strategy=training_args_config.get('eval_strategy', 'no'),
        seed=training_args_config.get('seed', 42),
        optim=training_args_config.get('optim', 'adamw_torch'),
        lr_scheduler_type=training_args_config.get('lr_scheduler_type', 'linear'),
        max_grad_norm=training_args_config.get('max_grad_norm', 1.0) if not use_fp16 else None,  # Disable grad clipping with fp16
        gradient_checkpointing=enable_checkpointing and adapter_type == 'none',  # Only for full FT
    )

    # Clear GPU cache before training
    clear_gpu_cache()
    mem_info = get_gpu_memory_info()
    print(f"\nGPU Memory before training:")
    print(f"  Allocated: {mem_info['allocated']:.2f} GB")
    print(f"  Reserved: {mem_info['reserved']:.2f} GB")
    print(f"  Free: {mem_info['free']:.2f} GB\n")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        monitor_gpu=True,
    )
    
    print("🚀 Starting training...")
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = train_result.metrics.get("train_runtime", time.time() - start_time)
        
        # Final GPU memory check
        clear_gpu_cache()
        mem_info = get_gpu_memory_info()
        print(f"\n✅ Training complete!")
        print(f"   Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"   GPU Memory after training:")
        print(f"     Allocated: {mem_info['allocated']:.2f} GB")
        print(f"     Reserved: {mem_info['reserved']:.2f} GB")
        
        # Log peak memory if available
        if hasattr(trainer, 'gpu_memory_logs') and trainer.gpu_memory_logs:
            peak_mem = max(log['reserved'] for log in trainer.gpu_memory_logs)
            print(f"     Peak Reserved: {peak_mem:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ GPU Out of Memory Error!")
            print(f"   Try reducing batch size or enabling gradient checkpointing.")
            print(f"   Current batch size: {batch_size}")
            print(f"   Current gradient accumulation: {gradient_accumulation}")
            raise
        else:
            raise
    
    return model, training_time