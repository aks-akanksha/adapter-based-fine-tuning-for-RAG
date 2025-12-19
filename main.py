import yaml
import argparse
import pandas as pd
import os
import torch
import math
from src.data_loader import load_and_preprocess_data
from src.rag_pipeline import build_rag_pipeline
from src.advanced_rag import build_advanced_rag_pipeline, AdvancedRAGRetriever
from src.model_loader import load_model_with_adapters
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.enhanced_evaluator import evaluate_model as enhanced_evaluate_model, analyze_errors, print_error_analysis
from src.gpu_utils import print_gpu_info, check_gpu_availability, suggest_batch_size, clear_gpu_cache

def main(config_path):
    """
    Main function to orchestrate the fine-tuning and evaluation process for all experiments.
    """
    # 0. Check GPU availability
    print("="*80)
    print("ADAPTER-BASED FINE-TUNING FOR RAG")
    print("="*80)
    gpu_available = print_gpu_info()
    
    if not gpu_available:
        print("⚠️  Warning: No GPU detected. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # 1. Load Configuration
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    global_settings = full_config['global_settings']
    experiments = full_config['experiments']
    results_log_path = 'results.csv'
    all_results = []

    # Check if a results file already exists
    if os.path.exists(results_log_path):
        print(f"Found existing results file at {results_log_path}. Loading previous results.")
        results_df = pd.read_csv(results_log_path)
        all_results = results_df.to_dict('records')
        completed_experiments = results_df['experiment_name'].tolist()
    else:
        completed_experiments = []
    
    # Suggest optimal batch sizes if GPU info is available
    if gpu_available:
        gpu_info = check_gpu_availability()
        if gpu_info['total_memory_gb']:
            print(f"\n💡 GPU Memory Optimization Suggestions:")
            for adapter_type in ['lora', 'none']:
                suggestions = suggest_batch_size(gpu_info['total_memory_gb'], adapter_type)
                print(f"   For {adapter_type.upper()}: batch_size={suggestions['per_device_batch_size']}, "
                      f"gradient_accumulation={suggestions['gradient_accumulation_steps']}")
            print()

    # 2. Load Data and Build RAG (once for all experiments)
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_data(global_settings['data']['dataset_name'], global_settings['data']['split'])
    knowledge_base_texts = [item['context'] for item in dataset]
    
    print("Building RAG pipeline...")
    rag_config = global_settings['rag']
    
    # Use advanced RAG if enabled, otherwise use basic RAG
    if rag_config.get('enable_reranking', False) or rag_config.get('enable_sparse', False):
        print("Using Advanced RAG with enhanced retrieval strategies...")
        retriever = build_advanced_rag_pipeline(
            knowledge_base_texts,
            model_name=rag_config['retriever_model'],
            enable_reranking=rag_config.get('enable_reranking', False),
            enable_sparse=rag_config.get('enable_sparse', False)
        )
    else:
        retriever = build_rag_pipeline(
            knowledge_base_texts,
            model_name=rag_config['retriever_model']
        )
    
    # Calculate total training steps for AdaLoRA
    num_samples = len(dataset)
    batch_size = global_settings['training'].get('per_device_train_batch_size', 2)
    gradient_accumulation = global_settings['training'].get('gradient_accumulation_steps', 1)
    epochs = global_settings['training']['num_train_epochs']
    effective_batch_size = batch_size * gradient_accumulation
    total_training_steps = math.ceil((num_samples / effective_batch_size) * epochs)
    print(f"Calculated total training steps for AdaLoRA: {total_training_steps}")


    # 3. Loop Through Experiments
    for i, exp_config in enumerate(experiments):
        print("\n" + "="*80)
        print(f"Running Experiment {i+1}/{len(experiments)}: {exp_config['experiment_name']}")
        print("="*80 + "\n")

        if exp_config['experiment_name'] in completed_experiments:
            print(f"Skipping '{exp_config['experiment_name']}' as results are already logged.")
            continue

        config = {**global_settings, **exp_config}
        adapter_type = config['model'].get('adapter_type', 'none')

        # Clear GPU cache before loading a new model
        clear_gpu_cache()

        print("Loading LLM and applying adapters...")
        model, tokenizer, params_info = load_model_with_adapters(
            base_model_name=config['model']['base_model'],
            adapter_type=adapter_type,
            adapter_config=config['model'],
            total_training_steps=total_training_steps # Pass total steps
        )

        print("Starting model training...")
        use_tensorboard = config['training'].get('use_tensorboard', True)
        enable_checkpointing = config['training'].get('enable_checkpointing', True)
        
        trained_model, training_time = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            retriever=retriever,
            training_args_config=config['training'],
            rag_config=config['rag'],
            adapter_type=adapter_type,
            experiment_name=exp_config['experiment_name'],
            use_tensorboard=use_tensorboard,
            enable_checkpointing=enable_checkpointing
        )

        print("Starting model evaluation...")
        eval_dataset = dataset.select(range(50))
        
        # Use enhanced evaluator with error analysis
        use_enhanced = config['rag'].get('enable_reranking', False) or config['rag'].get('enable_sparse', False)
        
        if use_enhanced:
            metrics, error_data = enhanced_evaluate_model(
                model=trained_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                retriever=retriever,
                rag_config=config['rag'],
                show_examples=True,
                num_examples=3,
                calculate_bertscore=False,  # Disable for speed, can enable if needed
                save_error_analysis=f"./results/error_analysis_{exp_config['experiment_name']}.csv"
            )
            
            # Perform error analysis
            if error_data:
                error_analysis = analyze_errors(error_data)
                print_error_analysis(error_analysis)
        else:
            metrics = evaluate_model(
                model=trained_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                retriever=retriever,
                rag_config=config['rag']
            )

        print("\n--- Evaluation Complete ---")
        print(f"Experiment: {config['experiment_name']}")
        print(f"Fine-tuning time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
        print(f"Trainable Params: {params_info['trainable_params']:,} ({params_info['trainable_percent']:.4f}%)")
        print(f"ROUGE-L Score: {metrics['rougeL']:.4f}")
        print(f"ROUGE-1 Score: {metrics.get('rouge1', 0):.4f}")
        print(f"ROUGE-2 Score: {metrics.get('rouge2', 0):.4f}")
        print(f"BLEU Score: {metrics['bleu']:.4f}")
        if 'meteor' in metrics:
            print(f"METEOR Score: {metrics['meteor']:.4f}")
        print(f"Semantic Accuracy: {metrics['semantic_accuracy']:.4f}")
        print(f"Exact Match: {metrics.get('exact_match', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        if 'retrieval_f1' in metrics:
            print(f"Retrieval F1: {metrics['retrieval_f1']:.4f}")
            print(f"Answer in Context Rate: {metrics.get('answer_in_context_rate', 0):.4f}")
        if 'bertscore_f1' in metrics:
            print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
        print("---------------------------\n")

        result_data = {
            'experiment_name': config['experiment_name'],
            'base_model': config['model']['base_model'],
            'adapter_type': config['model'].get('adapter_type', 'Full FT'),
            'trainable_params': params_info['trainable_params'],
            'total_params': params_info['total_params'],
            'trainable_percent': params_info['trainable_percent'],
            'training_time_sec': training_time,
            'rougeL': metrics['rougeL'],
            'rouge1': metrics.get('rouge1', 0),
            'rouge2': metrics.get('rouge2', 0),
            'bleu': metrics['bleu'],
            'meteor': metrics.get('meteor', 0),
            'semantic_accuracy': metrics['semantic_accuracy'],
            'exact_match': metrics.get('exact_match', 0),
            'f1_score': metrics.get('f1_score', 0),
            'retrieval_f1': metrics.get('retrieval_f1', 0),
            'answer_in_context_rate': metrics.get('answer_in_context_rate', 0),
            'bertscore_f1': metrics.get('bertscore_f1', 0)
        }
        all_results.append(result_data)

        pd.DataFrame(all_results).to_csv(results_log_path, index=False)
        print(f"Results for '{config['experiment_name']}' saved to {results_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning for RAG.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
