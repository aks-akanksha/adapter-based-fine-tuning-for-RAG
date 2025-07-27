import yaml
import argparse
import pandas as pd
import os
import torch
import math
from src.data_loader import load_and_preprocess_data
from src.rag_pipeline import build_rag_pipeline
from src.model_loader import load_model_with_adapters
from src.trainer import train_model
from src.evaluator import evaluate_model

def main(config_path):
    """
    Main function to orchestrate the fine-tuning and evaluation process for all experiments.
    """
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

    # 2. Load Data and Build RAG (once for all experiments)
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_data(global_settings['data']['dataset_name'], global_settings['data']['split'])
    knowledge_base_texts = [item['context'] for item in dataset]
    
    print("Building RAG pipeline...")
    retriever = build_rag_pipeline(
        knowledge_base_texts,
        model_name=global_settings['rag']['retriever_model']
    )
    
    # Calculate total training steps for AdaLoRA
    num_samples = len(dataset)
    batch_size = global_settings['training']['per_device_train_batch_size']
    epochs = global_settings['training']['num_train_epochs']
    total_training_steps = math.ceil((num_samples / batch_size) * epochs)
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
        torch.cuda.empty_cache()

        print("Loading LLM and applying adapters...")
        model, tokenizer, params_info = load_model_with_adapters(
            base_model_name=config['model']['base_model'],
            adapter_type=adapter_type,
            adapter_config=config['model'],
            total_training_steps=total_training_steps # Pass total steps
        )

        print("Starting model training...")
        trained_model, training_time = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            retriever=retriever,
            training_args_config=config['training'],
            rag_config=config['rag'],
            adapter_type=adapter_type # Pass adapter type to trainer
        )

        print("Starting model evaluation...")
        eval_dataset = dataset.select(range(50)) 
        
        metrics = evaluate_model(
            model=trained_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            retriever=retriever,
            rag_config=config['rag']
        )

        print("\n--- Evaluation Complete ---")
        print(f"Experiment: {config['experiment_name']}")
        print(f"Fine-tuning time: {training_time:.2f}s")
        print(f"Trainable Params: {params_info['trainable_params']:,} ({params_info['trainable_percent']:.4f}%)")
        print(f"ROUGE-L Score: {metrics['rougeL']:.4f}")
        print(f"BLEU Score: {metrics['bleu']:.4f}")
        print(f"Semantic Accuracy: {metrics['semantic_accuracy']:.4f}")
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
            'bleu': metrics['bleu'],
            'semantic_accuracy': metrics['semantic_accuracy']
        }
        all_results.append(result_data)

        pd.DataFrame(all_results).to_csv(results_log_path, index=False)
        print(f"Results for '{config['experiment_name']}' saved to {results_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning for RAG.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
