import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tabulate import tabulate
import numpy as np

def analyze(results_path):
    """
    Loads results from a CSV file and generates multiple, clean analysis plots.
    """
    try:
        results_df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{results_path}'")
        print("Please run main.py to generate results first.")
        return

    # Clean up adapter type names for better plotting
    results_df['adapter_type'] = results_df['adapter_type'].replace('none', 'Full FT')

    print("--- Experiment Results Summary ---")
    print(tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    print("--------------------------------\n")

    # Generate and save each plot as a separate file
    plot_performance_vs_efficiency(results_df)
    plot_performance_comparison(results_df)
    plot_cost_comparison(results_df)
    plot_deep_dive(results_df)


def plot_performance_vs_efficiency(results_df):
    """Plot 1: The main trade-off scatter plot."""
    plt.figure(figsize=(18, 12))
    sns.set_theme(style="whitegrid", palette="muted")
    
    ax = sns.scatterplot(
        data=results_df, x='trainable_params', y='semantic_accuracy', hue='base_model',
        style='adapter_type', size='training_time_sec', sizes=(200, 2000),
        palette='viridis', edgecolor='black', alpha=0.8
    )
    
    for i, row in results_df.iterrows():
        plt.text(row['trainable_params'] * 1.3, row['semantic_accuracy'], row['experiment_name'], 
                 fontsize=11, ha='left', va='center', rotation=15, weight='medium')
                 
    plt.title('Performance vs. Efficiency Trade-Off', fontsize=24, weight='bold', pad=20)
    plt.xlabel('Trainable Parameters (Log Scale)', fontsize=18)
    plt.ylabel('Semantic Accuracy Score', fontsize=18)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    
    filename = 'plot_1_performance_vs_efficiency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_performance_comparison(results_df):
    """Plot 2: Detailed comparison of performance metrics."""
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="whitegrid")
    
    df_melted = results_df.melt(
        id_vars=['experiment_name'], value_vars=['rougeL', 'bleu', 'semantic_accuracy'],
        var_name='metric', value_name='score'
    )
    
    ax = sns.barplot(data=df_melted, x='score', y='experiment_name', hue='metric', palette='magma', orient='h')
    
    plt.title('Performance Metrics Comparison', fontsize=24, weight='bold', pad=20)
    plt.xlabel('Score', fontsize=18)
    plt.ylabel('Experiment', fontsize=18)
    plt.xlim(0, max(0.5, df_melted['score'].max() * 1.15))
    plt.legend(title='Metric', loc='lower right', fontsize=14)
    
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.005, p.get_y() + p.get_height() / 2, f'{width:.3f}', va='center', ha='left', fontsize=11)
        
    plt.tight_layout()
    filename = 'plot_2_performance_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_cost_comparison(results_df):
    """Plot 3: Combined plot for training time and parameter costs."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    sns.set_theme(style="whitegrid")
    fig.suptitle('Cost Analysis: Time & Parameter Efficiency', fontsize=24, weight='bold')

    # Subplot A: Training Time
    sns.barplot(data=results_df.sort_values('training_time_sec', ascending=False), 
                x='training_time_sec', y='experiment_name', hue='experiment_name', ax=axes[0], 
                palette='plasma', orient='h', legend=False)
    axes[0].set_title('Fine-Tuning Time', fontsize=20, weight='bold')
    axes[0].set_xlabel('Time (seconds)', fontsize=16)
    axes[0].set_ylabel('')

    # Subplot B: Parameter Efficiency
    sns.barplot(data=results_df.sort_values('trainable_percent', ascending=False), 
                x='trainable_percent', y='experiment_name', hue='experiment_name', ax=axes[1], 
                palette='cividis', orient='h', legend=False)
    axes[1].set_title('Parameter Efficiency', fontsize=20, weight='bold')
    axes[1].set_xlabel('% of Parameters Trained (Log Scale)', fontsize=16)
    axes[1].set_ylabel('')
    axes[1].set_xscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = 'plot_3_cost_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_deep_dive(results_df):
    """Plot 4: Model-wise deep-dive comparison with corrected scaling."""
    unique_models = sorted(results_df['base_model'].unique())
    n_models = len(unique_models)
    
    fig, axes = plt.subplots(n_models, 3, figsize=(24, 8 * n_models), squeeze=False)
    fig.suptitle('Model-wise Deep Dive: Full Fine-Tuning vs. PEFT Methods', fontsize=28, weight='bold')

    # Determine global y-limits for fair comparison
    max_accuracy = results_df['semantic_accuracy'].max() * 1.2
    max_time = results_df['training_time_sec'].max() * 1.2

    for i, model_name in enumerate(unique_models):
        model_df = results_df[results_df['base_model'] == model_name].sort_values('adapter_type')
        
        # --- Column 1: Semantic Accuracy ---
        ax1 = axes[i, 0]
        sns.barplot(data=model_df, x='adapter_type', y='semantic_accuracy', hue='adapter_type', ax=ax1, palette='summer', legend=False)
        ax1.set_ylabel(f'{model_name}', fontsize=18, weight='bold')
        ax1.set_ylim(0, max_accuracy) # Use global max for fair scaling
        if i == 0: ax1.set_title('Performance (Semantic Accuracy)', fontsize=18, weight='bold')

        # --- Column 2: Training Time ---
        ax2 = axes[i, 1]
        sns.barplot(data=model_df, x='adapter_type', y='training_time_sec', hue='adapter_type', ax=ax2, palette='autumn', legend=False)
        ax2.set_ylim(0, max_time) # Use global max for fair scaling
        if i == 0: ax2.set_title('Cost (Training Time)', fontsize=18, weight='bold')

        # --- Column 3: Trainable Parameters ---
        ax3 = axes[i, 2]
        sns.barplot(data=model_df, x='adapter_type', y='trainable_params', hue='adapter_type', ax=ax3, palette='winter', legend=False)
        ax3.set_yscale('log')
        if i == 0: ax3.set_title('Cost (Trainable Parameters)', fontsize=18, weight='bold')
        
        # Formatting for all subplots in the row
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45, labelsize=14)
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:,.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 9),
                            textcoords='offset points', rotation=45)
            if i < n_models - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Adapter Type', fontsize=18)
            ax.yaxis.grid(True, linestyle='--', which='both', color='gray', alpha=.25)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = 'plot_4_model_deep_dive.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from LLM fine-tuning experiments.")
    parser.add_argument('--results_file', type=str, default='results.csv', help='Path to the CSV file containing experiment results.')
    args = parser.parse_args()
    analyze(args.results_file)