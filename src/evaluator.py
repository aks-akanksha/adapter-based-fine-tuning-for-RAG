# Import enhanced evaluator functions
from src.enhanced_evaluator import (
    evaluate_model as enhanced_evaluate_model,
    analyze_errors,
    print_error_analysis
)
from typing import Dict

# Wrapper to maintain backward compatibility
def evaluate_model(model, tokenizer, eval_dataset, retriever, rag_config, 
                   show_examples: bool = True, num_examples: int = 3) -> Dict:
    """
    Evaluates the model using enhanced metrics.
    Maintains backward compatibility with existing code.
    """
    metrics, error_data = enhanced_evaluate_model(
        model, tokenizer, eval_dataset, retriever, rag_config,
        show_examples=show_examples, num_examples=num_examples,
        calculate_bertscore=False,  # Disable by default for speed
        save_error_analysis=None
    )
    
    # Perform error analysis
    if error_data:
        error_analysis = analyze_errors(error_data)
        print_error_analysis(error_analysis)
    
    return metrics