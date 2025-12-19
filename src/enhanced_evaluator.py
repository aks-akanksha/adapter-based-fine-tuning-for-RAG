"""
Enhanced evaluator with additional metrics and error analysis capabilities.
"""
from tqdm import tqdm
import torch
import evaluate 
from sentence_transformers import SentenceTransformer, util
import numpy as np
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
from collections import defaultdict

# Load metrics
rouge_metric = evaluate.load('rouge') 
bleu_metric = evaluate.load('sacrebleu')
meteor_metric = evaluate.load('meteor')

# Lazy loading for expensive metrics
bert_score_model = None
semantic_sim_model = None

def format_prompt(question, context):
    """Formats the input for the LLM."""
    return f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def calculate_exact_match(prediction: str, reference: str) -> bool:
    """Calculate exact match (case-insensitive, whitespace-normalized)."""
    pred_clean = prediction.strip().lower()
    ref_clean = reference.strip().lower()
    return pred_clean == ref_clean


def calculate_f1_score(prediction: str, reference: str) -> float:
    """Calculate F1 score based on token overlap."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    intersection = pred_tokens & ref_tokens
    if len(intersection) == 0:
        return 0.0
    
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_retrieval_metrics(retrieved_contexts: List[str], reference_answer: str, 
                                question: str, top_k: int) -> Dict[str, float]:
    """
    Calculate retrieval quality metrics.
    
    Args:
        retrieved_contexts: List of retrieved context strings
        reference_answer: The correct answer
        question: The question
        top_k: Number of retrieved contexts
    
    Returns:
        Dictionary with retrieval metrics
    """
    if not retrieved_contexts:
        return {
            'retrieval_precision': 0.0,
            'retrieval_recall': 0.0,
            'retrieval_f1': 0.0,
            'answer_in_context': False
        }
    
    # Check if answer appears in any retrieved context
    answer_in_context = any(reference_answer.lower() in ctx.lower() for ctx in retrieved_contexts)
    
    # Simple keyword-based retrieval quality
    # Count how many contexts contain answer-related terms
    answer_terms = set(reference_answer.lower().split())
    relevant_contexts = 0
    
    for ctx in retrieved_contexts:
        ctx_terms = set(ctx.lower().split())
        if answer_terms & ctx_terms:  # Has overlap with answer
            relevant_contexts += 1
    
    precision = relevant_contexts / len(retrieved_contexts) if retrieved_contexts else 0.0
    recall = 1.0 if answer_in_context else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'retrieval_precision': precision,
        'retrieval_recall': recall,
        'retrieval_f1': f1,
        'answer_in_context': answer_in_context,
        'relevant_contexts_count': relevant_contexts
    }


def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BERTScore metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers (can be nested lists)
    
    Returns:
        Dictionary with BERTScore metrics
    """
    global bert_score_model
    
    try:
        from bert_score import score as bert_score_fn
        
        # Flatten references if nested
        flat_refs = [ref[0] if isinstance(ref, list) else ref for ref in references]
        
        P, R, F1 = bert_score_fn(predictions, flat_refs, lang='en', verbose=False)
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    except ImportError:
        print("⚠️  BERTScore not available. Install with: pip install bert-score")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }
    except Exception as e:
        print(f"⚠️  Error calculating BERTScore: {e}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }


def evaluate_model(model, tokenizer, eval_dataset, retriever, rag_config, 
                   show_examples: bool = True, num_examples: int = 3,
                   calculate_bertscore: bool = True,
                   save_error_analysis: Optional[str] = None) -> Dict:
    """
    Enhanced evaluation with additional metrics and error analysis.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        eval_dataset: Evaluation dataset
        retriever: RAG retriever
        rag_config: RAG configuration
        show_examples: Whether to show example predictions
        num_examples: Number of examples to show
        calculate_bertscore: Whether to calculate BERTScore (slower)
        save_error_analysis: Path to save error analysis CSV (optional)
    
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    global semantic_sim_model
    if semantic_sim_model is None:
        print("Loading semantic similarity model for evaluation...")
        semantic_sim_model = SentenceTransformer('all-mpnet-base-v2')

    model.eval()
    predictions, references, semantic_similarities = [], [], []
    exact_matches = []
    f1_scores = []
    examples = []
    error_analysis_data = []  # For detailed error analysis
    
    # Retrieval metrics
    retrieval_precisions = []
    retrieval_recalls = []
    retrieval_f1s = []
    answer_in_context_flags = []

    print(f"Evaluating on {len(eval_dataset)} samples...")
    
    for idx, item in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        question = item['question']
        reference_answer = item['answer']
        
        # Retrieve contexts (support advanced RAG strategies)
        from src.advanced_rag import AdvancedRAGRetriever
        strategy = rag_config.get('retrieval_strategy', 'dense')
        if isinstance(retriever, AdvancedRAGRetriever):
            # Advanced retriever with strategy support
            retrieved_contexts = retriever.retrieve(question, top_k=rag_config['top_k'], strategy=strategy)
        else:
            # Basic retriever
            retrieved_contexts = retriever.retrieve(question, top_k=rag_config['top_k'])
        retrieved_context = "\n".join(retrieved_contexts)
        prompt = format_prompt(question, retrieved_context)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            with autocast():
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=rag_config.get('max_new_tokens', 50),
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1.0,
                )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_answer = generated_text.split("Answer:")[-1].strip()
        
        predictions.append(generated_answer)
        references.append([reference_answer])

        # Calculate semantic similarity
        pred_embedding = semantic_sim_model.encode(generated_answer, convert_to_tensor=True)
        ref_embedding = semantic_sim_model.encode(reference_answer, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
        semantic_similarities.append(cosine_score)
        
        # Calculate exact match and F1
        exact_match = calculate_exact_match(generated_answer, reference_answer)
        f1_score = calculate_f1_score(generated_answer, reference_answer)
        exact_matches.append(exact_match)
        f1_scores.append(f1_score)
        
        # Calculate retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(
            retrieved_contexts, reference_answer, question, rag_config['top_k']
        )
        retrieval_precisions.append(retrieval_metrics['retrieval_precision'])
        retrieval_recalls.append(retrieval_metrics['retrieval_recall'])
        retrieval_f1s.append(retrieval_metrics['retrieval_f1'])
        answer_in_context_flags.append(retrieval_metrics['answer_in_context'])
        
        # Store for error analysis
        error_analysis_data.append({
            'question': question,
            'reference': reference_answer,
            'prediction': generated_answer,
            'semantic_sim': cosine_score,
            'exact_match': exact_match,
            'f1_score': f1_score,
            'answer_in_context': retrieval_metrics['answer_in_context'],
            'retrieval_f1': retrieval_metrics['retrieval_f1'],
            'prediction_length': len(generated_answer.split()),
            'reference_length': len(reference_answer.split()),
        })
        
        # Store examples
        if len(examples) < num_examples or random.random() < 0.1:
            examples.append({
                'question': question,
                'reference': reference_answer,
                'prediction': generated_answer,
                'semantic_sim': cosine_score,
                'exact_match': exact_match,
                'f1': f1_score,
                'retrieval_f1': retrieval_metrics['retrieval_f1'],
                'answer_in_context': retrieval_metrics['answer_in_context']
            })

    # Compute standard metrics
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    
    # Calculate METEOR
    try:
        meteor_score = meteor_metric.compute(predictions=predictions, references=references)
        meteor_value = meteor_score.get('meteor', 0.0)
    except:
        meteor_value = 0.0
    
    # Calculate BERTScore if requested
    bert_score_metrics = {}
    if calculate_bertscore:
        print("Calculating BERTScore (this may take a while)...")
        bert_score_metrics = calculate_bert_score(predictions, references)
    
    # Compile all metrics
    metrics = {
        "rougeL": rouge_score.get('rougeL', 0.0),
        "rouge1": rouge_score.get('rouge1', 0.0),
        "rouge2": rouge_score.get('rouge2', 0.0),
        "bleu": bleu_score['score'] / 100,
        "meteor": meteor_value,
        "semantic_accuracy": np.mean(semantic_similarities),
        "exact_match": np.mean(exact_matches),
        "f1_score": np.mean(f1_scores),
        "retrieval_precision": np.mean(retrieval_precisions),
        "retrieval_recall": np.mean(retrieval_recalls),
        "retrieval_f1": np.mean(retrieval_f1s),
        "answer_in_context_rate": np.mean(answer_in_context_flags),
        "num_samples": len(predictions),
        **bert_score_metrics
    }
    
    # Display examples
    if show_examples and examples:
        print("\n" + "="*80)
        print("EVALUATION EXAMPLES")
        print("="*80)
        for i, ex in enumerate(examples[:num_examples], 1):
            print(f"\nExample {i}:")
            print(f"  Question: {ex['question']}")
            print(f"  Reference: {ex['reference']}")
            print(f"  Prediction: {ex['prediction']}")
            print(f"  Metrics:")
            print(f"    - Semantic Similarity: {ex['semantic_sim']:.3f}")
            print(f"    - Exact Match: {'✅' if ex['exact_match'] else '❌'}")
            print(f"    - F1 Score: {ex['f1']:.3f}")
            print(f"    - Retrieval F1: {ex['retrieval_f1']:.3f}")
            print(f"    - Answer in Context: {'✅' if ex['answer_in_context'] else '❌'}")
        print("="*80 + "\n")
    
    # Save error analysis if requested
    if save_error_analysis:
        error_df = pd.DataFrame(error_analysis_data)
        error_df.to_csv(save_error_analysis, index=False)
        print(f"💾 Error analysis saved to: {save_error_analysis}")
    
    return metrics, error_analysis_data


def analyze_errors(error_data: List[Dict]) -> Dict:
    """
    Analyze errors and provide insights.
    
    Args:
        error_data: List of error analysis dictionaries
    
    Returns:
        Dictionary with error analysis insights
    """
    if not error_data:
        return {}
    
    df = pd.DataFrame(error_data)
    
    # Categorize errors
    low_semantic = df[df['semantic_sim'] < 0.3]
    no_exact_match = df[~df['exact_match']]
    low_f1 = df[df['f1_score'] < 0.3]
    retrieval_failures = df[~df['answer_in_context']]
    
    # Length analysis
    length_diff = df['prediction_length'] - df['reference_length']
    too_short = df[length_diff < -5]
    too_long = df[length_diff > 10]
    
    analysis = {
        'total_samples': len(df),
        'low_semantic_count': len(low_semantic),
        'low_semantic_rate': len(low_semantic) / len(df),
        'no_exact_match_count': len(no_exact_match),
        'no_exact_match_rate': len(no_exact_match) / len(df),
        'low_f1_count': len(low_f1),
        'low_f1_rate': len(low_f1) / len(df),
        'retrieval_failure_count': len(retrieval_failures),
        'retrieval_failure_rate': len(retrieval_failures) / len(df),
        'too_short_count': len(too_short),
        'too_long_count': len(too_long),
        'avg_semantic_sim': df['semantic_sim'].mean(),
        'avg_f1_score': df['f1_score'].mean(),
        'avg_retrieval_f1': df['retrieval_f1'].mean(),
    }
    
    return analysis


def print_error_analysis(analysis: Dict):
    """Print formatted error analysis."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"Total Samples: {analysis.get('total_samples', 0)}")
    print(f"\n📊 Performance Issues:")
    print(f"  - Low Semantic Similarity (<0.3): {analysis.get('low_semantic_count', 0)} ({analysis.get('low_semantic_rate', 0)*100:.1f}%)")
    print(f"  - No Exact Match: {analysis.get('no_exact_match_count', 0)} ({analysis.get('no_exact_match_rate', 0)*100:.1f}%)")
    print(f"  - Low F1 Score (<0.3): {analysis.get('low_f1_count', 0)} ({analysis.get('low_f1_rate', 0)*100:.1f}%)")
    print(f"\n🔍 Retrieval Issues:")
    print(f"  - Retrieval Failures: {analysis.get('retrieval_failure_count', 0)} ({analysis.get('retrieval_failure_rate', 0)*100:.1f}%)")
    print(f"  - Average Retrieval F1: {analysis.get('avg_retrieval_f1', 0):.3f}")
    print(f"\n📏 Length Issues:")
    print(f"  - Too Short Predictions: {analysis.get('too_short_count', 0)}")
    print(f"  - Too Long Predictions: {analysis.get('too_long_count', 0)}")
    print(f"\n📈 Averages:")
    print(f"  - Average Semantic Similarity: {analysis.get('avg_semantic_sim', 0):.3f}")
    print(f"  - Average F1 Score: {analysis.get('avg_f1_score', 0):.3f}")
    print("="*80 + "\n")

