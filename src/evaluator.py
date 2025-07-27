from tqdm import tqdm
import torch
import evaluate 
from sentence_transformers import SentenceTransformer, util
import numpy as np
from torch.cuda.amp import autocast

rouge_metric = evaluate.load('rouge') 
bleu_metric = evaluate.load('sacrebleu') 
semantic_sim_model = None

def format_prompt(question, context):
    """Formats the input for the LLM."""
    return f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def evaluate_model(model, tokenizer, eval_dataset, retriever, rag_config):
    """
    Evaluates the model on a given dataset and computes multiple metrics.
    """
    global semantic_sim_model
    if semantic_sim_model is None:
        print("Loading semantic similarity model for evaluation...")
        semantic_sim_model = SentenceTransformer('all-mpnet-base-v2')

    model.eval()
    predictions, references, semantic_similarities = [], [], []

    for item in tqdm(eval_dataset, desc="Evaluating"):
        question = item['question']
        reference_answer = item['answer']
        
        retrieved_context = "\n".join(retriever.retrieve(question, top_k=rag_config['top_k']))
        prompt = format_prompt(question, retrieved_context)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            with autocast():
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    max_new_tokens=50,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_answer = generated_text.split("Answer:")[-1].strip()
        
        predictions.append(generated_answer)
        references.append([reference_answer])

        pred_embedding = semantic_sim_model.encode(generated_answer, convert_to_tensor=True)
        ref_embedding = semantic_sim_model.encode(reference_answer, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
        semantic_similarities.append(cosine_score)

    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)

    return {
        "rougeL": rouge_score['rougeL'],
        "bleu": bleu_score['score'] / 100,
        "semantic_accuracy": np.mean(semantic_similarities)
    }