from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def format_prompt(question, context):
    """Formats the input for the LLM."""
    return f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def train_model(model, tokenizer, train_dataset, retriever, training_args_config, rag_config, adapter_type):
    """
    Sets up the Trainer, fine-tunes the model, and returns the training time.
    """
    def tokenize_and_format(examples):
        retrieved_contexts = ["\n".join(retriever.retrieve(q, top_k=rag_config['top_k'])) for q in examples['question']]
        prompts = [format_prompt(q, c) + " " + a for q, c, a in zip(examples['question'], retrieved_contexts, examples['answer'])]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

    processed_dataset = train_dataset.map(tokenize_and_format, batched=True)

    # Conditionally set fp16 based on whether we are doing full fine-tuning
    use_fp16 = True
    if adapter_type == 'none':
        print("Disabling fp16 for Full Fine-Tuning to prevent gradient scaling issues.")
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=training_args_config['output_dir'],
        num_train_epochs=training_args_config['num_train_epochs'],
        per_device_train_batch_size=training_args_config['per_device_train_batch_size'],
        learning_rate=training_args_config['learning_rate'],
        logging_dir=f"{training_args_config['output_dir']}/logs",
        logging_steps=10,
        save_steps=500, # Only save checkpoints less frequently
        report_to="none",
        fp16=use_fp16, # Use the conditional flag here
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    train_result = trainer.train()
    training_time = train_result.metrics.get("train_runtime", 0)
    
    print("Training complete.")
    return model, training_time