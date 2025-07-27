from datasets import load_dataset

def load_and_preprocess_data(dataset_name, split):
    """
    Loads a dataset from Hugging Face and performs minimal preprocessing.
    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        split (str): The dataset split to use (e.g., 'train', 'train[:5000]').
    Returns:
        A Hugging Face Dataset object.
    """
    print(f"Loading dataset '{dataset_name}' with split '{split}'...")
    dataset = load_dataset(dataset_name, split=split)
    
    # We need to ensure the columns are named consistently.
    # SQuAD has 'question', 'context', and 'answers' (with a nested 'text' field)
    def preprocess_function(examples):
        # Extract the first answer text
        answers = [ans['text'][0] if len(ans['text']) > 0 else "" for ans in examples['answers']]
        return {
            'question': examples['question'],
            'context': examples['context'],
            'answer': answers
        }

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset preprocessed.")
    return dataset