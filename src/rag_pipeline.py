import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class FaissRetriever:
    """A simple retriever using FAISS and SentenceTransformers."""
    def __init__(self, index, documents, vectorizer):
        self.index = index
        self.documents = documents
        self.vectorizer = vectorizer

    def retrieve(self, query, top_k):
        """Retrieves the top_k most relevant documents for a query."""
        query_vector = self.vectorizer.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        return [self.documents[i] for i in indices[0]]

def build_rag_pipeline(knowledge_base_texts, model_name='all-MiniLM-L6-v2'):
    """
    Builds a FAISS index from the knowledge base texts.
    Args:
        knowledge_base_texts (list[str]): A list of documents for the knowledge base.
        model_name (str): The SentenceTransformer model to use for vectorization.
    Returns:
        A FaissRetriever object.
    """
    print(f"Initializing vectorizer with '{model_name}'...")
    vectorizer = SentenceTransformer(model_name)
    
    print("Encoding knowledge base... (This might take a while)")
    embeddings = vectorizer.encode(knowledge_base_texts, show_progress_bar=True)
    
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    
    print(f"FAISS index built with {index.ntotal} documents.")
    
    return FaissRetriever(index, knowledge_base_texts, vectorizer)
