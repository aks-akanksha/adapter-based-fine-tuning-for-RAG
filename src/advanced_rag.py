"""
Advanced RAG pipeline with re-ranking, multi-query retrieval, and hybrid search.
"""
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from typing import List, Dict, Optional
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank-bm25 not available. Sparse retrieval will be disabled.")
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class AdvancedRAGRetriever:
    """
    Advanced RAG retriever with multiple retrieval strategies.
    """
    def __init__(self, index, documents, vectorizer, reranker=None, bm25_index=None):
        self.index = index
        self.documents = documents
        self.vectorizer = vectorizer
        self.reranker = reranker
        self.bm25_index = bm25_index
    
    def retrieve(self, query: str, top_k: int = 3, strategy: str = "dense") -> List[str]:
        """
        Retrieve documents using specified strategy.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            strategy: Retrieval strategy ('dense', 'sparse', 'hybrid', 'reranked')
        
        Returns:
            List of retrieved document strings
        """
        if strategy == "dense":
            return self._dense_retrieval(query, top_k)
        elif strategy == "sparse":
            return self._sparse_retrieval(query, top_k)
        elif strategy == "hybrid":
            return self._hybrid_retrieval(query, top_k)
        elif strategy == "reranked":
            return self._reranked_retrieval(query, top_k)
        else:
            return self._dense_retrieval(query, top_k)
    
    def _dense_retrieval(self, query: str, top_k: int) -> List[str]:
        """Dense retrieval using semantic similarity."""
        query_vector = self.vectorizer.encode([query])
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), 
            min(top_k * 2, len(self.documents))  # Get more for reranking
        )
        return [self.documents[i] for i in indices[0][:top_k]]
    
    def _sparse_retrieval(self, query: str, top_k: int) -> List[str]:
        """Sparse retrieval using BM25."""
        if self.bm25_index is None or not BM25_AVAILABLE:
            return self._dense_retrieval(query, top_k)  # Fallback
        
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]
    
    def _hybrid_retrieval(self, query: str, top_k: int, alpha: float = 0.5) -> List[str]:
        """
        Hybrid retrieval combining dense and sparse methods.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        # Get candidates from both methods
        dense_candidates = self._dense_retrieval(query, top_k * 2)
        sparse_candidates = self._sparse_retrieval(query, top_k * 2)
        
        # Combine and score
        all_candidates = list(set(dense_candidates + sparse_candidates))
        
        if len(all_candidates) == 0:
            return []
        
        # Score each candidate
        candidate_scores = {}
        
        # Dense scores
        if dense_candidates:
            dense_embeddings = self.vectorizer.encode(dense_candidates)
            query_embedding = self.vectorizer.encode([query])[0]
            dense_scores = np.dot(dense_embeddings, query_embedding)
            for doc, score in zip(dense_candidates, dense_scores):
                candidate_scores[doc] = candidate_scores.get(doc, 0) + alpha * score
        
        # Sparse scores
        if sparse_candidates and self.bm25_index:
            tokenized_query = word_tokenize(query.lower())
            for doc in sparse_candidates:
                doc_tokens = word_tokenize(doc.lower())
                # Simple overlap score
                overlap = len(set(tokenized_query) & set(doc_tokens))
                candidate_scores[doc] = candidate_scores.get(doc, 0) + (1 - alpha) * overlap
        
        # Sort by combined score
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [doc for doc, _ in sorted_candidates[:top_k]]
    
    def _reranked_retrieval(self, query: str, top_k: int) -> List[str]:
        """
        Retrieval with cross-encoder re-ranking.
        """
        # First, get more candidates using dense retrieval
        candidates = self._dense_retrieval(query, top_k * 3)
        
        if not candidates:
            return []
        
        # Re-rank using cross-encoder if available
        if self.reranker:
            try:
                # Create query-document pairs
                pairs = [[query, doc] for doc in candidates]
                
                # Get reranking scores
                scores = self.reranker.predict(pairs)
                
                # Sort by reranking scores
                ranked = sorted(
                    zip(candidates, scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                return [doc for doc, _ in ranked[:top_k]]
            except Exception as e:
                print(f"⚠️  Reranking failed: {e}, using original ranking")
        
        # Fallback to original ranking
        return candidates[:top_k]


def build_advanced_rag_pipeline(
    knowledge_base_texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    enable_reranking: bool = True,
    enable_sparse: bool = True,
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
) -> AdvancedRAGRetriever:
    """
    Build an advanced RAG pipeline with multiple retrieval strategies.
    
    Args:
        knowledge_base_texts: List of documents for the knowledge base
        model_name: SentenceTransformer model for dense retrieval
        enable_reranking: Whether to enable cross-encoder reranking
        enable_sparse: Whether to enable BM25 sparse retrieval
        reranker_model: Cross-encoder model for reranking
    
    Returns:
        AdvancedRAGRetriever object
    """
    print(f"Building advanced RAG pipeline...")
    print(f"  - Dense model: {model_name}")
    print(f"  - Reranking: {'Enabled' if enable_reranking else 'Disabled'}")
    print(f"  - Sparse (BM25): {'Enabled' if enable_sparse else 'Disabled'}")
    
    # Build dense index
    print("Initializing dense retriever...")
    vectorizer = SentenceTransformer(model_name)
    
    print("Encoding knowledge base...")
    embeddings = vectorizer.encode(knowledge_base_texts, show_progress_bar=True)
    
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    
    print(f"✅ Dense index built with {index.ntotal} documents.")
    
    # Build sparse index (BM25)
    bm25_index = None
    if enable_sparse and BM25_AVAILABLE:
        print("Building BM25 sparse index...")
        try:
            tokenized_docs = [word_tokenize(doc.lower()) for doc in knowledge_base_texts]
            bm25_index = BM25Okapi(tokenized_docs)
            print(f"✅ BM25 index built.")
        except Exception as e:
            print(f"⚠️  BM25 index build failed: {e}")
    elif enable_sparse and not BM25_AVAILABLE:
        print("⚠️  BM25 not available. Install with: pip install rank-bm25")
    
    # Load reranker
    reranker = None
    if enable_reranking:
        print(f"Loading reranker model: {reranker_model}...")
        try:
            reranker = CrossEncoder(reranker_model)
            print("✅ Reranker loaded.")
        except Exception as e:
            print(f"⚠️  Reranker loading failed: {e}")
            print("   Continuing without reranking...")
    
    return AdvancedRAGRetriever(
        index=index,
        documents=knowledge_base_texts,
        vectorizer=vectorizer,
        reranker=reranker,
        bm25_index=bm25_index
    )


def multi_query_retrieval(
    retriever: AdvancedRAGRetriever,
    query: str,
    top_k: int = 3,
    num_queries: int = 3
) -> List[str]:
    """
    Generate multiple query variations and retrieve from all.
    
    Args:
        retriever: AdvancedRAGRetriever instance
        query: Original query
        top_k: Number of documents to return
        num_queries: Number of query variations to generate
    
    Returns:
        List of retrieved documents
    """
    # Simple query expansion (can be enhanced with LLM)
    query_variations = [query]  # Start with original
    
    # Add simple variations
    words = query.lower().split()
    if len(words) > 1:
        # Add variations by reordering or removing stop words
        query_variations.append(" ".join(words[::-1]))  # Reverse order
        if len(words) > 2:
            query_variations.append(" ".join(words[1:]))  # Remove first word
    
    # Retrieve for each variation
    all_results = []
    for q in query_variations[:num_queries]:
        results = retriever.retrieve(q, top_k=top_k, strategy="dense")
        all_results.extend(results)
    
    # Deduplicate while preserving order
    seen = set()
    unique_results = []
    for doc in all_results:
        if doc not in seen:
            seen.add(doc)
            unique_results.append(doc)
    
    return unique_results[:top_k]

