"""
Dense Passage Retrieval (DPR) Module
Implements dense retrieval using BGE embeddings without external dependencies.
"""
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
import sys
import os

# Add src to path for prompts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.prompts import get_query_instruction, format_query_with_instruction


class DenseRetriever:
    """
    Dense Passage Retrieval using BGE embeddings.
    Implements dense retrieval without external dependencies.
    """
    
    def __init__(self, dataset: str, retriever_name: str = 'BAAI/bge-base-en-v1.5',
                 corpus_path: Optional[str] = None, top_k: int = 5,
                 embedding_cache_dir: Optional[str] = None):
        """
        Initialize DPR retriever.
        
        Args:
            dataset: Dataset name
            retriever_name: HuggingFace model name for embeddings
            corpus_path: Path to corpus JSON file
            top_k: Number of documents to retrieve
            embedding_cache_dir: Directory to cache document embeddings
        """
        self.dataset = dataset
        self.retriever_name = retriever_name
        self.top_k = top_k
        self.embedding_cache_dir = embedding_cache_dir or f'data/embeddings/{dataset}'
        
        # Load corpus
        if corpus_path is None:
            corpus_path = f'data/{dataset}_corpus.json'
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        # Initialize embedding model
        print(f"Loading embedding model: {retriever_name}")
        self.embed_model = SentenceTransformer(retriever_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model.to(self.device)
        
        # Prepare documents
        self.documents = self._prepare_documents()
        
        # Load or compute document embeddings
        self.doc_embeddings = self._load_or_compute_doc_embeddings()
    
    def _prepare_documents(self) -> List[str]:
        """Prepare document texts from corpus"""
        documents = []
        if isinstance(self.corpus, dict):
            # HotpotQA-style format
            for key, content in self.corpus.items():
                if isinstance(content, list):
                    doc_text = key + '\n' + ''.join(content)
                else:
                    doc_text = key + '\n' + str(content)
                documents.append(doc_text)
        else:
            # List format
            for item in self.corpus:
                if isinstance(item, dict):
                    doc_text = item.get('title', '') + '\n' + item.get('text', '')
                else:
                    doc_text = str(item)
                documents.append(doc_text)
        return documents
    
    def _load_or_compute_doc_embeddings(self) -> np.ndarray:
        """Load cached embeddings or compute new ones"""
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        cache_path = os.path.join(self.embedding_cache_dir, 
                                  f'doc_embeddings_{self.retriever_name.replace("/", "_")}.npy')
        
        if os.path.exists(cache_path):
            print(f"Loading cached document embeddings from {cache_path}")
            return np.load(cache_path)
        else:
            print(f"Computing document embeddings for {len(self.documents)} documents...")
            embeddings = self.embed_model.encode(
                self.documents,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            np.save(cache_path, embeddings)
            print(f"Saved embeddings to {cache_path}")
            return embeddings
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using dense retrieval.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides initialization)
            
        Returns:
            Tuple of (retrieved_documents, scores)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Get instruction for query-to-passage encoding
        instruction = get_query_instruction('query_to_passage')
        formatted_query = format_query_with_instruction(query, instruction, self.retriever_name)
        
        # Encode query with instruction
        query_embedding = self.embed_model.encode(
            formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Compute similarity scores
        scores = np.dot(self.doc_embeddings, query_embedding)
        
        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_docs = [self.documents[idx] for idx in top_indices]
        retrieved_scores = scores[top_indices].tolist()
        
        return retrieved_docs, retrieved_scores
