"""
Complexity-Aware Reciprocal Rank Fusion (RRF) for EA-GraphRAG
Fuses results from dense and graph-based retrieval based on query complexity.
"""
from typing import List, Dict, Tuple
import numpy as np


class ComplexityAwareRRF:
    """
    Complexity-Aware Reciprocal Rank Fusion.
    Combines dense and graph retrieval results with weights based on query complexity.
    """
    
    def __init__(self, k: int = 60, weight_mode: str = 'linear'):
        """
        Initialize complexity-aware RRF.
        
        Args:
            k: RRF constant (default 60)
            weight_mode: Weighting mode ('linear', 'exponential', 'sigmoid')
        """
        self.k = k
        self.weight_mode = weight_mode
    
    def compute_weights(self, complexity_prob: float) -> Tuple[float, float]:
        """
        Compute weights for dense and graph retrieval based on complexity.
        
        Args:
            complexity_prob: Probability that query is complex (0-1)
            
        Returns:
            Tuple of (dense_weight, graph_weight)
        """
        if self.weight_mode == 'linear':
            # Linear interpolation: low complexity -> more weight on dense
            graph_weight = complexity_prob
            dense_weight = 1.0 - complexity_prob
        elif self.weight_mode == 'exponential':
            # Exponential weighting
            graph_weight = complexity_prob ** 2
            dense_weight = (1.0 - complexity_prob) ** 2
        elif self.weight_mode == 'sigmoid':
            # Sigmoid-based weighting
            graph_weight = 1.0 / (1.0 + np.exp(-5.0 * (complexity_prob - 0.5)))
            dense_weight = 1.0 - graph_weight
        else:
            # Default: linear
            graph_weight = complexity_prob
            dense_weight = 1.0 - complexity_prob
        
        # Normalize to sum to 1
        total = dense_weight + graph_weight
        if total > 0:
            dense_weight /= total
            graph_weight /= total
        
        return dense_weight, graph_weight
    
    def fuse(self, dense_results: List[Tuple[str, float]], 
             graph_results: List[Tuple[str, float]],
             complexity_prob: float) -> List[Tuple[str, float]]:
        """
        Fuse dense and graph retrieval results using complexity-aware RRF.
        
        Args:
            dense_results: List of (document, score) from dense retrieval
            graph_results: List of (document, score) from graph retrieval
            complexity_prob: Complexity probability (0-1)
            
        Returns:
            Fused list of (document, score) sorted by score
        """
        # Compute weights
        dense_weight, graph_weight = self.compute_weights(complexity_prob)
        
        # Create document -> RRF score mapping
        doc_scores = {}
        
        # Add dense retrieval results
        for rank, (doc, score) in enumerate(dense_results, start=1):
            rrf_score = dense_weight / (self.k + rank)
            if doc not in doc_scores:
                doc_scores[doc] = 0.0
            doc_scores[doc] += rrf_score
        
        # Add graph retrieval results
        for rank, (doc, score) in enumerate(graph_results, start=1):
            rrf_score = graph_weight / (self.k + rank)
            if doc not in doc_scores:
                doc_scores[doc] = 0.0
            doc_scores[doc] += rrf_score
        
        # Sort by score
        fused_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def fuse_with_ranks(self, dense_docs: List[str], graph_docs: List[str],
                       complexity_prob: float) -> List[str]:
        """
        Fuse retrieval results given only document lists (no scores).
        
        Args:
            dense_docs: List of documents from dense retrieval (ordered by rank)
            graph_docs: List of documents from graph retrieval (ordered by rank)
            complexity_prob: Complexity probability (0-1)
            
        Returns:
            Fused list of documents sorted by RRF score
        """
        # Convert to (doc, score) format with dummy scores
        dense_results = [(doc, 1.0) for doc in dense_docs]
        graph_results = [(doc, 1.0) for doc in graph_docs]
        
        fused = self.fuse(dense_results, graph_results, complexity_prob)
        return [doc for doc, score in fused]

