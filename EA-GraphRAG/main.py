"""
Main Workflow Script for EA-GraphRAG
Efficient and Adaptive GraphRAG framework that integrates query complexity classification, 
dense retrieval, graph retrieval, and fusion.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Optional
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classifier.mlp_classifier import load_classifier, predict_complexity
from classifier.feature_extractor import QueryComplexityFeaturizer
from retrieval.dense_retriever import DenseRetriever
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ComplexityAwareRRF
import stanza
from stanza.server import CoreNLPClient


class EAGraphRAGSystem:
    """
    Main system for EA-GraphRAG: Efficient and Adaptive GraphRAG retrieval routing.
    """
    
    def __init__(self, 
                 classifier_path: str,
                 dataset: str,
                 dense_retriever_name: str = 'BAAI/bge-base-en-v1.5',
                 graph_llm_name: str = 'gpt-4o-mini',
                 graph_embedding_name: str = 'BAAI/bge-base-en-v1.5',
                 corpus_path: Optional[str] = None,
                 top_k: int = 5,
                 damping: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize the EA-GraphRAG system.
        
        Args:
            classifier_path: Path to trained MLP classifier checkpoint
            dataset: Dataset name
            dense_retriever_name: Dense retriever model name
            graph_llm_name: LLM name for graph construction
            graph_embedding_name: Embedding model name for graph
            corpus_path: Path to corpus JSON file
            top_k: Number of documents to retrieve
            damping: Damping factor for graph retrieval
            device: Device for classifier (cpu/cuda)
        """
        self.dataset = dataset
        self.top_k = top_k
        self.device = device
        
        # Load classifier
        print("Loading complexity classifier...")
        self.classifier, metadata = load_classifier(classifier_path, device)
        
        # Initialize featurizer
        self.featurizer = QueryComplexityFeaturizer(
            feature_keys=metadata['feature_keys'],
            scaler_mean=metadata['scaler_mean'],
            scaler_std=metadata['scaler_std'],
            selected_features=metadata['selected_features']
        )
        
        # Initialize retrievers
        print("Initializing retrievers...")
        self.dense_retriever = DenseRetriever(
            dataset=dataset,
            retriever_name=dense_retriever_name,
            corpus_path=corpus_path,
            top_k=top_k
        )
        
        self.graph_retriever = GraphRetriever(
            dataset=dataset,
            llm_model_name=graph_llm_name,
            embedding_model_name=graph_embedding_name,
            corpus_path=corpus_path,
            top_k=top_k,
            damping=damping
        )
        
        # Initialize fusion
        self.fusion = ComplexityAwareRRF(k=60, weight_mode='linear')
        
        # Initialize NLP tools (for feature extraction)
        print("Initializing NLP tools...")
        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        self.corenlp_client = None  # Will be initialized when needed
    
    def _get_corenlp_client(self):
        """Get or create CoreNLP client"""
        if self.corenlp_client is None:
            self.corenlp_client = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse'],
                timeout=30000,
                memory='16G',
                be_quiet=True,
                max_char_length=100000,
                threads=8
            )
        return self.corenlp_client
    
    def retrieve(self, query: str, use_corenlp: bool = False) -> Dict:
        """
        Retrieve documents for a query based on complexity.
        
        Args:
            query: Query string
            use_corenlp: Whether to use CoreNLP for feature extraction (slower but more accurate)
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        # Extract features and predict complexity
        if use_corenlp:
            client = self._get_corenlp_client()
            features = self.featurizer.extract_features(query, client, self.stanza_nlp)
        else:
            # Use only SpaCy features (faster)
            features = self.featurizer.extract_features(query, None, None)
        
        complexity_prob, complexity_level = predict_complexity(self.classifier, features)
        
        # Route based on complexity
        if complexity_level == 'low':
            # Use dense retrieval only
            retrieved_docs, scores = self.dense_retriever.retrieve(query, self.top_k)
            method = 'dense'
            fusion_weight = None
            
        elif complexity_level == 'high':
            # Use graph retrieval only
            retrieved_docs, scores = self.graph_retriever.retrieve(
                query, self.top_k, complexity_score=complexity_prob
            )
            method = 'graph'
            fusion_weight = None
            
        else:  # medium
            # Use fusion
            dense_docs, dense_scores = self.dense_retriever.retrieve(query, self.top_k)
            graph_docs, graph_scores = self.graph_retriever.retrieve(
                query, self.top_k, complexity_score=complexity_prob
            )
            
            # Fuse results
            dense_results = list(zip(dense_docs, dense_scores))
            graph_results = list(zip(graph_docs, graph_scores))
            fused = self.fusion.fuse(dense_results, graph_results, complexity_prob)
            
            retrieved_docs = [doc for doc, score in fused]
            scores = [score for doc, score in fused]
            method = 'fusion'
            fusion_weight = self.fusion.compute_weights(complexity_prob)
        
        return {
            'query': query,
            'complexity_prob': float(complexity_prob),
            'complexity_level': complexity_level,
            'method': method,
            'retrieved_documents': retrieved_docs,
            'scores': [float(s) for s in scores],
            'fusion_weights': fusion_weight,
            'top_k': self.top_k
        }
    
    def retrieve_batch(self, queries: List[str], use_corenlp: bool = False) -> List[Dict]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            use_corenlp: Whether to use CoreNLP
            
        Returns:
            List of retrieval result dictionaries
        """
        results = []
        for query in queries:
            result = self.retrieve(query, use_corenlp)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='EA-GraphRAG: Efficient and Adaptive GraphRAG System')
    
    # Required arguments
    parser.add_argument('--classifier_path', type=str, required=True,
                       help='Path to trained MLP classifier checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to input data file (JSON)')
    
    # Optional arguments
    parser.add_argument('--corpus_path', type=str, default=None,
                       help='Path to corpus JSON file (default: data/{dataset}_corpus.json)')
    parser.add_argument('--dense_retriever', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Dense retriever model name')
    parser.add_argument('--graph_llm', type=str, default='gpt-4o-mini',
                       help='LLM name for graph construction')
    parser.add_argument('--graph_embedding', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Embedding model name for graph')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--damping', type=float, default=0.5,
                       help='Damping factor for graph retrieval')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save results (default: output/{dataset}_results.json)')
    parser.add_argument('--use_corenlp', action='store_true',
                       help='Use CoreNLP for feature extraction (slower but more accurate)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for classifier (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = EAGraphRAGSystem(
        classifier_path=args.classifier_path,
        dataset=args.dataset,
        dense_retriever_name=args.dense_retriever,
        graph_llm_name=args.graph_llm,
        graph_embedding_name=args.graph_embedding,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        damping=args.damping,
        device=args.device
    )
    
    # Load data
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    # Process queries
    print(f"Processing {len(data)} queries...")
    results = []
    for sample in data:
        query = sample.get('question', sample.get('query', ''))
        if not query:
            continue
        
        result = system.retrieve(query, use_corenlp=args.use_corenlp)
        result['sample_id'] = sample.get('id', sample.get('_id', sample.get('idx', '')))
        results.append(result)
    
    # Save results
    if args.output_path is None:
        args.output_path = f'output/{args.dataset}_results.json'
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_path}")
    print(f"Processed {len(results)} queries")
    
    # Print statistics
    method_counts = {}
    for result in results:
        method = result['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("\nRetrieval method distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} ({count/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()

