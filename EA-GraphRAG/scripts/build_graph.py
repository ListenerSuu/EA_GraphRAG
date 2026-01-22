"""
Script to build knowledge graph from corpus and OpenIE results.
"""
import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.graph_builder import build_graph_from_corpus


def main():
    parser = argparse.ArgumentParser(description='Build knowledge graph from corpus')
    
    parser.add_argument('--corpus_path', type=str, required=True,
                       help='Path to corpus JSON file')
    parser.add_argument('--openie_results_path', type=str, required=True,
                       help='Path to OpenIE extraction results JSON')
    parser.add_argument('--output_graph_path', type=str, required=True,
                       help='Path to save the built graph (pickle format)')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Embedding model for synonymy detection')
    parser.add_argument('--synonymy_threshold', type=float, default=0.8,
                       help='Similarity threshold for synonymy edges')
    parser.add_argument('--synonymy_top_k', type=int, default=10,
                       help='Top-k similar entities to consider for synonymy')
    
    args = parser.parse_args()
    
    # Build graph
    build_graph_from_corpus(
        corpus_path=args.corpus_path,
        openie_results_path=args.openie_results_path,
        output_graph_path=args.output_graph_path,
        embedding_model_name=args.embedding_model,
        synonymy_threshold=args.synonymy_threshold,
        synonymy_top_k=args.synonymy_top_k
    )
    
    print("Graph construction completed!")


if __name__ == '__main__':
    main()

