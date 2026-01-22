"""
Graph Construction Utilities
Builds entity-fact knowledge graph from corpus documents.
"""
import json
import os
import re
import hashlib
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
import igraph as ig
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID for content"""
    content_str = str(content).lower().strip()
    hash_obj = hashlib.md5(content_str.encode('utf-8'))
    return prefix + hash_obj.hexdigest()


def text_processing(text: str) -> str:
    """Basic text processing for normalization"""
    return text.lower().strip()


def extract_entity_nodes(chunk_triples: List[List[Tuple]]) -> Tuple[List[str], List[List[str]]]:
    """
    Extract unique entity nodes from triples.
    
    Args:
        chunk_triples: List of lists of triples, one per chunk
        
    Returns:
        Tuple of (unique_entities, chunk_triple_entities)
    """
    all_entities = set()
    chunk_triple_entities = []
    
    for triples in chunk_triples:
        chunk_entities = []
        for triple in triples:
            if len(triple) == 3:
                subj = text_processing(triple[0])
                obj = text_processing(triple[2])
                all_entities.add(subj)
                all_entities.add(obj)
                chunk_entities.extend([subj, obj])
        chunk_triple_entities.append(list(set(chunk_entities)))
    
    return list(all_entities), chunk_triple_entities


def flatten_facts(chunk_triples: List[List[Tuple]]) -> List[Tuple]:
    """Flatten nested list of triples into single list"""
    facts = []
    for triples in chunk_triples:
        for triple in triples:
            if len(triple) == 3:
                facts.append(triple)
    return facts


class GraphBuilder:
    """
    Builds entity-fact knowledge graph from corpus.
    """
    
    def __init__(self, embedding_model_name: str = 'BAAI/bge-base-en-v1.5',
                 synonymy_threshold: float = 0.8,
                 synonymy_top_k: int = 10):
        """
        Initialize graph builder.
        
        Args:
            embedding_model_name: Embedding model for synonymy detection
            synonymy_threshold: Similarity threshold for synonymy edges
            synonymy_top_k: Top-k similar entities to consider
        """
        self.embedding_model_name = embedding_model_name
        self.synonymy_threshold = synonymy_threshold
        self.synonymy_top_k = synonymy_top_k
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model.to(self.device)
        
        # Graph structure
        self.graph = ig.Graph()
        self.node_to_node_stats = {}  # (node1, node2) -> weight
        self.ent_node_to_chunk_ids = defaultdict(set)
    
    def build_from_openie_results(self, openie_results: List[Dict], 
                                  output_graph_path: str,
                                  output_openie_path: str = None):
        """
        Build graph from OpenIE extraction results.
        
        Args:
            openie_results: List of OpenIE results with 'idx', 'passage', 'extracted_triples'
            output_graph_path: Path to save the graph
            output_openie_path: Optional path to save OpenIE results
        """
        print(f"Building graph from {len(openie_results)} documents...")
        
        # Save OpenIE results if path provided
        if output_openie_path:
            os.makedirs(os.path.dirname(output_openie_path), exist_ok=True)
            with open(output_openie_path, 'w') as f:
                json.dump({'docs': openie_results}, f, indent=2, ensure_ascii=False)
            print(f"Saved OpenIE results to {output_openie_path}")
        
        # Extract chunk IDs and triples
        chunk_ids = []
        chunk_triples = []
        
        for doc in openie_results:
            chunk_id = doc.get('idx', compute_mdhash_id(doc.get('passage', ''), 'chunk-'))
            chunk_ids.append(chunk_id)
            
            triples = doc.get('extracted_triples', [])
            processed_triples = [[text_processing(t) for t in triple] if isinstance(triple, (list, tuple)) else triple 
                                for triple in triples]
            chunk_triples.append(processed_triples)
        
        # Extract entities
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        print(f"Extracted {len(entity_nodes)} unique entities")
        
        # Add fact edges (entity-entity connections from triples)
        print("Adding fact edges...")
        self._add_fact_edges(chunk_ids, chunk_triples)
        
        # Add passage edges (chunk-entity connections)
        # Note: We skip passage nodes as per requirements
        # self._add_passage_edges(chunk_ids, chunk_triple_entities)
        
        # Add synonymy edges
        print("Adding synonymy edges...")
        self._add_synonymy_edges(entity_nodes)
        
        # Build graph structure
        print("Constructing graph structure...")
        self._build_graph_structure(entity_nodes, chunk_ids)
        
        # Save graph
        os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
        self.graph.write_pickle(output_graph_path)
        print(f"Graph saved to {output_graph_path}")
        print(f"Graph info: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
    
    def _add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[List[Tuple]]):
        """Add edges from triples (entity-entity connections)"""
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            entities_in_chunk = set()
            
            for triple in triples:
                if len(triple) == 3:
                    subj = text_processing(triple[0])
                    obj = text_processing(triple[2])
                    
                    node_key = compute_mdhash_id(subj, 'entity-')
                    node_2_key = compute_mdhash_id(obj, 'entity-')
                    
                    # Add bidirectional edge
                    self.node_to_node_stats[(node_key, node_2_key)] = \
                        self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1.0
                    self.node_to_node_stats[(node_2_key, node_key)] = \
                        self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1.0
                    
                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
            
            # Track which chunks contain each entity
            for node in entities_in_chunk:
                self.ent_node_to_chunk_ids[node].add(chunk_key)
    
    def _add_synonymy_edges(self, entity_nodes: List[str]):
        """Add synonymy edges between similar entities"""
        if len(entity_nodes) < 2:
            return
        
        print(f"Computing embeddings for {len(entity_nodes)} entities...")
        # Compute embeddings for entities
        entity_embeddings = self.embed_model.encode(
            entity_nodes,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Build entity to embedding mapping
        entity_to_emb = {entity: emb for entity, emb in zip(entity_nodes, entity_embeddings)}
        
        print("Finding similar entities...")
        # Find similar entities using cosine similarity
        entity_array = np.array(entity_embeddings)
        
        # Compute similarity matrix in batches
        batch_size = 1000
        num_synonym_edges = 0
        
        for i in tqdm(range(0, len(entity_nodes), batch_size), desc="Computing similarities"):
            batch_entities = entity_nodes[i:i+batch_size]
            batch_embs = entity_array[i:i+batch_size]
            
            # Compute similarity with all entities
            similarities = np.dot(batch_embs, entity_array.T)
            
            for j, entity in enumerate(batch_entities):
                entity_key = compute_mdhash_id(entity, 'entity-')
                
                # Get top-k similar entities
                sim_scores = similarities[j]
                top_indices = np.argsort(sim_scores)[::-1][:self.synonymy_top_k + 1]  # +1 to exclude self
                
                for idx in top_indices:
                    if idx == i + j:  # Skip self
                        continue
                    
                    similar_entity = entity_nodes[idx]
                    similar_key = compute_mdhash_id(similar_entity, 'entity-')
                    sim_score = float(sim_scores[idx])
                    
                    # Add edge if similarity exceeds threshold
                    if sim_score >= self.synonymy_threshold and entity_key != similar_key:
                        # Check if edge already exists (from fact edges)
                        if (entity_key, similar_key) not in self.node_to_node_stats:
                            self.node_to_node_stats[(entity_key, similar_key)] = sim_score
                            self.node_to_node_stats[(similar_key, entity_key)] = sim_score
                            num_synonym_edges += 1
        
        print(f"Added {num_synonym_edges} synonymy edges")
    
    def _build_graph_structure(self, entity_nodes: List[str], chunk_ids: List[str]):
        """Build igraph structure from node_to_node_stats"""
        # Collect all node keys
        all_node_keys = set()
        for edge in self.node_to_node_stats.keys():
            all_node_keys.add(edge[0])
            all_node_keys.add(edge[1])
        
        # Add chunk IDs (passage nodes) - but we skip them as per requirements
        # all_node_keys.update([compute_mdhash_id(chunk_id, 'chunk-') for chunk_id in chunk_ids])
        
        all_node_keys = sorted(list(all_node_keys))
        
        # Create graph with nodes
        num_nodes = len(all_node_keys)
        self.graph.add_vertices(num_nodes)
        
        # Set node names
        self.graph.vs['name'] = all_node_keys
        
        # Create node key to index mapping
        node_key_to_idx = {key: idx for idx, key in enumerate(all_node_keys)}
        
        # Add edges
        edges = []
        edge_weights = []
        
        for (node1, node2), weight in self.node_to_node_stats.items():
            if node1 == node2:  # Skip self-loops
                continue
            
            if node1 in node_key_to_idx and node2 in node_key_to_idx:
                idx1 = node_key_to_idx[node1]
                idx2 = node_key_to_idx[node2]
                edges.append((idx1, idx2))
                edge_weights.append(float(weight))
        
        # Add edges to graph
        if len(edges) > 0:
            self.graph.add_edges(edges, attributes={'weight': edge_weights})
        
        print(f"Graph structure: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")


def build_graph_from_corpus(corpus_path: str,
                            openie_results_path: str,
                            output_graph_path: str,
                            embedding_model_name: str = 'BAAI/bge-base-en-v1.5',
                            synonymy_threshold: float = 0.8,
                            synonymy_top_k: int = 10):
    """
    Build knowledge graph from corpus and OpenIE results.
    
    Args:
        corpus_path: Path to corpus JSON file
        openie_results_path: Path to OpenIE extraction results JSON
        output_graph_path: Path to save the built graph
        embedding_model_name: Embedding model for synonymy detection
        synonymy_threshold: Similarity threshold for synonymy edges
        synonymy_top_k: Top-k similar entities to consider
    """
    # Load OpenIE results
    with open(openie_results_path, 'r') as f:
        openie_data = json.load(f)
        openie_results = openie_data.get('docs', [])
    
    # Build graph
    builder = GraphBuilder(
        embedding_model_name=embedding_model_name,
        synonymy_threshold=synonymy_threshold,
        synonymy_top_k=synonymy_top_k
    )
    
    builder.build_from_openie_results(
        openie_results=openie_results,
        output_graph_path=output_graph_path,
        output_openie_path=openie_results_path  # Save updated results
    )

