"""
Graph-based Retrieval Module
Implements graph-based retrieval using entity-fact graph and PPR without external dependencies.
"""
import json
import os
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
import igraph as ig
from collections import defaultdict
import hashlib
import sys
import os

# Add src to path for prompts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.prompts import get_query_instruction, format_query_with_instruction


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID for content"""
    content_str = str(content).lower().strip()
    hash_obj = hashlib.md5(content_str.encode('utf-8'))
    return prefix + hash_obj.hexdigest()


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range"""
    if len(scores) == 0:
        return scores
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score > 1e-8:
        return (scores - min_score) / (max_score - min_score)
    else:
        return np.ones_like(scores) / len(scores)


class GraphRetriever:
    """
    Graph-based retrieval using entity-fact graph and Personalized PageRank.
    Implements graph retrieval without external dependencies.
    """
    
    def __init__(self, dataset: str, 
                 llm_model_name: str = 'gpt-4o-mini',
                 embedding_model_name: str = 'BAAI/bge-base-en-v1.5',
                 corpus_path: Optional[str] = None,
                 graph_path: Optional[str] = None,
                 openie_results_path: Optional[str] = None,
                 top_k: int = 5,
                 damping: float = 0.5,
                 linking_top_k: int = 10):
        """
        Initialize graph retriever.
        
        Args:
            dataset: Dataset name
            llm_model_name: LLM model name (for compatibility, not used directly)
            embedding_model_name: Embedding model name
            corpus_path: Path to corpus JSON file
            graph_path: Path to pre-built graph pickle file
            openie_results_path: Path to OpenIE extraction results
            top_k: Number of documents to retrieve
            damping: Damping factor for PPR
            linking_top_k: Top-k entities to link for graph search
        """
        self.dataset = dataset
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.damping = damping
        self.linking_top_k = linking_top_k
        
        # Load corpus
        if corpus_path is None:
            corpus_path = f'data/{dataset}_corpus.json'
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model.to(self.device)
        
        # Prepare documents
        self.documents = self._prepare_documents()
        
        # Load graph and OpenIE results
        if graph_path is None:
            graph_path = f'data/graphs/{dataset}_graph.pickle'
        if openie_results_path is None:
            openie_results_path = f'data/openie/{dataset}_openie.json'
        
        self.graph_path = graph_path
        self.openie_results_path = openie_results_path
        
        # Load graph structure
        self._load_graph_structure()
        
        # Prepare embeddings
        self._prepare_embeddings()
    
    def _prepare_documents(self) -> List[str]:
        """Prepare document texts from corpus"""
        documents = []
        if isinstance(self.corpus, dict):
            for key, content in self.corpus.items():
                if isinstance(content, list):
                    doc_text = key + '\n' + ''.join(content)
                else:
                    doc_text = key + '\n' + str(content)
                documents.append(doc_text)
        else:
            for item in self.corpus:
                if isinstance(item, dict):
                    doc_text = item.get('title', '') + '\n' + item.get('text', '')
                else:
                    doc_text = str(item)
                documents.append(doc_text)
        return documents
    
    def _load_graph_structure(self):
        """Load graph structure from files"""
        # Try to load pre-built graph
        if os.path.exists(self.graph_path):
            print(f"Loading graph from {self.graph_path}")
            self.graph = ig.Graph.Read_Pickle(self.graph_path)
        else:
            print(f"Graph file not found: {self.graph_path}")
            print("Creating empty graph. Please build graph first using indexing script.")
            self.graph = ig.Graph()
        
        # Load OpenIE results
        if os.path.exists(self.openie_results_path):
            with open(self.openie_results_path, 'r') as f:
                openie_data = json.load(f)
                self.openie_results = openie_data.get('docs', [])
        else:
            print(f"OpenIE results not found: {self.openie_results_path}")
            self.openie_results = []
        
        # Build node mappings
        self._build_node_mappings()
    
    def _build_node_mappings(self):
        """Build mappings between nodes and documents"""
        # Map from node name to vertex index
        if self.graph.vcount() > 0 and 'name' in self.graph.vs.attribute_names():
            self.node_name_to_vertex_idx = {
                node['name']: idx for idx, node in enumerate(self.graph.vs)
            }
        else:
            self.node_name_to_vertex_idx = {}
        
        # Map from entity nodes to chunk IDs
        self.ent_node_to_chunk_ids = defaultdict(set)
        
        # Map from passage node keys to indices
        self.passage_node_keys = []
        self.entity_node_keys = []
        
        for doc in self.openie_results:
            chunk_id = doc.get('idx', compute_mdhash_id(doc.get('passage', ''), 'chunk-'))
            self.passage_node_keys.append(chunk_id)
            
            # Extract entities from triples
            triples = doc.get('extracted_triples', [])
            for triple in triples:
                if len(triple) == 3:
                    subj_entity = compute_mdhash_id(triple[0].lower(), 'entity-')
                    obj_entity = compute_mdhash_id(triple[2].lower(), 'entity-')
                    
                    self.entity_node_keys.append(subj_entity)
                    self.entity_node_keys.append(obj_entity)
                    
                    self.ent_node_to_chunk_ids[subj_entity].add(chunk_id)
                    self.ent_node_to_chunk_ids[obj_entity].add(chunk_id)
        
        # Remove duplicates
        self.entity_node_keys = list(set(self.entity_node_keys))
        self.passage_node_keys = list(set(self.passage_node_keys))
        
        # Get passage node indices in graph
        self.passage_node_idxs = [
            self.node_name_to_vertex_idx.get(key, -1) 
            for key in self.passage_node_keys
            if key in self.node_name_to_vertex_idx
        ]
        self.passage_node_idxs = [idx for idx in self.passage_node_idxs if idx >= 0]
    
    def _prepare_embeddings(self):
        """Prepare embeddings for facts and entities"""
        # For simplicity, we'll compute embeddings on-the-fly
        # In a full implementation, these would be pre-computed and cached
        self.fact_embeddings = None
        self.entity_embeddings = None
        
        # Extract facts from OpenIE results
        self.facts = []
        self.fact_node_keys = []
        
        for doc in self.openie_results:
            triples = doc.get('extracted_triples', [])
            for triple in triples:
                if len(triple) == 3:
                    fact_str = str(triple)
                    fact_key = compute_mdhash_id(fact_str, 'fact-')
                    self.facts.append(triple)
                    self.fact_node_keys.append(fact_key)
    
    def _get_fact_scores(self, query: str) -> np.ndarray:
        """Get similarity scores between query and facts"""
        if len(self.facts) == 0:
            return np.array([])
        
        # Get instruction for query-to-fact encoding
        instruction = get_query_instruction('query_to_fact')
        formatted_query = format_query_with_instruction(query, instruction, self.embedding_model_name)
        
        # Encode query with instruction
        query_embedding = self.embed_model.encode(
            formatted_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Encode facts (no instruction needed for facts)
        fact_texts = [f"{f[0]} {f[1]} {f[2]}" for f in self.facts]
        fact_embeddings = self.embed_model.encode(
            fact_texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Compute similarity
        scores = np.dot(fact_embeddings, query_embedding)
        return min_max_normalize(scores)
    
    def _run_ppr(self, reset_prob: np.ndarray, damping: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run Personalized PageRank on the graph"""
        if damping is None:
            damping = self.damping
        
        if self.graph.vcount() == 0:
            # Fallback to uniform distribution if graph is empty
            doc_scores = np.ones(len(self.passage_node_keys)) / len(self.passage_node_keys)
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            return sorted_doc_ids, doc_scores[sorted_doc_ids]
        
        # Normalize reset probabilities
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        if reset_prob.sum() > 0:
            reset_prob = reset_prob / reset_prob.sum()
        
        # Run PPR
        try:
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(self.graph.vcount()),
                damping=damping,
                directed=False,
                weights='weight' if 'weight' in self.graph.es.attribute_names() else None,
                reset=reset_prob.tolist() if len(reset_prob) == self.graph.vcount() else None,
                implementation='prpack'
            )
            
            # Extract document scores
            doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs 
                                  if idx < len(pagerank_scores)])
            
            if len(doc_scores) == 0:
                doc_scores = np.ones(len(self.passage_node_keys)) / len(self.passage_node_keys)
            
        except Exception as e:
            print(f"Error in PPR computation: {e}")
            doc_scores = np.ones(len(self.passage_node_keys)) / len(self.passage_node_keys)
        
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]
        
        return sorted_doc_ids, sorted_doc_scores
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                 complexity_score: Optional[float] = None) -> Tuple[List[str], List[float]]:
        """
        Retrieve documents using graph-based retrieval.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides initialization)
            complexity_score: Optional complexity score for adaptive PPR damping
            
        Returns:
            Tuple of (retrieved_documents, scores)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Adapt damping based on complexity if provided
        damping = self.damping
        if complexity_score is not None:
            # Map complexity to damping: low complexity -> higher damping, high complexity -> lower damping
            c = max(0.0, min(1.0, float(complexity_score)))
            dmin, dmax = 0.30, 0.55
            m, tau = 0.50, 0.25
            z = 1.0 / (1.0 + math.exp(-(c - m) / max(tau, 1e-6)))
            damping = dmin + (dmax - dmin) * z
            damping = max(0.05, min(0.95, damping))
        
        # Get fact scores
        fact_scores = self._get_fact_scores(query)
        
        if len(fact_scores) == 0 or self.graph.vcount() == 0:
            # Fallback to simple keyword matching if no facts or graph
            return self._fallback_retrieve(query, top_k)
        
        # Get top-k facts
        top_k_fact_indices = np.argsort(fact_scores)[::-1][:self.linking_top_k]
        top_k_facts = [self.facts[idx] for idx in top_k_fact_indices]
        
        # Build phrase weights from facts
        phrase_weights = np.zeros(self.graph.vcount())
        phrase_scores = defaultdict(list)
        
        for rank, fact in enumerate(top_k_facts):
            fact_score = fact_scores[top_k_fact_indices[rank]]
            subj_phrase = fact[0].lower()
            obj_phrase = fact[2].lower()
            
            for phrase in [subj_phrase, obj_phrase]:
                phrase_key = compute_mdhash_id(phrase, 'entity-')
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                
                if phrase_id is not None and phrase_id < len(phrase_weights):
                    # Normalize by number of chunks containing this entity
                    num_chunks = len(self.ent_node_to_chunk_ids.get(phrase_key, set()))
                    if num_chunks > 0:
                        phrase_weights[phrase_id] = fact_score / num_chunks
                    else:
                        phrase_weights[phrase_id] = fact_score
                    
                    phrase_scores[phrase].append(fact_score)
        
        # Normalize phrase weights
        if phrase_weights.sum() > 0:
            phrase_weights = phrase_weights / phrase_weights.sum()
        
        # Run PPR
        sorted_doc_ids, sorted_doc_scores = self._run_ppr(phrase_weights, damping)
        
        # Get top-k documents
        top_indices = sorted_doc_ids[:top_k]
        retrieved_docs = [self.documents[idx] if idx < len(self.documents) else '' 
                         for idx in top_indices]
        retrieved_scores = sorted_doc_scores[:top_k].tolist()
        
        return retrieved_docs, retrieved_scores
    
    def _fallback_retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """Fallback retrieval method when graph is not available"""
        query_words = set(query.lower().split())
        scores = []
        
        for idx, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append((overlap, idx))
        
        scores.sort(reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]
        retrieved = [self.documents[idx] for idx in top_indices]
        retrieval_scores = [float(score) for score, _ in scores[:top_k]]
        
        return retrieved, retrieval_scores
