"""
Query Complexity Feature Extractor
Extracts syntactic, semantic, and dependency-based features from queries.
"""
import spacy
import stanza
from stanza.server import CoreNLPClient
from collections import defaultdict, Counter
from typing import Dict, List
import numpy as np

try:
    from .syntactic_features import SyntacticComplexityAnalyzer
except ImportError:
    SyntacticComplexityAnalyzer = None

try:
    from .advanced_features import extract_advanced_features
except ImportError:
    def extract_advanced_features(query):
        return {}


class QueryComplexityFeaturizer:
    """
    Comprehensive featurizer for query complexity analysis.
    Combines syntactic, dependency, semantic, and lexical features.
    """
    
    def __init__(self, feature_keys: List[str] = None, 
                 scaler_mean: np.ndarray = None,
                 scaler_std: np.ndarray = None,
                 selected_features: np.ndarray = None):
        """
        Initialize featurizer.
        
        Args:
            feature_keys: Ordered list of feature names (from training)
            scaler_mean: Mean for normalization (from training)
            scaler_std: Std for normalization (from training)
            selected_features: Indices of selected features (from feature selection)
        """
        self.feature_keys = feature_keys
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.selected_features = selected_features
        
        # Initialize analyzers
        self.syntactic_analyzer = SyntacticComplexityAnalyzer()
        self.spacy_nlp = spacy.load("en_core_web_md")
        
    def extract_features(self, query: str, client: CoreNLPClient = None, 
                        stanza_nlp = None) -> np.ndarray:
        """
        Extract all features for a query.
        
        Args:
            query: Input query string
            client: CoreNLP client (optional, will create if needed)
            stanza_nlp: Stanza pipeline (optional, will create if needed)
            
        Returns:
            Normalized feature vector
        """
        # Extract advanced features (dependency, semantic, lexical)
        feat_dict_advanced = extract_advanced_features(query)
        
        # Extract syntactic features (if client and nlp provided)
        if client is not None and stanza_nlp is not None:
            feat_dict_syntactic = self.syntactic_analyzer.analyze_text(query, client, stanza_nlp)
        else:
            # Use defaults if not available
            feat_dict_syntactic = {}
        
        # Combine features
        feat_dict = {**feat_dict_advanced, **feat_dict_syntactic}
        
        # Convert to vector
        if self.feature_keys is not None:
            feature_vector = np.array([float(feat_dict.get(k, 0.0)) for k in self.feature_keys], 
                                     dtype=np.float32)
        else:
            # If no feature keys, use all features in sorted order
            feature_vector = np.array([float(v) for k, v in sorted(feat_dict.items())], 
                                     dtype=np.float32)
        
        # Normalize
        if self.scaler_mean is not None and self.scaler_std is not None:
            feature_vector = (feature_vector - self.scaler_mean) / self.scaler_std
            # Handle division by zero
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply feature selection
        if self.selected_features is not None:
            feature_vector = feature_vector[self.selected_features]
        
        return feature_vector
    
    def extract_features_batch(self, queries: List[str], 
                              client: CoreNLPClient = None,
                              stanza_nlp = None) -> np.ndarray:
        """
        Extract features for multiple queries.
        
        Args:
            queries: List of query strings
            client: CoreNLP client
            stanza_nlp: Stanza pipeline
            
        Returns:
            Feature matrix (n_queries, n_features)
        """
        features_list = []
        for query in queries:
            feat_vec = self.extract_features(query, client, stanza_nlp)
            features_list.append(feat_vec)
        return np.stack(features_list, axis=0)

