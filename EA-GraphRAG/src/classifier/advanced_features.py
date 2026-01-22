"""
Advanced feature extraction for query complexity classification.
Adds graph-aware features, semantic features, and dependency-based features.
"""
import spacy
import re
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import math

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

WH_WORDS = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 
             'whether', 'whatever', 'whenever', 'wherever', 'whichever', 'whoever', 'whomever'}

# Logical connectors that indicate multi-hop reasoning
LOGICAL_CONNECTORS = {
    'and', 'or', 'but', 'then', 'after', 'before', 'when', 'while', 'where', 
    'both', 'either', 'neither', 'also', 'as well', 'furthermore', 'moreover',
    'however', 'therefore', 'thus', 'hence', 'consequently', 'whereas', 'although',
    'because', 'since', 'if', 'unless', 'until', 'during', 'through'
}

def is_wh_word(token):
    return token.text.lower() in WH_WORDS

def is_logical_connector(token):
    return token.text.lower() in LOGICAL_CONNECTORS

def tree_depth(node):
    """Calculate maximum depth of dependency tree."""
    if not list(node.children):
        return 1
    return 1 + max(tree_depth(child) for child in node.children)

def tree_width(node):
    """Calculate maximum width of dependency tree."""
    if not list(node.children):
        return 0
    width = len(list(node.children))
    return max(width, max(tree_width(child) for child in node.children))

def count_nodes(node):
    """Count leaf and non-leaf nodes."""
    if not list(node.children):
        return (1, 0)
    else:
        leaf_count = 0
        non_leaf_count = 1
        for child in node.children:
            child_leaf, child_non_leaf = count_nodes(child)
            leaf_count += child_leaf
            non_leaf_count += child_non_leaf
        return (leaf_count, non_leaf_count)

def count_clauses(doc):
    """Count clauses (finite verbs)."""
    clause_count = 0
    for sent in doc.sents:
        clause_count += sum(1 for token in sent if token.pos_ == "VERB" and token.dep_ != "aux")
    return clause_count

def extract_dependency_features(doc):
    """Extract dependency parsing features."""
    features = {}
    
    if len(doc) == 0:
        return {
            'max_dependency_distance': 0,
            'avg_dependency_distance': 0,
            'num_long_dependencies': 0,
            'num_subject_verb_relations': 0,
            'num_object_verb_relations': 0,
            'num_modifier_relations': 0,
            'num_coordination_relations': 0,
            'num_subordination_relations': 0,
            'dependency_tree_imbalance': 0
        }
    
    # Calculate dependency distances
    distances = []
    long_deps = 0
    subj_verb = 0
    obj_verb = 0
    modifiers = 0
    coordinations = 0
    subordinations = 0
    
    for token in doc:
        if token.head != token:  # Not root
            distance = abs(token.i - token.head.i)
            distances.append(distance)
            if distance > 5:
                long_deps += 1
            
            # Relation type counts
            if token.dep_ in ['nsubj', 'nsubjpass']:
                subj_verb += 1
            elif token.dep_ in ['dobj', 'pobj']:
                obj_verb += 1
            elif token.dep_ in ['amod', 'advmod', 'nmod']:
                modifiers += 1
            elif token.dep_ == 'conj':
                coordinations += 1
            elif token.dep_ in ['mark', 'advcl', 'acl']:
                subordinations += 1
    
    features['max_dependency_distance'] = max(distances) if distances else 0
    features['avg_dependency_distance'] = np.mean(distances) if distances else 0
    features['num_long_dependencies'] = long_deps
    features['num_subject_verb_relations'] = subj_verb
    features['num_object_verb_relations'] = obj_verb
    features['num_modifier_relations'] = modifiers
    features['num_coordination_relations'] = coordinations
    features['num_subordination_relations'] = subj_verb
    
    # Tree imbalance (left vs right children)
    root = doc[:].root
    left_children = sum(1 for child in root.children if child.i < root.i)
    right_children = sum(1 for child in root.children if child.i > root.i)
    total_children = left_children + right_children
    if total_children > 0:
        features['dependency_tree_imbalance'] = abs(left_children - right_children) / total_children
    else:
        features['dependency_tree_imbalance'] = 0
    
    return features

def extract_semantic_role_features(doc):
    """Extract semantic role-like features from dependency parsing."""
    features = {}
    
    # Count different semantic roles based on dependency labels
    role_counts = Counter([token.dep_ for token in doc])
    
    # Agent/Subject indicators
    features['num_agents'] = role_counts.get('nsubj', 0) + role_counts.get('nsubjpass', 0)
    
    # Patient/Object indicators
    features['num_patients'] = role_counts.get('dobj', 0) + role_counts.get('pobj', 0)
    
    # Temporal indicators
    features['num_temporal'] = role_counts.get('npadvmod', 0) + sum(1 for token in doc 
                                                                    if token.ent_type_ in ['DATE', 'TIME'])
    
    # Locative indicators
    features['num_locative'] = sum(1 for token in doc if token.ent_type_ in ['GPE', 'LOC'])
    
    # Instrumental/Causal indicators
    features['num_causal'] = role_counts.get('agent', 0) + role_counts.get('prep', 0)
    
    return features

def extract_advanced_features(question: str) -> Dict[str, float]:
    """
    Extract comprehensive features including syntactic, semantic, and graph-aware features.
    """
    doc = nlp(question)
    
    features = {}
    
    # ========== Basic Length Features ==========
    features['length'] = len(question)
    features['num_tokens'] = len(doc)
    features['num_chars'] = len(question.replace(' ', ''))
    features['avg_token_length'] = features['num_chars'] / max(features['num_tokens'], 1)
    features['num_sentences'] = len(list(doc.sents))
    
    # ========== POS Tag Features ==========
    pos_counts = Counter([token.pos_ for token in doc])
    features['num_nouns'] = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    features['num_verbs'] = pos_counts.get('VERB', 0)
    features['num_adjs'] = pos_counts.get('ADJ', 0)
    features['num_adverbs'] = pos_counts.get('ADV', 0)
    features['num_pronouns'] = pos_counts.get('PRON', 0)
    features['num_determiners'] = pos_counts.get('DET', 0)
    features['num_prepositions'] = pos_counts.get('ADP', 0)
    features['num_conjunctions'] = pos_counts.get('CCONJ', 0) + pos_counts.get('SCONJ', 0)
    features['num_punctuation'] = pos_counts.get('PUNCT', 0)
    
    # POS ratios
    num_tokens = max(features['num_tokens'], 1)
    features['noun_ratio'] = features['num_nouns'] / num_tokens
    features['verb_ratio'] = features['num_verbs'] / num_tokens
    features['adj_ratio'] = features['num_adjs'] / num_tokens
    features['content_word_ratio'] = (features['num_nouns'] + features['num_verbs'] + 
                                      features['num_adjs']) / num_tokens
    features['function_word_ratio'] = (features['num_determiners'] + features['num_prepositions'] + 
                                       features['num_conjunctions']) / num_tokens
    
    # ========== Syntactic Structure Features ==========
    features['num_wh_words'] = sum(1 for token in doc if is_wh_word(token))
    features['num_entities'] = len(doc.ents)
    features['num_stopwords'] = sum(1 for token in doc if token.is_stop)
    features['stopword_ratio'] = features['num_stopwords'] / num_tokens
    
    # Logical connectors (indicate multi-hop reasoning)
    features['num_logical_connectors'] = sum(1 for token in doc if is_logical_connector(token))
    features['logical_connector_ratio'] = features['num_logical_connectors'] / num_tokens
    
    # ========== Dependency Features ==========
    dep_features = extract_dependency_features(doc)
    features.update(dep_features)
    
    # ========== Tree Structure Features ==========
    if len(doc) > 0:
        root = doc[:].root
        features['tree_depth'] = tree_depth(root)
        features['tree_width'] = tree_width(root)
        features['num_leaf_nodes'], features['num_non_leaf_nodes'] = count_nodes(root)
        features['num_clauses'] = count_clauses(doc)
        
        # Tree complexity ratios
        total_nodes = features['num_leaf_nodes'] + features['num_non_leaf_nodes']
        features['leaf_ratio'] = features['num_leaf_nodes'] / max(total_nodes, 1)
        features['branching_factor'] = features['num_non_leaf_nodes'] / max(features['num_leaf_nodes'], 1)
        features['depth_width_ratio'] = features['tree_depth'] / max(features['tree_width'], 1)
        features['depth_per_token'] = features['tree_depth'] / num_tokens
    else:
        features.update({
            'tree_depth': 0, 'tree_width': 0, 'num_leaf_nodes': 0,
            'num_non_leaf_nodes': 0, 'num_clauses': 0,
            'leaf_ratio': 0, 'branching_factor': 0, 'depth_width_ratio': 0, 'depth_per_token': 0
        })
    
    # ========== Semantic Features ==========
    # Named entity types
    entity_types = Counter([ent.label_ for ent in doc.ents])
    features['num_person_entities'] = entity_types.get('PERSON', 0)
    features['num_org_entities'] = entity_types.get('ORG', 0)
    features['num_location_entities'] = entity_types.get('GPE', 0) + entity_types.get('LOC', 0)
    features['num_date_entities'] = entity_types.get('DATE', 0)
    features['num_entity_types'] = len(entity_types)
    features['entity_density'] = features['num_entities'] / num_tokens
    
    # Semantic role features
    role_features = extract_semantic_role_features(doc)
    features.update(role_features)
    
    # ========== Question Type Features ==========
    wh_tokens = [token for token in doc if is_wh_word(token)]
    if wh_tokens:
        features['wh_word_position'] = wh_tokens[0].i / num_tokens
        wh_word_map = {'what': 1, 'who': 2, 'when': 3, 'where': 4, 'why': 5, 
                      'how': 6, 'which': 7, 'whom': 8, 'whose': 9, 'whether': 10}
        features['first_wh_word_encoded'] = float(wh_word_map.get(wh_tokens[0].text.lower(), 0))
    else:
        features['wh_word_position'] = 0.0
        features['first_wh_word_encoded'] = 0.0
    
    question_lower = question.lower()
    features['is_what_question'] = 1.0 if question_lower.startswith('what') else 0.0
    features['is_who_question'] = 1.0 if question_lower.startswith('who') else 0.0
    features['is_when_question'] = 1.0 if question_lower.startswith('when') else 0.0
    features['is_where_question'] = 1.0 if question_lower.startswith('where') else 0.0
    features['is_why_question'] = 1.0 if question_lower.startswith('why') else 0.0
    features['is_how_question'] = 1.0 if question_lower.startswith('how') else 0.0
    features['is_which_question'] = 1.0 if question_lower.startswith('which') else 0.0
    
    # ========== Complexity Indicators ==========
    features['has_coordination'] = 1.0 if features['num_conjunctions'] > 0 else 0.0
    features['has_subordination'] = 1.0 if any(token.dep_ == 'mark' for token in doc) else 0.0
    features['has_relative_clause'] = 1.0 if any(token.tag_ == 'WDT' or token.tag_ == 'WP' for token in doc) else 0.0
    features['has_negation'] = 1.0 if any(token.dep_ == 'neg' for token in doc) else 0.0
    features['has_passive'] = 1.0 if any(token.dep_ == 'nsubjpass' for token in doc) else 0.0
    features['has_question_mark'] = 1.0 if '?' in question else 0.0
    
    # ========== Lexical Diversity ==========
    unique_tokens = set([token.lemma_.lower() for token in doc if not token.is_punct])
    features['unique_token_count'] = len(unique_tokens)
    features['lexical_diversity'] = len(unique_tokens) / max(num_tokens, 1)
    
    # ========== Information Density ==========
    content_words = features['num_nouns'] + features['num_verbs'] + features['num_adjs'] + features['num_adverbs']
    features['content_word_count'] = content_words
    features['information_density'] = content_words / max(num_tokens, 1)
    
    # ========== Embedding-based Features (using SpaCy vectors) ==========
    if len(doc) > 0:
        # Calculate embedding statistics
        vectors = [token.vector for token in doc if token.has_vector]
        if vectors:
            vec_array = np.array(vectors)
            features['embedding_mean_norm'] = float(np.mean(np.linalg.norm(vec_array, axis=1)))
            features['embedding_std_norm'] = float(np.std(np.linalg.norm(vec_array, axis=1)))
            features['embedding_variance'] = float(np.var(vec_array))
        else:
            features['embedding_mean_norm'] = 0.0
            features['embedding_std_norm'] = 0.0
            features['embedding_variance'] = 0.0
    else:
        features['embedding_mean_norm'] = 0.0
        features['embedding_std_norm'] = 0.0
        features['embedding_variance'] = 0.0
    
    # ========== Interaction Features ==========
    features['tokens_per_clause'] = num_tokens / max(features['num_clauses'], 1)
    features['entities_per_token'] = features['num_entities'] / num_tokens
    features['depth_per_token'] = features['tree_depth'] / num_tokens
    features['connectors_per_clause'] = features['num_logical_connectors'] / max(features['num_clauses'], 1)
    features['complexity_score'] = (features['tree_depth'] * 0.3 + 
                                    features['num_clauses'] * 0.2 + 
                                    features['num_logical_connectors'] * 0.2 + 
                                    features['num_entities'] * 0.15 + 
                                    features['max_dependency_distance'] * 0.15)
    
    return features

