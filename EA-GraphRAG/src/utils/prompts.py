"""
Prompt Templates and Instructions
Defines instructions for different embedding tasks, following the design from graph-based retrieval systems.
"""
from typing import Dict


def get_query_instruction(linking_method: str) -> str:
    """
    Get instruction for query encoding based on linking method.
    These instructions help the embedding model understand the task context.
    
    Args:
        linking_method: Type of linking task
            - 'query_to_fact': Encode query for fact retrieval
            - 'query_to_passage': Encode query for passage retrieval
            - 'query_to_node': Encode query for entity node retrieval
            - 'ner_to_node': Encode named entity for node retrieval
    
    Returns:
        Instruction string for the embedding model
    """
    instructions = {
        'ner_to_node': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_node': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
        'query_to_fact': 'Given a question, retrieve relevant triplet facts that matches this question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'query_to_passage': 'Given a question, retrieve relevant documents that best answer the question.',
    }
    default_instruction = 'Given a question, retrieve relevant documents that best answer the question.'
    return instructions.get(linking_method, default_instruction)


def format_query_with_instruction(query: str, instruction: str, model_name: str = None) -> str:
    """
    Format query with instruction based on model type.
    
    Different embedding models have different instruction formats:
    - BGE models: "Instruct: {instruction}\nQuery: {query}"
    - Some models: "{instruction}\n{query}"
    - Others: Just the query
    
    Args:
        query: Original query string
        instruction: Instruction for the task
        model_name: Name of the embedding model (optional, for model-specific formatting)
    
    Returns:
        Formatted query string
    """
    if not instruction:
        return query
    
    # BGE models use specific instruction format
    if model_name and 'bge' in model_name.lower():
        return f"Instruct: {instruction}\nQuery: {query}"
    
    # Default format: instruction + query
    return f"{instruction}\n{query}"



