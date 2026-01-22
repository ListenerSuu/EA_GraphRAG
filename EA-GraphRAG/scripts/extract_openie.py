"""
Script to extract OpenIE triples from corpus using LLM.
This is a simplified version - in practice, you may use more sophisticated OpenIE tools.
"""
import argparse
import json
import os
import sys
import re
from typing import List, Dict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def extract_triples_simple(passage: str) -> List[tuple]:
    """
    Simple triple extraction using pattern matching.
    For production use, replace this with proper OpenIE extraction (e.g., using LLM).
    
    Args:
        passage: Text passage to extract triples from
        
    Returns:
        List of (subject, predicate, object) tuples
    """
    # This is a placeholder - in practice, use proper OpenIE extraction
    # For example, using OpenAI API or other OpenIE tools
    triples = []
    
    # Simple pattern-based extraction (very basic)
    # In practice, use LLM-based extraction or specialized OpenIE tools
    sentences = passage.split('.')
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        
        # Very basic pattern: "X is Y", "X has Y", etc.
        patterns = [
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(.+)', 'is'),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(.+)', 'has'),
            (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+(.+)', 'was'),
        ]
        
        for pattern, pred in patterns:
            matches = re.finditer(pattern, sent)
            for match in matches:
                subj = match.group(1).strip()
                obj = match.group(2).strip()
                if len(subj) > 2 and len(obj) > 2:
                    triples.append((subj, pred, obj))
    
    return triples


def extract_openie_from_corpus(corpus_path: str, output_path: str, 
                               use_llm: bool = False, llm_model: str = None):
    """
    Extract OpenIE triples from corpus.
    
    Args:
        corpus_path: Path to corpus JSON file
        output_path: Path to save OpenIE results
        use_llm: Whether to use LLM for extraction (requires API key)
        llm_model: LLM model name if use_llm is True
    """
    # Load corpus
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    # Prepare documents
    documents = []
    if isinstance(corpus, dict):
        for key, content in corpus.items():
            if isinstance(content, list):
                doc_text = key + '\n' + ''.join(content)
            else:
                doc_text = key + '\n' + str(content)
            documents.append(doc_text)
    else:
        for item in corpus:
            if isinstance(item, dict):
                doc_text = item.get('title', '') + '\n' + item.get('text', '')
            else:
                doc_text = str(item)
            documents.append(doc_text)
    
    # Extract triples
    print(f"Extracting triples from {len(documents)} documents...")
    openie_results = []
    
    for idx, doc in enumerate(tqdm(documents, desc="Extracting triples")):
        # Extract triples (simplified - replace with proper OpenIE)
        triples = extract_triples_simple(doc)
        
        # Extract named entities (simplified)
        # In practice, use proper NER
        named_entities = []
        words = doc.split()
        # Simple heuristic: capitalized words might be entities
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                if i == 0 or words[i-1][-1] in '.!?':
                    named_entities.append(word)
        
        # Create OpenIE result
        import hashlib
        chunk_id = hashlib.md5(doc.encode('utf-8')).hexdigest()
        
        openie_result = {
            'idx': f'chunk-{chunk_id}',
            'passage': doc,
            'extracted_entities': list(set(named_entities)),
            'extracted_triples': triples
        }
        openie_results.append(openie_result)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'docs': openie_results,
            'avg_ent_chars': 0.0,  # Placeholder
            'avg_ent_words': 0.0   # Placeholder
        }, f, indent=2, ensure_ascii=False)
    
    print(f"OpenIE results saved to {output_path}")
    print(f"Extracted {sum(len(r['extracted_triples']) for r in openie_results)} triples")


def main():
    parser = argparse.ArgumentParser(description='Extract OpenIE triples from corpus')
    
    parser.add_argument('--corpus_path', type=str, required=True,
                       help='Path to corpus JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save OpenIE results')
    parser.add_argument('--use_llm', action='store_true',
                       help='Use LLM for extraction (requires API key)')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                       help='LLM model name if using LLM')
    
    args = parser.parse_args()
    
    extract_openie_from_corpus(
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        use_llm=args.use_llm,
        llm_model=args.llm_model
    )


if __name__ == '__main__':
    main()

