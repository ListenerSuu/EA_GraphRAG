"""
OpenIE Extraction Prompts
Prompts for LLM-based named entity recognition and triple extraction.
These follow the same design as graph-based retrieval systems.
"""
import json
from typing import List, Dict


# NER (Named Entity Recognition) Prompts
NER_SYSTEM_PROMPT = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

NER_EXAMPLE_PARAGRAPH = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

NER_EXAMPLE_OUTPUT = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""


# Triple Extraction Prompts
TRIPLE_EXTRACTION_SYSTEM_PROMPT = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
"""

TRIPLE_EXTRACTION_EXAMPLE_OUTPUT = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""


def get_ner_prompt(passage: str) -> List[Dict[str, str]]:
    """
    Get NER prompt for extracting named entities from a passage.
    
    Args:
        passage: Text passage to extract entities from
        
    Returns:
        List of message dictionaries for LLM API
    """
    return [
        {"role": "system", "content": NER_SYSTEM_PROMPT},
        {"role": "user", "content": NER_EXAMPLE_PARAGRAPH},
        {"role": "assistant", "content": NER_EXAMPLE_OUTPUT},
        {"role": "user", "content": passage}
    ]


def get_triple_extraction_prompt(passage: str, named_entities: List[str]) -> List[Dict[str, str]]:
    """
    Get triple extraction prompt for extracting triples from a passage.
    
    Args:
        passage: Text passage to extract triples from
        named_entities: List of named entities extracted from the passage
        
    Returns:
        List of message dictionaries for LLM API
    """
    named_entity_json = json.dumps({"named_entities": named_entities})
    
    prompt_frame = f"""Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""
    
    example_input = prompt_frame.replace(passage, NER_EXAMPLE_PARAGRAPH)
    example_input = example_input.replace(named_entity_json, NER_EXAMPLE_OUTPUT)
    
    return [
        {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": TRIPLE_EXTRACTION_EXAMPLE_OUTPUT},
        {"role": "user", "content": prompt_frame}
    ]

