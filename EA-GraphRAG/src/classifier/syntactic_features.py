import stanza
from stanza.server import CoreNLPClient
from collections import defaultdict
import math

class SyntacticComplexityAnalyzer:
    def __init__(self):
        self.stats = defaultdict(int)
        self.measures = {
            # Type 1: Length of production unit
            'MLC': {'numerator': 'words', 'denominator': 'C'},
            'MLS': {'numerator': 'words', 'denominator': 'S'},
            'MLT': {'numerator': 'words', 'denominator': 'T'},
            
            # Type 2: Sentence complexity
            'C/S': {'numerator': 'C', 'denominator': 'S'},
            
            # Type 3: Subordination
            'C/T': {'numerator': 'C', 'denominator': 'T'},
            'CT/T': {'numerator': 'CT', 'denominator': 'T'},
            'DC/C': {'numerator': 'DC', 'denominator': 'C'},
            'DC/T': {'numerator': 'DC', 'denominator': 'T'},
            
            # Type 4: Coordination
            'CP/C': {'numerator': 'CP', 'denominator': 'C'},
            'CP/T': {'numerator': 'CP', 'denominator': 'T'},
            'T/S': {'numerator': 'T', 'denominator': 'S'},
            
            # Type 5: Particular structures
            'CN/C': {'numerator': 'CN', 'denominator': 'C'},
            'CN/T': {'numerator': 'CN', 'denominator': 'T'},
            'VP/T': {'numerator': 'VP', 'denominator': 'T'}
        }
    
    def analyze_text(self, text, client, nlp):
        """Analyze the given text and compute all syntactic complexity measures"""
        # Reset statistics
        self.stats = defaultdict(int)
        
        # First parse the text to get the parse trees

        
        doc = nlp(text)
        trees = [sentence.constituency for sentence in doc.sentences]

        # ann = client.annotate(text)
        
        # Count words (excluding punctuation)
        self.count_words(doc)
        
        # Count sentences
        self.count_sentences(doc)
        
        # Count clauses (C)
        self.count_clauses(client, trees)
        
        # Count dependent clauses (DC)
        self.count_dependent_clauses(client, trees)
        
        # Count T-units (T)
        self.count_tunits(client, trees)
        
        # Count complex T-units (CT)
        self.count_complex_tunits(client, trees)
        
        # Count coordinate phrases (CP)
        self.count_coordinate_phrases(client, trees)
        
        # Count complex nominals (CN)
        self.count_complex_nominals(client, trees)
        
        # Count verb phrases (VP)
        self.count_verb_phrases(client, trees)
        

        # for measure, value in self.stats.items():
        #     print(f"{measure}: {value:.3f}")

        # Calculate all measures
        results = {}
        for measure, params in self.measures.items():
            numerator = self.stats[params['numerator']]
            denominator = self.stats[params['denominator']]
            
            if denominator == 0:
                results[measure] = 0.0
            else:
                results[measure] = numerator / denominator
        
        return results

    
    def count_words(self, doc):
        """Count words excluding punctuation"""
        for sentence in doc.sentences:
            for word in sentence.words:
                # Exclude punctuation
                if word.upos not in ['PUNCT', 'SYM']:
                    self.stats['words'] += 1
    
    def count_sentences(self, doc):
        """Count sentences"""
        self.stats['S'] = len(doc.sentences)
    
    def count_clauses(self, client, trees):
        """Count clauses using Tregex pattern: S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ)"""
        
        
        pattern = 'S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ)'
        matches = client.tregex(pattern=pattern, trees=trees)
        # matches = client.tregex(text, pattern, trees)
        self.stats['C'] = sum(len(sent_matches) for sent_matches in matches['sentences'])

        
        # Also count sentence fragments (FRAG > ROOT !<< VP)
        frag_pattern = 'FRAG > ROOT !<< VP'
        frag_matches = client.tregex(pattern=frag_pattern, trees=trees)
        # frag_matches = client.tregex(text, frag_pattern, trees)
        self.stats['C'] += sum(len(sent_matches) for sent_matches in frag_matches['sentences'])

    
    def count_dependent_clauses(self, client, trees):
        """Count dependent clauses using Tregex pattern: SBAR < (S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ))"""
        

        pattern = 'SBAR < (S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ))'
        matches = client.tregex(pattern=pattern, trees=trees)
        # matches = client.tregex(text, pattern, trees)
        self.stats['DC'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
    def count_tunits(self, client, trees):
        """Count T-units using Tregex pattern: S|SBARQ|SINV|SQ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]"""
        
        pattern = 'S|SBARQ|SINV|SQ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]'
        matches = client.tregex(pattern=pattern, trees=trees)
        # matches = client.tregex(text, pattern, trees)
        self.stats['T'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
        
        # Also count sentence fragments (FRAG > ROOT)
        frag_pattern = 'FRAG > ROOT'
        frag_matches = client.tregex(pattern=frag_pattern,trees=trees)
        # frag_matches = client.tregex(text, frag_pattern,trees)
        self.stats['T'] += sum(len(sent_matches) for sent_matches in frag_matches['sentences'])
    
    def count_complex_tunits(self, client,  trees):
        """Count complex T-units using Tregex pattern: 
        S|SBARQ|SINV|SQ[ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]] << (SBAR < (S|SINV|SQ < (VP <# MD|VBP|VBZ|VBD)))"""
        

        pattern = 'S|SBARQ|SINV|SQ[ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]] << (SBAR < (S|SINV|SQ < (VP <# MD|VBP|VBZ|VBD)))'
        matches = client.tregex(pattern=pattern, trees=trees)
        # matches = client.tregex(text, pattern, trees)
        self.stats['CT'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
    def count_coordinate_phrases(self, client, trees):
        """Count coordinate phrases using Tregex pattern: ADJP|ADVP|NP|VP < CC"""
        

        pattern = 'ADJP|ADVP|NP|VP < CC'
        # matches = client.tregex(text, pattern, trees)
        matches = client.tregex(pattern=pattern, trees=trees)
        self.stats['CP'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
    def count_complex_nominals(self, client, trees):
        """Count complex nominals using multiple Tregex patterns"""
        

        # Pattern 1: NP!> NP <<JJ|POS|PP|S|VBG|<<(NP $++ NP!$(+CC))
        pattern1 = 'NP!> NP <<JJ|POS|PP|S|VBG|<<(NP $++ NP!$(+CC))'
        # matches1 = client.tregex(text, pattern1, trees)
        matches1 = client.tregex(pattern=pattern1, trees=trees)
        count1 = sum(len(sent_matches) for sent_matches in matches1['sentences'])
        
        # Pattern 2: SBAR[$+VP|>VP] & [<# WHNP | <# (IN < that|That|For|for) | <, S]
        pattern2 = 'SBAR[$+VP|>VP] & [<# WHNP | <# (IN < that|That|For|for) | <, S]'
        matches2 = client.tregex(pattern=pattern2, trees=trees)
        # matches2 = client.tregex(text, pattern2, trees)
        count2 = sum(len(sent_matches) for sent_matches in matches2['sentences'])
        
        # Pattern 3: S < (VP <# VBG|TO) $+ VP
        pattern3 = 'S < (VP <# VBG|TO) $+ VP'
        matches3 = client.tregex(pattern=pattern3, trees=trees)
        # matches3 = client.tregex(text, pattern3, trees)
        count3 = sum(len(sent_matches) for sent_matches in matches3['sentences'])
        
        self.stats['CN'] = count1 + count2 + count3
    
    def count_verb_phrases(self, client, trees):
        """Count verb phrases using Tregex pattern: VP > S|SQ|SINV"""
        

        pattern = 'VP > S|SQ|SINV'
        matches = client.tregex(pattern=pattern, trees=trees)
        # matches = client.tregex(text, pattern, trees)
        self.stats['VP'] = sum(len(sent_matches) for sent_matches in matches['sentences'])

# Example usage
if __name__ == "__main__":
    text = """We use it when a girl in our dorm is acting like a spoiled child. 
    Saving energy is really important. I know you like to read."""
    
    analyzer = SyntacticComplexityAnalyzer()
    
    with CoreNLPClient(
            annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse'],
            # annotators=['parse'],
            timeout=30000,
            memory='16G',
            # be_quiet=True,          # 减少日志
            max_char_length=100000, # 调大单次请求上限
            threads=8               # 调大线程池
            ) as client:
        
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        results = analyzer.analyze_text(text, client, nlp)
        
        print("Syntactic Complexity Measures:")
        for measure, value in results.items():
            print(f"{measure}: {value:.3f}")


# import stanza
# from stanza.server import CoreNLPClient
# from collections import defaultdict
# import math

# class SyntacticComplexityAnalyzer:
#     def __init__(self):
#         self.stats = defaultdict(int)
#         self.measures = {
#             # Type 1: Length of production unit
#             'MLC': {'numerator': 'words', 'denominator': 'C'},
#             'MLS': {'numerator': 'words', 'denominator': 'S'},
#             'MLT': {'numerator': 'words', 'denominator': 'T'},
            
#             # Type 2: Sentence complexity
#             'C/S': {'numerator': 'C', 'denominator': 'S'},
            
#             # Type 3: Subordination
#             'C/T': {'numerator': 'C', 'denominator': 'T'},
#             'CT/T': {'numerator': 'CT', 'denominator': 'T'},
#             'DC/C': {'numerator': 'DC', 'denominator': 'C'},
#             'DC/T': {'numerator': 'DC', 'denominator': 'T'},
            
#             # Type 4: Coordination
#             'CP/C': {'numerator': 'CP', 'denominator': 'C'},
#             'CP/T': {'numerator': 'CP', 'denominator': 'T'},
#             'T/S': {'numerator': 'T', 'denominator': 'S'},
            
#             # Type 5: Particular structures
#             'CN/C': {'numerator': 'CN', 'denominator': 'C'},
#             'CN/T': {'numerator': 'CN', 'denominator': 'T'},
#             'VP/T': {'numerator': 'VP', 'denominator': 'T'}
#         }
    
#     def analyze_text(self, text, client):
#         """Analyze the given text and compute all syntactic complexity measures"""
#         # Reset statistics
#         self.stats = defaultdict(int)
        
#         # First parse the text to get the parse trees
#         ann = client.annotate(text)
        
#         # Count words (excluding punctuation)
#         self.count_words(ann)
        
#         # Count sentences
#         self.count_sentences(ann)
        
#         # Count clauses (C)
#         self.count_clauses(client, text, ann)
        
#         # Count dependent clauses (DC)
#         self.count_dependent_clauses(client, text, ann)
        
#         # Count T-units (T)
#         self.count_tunits(client, text, ann)
        
#         # Count complex T-units (CT)
#         self.count_complex_tunits(client, text, ann)
        
#         # Count coordinate phrases (CP)
#         self.count_coordinate_phrases(client, text, ann)
        
#         # Count complex nominals (CN)
#         self.count_complex_nominals(client, text, ann)
        
#         # Count verb phrases (VP)
#         self.count_verb_phrases(client, text, ann)
        

#         for measure, value in self.stats.items():
#             print(f"{measure}: {value:.3f}")

#         # Calculate all measures
#         results = {}
#         for measure, params in self.measures.items():
#             numerator = self.stats[params['numerator']]
#             denominator = self.stats[params['denominator']]
            
#             if denominator == 0:
#                 results[measure] = 0.0
#             else:
#                 results[measure] = numerator / denominator
        
#         return results

    
#     def count_words(self, ann):
#         """Count words excluding punctuation"""
#         for sentence in ann.sentence:
#             for token in sentence.token:
#                 # Exclude punctuation
#                 if not any(cat in token.pos for cat in ['.', ',', ':', "''", "``", '-', '(', ')']):
#                     self.stats['words'] += 1
    
#     def count_sentences(self, ann):
#         """Count sentences"""
#         self.stats['S'] = len(ann.sentence)
    
#     def count_clauses(self, client, text, ann):
#         """Count clauses using Tregex pattern: S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ)"""
#         trees = [sen.parseTree for sen in ann.sentence]
        
#         pattern = 'S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ)'
#         matches = client.tregex(pattern, trees)
#         # matches = client.tregex(text, pattern, trees)
#         self.stats['C'] = sum(len(sent_matches) for sent_matches in matches['sentences'])

        
#         # Also count sentence fragments (FRAG > ROOT !<< VP)
#         frag_pattern = 'FRAG > ROOT !<< VP'
#         frag_matches = client.tregex(frag_pattern, trees)
#         # frag_matches = client.tregex(text, frag_pattern, trees)
#         self.stats['C'] += sum(len(sent_matches) for sent_matches in frag_matches['sentences'])

    
#     def count_dependent_clauses(self, client, text, ann):
#         """Count dependent clauses using Tregex pattern: SBAR < (S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ))"""
#         trees = [sen.parseTree for sen in ann.sentence]

#         pattern = 'SBAR < (S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ))'
#         matches = client.tregex(pattern, trees)
#         # matches = client.tregex(text, pattern, trees)
#         self.stats['DC'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
#     def count_tunits(self, client, text, ann):
#         """Count T-units using Tregex pattern: S|SBARQ|SINV|SQ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]"""
#         trees = [sen.parseTree for sen in ann.sentence]
#         pattern = 'S|SBARQ|SINV|SQ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]'
#         matches = client.tregex(pattern, trees)
#         # matches = client.tregex(text, pattern, trees)
#         self.stats['T'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
        
#         # Also count sentence fragments (FRAG > ROOT)
#         frag_pattern = 'FRAG > ROOT'
#         frag_matches = client.tregex(frag_pattern,trees)
#         # frag_matches = client.tregex(text, frag_pattern,trees)
#         self.stats['T'] += sum(len(sent_matches) for sent_matches in frag_matches['sentences'])
    
#     def count_complex_tunits(self, client, text, ann):
#         """Count complex T-units using Tregex pattern: 
#         S|SBARQ|SINV|SQ[ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]] << (SBAR < (S|SINV|SQ < (VP <# MD|VBP|VBZ|VBD)))"""
#         trees = [sen.parseTree for sen in ann.sentence]

#         pattern = 'S|SBARQ|SINV|SQ[ > ROOT | [ $- S|SBARQ|SINV|SQ !>> SBAR|VP ]] << (SBAR < (S|SINV|SQ < (VP <# MD|VBP|VBZ|VBD)))'
#         matches = client.tregex(pattern, trees)
#         # matches = client.tregex(text, pattern, trees)
#         self.stats['CT'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
#     def count_coordinate_phrases(self, client, text, ann):
#         """Count coordinate phrases using Tregex pattern: ADJP|ADVP|NP|VP < CC"""
#         trees = [sen.parseTree for sen in ann.sentence]

#         pattern = 'ADJP|ADVP|NP|VP < CC'
#         # matches = client.tregex(text, pattern, trees)
#         matches = client.tregex(pattern, trees)
#         self.stats['CP'] = sum(len(sent_matches) for sent_matches in matches['sentences'])
    
#     def count_complex_nominals(self, client, text, ann):
#         """Count complex nominals using multiple Tregex patterns"""
#         trees = [sen.parseTree for sen in ann.sentence]

#         # Pattern 1: NP!> NP <<JJ|POS|PP|S|VBG|<<(NP $++ NP!$(+CC))
#         pattern1 = 'NP!> NP <<JJ|POS|PP|S|VBG|<<(NP $++ NP!$(+CC))'
#         # matches1 = client.tregex(text, pattern1, trees)
#         matches1 = client.tregex(pattern1, trees)
#         count1 = sum(len(sent_matches) for sent_matches in matches1['sentences'])
        
#         # Pattern 2: SBAR[$+VP|>VP] & [<# WHNP | <# (IN < that|That|For|for) | <, S]
#         pattern2 = 'SBAR[$+VP|>VP] & [<# WHNP | <# (IN < that|That|For|for) | <, S]'
#         matches2 = client.tregex(pattern2, trees)
#         # matches2 = client.tregex(text, pattern2, trees)
#         count2 = sum(len(sent_matches) for sent_matches in matches2['sentences'])
        
#         # Pattern 3: S < (VP <# VBG|TO) $+ VP
#         pattern3 = 'S < (VP <# VBG|TO) $+ VP'
#         matches3 = client.tregex(pattern3, trees)
#         # matches3 = client.tregex(text, pattern3, trees)
#         count3 = sum(len(sent_matches) for sent_matches in matches3['sentences'])
        
#         self.stats['CN'] = count1 + count2 + count3
    
#     def count_verb_phrases(self, client, text, ann):
#         """Count verb phrases using Tregex pattern: VP > S|SQ|SINV"""
#         trees = [sen.parseTree for sen in ann.sentence]

#         pattern = 'VP > S|SQ|SINV'
#         matches = client.tregex(pattern, trees)
#         # matches = client.tregex(text, pattern, trees)
#         self.stats['VP'] = sum(len(sent_matches) for sent_matches in matches['sentences'])

# # Example usage
# if __name__ == "__main__":
#     text = """We use it when a girl in our dorm is acting like a spoiled child. 
#     Saving energy is really important. I know you like to read."""
    
#     analyzer = SyntacticComplexityAnalyzer()
    
#     with CoreNLPClient(
#             annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse'],
#             # annotators=['parse'],
#             timeout=30000,
#             memory='16G',
#             # be_quiet=True,          # 减少日志
#             max_char_length=100000, # 调大单次请求上限
#             threads=8               # 调大线程池
#             ) as client:
        
        
#         results = analyzer.analyze_text(text, client)
        
#         print("Syntactic Complexity Measures:")
#         for measure, value in results.items():
#             print(f"{measure}: {value:.3f}")

        # ann = client.annotate(text)
        # trees = ann.sentence[0].parseTree
        # for tree_idx, tree in enumerate(trees):
        #     print("--------------------------------")
        #     print(tree_idx)
        #     print(tree)
        #     leaves = tree.leaf_labels()
        #     self.fill_tree_proto(tree, sentence.parseTree)
        #     for word in leaves:
        #         token = sentence.token.add()
        #         # the other side uses both value and word, weirdly enough
        #         token.value = word
        #         token.word = word
        #         # without the actual tokenization, at least we can
        #         # stop the words from running together
        #         token.after = " "
        # doc.text = " ".join(full_text)


# import stanza
# import json
# from stanza.server import CoreNLPClient
# text = "We use it when a girl in our dorm is acting like a spoiled child"
# with CoreNLPClient(
#         annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse'],
#         timeout=30000,
#         # start_server=stanza.server.StartServer.TRY_START,
#         memory='16G') as client:


#     # Tregex example
#     pattern = 'S|SINV|SQ < (VP <# MD|VBD|VBP|VBZ)'
#     matches = client.tregex(text, pattern)
#     # You can access matches similarly
#     with open("matches.json", "w") as f:
#         json.dump(matches, f, indent=2)
#     print(matches['sentences'][0]['1']['match']) # prints: "(NP (DT a) (JJ simple) (NN sentence))\n"
#     print(len(matches['sentences'][0]))