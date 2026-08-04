# computational_consciousness_engine/input_mapping/intent_parser_nltk.py
import nltk
from nltk import word_tokenize, pos_tag

class IntentParserNLTK:
    def __init__(self):
        pass

    def parse(self, text: str) -> dict:
        lower = (text or "").lower()
        if "limit" in lower:
            return {"operation": "limit"}
        if "derivative" in lower:
            return {"operation": "derivative"}
        if "surface" in lower and "area" in lower:
            return {"operation": "surface_area"}
        if "unfold" in lower:
            return {"operation": "conal_unfolding"}
        if "mutation" in lower:
            return {"operation": "mutation_count"}
        
        try:
            tokens = word_tokenize(text)
            tags = pos_tag(tokens)
            return {"operation": "unknown", "tokens": tokens, "tags": tags}
        except Exception:
            return {"operation": "unknown", "raw": text}