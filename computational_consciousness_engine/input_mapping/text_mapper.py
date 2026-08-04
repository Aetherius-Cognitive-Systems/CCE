import string
import numpy as np
from typing import List, Dict, Set, Iterable

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

class TextToChaosMapper:
    def __init__(self, num_shards: int = 1024):
        self.shard_to_word: Dict[int, str] = {}
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = set()
            
        self.num_shards = max(1, int(num_shards))
        
        # [NEW] Load the ML Embedding Model (Downloads ~80MB model automatically)
        print("Loading AI Semantic Embedding Model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Fixed random projection matrix for stable LSH (Locality Sensitive Hashing)
        np.random.seed(42)
        self.projection_matrix = np.random.randn(384) 

    def _clean_and_tokenize(self, text: str) -> List[str]:
        # [Keep this exactly as you have it right now]
        if not text:
            return []
        text = text.lower()
        raw_tokens = word_tokenize(text)
        cleaned_tokens = []
        for t in raw_tokens:
            t = t.strip(string.punctuation)
            t = t.translate(str.maketrans("", "", string.punctuation))
            if not t or t in self.stop_words:
                continue
            base_word = self.lemmatizer.lemmatize(t)
            cleaned_tokens.append(base_word)
        return cleaned_tokens

    def _hash_token(self, token: str) -> int:
        """
        [NEW] AI Semantic Locality-Sensitive Hashing.
        Instead of arbitrary SHA256, we calculate a 384-dimensional semantic 
        meaning vector, and project it into an integer space.
        """
        if not token:
            return 0
            
        # 1. Calculate dense neural representation of the word
        embedding = self.encoder.encode(token)
        
        # 2. Project 384-dimensions down to a scalar using our fixed matrix
        semantic_scalar = np.dot(embedding, self.projection_matrix)
        
        # 3. Map into the shard integer space
        val = int(abs(semantic_scalar) * 1000000)
        return val % self.num_shards

    # [Keep map_text_to_shards, seed_chaos_pool, get_shard_word, text_to_shard_list exactly as they are]

    def map_text_to_shards(self, text: str) -> Dict[int, Set[str]]:
        """
        Map cleaned tokens to shard indices. Returns a dict: shard -> set(tokens).
        Useful for building inverted indices or seeding chaos pools.
        """
        tokens = self._clean_and_tokenize(text)
        shard_map: Dict[int, Set[str]] = {}
        for tok in tokens:
            shard = self._hash_token(tok)
            if shard not in shard_map:
                shard_map[shard] = set()
            shard_map[shard].add(tok)
        return shard_map

    def seed_chaos_pool(self, texts: Iterable[str]) -> None:
        """
        Populate self.shard_to_word with a representative token for each shard.
        If multiple tokens map to the same shard, the first seen token wins.
        """
        for text in texts:
            shard_map = self.map_text_to_shards(text)
            for shard, toks in shard_map.items():
                if shard not in self.shard_to_word:
                    # choose a deterministic representative (sorted)
                    rep = sorted(toks)[0]
                    self.shard_to_word[shard] = rep

    def get_shard_word(self, shard: int) -> str:
        """
        Return the representative word for a shard, or empty string if none.
        """
        return self.shard_to_word.get(shard, "")

    def text_to_shard_list(self, text: str) -> List[int]:
        """
        Convenience: return sorted list of shard indices for a text.
        """
        return sorted(self.map_text_to_shards(text).keys())
