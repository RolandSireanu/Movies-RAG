"""Inverted index and BM25-style scoring for movie keyword search."""
import math
import os
import ipdb
import pickle
from collections import Counter, defaultdict

from text_processor import preprocess_text


class InvertedIndex:
    """Inverted index for keyword-based movie search with BM25-style scoring."""

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, str] = {}
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)
        self.index_pkl_path = "cache/index.pkl"
        self.docmap_pkl_path = "cache/docmap.pkl"
        self.term_frequencies_pkl_path = "cache/term_frequencies.pkl"

    def build(self, arg_movies: list[dict[int | str, str]]):
        """Build the inverted index from a list of movie dicts."""
        for movie in arg_movies:
            entire_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], entire_text)
            self.docmap[movie["id"]] = entire_text

    def save(self):
        """Persist the index, docmap, and term frequencies to disk."""
        os.makedirs("cache", exist_ok=True)
        with open(self.index_pkl_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_pkl_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_pkl_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        """Load index, docmap, and term frequencies from disk if not already loaded."""
        self.__load_pickle(self.index_pkl_path)
        self.__load_pickle(self.docmap_pkl_path, "docmap")
        self.__load_pickle(self.term_frequencies_pkl_path, "term_frequencies")

    def __load_pickle(self, pickle_path, attr):
        """ Load the pickle file into specified attribute"""
        if os.path.exists(pickle_path):
            if not getattr(self, attr):
                with open(self.pickle_path, "rb") as f:
                    setattr(self, attr, pickle.load(f)) 
        else:
            raise FileNotFoundError()

    def get_tf(self, doc_id, term):
        """Print the term frequency of a single term in the given document."""
        self.load()
        term_list = preprocess_text(term)
        if len(term_list) > 1:
            raise Exception("More than one token given to get_rf")

        term_freq = self.term_frequencies[doc_id][term_list[0]]
        print(f"{term_list[0]} appeared {term_freq} times")

    def idf(self, term):
        """Print the inverse document frequency of a term."""
        self.load()
        term = preprocess_text(term)[0]
        term_match_doc_count = len(self.index.get(term, set()))
        total_doc_count = len(self.docmap)
        idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        print(f"Inverse document frequency of '{term}': {idf_value:.2f}")

    def get_documents(self, term):
        """Return a sorted list of doc IDs containing the given term."""
        return sorted(self.index.get(term.lower(), set()))

    def __add_document(self, doc_id: int, text: str):
        """Tokenize text and add tokens to the index and term frequency map."""
        text = preprocess_text(text)
        for token in text:
            self.term_frequencies[doc_id][token] += 1
            self.index.setdefault(token, set()).add(doc_id)
