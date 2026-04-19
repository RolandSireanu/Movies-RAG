"""Inverted index and BM25-style scoring for movie keyword search."""
import math
import os
import ipdb
import pickle
from collections import Counter, defaultdict

from text_processor import preprocess_text
from constants import BM25_K1, BM25_B


class InvertedIndex:
    """Inverted index for keyword-based movie search with BM25-style scoring."""

    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, str] = {}
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_pkl_path = "cache/index.pkl"
        self.docmap_pkl_path = "cache/docmap.pkl"
        self.term_frequencies_pkl_path = "cache/term_frequencies.pkl"
        self.doc_lengths_path = "cache/doc_lengths.pkl"

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        """Load index, docmap, and term frequencies from disk if not already loaded."""
        self.__load_pickle(self.index_pkl_path, "index")
        self.__load_pickle(self.docmap_pkl_path, "docmap")
        self.__load_pickle(self.term_frequencies_pkl_path, "term_frequencies")
        self.__load_pickle(self.doc_lengths_path, "doc_lengths")

    def __load_pickle(self, pickle_path, attr):
        """ Load the pickle file into specified attribute"""
        if os.path.exists(pickle_path):
            if not getattr(self, attr):
                with open(pickle_path, "rb") as f:
                    setattr(self, attr, pickle.load(f)) 
        else:
            raise FileNotFoundError()

    def get_tf(self, doc_id, term):
        """Print and return the term frequency of a single term in the given document."""
        self.load()
        term_list = preprocess_text(term)
        if len(term_list) > 1:
            raise Exception("More than one token given to get_tf")

        term_freq = self.term_frequencies[doc_id][term_list[0]]
        print(f"{term_list[0]} appeared {term_freq} times")
        return term_freq

    def __get_avg_doc_length(self) -> float:
        """Return the average document length across all documents."""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """Return the BM25 saturated and length-normalized term frequency."""
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25_tf_command(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """Load the index from disk and return the BM25 TF score."""
        self.load()
        return self.get_bm25_tf(doc_id, term, k1, b)

    def idf(self, term):
        """Print the inverse document frequency of a term."""
        self.load()
        term = preprocess_text(term)[0]
        term_match_doc_count = len(self.index.get(term, set()))
        total_doc_count = len(self.docmap)
        idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        print(f"Inverse document frequency of '{term}': {idf_value:.2f}")

    def get_bm25_idf(self, term: str) -> float:
        """Return the BM25 IDF score for a single term."""
        tokens = preprocess_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        stemmed = tokens[0]
        df = len(self.index.get(stemmed, set()))
        n = len(self.docmap)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def bm25_idf_command(self, term: str) -> float:
        """Load the index from disk and return the BM25 IDF score for term."""
        self.load()
        return self.get_bm25_idf(term)

    def get_documents(self, term):
        """Return a sorted list of doc IDs containing the given term."""
        return sorted(self.index.get(term.lower(), set()))

    def __add_document(self, doc_id: int, text: str):
        """Tokenize text and add tokens to the index and term frequency map."""
        tokens = preprocess_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
            self.index.setdefault(token, set()).add(doc_id)
