import ipdb
import os
import pickle
from text_processor import preprocess_text

class InvertedIndex:
    
    def __init__(self):        
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, str] = {}
        self.index_pkl_path = "cache/index.pkl"
        self.docmap_pkl_path = "cache/docmap.pkl"

    def build(self, arg_movies: list[dict[int | str, str]]):
        for movie in arg_movies:
            entire_text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie["id"], entire_text)
            self.docmap[movie["id"]] = entire_text        

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open(self.index_pkl_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_pkl_path, "wb") as f:
            pickle.dump(self.docmap, f)
    
    def load(self):
        if os.path.exists(self.index_pkl_path):
            with open(self.index_pkl_path, "rb") as f:
                self.index = pickle.load(f)
        else:
            raise FileNotFoundError()
        
        if os.path.exists(self.docmap_pkl_path):
            with open(self.docmap_pkl_path, "rb") as f:
                self.docmap = pickle.load(f)
        else:
            raise FileNotFoundError()

    def get_documents(self, term):
        return sorted(self.index.get(term.lower(), set()))

    def __add_document(self, doc_id: int, text: str):        
        text = preprocess_text(text)
        for token in text:            
            self.index.setdefault(token, set()).add(doc_id)                