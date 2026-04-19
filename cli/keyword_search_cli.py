import argparse
import json
import math
from constants import BM25_K1
from inverted_index import InvertedIndex
from text_processor import preprocess_text



def match_logic(arg_querry_list, arg_title_list):
    return any(q in t for q in arg_querry_list for t in arg_title_list)


def main() -> None:    
    invIndex = InvertedIndex()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Builds the inverted index and save it to disk")
    build_parser.add_argument("build", action="store_true", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency of specified term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser("idf", help="Get the idf value of a term")
    idf_parser.add_argument("term", type=str, help="Term to compute the idf value for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to compute the TF-IDF score for")

    args = parser.parse_args()

    with open("data/movies.json") as f:
        moviesDB=json.load(f)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                invIndex.load()
            except FileNotFoundError:
                print("Error: First you have to build and save the inverted index")
                return 
            querry_list = preprocess_text(args.query)
            list_of_ids = []
            
            for q in querry_list:
                list_of_ids.extend(invIndex.get_documents(q))
                if len(list_of_ids) >= 5:
                    break
            for movie in moviesDB["movies"]:                                
                if movie["id"] in list_of_ids:                                        
                    print(f"ID: {movie["id"]} \nTITLE: {movie["title"]}")


        case "build":            
            invIndex.build(moviesDB["movies"])
            invIndex.save()
            print(f"First document for token 'merida' = {invIndex.get_documents("merida")[0]}")
        case "tf":
            invIndex.get_tf(args.doc_id, args.term)
        case "idf":
            invIndex.idf(args.term)
        case "bm25tf":
            bm25tf = invIndex.bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25idf":
            bm25idf = invIndex.bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "tfidf":
            try:
                invIndex.load()
            except FileNotFoundError:
                print("Error: First you have to build and save the inverted index")
                return
            tokens = preprocess_text(args.term)
            if not tokens:
                print("0.00")
                return
            stemmed_term = tokens[0]
            tf = invIndex.term_frequencies[args.doc_id][stemmed_term]
            matching_docs = len(invIndex.index.get(stemmed_term, set()))
            total_docs = len(invIndex.docmap)
            idf = math.log((total_docs + 1) / (matching_docs + 1))
            print(f"{tf * idf:.2f}")
        case _:
            parser.print_help()
    

if __name__ == "__main__":
    main()