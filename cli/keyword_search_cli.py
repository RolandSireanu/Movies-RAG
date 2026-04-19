import argparse
import json
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
        case _:
            parser.print_help()    
    



    # result_list = []
    # for movie in moviesDB["movies"]:                
    #     # if args.query.lower() in movie["title"].lower():
    #     if match_logic(preprocess_text(args.query), preprocess_text(movie["title"])):
    #         result_list.append(movie["title"])
        
            
    # for index, movie in enumerate(result_list, start=1):
    #     print(f"{index}. "+movie)
            
    # result_list = result_list[:5]
    

if __name__ == "__main__":
    main()