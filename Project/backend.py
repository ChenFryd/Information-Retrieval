import inverted_index_gcp


def search_title(query):
    # Assuming the index file is named "postings_gcp_title_index.pkl" and is located in the specified directory
    index_path = r"C:\Users\Chen\Documents\GitHub\Information-Retrieval\Project\indices\titles"
    index_name = 'postings_gcp_title_index'

    # Load the inverted index from the specified path
    inverted_index = inverted_index_gcp.InvertedIndex.read_index(index_path, index_name)
    print(inverted_index)

    pass


search_title('hello')
