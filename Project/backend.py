import inverted_index_gcp
import os
import re
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_path.json"
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict
stop_words = set(stopwords.words('english'))


def remove_stop_words_nltk(query):
    # Tokenize the query and remove stop words
    tokens = re.findall(r'\w+', query.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def search_title(query):
    # Tokenize the query
    tokens = remove_stop_words_nltk(query)
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)
    # Iterate over tokens in the title
    for token in tokens:
        # Retrieve the posting list for the token from the inverted index
        posting_list = inverted_index.read_a_posting_list(base_dir, token, bucket_name)
        # Update document scores based on the posting list
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += tf  # You can use more sophisticated scoring methods here

    # Sort documents by score (optional)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return list(map(lambda x: x[0], sorted_docs))[:100]

# Example usage:
base_dir = "titlesIndex"
bucket_name = "bucket_title"
index_name = 'index'

# Load the inverted index from the specified path
inverted_index = inverted_index_gcp.InvertedIndex.read_index(base_dir, index_name,bucket_name)
query = "Star wars"
results = search_title(query)
print(len(results))
