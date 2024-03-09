import inverted_index_gcp
import os
import re
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_path.json"
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer, PorterStemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "best"]
ALL_STOPWORDS = stopwords_frozen.union(corpus_stopwords)
def tokenize(text, stem=False, lemm=False):
    """
    This function turns text into a list of tokens. Moreover, it filters stopwords.
    Parameters:
        lemm: lemmatize tokens
        stem: stem tokens
        text: string , represting the text to tokenize.
    Returns:
         list of tokens (e.g., list of tokens).
    """

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in ALL_STOPWORDS]
    if stem:
        return [stemmer.stem(tok) for tok in list_of_tokens]
    if lemm:
        return [lemmatizer.lemmatize(tok) for tok in list_of_tokens]
    return list_of_tokens



def search_title(query):
    # Tokenize the query
    tokens = tokenize(query)
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    # Iterate over tokens in the title
    for token in tokens:
        # Retrieve the posting list for the token from the inverted index
        posting_list = index_title.read_a_posting_list(token)
        # Update document scores based on the posting list
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1

        # Normalize document scores by the length of the title
        doc_scores[doc_id] = doc_scores[doc_id] / index_title.docID_to_title_dict[doc_id]
    sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    return list(map(lambda x: x[0], sorted_match_counter.items()))[:100]


base_dir = "index_title"
bucket_name = "bucket_title"
index_name = 'index_title'
# Load the inverted index from the specified path
index_title = inverted_index_gcp.InvertedIndex.read_index(base_dir, index_name,bucket_name)
