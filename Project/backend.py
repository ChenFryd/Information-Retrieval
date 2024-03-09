import inverted_index_gcp
import os
import re
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_path.json"
import nltk
from math import sqrt
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict, Counter
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

def search_anchor(query):
    # Tokenize the query
    tokens = tokenize(query)
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    # Iterate over tokens in the title
    for token in tokens:
        # Retrieve the posting list for the token from the inverted index
        posting_list = index_anchor.read_a_posting_list(token)
        # Update document scores based on the posting list
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1 # boolean model

    sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    return list(map(lambda x: x[0], sorted_match_counter.items()))[:100]


def search_body(query):
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    tokens = tokenize(query)
    doc_scores_sorted = cosin_similarity_score(tokens, index_text)
    return list(map(lambda x: x[0], doc_scores_sorted.items()))[:100]
def cosin_similarity_score(tokenized_query, index):
    """
    Support function to calculate cosin similarity
    :param index: InvertedIndex
    :param tokenized_query: list of query tokens post-processing
    :return: sorted dictionary of doc_id: cosin_similarity_score
    """
    cosine_sim_numerator = defaultdict(float)
    query_len = len(tokenized_query)
    tf_query = Counter(tokenized_query)
    query_norm = sum([pow((tf_term / query_len) * index.get_idf(term), 2) for term, tf_term in tf_query.items()])
    for term, count in tf_query.items():
        pls = index.read_a_posting_list(term)
        if pls is None:
            pls = []
        term_idf = index.get_idf(term)
        # normalized query tfidf
        query_tfidf = count / query_len * term_idf
        for doc_id, doc_tf in pls:
            doc_len = index.doc_data[doc_id][1]
            # normalized document tfidf
            doc_tfidf = doc_tf / doc_len * term_idf
            cosine_sim_numerator[doc_id] += doc_tfidf * query_tfidf

    # vector length of each doc is calculated at index creation
    cosine_sim = {doc_id: numerator /index.doc_data[doc_id][0] * sqrt(query_norm) for doc_id, numerator in
                  cosine_sim_numerator.items()}
    sorted_cosin_sim = {k: v for k, v in sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)}
    return sorted_cosin_sim

# def bm25_score(tokenized_query, index, b=0.75, k1=1.5, k3=1.5):
#     """
#     Support function to calculate bm25 score
#     :param index: InvertedIndex
#     :param tokenized_query: list of query tokens post-processing
#     :return: sorted dictionary of doc_id: bm25_similarity_score
#     """
#     score = defaultdict(float)
#     query_len = len(tokenized_query)
#     tf_query = Counter(tokenized_query)
#     index_pages_count = index_text._N
#     index_dict = {"index_text": index_text, "index_title": index_title, "index_anchor": index_anchor}
#     avg_dl = {index_name: sum([data[1] for doc_id, data in index_text.doc_data.items()]) / index_pages_count for index_name, index in index_dict.items()}[index_folder]
#     for term, count in tf_query.items():
#         pls = index_text.read_a_posting_list(term)
#         if pls is None:
#             pls = []
#         term_idf = index.get_idf(term)
#         # normalized query tfidf
#         query_tf = count / query_len
#         for doc_id, doc_tf in pls:
#             doc_len = index.doc_data[doc_id][1]
#             numerator = ((k1 + 1) * doc_tf / doc_len) * term_idf * ((k3 + 1) * query_tf)
#             denumerator = (k1 * (1 - b + b * doc_len / avg_dl) + doc_tf / doc_len) * (k3 + query_tf)
#             score[doc_id] += numerator / denumerator
#
#     sorted_bm25 = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
#     return sorted_bm25



bucket_name = "bucket_title"
# Load the inverted index from the specified path
index_title = inverted_index_gcp.InvertedIndex.read_index("index_title", "index_title",bucket_name)
index_anchor = inverted_index_gcp.InvertedIndex.read_index("index_anchor", "index_anchor",bucket_name)
index_text = inverted_index_gcp.InvertedIndex.read_index("index_text", "index_text",bucket_name)