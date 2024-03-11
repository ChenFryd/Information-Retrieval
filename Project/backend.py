import math

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
from concurrent.futures import ThreadPoolExecutor,wait
import threading

stop_words = set(stopwords.words('english'))

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "best"]
ALL_STOPWORDS = stopwords_frozen.union(corpus_stopwords)
import concurrent.futures


def tokenize(text):
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
    return list_of_tokens

def search_title(query):
    # Tokenize the query
    tokens = tokenize(query)
    dict_doc_scores_sorted = search_title_scores(tokens)
    return list(map(lambda x: x[0], dict_doc_scores_sorted))[:30]


def search_title_scores(tokens):
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    lock = threading.Lock()
    # Define a function to process each token and update doc_scores
    def process_token(token):
        posting_list = index_title.read_a_posting_list(token)
        for doc_id, _ in posting_list:
            with lock:
                doc_scores[doc_id] += 1

    # Use ThreadPoolExecutor to concurrently process tokens
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_token, token) for token in tokens]
        wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    # Normalize document scores
    for doc_id in doc_scores:
        doc_scores[doc_id] = doc_scores[doc_id] / index_title.docID_to_length[doc_id]

    # Sort and return the top 100 document IDs
    #sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    sorted_match_counter = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_match_counter

def search_anchor(query):
    # Tokenize the query
    tokens = tokenize(query)
    dict_doc_scores_sorted = search_anchor_scores(tokens)
    return list(map(lambda x: x[0], dict_doc_scores_sorted))[:30]

def search_anchor_scores(tokens):
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    lock = threading.Lock()
    # Iterate over tokens in the title
    def process_token(token):
        # Retrieve the posting list for the token from the inverted index
        posting_list = index_anchor.read_a_posting_list(token)
        # Update document scores based on the posting list
        for doc_id, _ in posting_list:
            with lock:
                doc_scores[doc_id] += 1  # boolean model

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_token, token) for token in tokens]
        wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

    #sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    #return list(map(lambda x: x[0], sorted_match_counter.items()))[:30]
    sorted_match_counter = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_match_counter

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
    return list(map(lambda x: x[0], doc_scores_sorted))[:30]


def cosin_similarity_score(tokens, index):
    """
    Support function to calculate cosin similarity
    :param index: InvertedIndex
    :param tokenized_query: list of query tokens post-processing
    :return: sorted dictionary of doc_id: cosin_similarity_score
    """
    cosine_sim_numerator = defaultdict(float)
    query_len = len(tokens)
    tf_query = Counter(tokens)
    query_norm = sum([math.pow((tf_term / query_len) * index.get_idf(term), 2) for term, tf_term in tf_query.items()])
    lock = threading.Lock()

    def process_token(token):
        pls = index.read_a_posting_list(token)
        if pls is None:
            pls = []
        term_idf = index.get_idf(token)
        for doc_id, doc_tf in pls:
            doc_len = index.doc_data[doc_id][1]
            doc_tfidf = (doc_tf / doc_len) * term_idf
            score = doc_tfidf * ((tf_query[token] / query_len) * term_idf)
            with lock:
                cosine_sim_numerator[doc_id] += score

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_token, token) for token in tokens]
        wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        executor.shutdown(wait=True)

    # vector length of each doc is calculated at index creation
    cosine_sim = {doc_id: numerator / (index.doc_data[doc_id][0] * sqrt(query_norm)) for doc_id, numerator in
                  cosine_sim_numerator.items()}
    sorted_cosin_sim = sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)
    return sorted_cosin_sim

def get_first_letter(word):
    """
    This function retrieves the first valid letter or number from a word.

    Args:
        word: The word to check.

    Returns:
        The first letter or number from the word, or None if no valid character found.
    """
    for char in word:
        if char.isalnum():  # Check if character is alphanumeric (letter or number)
            return char
    return word[0]

def search(query):
    res = []

    # Adjust the weight assignments as needed
    #title_weight, body_weight, anchor_weight = 0.3, 0.6, 0.1
    if len(query) == 1:
        title_weight, body_weight, anchor_weight = 0.6, 0.1, 0.3
    elif len(query) == 2:
        title_weight, body_weight, anchor_weight = 0.4, 0.2, 0.4
    elif len(re.findall(r"\d{4}", query)) > 0:
        title_weight, body_weight, anchor_weight = 0.35, 0.15, 0.5
    else:
        title_weight, body_weight, anchor_weight = 0.3, 0.6, 0.1

    words = query.split()
    for i in range(len(words)-1):
        letter_a = get_first_letter(words[i])
        letter_b = get_first_letter(words[i+1])
        if letter_a.isupper() and letter_b.isupper():
            title_weight, body_weight, anchor_weight = 0.4, 0.2, 0.4
            break

    merged_score = defaultdict(float)
    tokens = tokenize(query)
    #These functions now return a list of (doc_id, score) tuples
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(cosin_similarity_score, tokens, index_text)
        future2 = executor.submit(search_title_scores, tokens)
        future3 = executor.submit(search_anchor_scores, tokens)
        futures = [future1, future2, future3]

        # Wait for all tasks to complete
        wait(futures, return_when='ALL_COMPLETED')

        # Retrieve results
        sorted_score_body = future1.result()
        sorted_score_title = future2.result()
        sorted_score_anchor = future3.result()

    #single thread
    # sorted_score_body = cosin_similarity_score(tokens, index_text)
    # sorted_score_title = search_title_scores(tokens)
    # sorted_score_anchor = search_anchor_scores(tokens)
    all_scores = [sorted_score_title, sorted_score_anchor, sorted_score_body]

    # Process returned list of tuples for each function

    limit_docs = 30
    for sorted_score in all_scores:
        for i, (doc_id, score) in enumerate(sorted_score):
            if i == limit_docs:
                continue
            merged_score[doc_id] += score * body_weight

    # Sort merged scores and return the top 100
    sorted_merged_scores = sorted(merged_score.items(), key=lambda item: item[1], reverse=True)[:30]

    # Assuming you want to return a list of doc_id based on the sorted merged scores
    # Modify this as per your requirement
    #return list(map(lambda x: int(x[0]), sorted_merged_scores))[:30]
    return [(str(doc_id),index_title.docID_to_title_dict[doc_id]) for doc_id, score in sorted_merged_scores]


bucket_name = "bucket_title"
# Load the inverted index from the specified path
index_title = inverted_index_gcp.InvertedIndex.read_index("index_title", "index_title", bucket_name)
index_anchor = inverted_index_gcp.InvertedIndex.read_index("index_anchor", "index_anchor", bucket_name)
index_text = inverted_index_gcp.InvertedIndex.read_index("index_text", "index_text",bucket_name)
print("finished loading the indecies")
