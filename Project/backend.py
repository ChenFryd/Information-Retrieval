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
    #tokens = tokenize(query)
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    # Iterate over tokens in the title
    for token in query:
        # Retrieve the posting list for the token from the inverted index
        posting_list = index_title.read_a_posting_list(token)
        # Update document scores based on the posting list
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1

        # Normalize document scores by the length of the title
        doc_scores[doc_id] = doc_scores[doc_id] / index_title.docID_to_length[doc_id]
    #sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    #return list(map(lambda x: x[0], sorted_match_counter.items()))[:100]
    sorted_match_counter = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:100]
    return sorted_match_counter


def search_anchor(query):
    # Tokenize the query
    #tokens = tokenize(query)
    # Initialize a dictionary to store document scores
    doc_scores = defaultdict(int)  # Ensures that each key starts with a default value of 0
    # Iterate over tokens in the title
    for token in query:
        # Retrieve the posting list for the token from the inverted index
        posting_list = index_anchor.read_a_posting_list(token)
        # Update document scores based on the posting list
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1 # boolean model

    #sorted_match_counter = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
    #return list(map(lambda x: x[0], sorted_match_counter.items()))[:100]
    sorted_match_counter = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:100]
    return sorted_match_counter[:100]


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
    #tokens = tokenize(query)
    doc_scores_sorted = cosin_similarity_score(query, index_text)
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
    cosine_sim = {doc_id: numerator / index.doc_data[doc_id][0] * sqrt(query_norm) for doc_id, numerator in
                  cosine_sim_numerator.items()}
    #sorted_cosin_sim = {k: v for k, v in sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)}
    sorted_cosin_sim = sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)[:100]
    return sorted_cosin_sim

def search(query):
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    # query = request.args.get('query', '')
    # if len(query) == 0:
    #   return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if len(tokens) == 1:
        title_weight, body_weight, anchor_weight = 0.6, 0.1, 0.3
    elif len(re.findall(r"(\w+)\s(and|the)\s(\w+)", query)) > 0 or len(re.findall(r"\d{4}", query)) > 0:
        title_weight, body_weight, anchor_weight = 0.3, 0.2, 0.5
    else:
        title_weight, body_weight, anchor_weight = 0.3, 0.6, 0.1

    merged_score = defaultdict(float)

    sorted_score_body = cosin_similarity_score(tokens, index_text)
    sorted_score_title = search_title(tokens)
    sorted_score_anchor = search_anchor(tokens)

    for doc_id, cosine_sim in sorted_score_body:
        merged_score[doc_id] += cosine_sim * body_weight
    for doc_id, doc_scores in sorted_score_title:
        merged_score[doc_id] += doc_scores * title_weight
    for doc_id, doc_scores in sorted_score_anchor:
        merged_score[doc_id] += doc_scores * anchor_weight

    #sorted_merged_scores = {k: v for k, v in sorted(merged_score.items(), key=lambda item: item[1], reverse=True)[:100]}
    sorted_merged_scores = sorted(merged_score.items(), key=lambda item: item[1], reverse=True)[:100]
    # END SOLUTION
    return [doc_id for doc_id, score in sorted_merged_scores]
    #return sorted_merged_scores(res)


bucket_name = "bucket_title"
# Load the inverted index from the specified path
index_title = inverted_index_gcp.InvertedIndex.read_index("index_title", "index_title",bucket_name)
index_anchor = inverted_index_gcp.InvertedIndex.read_index("index_anchor", "index_anchor",bucket_name)
index_text = inverted_index_gcp.InvertedIndex.read_index("index_text", "index_text",bucket_name)