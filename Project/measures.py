import json

from Project.backend import search_title
import time


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)


def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)


def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p + 1.0 / r), 3)


def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)

# Modified loop to process each query and calculate metrics correctly
for query, true_doc_ids_str in queries.items():
    true_doc_ids = [int(doc_id) for doc_id in true_doc_ids_str]  # Ensure integer comparison
    #predicted_doc_ids_anchor = search_anchor(tokens)  # Assuming this returns a list of integers
    start_time = time.time()  # Start the timer
    predicted_doc_ids = search_title(query)
    end_time = time.time()  # Stop the timer
    # Convert predicted_doc_ids to integers if they are not already
  #  predicted_doc_ids_anchor = [int(doc_id) for doc_id in predicted_doc_ids_anchor]
    predicted_doc_ids = [int(doc_id) for doc_id in predicted_doc_ids]
    #predicted_doc_ids = list(set(predicted_doc_ids_anchor) & set(predicted_doc_ids_titles))
    # Calculate metrics
    avg_prec = average_precision(true_doc_ids, predicted_doc_ids)
    r_at_k_value = recall_at_k(true_doc_ids, predicted_doc_ids, 10)  # Assuming you want recall at 10
    f1_value = f1_at_k(true_doc_ids, predicted_doc_ids, 10)  # Assuming you want F1 at 10
    p_at_k_value = precision_at_k(true_doc_ids, predicted_doc_ids, 10)  # Precision at 10
    results_qual = results_quality(true_doc_ids, predicted_doc_ids)  # Custom quality metric

    # Print calculated metrics for each query
    print(f"Query: {query}")
    print(f"Average Precision: {avg_prec}, Recall@10: {r_at_k_value}, F1@10: {f1_value}, Precision@10: {p_at_k_value}, Results Quality: {results_qual}, "
          f"Search completed in {end_time - start_time:.2f} seconds.")
