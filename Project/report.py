import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')  # Ensure stopwords are downloaded

# Define stopwords and the regex for word detection
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
def extract_metrics_from_lines(lines):
    metrics = {
        "Average Precision": [],
        "Recall@10": [],
        "F1@10": [],
        "Precision@10": [],
        "Results Quality": [],
        "Search completed in": []  # Added this to capture search completion times
    }
    search_times_for_queries_with_more_than_three_words = []

    current_query_more_than_three_non_stopwords = False

    for line in lines:
        if line.startswith("Query:"):
            query = line.split("Query:")[1].strip()
            tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
            current_query_more_than_three_non_stopwords = len(tokens) > 3
        else:
            parts = line.split(", ")
            for part in parts:
                if "Search completed in" in part:
                    # Special handling for "Search completed in"
                    time_taken = part.split("Search completed in ")[1].split(" seconds")[0]
                    metrics["Search completed in"].append(float(time_taken))
                    if current_query_more_than_three_non_stopwords:
                        search_times_for_queries_with_more_than_three_words.append(float(time_taken))
                else:
                    # Regular metric processing
                    metric_name, metric_value = part.split(": ")
                    if metric_name in metrics:
                        metrics[metric_name].append(float(metric_value))

    # Add a special entry for the average time of queries with more than three words
    if search_times_for_queries_with_more_than_three_words:
        metrics["Average Time for >3 Word Queries"] = [sum(search_times_for_queries_with_more_than_three_words) / len(search_times_for_queries_with_more_than_three_words)]
    else:
        metrics["Average Time for >3 Word Queries"] = [0]

    return metrics

def calculate_stats(metrics):
    stats = {}
    for metric, values in metrics.items():
        if values:  # Check if the list is not empty
            stats[metric] = {
                "Average": sum(values) / len(values),
                "Lowest Value": min(values),
                "Highest Value": max(values)
            }
    return stats

def print_stats(stats):
    for metric, metric_stats in stats.items():
        print(f"{metric}:")
        for stat_name, stat_value in metric_stats.items():
            print(f"  {stat_name}: {stat_value:.3f}")
        print()  # Blank line for readability

    # Sample usage with the data provided in the question
multiline_string1  = """
Query: genetics
Average Precision: 0.097, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 2.04 seconds.
Query: Who is considered the "Father of the United States"?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 44.26 seconds.
Query: economic
Average Precision: 0.211, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.154, Search completed in 3.90 seconds.
Query: When was the United Nations founded?
Average Precision: 0.211, Recall@10: 0.043, F1@10: 0.071, Precision@10: 0.2, Results Quality: 0.083, Search completed in 28.63 seconds.
Query: video gaming
Average Precision: 0.111, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 9.79 seconds.
Query: 3D printing technology
Average Precision: 0.181, Recall@10: 0.045, F1@10: 0.073, Precision@10: 0.2, Results Quality: 0.0, Search completed in 9.42 seconds.
Query: Who is the author of "1984"?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 8.67 seconds.
Query: bioinformatics
Average Precision: 0.101, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 2.07 seconds.
Query: Who is known for proposing the heliocentric model of the solar system?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 35.24 seconds.
Query: Describe the process of water erosion.
Average Precision: 0.522, Recall@10: 0.104, F1@10: 0.172, Precision@10: 0.5, Results Quality: 0.348, Search completed in 21.49 seconds.
Query: When was the Berlin Wall constructed?
Average Precision: 0.572, Recall@10: 0.102, F1@10: 0.169, Precision@10: 0.5, Results Quality: 0.273, Search completed in 10.93 seconds.
Query: What is the meaning of the term "Habeas Corpus"?
Average Precision: 0.579, Recall@10: 0.122, F1@10: 0.203, Precision@10: 0.6, Results Quality: 0.402, Search completed in 14.92 seconds.
Query: telecommunications
Average Precision: 0.071, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 2.04 seconds.
Query: internet
Average Precision: 0.158, Recall@10: 0.024, F1@10: 0.039, Precision@10: 0.1, Results Quality: 0.087, Search completed in 2.89 seconds.
Query: What are the characteristics of a chemical element?
Average Precision: 0.188, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.08, Search completed in 7.24 seconds.
Query: Describe the structure of a plant cell.
Average Precision: 0.133, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.0, Search completed in 17.59 seconds.
Query: Who painted "Starry Night"?
Average Precision: 0.289, Recall@10: 0.061, F1@10: 0.093, Precision@10: 0.2, Results Quality: 0.155, Search completed in 11.87 seconds.
Query: computer
Average Precision: 0.212, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.113, Search completed in 3.53 seconds.
Query: What is the structure of the Earth's layers?
Average Precision: 0.288, Recall@10: 0.023, F1@10: 0.037, Precision@10: 0.1, Results Quality: 0.085, Search completed in 12.66 seconds.
Query: When did World War II end?
Average Precision: 0.129, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.044, Search completed in 34.14 seconds.
Query: When was the Gutenberg printing press invented?
Average Precision: 0.429, Recall@10: 0.109, F1@10: 0.179, Precision@10: 0.5, Results Quality: 0.366, Search completed in 16.59 seconds.
Query: medicine
Average Precision: 0.124, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 5.50 seconds.
Query: Describe the water cycle.
Average Precision: 0.155, Recall@10: 0.021, F1@10: 0.035, Precision@10: 0.1, Results Quality: 0.112, Search completed in 14.14 seconds.
Query: artificial intelligence
Average Precision: 0.457, Recall@10: 0.093, F1@10: 0.151, Precision@10: 0.4, Results Quality: 0.259, Search completed in 6.62 seconds.
Query: physics
Average Precision: 0.098, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 3.41 seconds.
Query: nanotechnology
Average Precision: 0.65, Recall@10: 0.128, F1@10: 0.211, Precision@10: 0.6, Results Quality: 0.453, Search completed in 2.83 seconds.
Query: When did the Black Death pandemic occur?
Average Precision: 0.05, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 24.18 seconds.
Query: neuroscience
Average Precision: 0.575, Recall@10: 0.085, F1@10: 0.14, Precision@10: 0.4, Results Quality: 0.309, Search completed in 3.13 seconds.
Query: snowboard
Average Precision: 0.116, Recall@10: 0.062, F1@10: 0.077, Precision@10: 0.1, Results Quality: 0.0, Search completed in 3.19 seconds.
Query: Who is the founder of modern psychology?
Average Precision: 0.135, Recall@10: 0.041, F1@10: 0.068, Precision@10: 0.2, Results Quality: 0.0, Search completed in 14.13 seconds.
"""

multiline_string2 = """
Query: genetics
Average Precision: 0.108, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 3.40 seconds.
Query: Who is considered the "Father of the United States"?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 45.77 seconds.
Query: economic
Average Precision: 0.211, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.154, Search completed in 5.62 seconds.
Query: When was the United Nations founded?
Average Precision: 0.211, Recall@10: 0.043, F1@10: 0.071, Precision@10: 0.2, Results Quality: 0.083, Search completed in 29.50 seconds.
Query: video gaming
Average Precision: 0.111, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 8.12 seconds.
Query: 3D printing technology
Average Precision: 0.181, Recall@10: 0.045, F1@10: 0.073, Precision@10: 0.2, Results Quality: 0.0, Search completed in 9.09 seconds.
Query: Who is the author of "1984"?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 11.04 seconds.
Query: bioinformatics
Average Precision: 0.101, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 2.86 seconds.
Query: Who is known for proposing the heliocentric model of the solar system?
Average Precision: 0.0, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 39.39 seconds.
Query: Describe the process of water erosion.
Average Precision: 0.522, Recall@10: 0.104, F1@10: 0.172, Precision@10: 0.5, Results Quality: 0.348, Search completed in 20.13 seconds.
Query: When was the Berlin Wall constructed?
Average Precision: 0.62, Recall@10: 0.102, F1@10: 0.169, Precision@10: 0.5, Results Quality: 0.273, Search completed in 16.92 seconds.
Query: What is the meaning of the term "Habeas Corpus"?
Average Precision: 0.584, Recall@10: 0.122, F1@10: 0.203, Precision@10: 0.6, Results Quality: 0.402, Search completed in 17.68 seconds.
Query: telecommunications
Average Precision: 0.071, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 3.50 seconds.
Query: internet
Average Precision: 0.2, Recall@10: 0.024, F1@10: 0.039, Precision@10: 0.1, Results Quality: 0.087, Search completed in 2.91 seconds.
Query: What are the characteristics of a chemical element?
Average Precision: 0.188, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.08, Search completed in 10.73 seconds.
Query: Describe the structure of a plant cell.
Average Precision: 0.137, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.0, Search completed in 14.70 seconds.
Query: Who painted "Starry Night"?
Average Precision: 0.289, Recall@10: 0.061, F1@10: 0.093, Precision@10: 0.2, Results Quality: 0.155, Search completed in 11.42 seconds.
Query: computer
Average Precision: 0.212, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.113, Search completed in 2.77 seconds.
Query: What is the structure of the Earth's layers?
Average Precision: 0.288, Recall@10: 0.023, F1@10: 0.037, Precision@10: 0.1, Results Quality: 0.085, Search completed in 9.63 seconds.
Query: When did World War II end?
Average Precision: 0.2, Recall@10: 0.02, F1@10: 0.033, Precision@10: 0.1, Results Quality: 0.044, Search completed in 38.00 seconds.
Query: When was the Gutenberg printing press invented?
Average Precision: 0.449, Recall@10: 0.109, F1@10: 0.179, Precision@10: 0.5, Results Quality: 0.366, Search completed in 15.83 seconds.
Query: medicine
Average Precision: 0.126, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 3.60 seconds.
Query: Describe the water cycle.
Average Precision: 0.168, Recall@10: 0.021, F1@10: 0.035, Precision@10: 0.1, Results Quality: 0.112, Search completed in 11.70 seconds.
Query: artificial intelligence
Average Precision: 0.457, Recall@10: 0.093, F1@10: 0.151, Precision@10: 0.4, Results Quality: 0.259, Search completed in 6.31 seconds.
Query: physics
Average Precision: 0.103, Recall@10: 0.022, F1@10: 0.036, Precision@10: 0.1, Results Quality: 0.0, Search completed in 2.40 seconds.
Query: nanotechnology
Average Precision: 0.683, Recall@10: 0.128, F1@10: 0.211, Precision@10: 0.6, Results Quality: 0.453, Search completed in 2.48 seconds.
Query: When did the Black Death pandemic occur?
Average Precision: 0.05, Recall@10: 0.0, F1@10: 0.0, Precision@10: 0.0, Results Quality: 0.0, Search completed in 23.73 seconds.
Query: neuroscience
Average Precision: 0.575, Recall@10: 0.085, F1@10: 0.14, Precision@10: 0.4, Results Quality: 0.309, Search completed in 3.12 seconds.
Query: snowboard
Average Precision: 0.116, Recall@10: 0.062, F1@10: 0.077, Precision@10: 0.1, Results Quality: 0.0, Search completed in 1.91 seconds.
Query: Who is the founder of modern psychology?
Average Precision: 0.174, Recall@10: 0.041, F1@10: 0.068, Precision@10: 0.2, Results Quality: 0.0, Search completed in 13.86 seconds.
"""
lines1 = multiline_string1.strip().splitlines()
metrics1 = extract_metrics_from_lines(lines1)
stats1 = calculate_stats(metrics1)

lines2 = multiline_string2.strip().splitlines()
metrics2 = extract_metrics_from_lines(lines2)
stats2 = calculate_stats(metrics2)

# Print stats for the first test
print("Stats for Test 1:")
print_stats(stats1)

# Print stats for the second test
print("Stats for Test 2:")
print_stats(stats2)

# Compare metrics
print("\nDifferences in Metrics (Test 2 - Test 1):")
for metric in stats1:
    if metric in stats2:
        diff_average = stats2[metric]["Average"] - stats1[metric]["Average"]
        print(f"{metric}:")
        print(f"  Average Difference: {diff_average:.3f}")
