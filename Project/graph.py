import matplotlib.pyplot as plt
import numpy as np

metrics = ['Average Precision', 'Recall@10', 'F1@10', 'Precision@10', 'Results Quality']
averages = [0.238, 0.042, 0.068, 0.187, 0.111]
min_values = [0.0, 0.0, 0.0, 0.0, 0.0]
max_values = [0.683, 0.128, 0.211, 0.600, 0.453]

x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, min_values, width, label='Min')
rects2 = ax.bar(x, averages, width, label='Average')
rects3 = ax.bar(x + width, max_values, width, label='Max')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Test 1 search_body')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()

plt.xticks(rotation=45)
plt.show()

