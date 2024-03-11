import matplotlib.pyplot as plt
import numpy as np

metric = ['Search completed in']
averages = [12.568]
min_values = [2.040]
max_values = [44.260]

x = np.arange(len(metric))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, min_values, width, label='Min')
rects2 = ax.bar(x, averages, width, label='Average')
rects3 = ax.bar(x + width/3, max_values, width, label='Max')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Seconds')
ax.set_title('Search Completion Time')
ax.set_xticks(x)
ax.set_xticklabels(metric)
ax.legend()

fig.tight_layout()

plt.show()