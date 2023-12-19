 # "Television Data Analysis" project

```python
# Television Data Analysis

## Importing Dependencies and CSV File

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.lines as mlines

filepath = os.path.join('Resources/TV data index.csv')
index_df = pd.read_csv(filepath, header=0) 
index_df.head()
```

## Network Performance based on Engagement Index

### Preparing Dataset

```python
indexNet = index_df[['series', 'network', 'Engagement_index']]
indexNet_grp = indexNet.groupby(['network']).agg(['mean', 'sem'])
indexNet_grp.columns = indexNet_grp.columns.map('_'.join)
indexNet_grp = indexNet_grp.reset_index()
sorted_indexNet = indexNet_grp.sort_values('Engagement_index_mean', ascending=False)
sorted_indexNet.head()
```

### Plotting Network and Engagement Index

```python
# Data Lists
network = sorted_indexNet['network'].tolist()
index_network = sorted_indexNet['Engagement_index_mean'].tolist()
sem = sorted_indexNet['Engagement_index_sem'].tolist()

# Bar Chart
fig, ax = plt.subplots(figsize=(15, 6))
x_axis = np.arange(len(network))
ax.set_xticks(x_axis)
ax.set_xticklabels(network, rotation=90)
plt.ylim(0, max(index_network)+0.5)
plt.axhline(y=1.0, linewidth=0.8, linestyle='-', color='blue', alpha=2, zorder=0)

plt.grid(linestyle='dotted', zorder=0)

plt.xlabel('Network', fontsize=12)
plt.ylabel('Engagement Index', fontsize=12)
plt.title('Network Performance based on Engagement Index', fontsize=16)

ax.bar(x_axis, index_network, yerr=sem, color='cyan', edgecolor='k')

plt.tight_layout()
img_path = os.path.join('Images', 'network_performance.png')
plt.savefig(img_path)
plt.show()
```

### Observations

```python
print(f'''
Better performing network is presented by all the bars that crossed the blue dotted line
''')
```

## Exploring Performance of Cable and Broadcast

### Preparing Dataset

```python
indexType_grp = index_df[['series', 'Type', 'Engagement_index']]
indexType_grp = indexType_grp.groupby(['Type']).agg(['mean', 'sem'])
indexType_grp.columns = indexType_grp.columns.map('_'.join)
indexType_grp = indexType_grp.reset_index()
sorted_indexType = indexType_grp.sort_values('Engagement_index_mean', ascending=False)
sorted_indexType.head()
```

### Plotting Broadcast and Cable Performance

```python
# Data Lists
Type = sorted_indexType['Type'].tolist()
index_Type = sorted_indexType['Engagement_index_mean'].tolist()
sem = sorted_indexType['Engagement_index_sem'].tolist()

# Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = np.arange(len(Type))
ax.set_xticks(x_axis)
ax.set_xticklabels(Type)
plt.ylim(0, max(index_Type)+0.5)

plt.grid(linestyle='dotted', zorder=0)

plt.xlabel('Broadcast Type', fontsize=12)
plt.ylabel('Engagement Index', fontsize=12)
plt.title('Broadcast/Cable Performance based on Engagement Index', fontsize=16)
plt.axhline(y=1.0, linewidth=0.8, linestyle='-', color='blue', alpha=2, zorder=0)

ax.bar(x_axis, index_Type, width=0.9, yerr=sem, color='cyan', edgecolor='k')

plt.tight_layout()
img_path = os.path.join('Images', 'broadcast_performance.png')
plt.savefig(img_path)
plt.show()
```

### Observations

```python
print(f'''
Broadcast is performing slightly better than the average while cable TV is below average
''')
```

## Viewer Engagement by Daypart

### Preparing Dataset

```python
indexDay_grp = index_df[['series', 'daypart', 'Engagement_index']]
indexDay_grp = indexDay_grp.groupby(['daypart']).agg(['mean', 'sem'])
indexDay_grp.columns = indexDay_grp.columns.map('_'.join)
indexDay_grp = indexDay_grp.reset_index()
sorted_indexDay = indexDay_grp.sort_values('Engagement_index_mean', ascending=False)
sorted_indexDay.head()
```

### Plotting Daypart and Engagement Index

```python
# Data Lists
Daypart = sorted_indexDay['daypart'].tolist()
index_Day = sorted_indexDay['Engagement_index_mean'].tolist()
sem = sorted_indexDay['Engagement_index_sem'].tolist()

# Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = np.arange(len(Daypart))
ax.set_xticks(x_axis)
ax.set_xticklabels(Daypart)
plt.ylim(0, max(index_Day)+0.5)

plt.grid(linestyle='dotted', zorder=0)

plt.xlabel('Day Part', fontsize=12)
plt.ylabel('Engagement Index', fontsize=12)
plt.title('Viewership by Daypart', fontsize=16)
plt.axhline(y=1.0, linewidth=0.8, linestyle='-', color='blue', alpha=2, zorder=0)

ax.bar(x_axis, index_Day, width=0.9, yerr=sem, color='cyan', edgecolor='k')

plt.tight_layout()
img_path = os.path.join('Images', 'TimeBlock_performance.png')
plt.savefig(img_path)
plt.show()
```

### Observations

```python
print(f'''
Seems unusual but overnight and latefringe viewers seem to have better viewer engagement than other day-time periods
''')
```

## Relationship Between Run Time and Engagement Index

### Defining Variables

```python
x = index_df['Run_time (min)'].values.tolist()
y = index_df['Engagement_index'].values.tolist()

# Linear Regression Stats and Fitline
slope, intercept, p_value, r_value, _ = stats.linregress(x, y)
fit = intercept + slope*(np.array(x))

# Scatter Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, color='cyan', edgecolor='grey', zorder=2)
plt.plot(x, fit, 'black', label='fitted line', linewidth=0.6, color='grey')
plt.grid(linestyle='dotted', zorder=0)

plt.xlabel('Run-time')
plt.ylabel('Engagement Index')
plt.title("Relationship between Run-time and Engagement Index", fontsize=16)

# Show the figure
plt.tight_layout()
img_path = os.path.join('Images', 'Runtime_index.png')
plt.savefig(img_path)
plt.show()
```

### Observations

```python
print(f'''
There is no relationship between Engagement Index and Run-time shown by r2_value = {r_value**2}
''')
```

Feel free to adapt this documentation to your specific needs!
