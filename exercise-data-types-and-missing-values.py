#!/usr/bin/env python
# coding: utf-8

# **This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/data-types-and-missing-values).**
# 
# ---
# 

# # Introduction
# 
# Run the following cell to load your data and some utility functions.

# In[1]:


import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.data_types_and_missing_data import *
print("Setup complete.")


# # Exercises

# ## 1. 
# What is the data type of the `points` column in the dataset?

# In[3]:


# Your code here
dtype = reviews.points.dtype

# Check your answer
q1.check()
dtype


# In[4]:


#q1.hint()
#q1.solution()


# ## 2. 
# Create a Series from entries in the `points` column, but convert the entries to strings. Hint: strings are `str` in native Python.

# In[6]:


point_strings = reviews.points.astype(str)

# Check your answer
q2.check()
point_strings


# In[7]:


#q2.hint()
#q2.solution()


# ## 3.
# Sometimes the price column is null. How many reviews in the dataset are missing a price?

# In[9]:


missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
n_missing_prices = reviews.price.isnull().sum()

# Check your answer
q3.check()
n_missing_prices 


# In[10]:


#q3.hint()
#q3.solution()


# ## 4.
# What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:
# 
# ```
# Unknown                    21247
# Napa Valley                 4480
#                            ...  
# Bardolino Superiore            1
# Primitivo del Tarantino        1
# Name: region_1, Length: 1230, dtype: int64
# ```

# In[13]:


reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)

# Check your answer
q4.check()
reviews_per_region


# In[12]:


#q4.hint()
#q4.solution()


# # Keep going
# 
# Move on to **[renaming and combining](https://www.kaggle.com/residentmario/renaming-and-combining)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
