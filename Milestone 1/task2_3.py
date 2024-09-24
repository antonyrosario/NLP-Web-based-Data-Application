#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: ANTONY ROSARIO JOHN PETER
# #### Student ID: S3940203
# 
# Date: 01-10-2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# 
# In Task 2, we focus on generating different types of feature representations for job advertisement descriptions. These feature representations will be used as input data for building machine learning models in Task 3 to classify job advertisements into their respective categories.
# 
# Here is a summary of the steps and processes involved in Task 2:
# 
# Loading Data: We start by loading the necessary data from text files, including job descriptions, job titles, web indices, job categories, and the vocabulary generated in Task 1.
# Count Vector Representation: We use the Count Vectorizer from scikit-learn to generate the Count Vector representation for each job advertisement description. This representation is based on the vocabulary created in Task 1. The generated Count Vectors are saved into a file named "count_vectors.txt" in the specified format.
# Word Embeddings: We initialize a FastText model to generate word embeddings for the job advertisement descriptions. Both weighted (TF-IDF weighted) and unweighted versions of the embeddings are generated.
# Saving Outputs: We save the FastText model and the Count Vector representation as specified in the task requirements.
# Task 3, on the other hand, focuses on building machine learning models for job advertisement classification and addresses two specific questions:
# 
# Q1: Language Model Comparisons
# 
# We compare the performance of different language models (Count Vector, TF-IDF, and FastText embeddings) when combined with a logistic regression model for job advertisement classification.
# We evaluate and compare the accuracy and effectiveness of these models using 5-fold cross-validation.
# 
# Q2: Does More Information Provide Higher Accuracy?
# 
# We investigate whether including additional information, specifically the job title, improves the accuracy of the classification models.
# We compare the performance of models that consider:
# Only the job title
# Only the job description (as generated in Task 2)
# Both the job title and job description (combined)
# We evaluate these models using 5-fold cross-validation.
# The provided code outlines the steps and processes for Task 2 and Task 3, including data loading, feature representation generation, model building, and evaluation. The final results and comparisons are provided for each question addressed in Task 3.
# 

# ## Importing libraries 

# In[1]:


# Code to import libraries as needed in this assessment

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# In[2]:


# Load data from text files
job_description = './Description.txt'
with open(job_description) as f: 
    job_description = f.read().splitlines()  # Read all the job descriptions into a list
len(job_description)


# In[3]:


# Split job descriptions into tokens
job_description = [content.split(" ") for content in job_description]


# In[4]:


job_title = './Title.txt'
with open(job_title) as f: 
    job_title = f.read().splitlines()  # Read all the job titles into a list
len(job_title)


# In[5]:


job_webindices = './WebIndex.txt'
with open(job_webindices) as f: 
    job_webindices = f.read().splitlines()  # Read all the web indices into a list
len(job_webindices)


# In[6]:


job_category = './Categories.txt'
with open(job_category) as f: 
    job_category = f.read().splitlines()  # Read all the job categories into a list
len(job_category)


# In[7]:


vocab = './vocab.txt'
with open(vocab) as f: 
    vocab = f.read().splitlines()

# Split vocabulary into words
vocab = [word.split(':') for word in vocab]
vocabulary = []
for i in range(len(vocab)):
    vocabulary.append(vocab[i][0])


# In[8]:


# Create a dictionary to store data
data = dict()
data['category'] = list(job_category)
data['webIndex'] = list(job_webindices)
data['description'] = list(job_description)
data['title'] = list(job_title)


# In[9]:


data


# ### A) Count Vectorization 

# In[10]:


# Load your vocabulary 'voc', job descriptions 'job_description', and other necessary data

# Initialize the Count Vectorizer
count_vectorizer = CountVectorizer(analyzer="word", vocabulary=vocabulary)
count_vectorizer


# In[11]:


# Generate the Count Vector representation for all job descriptions
count_attributes = count_vectorizer.fit_transform([' '.join(des) for des in job_description])
print(count_attributes.shape)

count_attributes


# In[12]:


count_vec = count_attributes.toarray()


# ### B) TF-IDF Vectorization 

# In[13]:


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocabulary)

# Generate the TF-IDF representation for all job descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(des) for des in job_description])
print(tfidf_matrix.shape)


# ### C) Word Embedding with FastText

# In[14]:



# generating word embeddings

#corpus file names/path
corpus_file =  './Description.txt'

# Initialize the FastText model
job_FT = FastText(vector_size=100)

# Build the vocabulary
job_FT.build_vocab(corpus_file=corpus_file)

# Train the model
job_FT.train(
    corpus_file=corpus_file, epochs=job_FT.epochs,
    total_examples=job_FT.corpus_count, total_words=job_FT.corpus_total_words,
)

print(job_FT)


# In[15]:


# Initialize an empty list to store TF-IDF weighted embeddings
fasttext_embeddings_weighted = []

# Generate FastText embeddings for each job description (TF-IDF weighted)
def generate_fasttext_embeddings(description, model):
    tokens = word_tokenize(description.lower())
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no valid tokens

for i in range(len(job_description)):
    description = " ".join(job_description[i])
    tfidf_weights = tfidf_matrix[i].toarray()[0]
    weighted_embeddings = generate_fasttext_embeddings(description, job_FT)
    tfidf_weighted_embeddings = weighted_embeddings * tfidf_weights[:, np.newaxis]
    fasttext_embeddings_weighted.append(tfidf_weighted_embeddings)

# Initialize an empty list to store unweighted FastText embeddings
fasttext_embeddings_unweighted = []

# Generate FastText embeddings for each job description (unweighted)
for i in range(len(job_description)):
    description = " ".join(job_description[i])
    unweighted_embeddings = generate_fasttext_embeddings(description, job_FT)
    fasttext_embeddings_unweighted.append(unweighted_embeddings)


# In[16]:


job_FT_wv = job_FT.wv
print(job_FT_wv)


# In[17]:


# Save the FastText model
job_FT.save("job_FT.model")


# ### D) Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[18]:


# Save the Count Vector representation as 'count_vectors.txt'
df_X_count = pd.DataFrame(count_vec)

with open('count_vectors.txt', 'w') as f_count:
    for k in range(df_X_count.shape[0]):
        x_count = str(job_webindices[k]).strip()
        if k == 0:
            f_count.write(f'#{x_count},')
        else:
            f_count.write(f'\n#{x_count},')

        dict_count = dict()
        for i_count, j_count in zip(range(len(list(df_X_count.iloc[k]))), list(df_X_count.iloc[k])):
            if j_count > 0:
                dict_count.update({i_count: j_count})

        for i_count in dict_count.keys():
            if i_count != list(dict_count.keys())[-1]:
                f_count.write(f'{i_count}:{dict_count[i_count]},')
            if i_count == list(dict_count.keys())[-1]:
                f_count.write(f'{i_count}:{dict_count[i_count]}')
    f_count.close()


# ## Task 3. Job Advertisement Classification

# ### A) Count Vector Modeling

# In[19]:


# Initialize random seed for reproducibility
seed = 20

# Split data into training and testing sets for Count Vector model
X_train, X_test, y_train, y_test = train_test_split(count_vec, job_category, test_size=0.33, random_state=seed)

# Initialize and train a logistic regression model for Count Vector model
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[20]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy_score(y_test, y_pred, normalize=False)


# In[21]:


# Generate the confusion matrix
con_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", con_matrix)


# In[22]:


# Sort job categories for better visualization
category_sorted = sorted(list(set(job_category)))

# Plot a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(con_matrix, annot=True, fmt='d',
            xticklabels=category_sorted, yticklabels=category_sorted)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# ### B) TF-IDF Modeling

# In[23]:


# Split data into training and testing sets for TF-IDF model
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, job_category, test_size=0.33, random_state=seed)

# Initialize and train a logistic regression model for TF-IDF model
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[24]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy_score(y_test, y_pred, normalize=False)


# In[25]:


# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[26]:


# Sort job categories for better visualization
category_sorted = sorted(list(set(job_category)))

# Plot a heatmap of the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=category_sorted, yticklabels=category_sorted)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# ### C) FastText Modeling

# In[27]:


# Generate document vectors for FastText model
def generate_document_vectors(wv, text_tokens):
    doc_vectors = pd.DataFrame()
    
    for i in range(0, len(text_tokens)):
        tokens = text_tokens[i]
        temp = pd.DataFrame()
        for w in range(0, len(tokens)):
            try:
                word = tokens[w]
                word_vec = wv[word]
                temp = temp.append(pd.Series(word_vec), ignore_index=True)
            except:
                pass
        doc_vector = temp.sum()
        doc_vectors = doc_vectors.append(doc_vector, ignore_index=True)
    return doc_vectors


# In[28]:


job_FT = FastText.load("job_FT.model")
print(job_FT)
job_FT_wv = job_FT.wv


# In[29]:


# Generate document vectors for job descriptions
job_FT_document_vectors = generate_document_vectors(job_FT_wv, data['description'])
job_FT_document_vectors.isna().any().sum()


# In[30]:


# Convert document vectors to a NumPy array
ft_attributes = job_FT_document_vectors.to_numpy()


# In[31]:


# Create training and test splits for FastText model
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(job_FT_document_vectors, data['category'], list(range(0,len(data['category']))), test_size=0.33, random_state=seed)

# Initialize and train a logistic regression model for FastText model
model = LogisticRegression(max_iter=4000, random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ### D) K-fold cross-validation

# In[32]:


# K fold

number_folds = 5
seed = 20
kfold = KFold(n_splits=number_folds, random_state=seed, shuffle=True)  # Initialize a 5-fold validation
print(kfold)


# In[33]:


# Function to evaluate a model using K-fold cross-validation
def evaluate(X_train, X_test, y_train, y_test, seed):
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[34]:


# Number of models and data storage
number_models = 3
count_vec_data = pd.DataFrame(columns=['count', 'tfidf', 'fasttext'], index=range(number_folds))

# Initialize fold counter
folds = 0

# Perform K-fold cross-validation
for train_index, test_index in kfold.split(list(range(0,len(job_category)))):
    y_train = [job_category[i] for i in train_index]
    y_test = [job_category[i] for i in test_index]
 
    X_train_count, X_test_count = count_attributes[train_index], count_attributes[test_index]
    count_vec_data.loc[folds,'count'] = evaluate(count_attributes[train_index], count_attributes[test_index], y_train, y_test, seed)

    X_train_tfidf, X_test_tfidf = tfidf_matrix[train_index], tfidf_matrix[test_index]
    count_vec_data.loc[folds,'tfidf'] = evaluate(tfidf_matrix[train_index], tfidf_matrix[test_index], y_train, y_test, seed)
    
    X_train_fasttext, X_test_fasttext = ft_attributes[train_index], ft_attributes[test_index]
    count_vec_data.loc[folds,'fasttext'] = evaluate(ft_attributes[train_index], ft_attributes[test_index], y_train, y_test, seed)
    
    folds += 1


# In[35]:


count_vec_data


# In[36]:


# Calculate mean accuracy scores for each feature representation
count_vec_data.mean()


# ### E) Comparing with only title

# In[37]:


# Initialize the TF-IDF vectorizer for job titles
tfidf_vectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocabulary)

# Generate the TF-IDF representation for all job titles
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(des) for des in job_title])
tfidf_matrix.shape


# In[38]:


seed = 1000

# Split data into training and testing sets for TF-IDF model (job titles only)
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, job_category, test_size=0.33, random_state=seed)

# Initialize and train a logistic regression model for TF-IDF model (job titles only)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ### F) Comparing job titles and descriptions

# In[39]:


# Combine job titles and descriptions
combined_text = [" ".join(title) + " " + " ".join(description) for title, description in zip(job_title, job_description)]


# In[40]:


# Initialize the Count Vectorizer for combined text
count_vectorizer = CountVectorizer(analyzer="word", vocabulary=vocabulary)

# Generate the Count Vector representation for the combined text
count_features = count_vectorizer.fit_transform(combined_text)

# Split the data into training and testing sets for combined text
seed = 20
X_train, X_test, y_train, y_test = train_test_split(count_features, job_category, test_size=0.33, random_state=seed)

# Initialize and train a logistic regression model for combined text
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ## Summary
# 
# 
# #### Q1: Language model comparisons
# #### Which language model we built previously (based on job advertisement descriptions) performs the best with the chosen machine learning model? 
# 
# We sought to determine the most effective language model for job advertisement classification when paired with the Logistic Regression machine learning model. Three distinct models were compared:
# 
# Count Vectorization achieved an mean accuracy score of 0.866013.
# 
# TF-IDF Vectorization outperformed others with an mean accuracy score of 0.890422.
# 
# FastText Embeddings exhibited an mean accuracy score of 0.730678.
# 
# The TF-IDF model demonstrated superior performance, achieving an accuracy score of 0.923077 in comparison to Count Vectorization (0.839744) and FastText (0.724359). This indicates that TF-IDF is the optimal choice for representing job advertisement descriptions when coupled with Logistic Regression.
# 
# ####  Q2: Does more information provide higher accuracy?
# ####  In Task 2, we have built a number of feature representations of documents based on job advertisement descriptions. However, we have not explored other features of a job advertisement, e.g., the title of the job position. Will adding extra information help to boost up the accuracy of the model?
# 
# Only Title: The accuracy score was a mere 0.315, suggesting that using job titles alone is insufficient for accurate classification.
# 
# Both Title and Description: By combining job titles and descriptions, the accuracy notably improved to 0.848249. This illustrates the significance of incorporating both sets of information.
# 
# Only Description: Surprisingly, the model performed best when utilizing job descriptions exclusively, achieving an accuracy score of 0.923077.
# 
# In conclusion, our findings underscore the paramount importance of job descriptions for accurate classification. While including job titles can be beneficial, the description alone yielded the highest accuracy, reaffirming its central role in the classification process.
