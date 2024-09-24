#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: ANTONY ROSARIO JOHN PETER
# #### Student ID: S3940203
# 
# Date: 01/10/2023
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
# 
# In Task 1 of this project, we focus on performing basic text pre-processing on a dataset of job advertisements. The goal of this task is to prepare the job advertisement descriptions for further analysis. The key pre-processing steps include tokenization, lowercasing, removal of short words, elimination of stopwords, filtering out words that appear only once, and discarding the top 50 most frequent words based on document frequency. The cleaned and pre-processed job advertisement descriptions are then saved for later use, and a vocabulary of the cleaned descriptions is built and saved in a specific format.
# 
# This initial task is crucial as it lays the foundation for subsequent tasks, allowing us to work with more structured and meaningful textual data. The output of this task includes important files, such as 'vocab.txt,' which contains the vocabulary of words used in the descriptions, and it serves as a reference for encoding text data. These files are essential for interpreting and analyzing the job advertisement descriptions effectively in later stages of the project.

# ## Importing libraries 

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import regex as re
import os


# In[2]:


import nltk
nltk.download('punkt')


# ### 1.1 Examining and loading data
# 
# Findings from the Data Folder:
# 
# The data folder contains a small collection of job advertisement documents, totaling around 770 jobs.
# Inside the data folder, there are four different subfolders, each representing a job category. These subfolders are named: "Accounting_Finance," "Engineering," "Healthcare_Nursing," and "Sales."
# Each subfolder's name corresponds to a specific job category, suggesting that the job advertisements are categorized into these four fields.
# The job advertisement text documents for a particular category are stored in the respective subfolder.
# Format of Text Files:
# 
# Each job advertisement document is stored as a text file with the following format: "Job_<ID>.txt," where "<ID>" represents a unique identifier for the job advertisement.
# Contents of Text Files:
# 
# Each job advertisement text file contains the following information:
# Title: The job title.
# Webindex: The web index associated with the job advertisement (some may also have information on the company name).
# Description: The full description of the job advertisement.
#     
# Data Preparation:
# The provided code performs the following steps to prepare the data for further processing:
# 
# Initializes empty lists (Job_ID, Job_Category, job_titles, job_webindices, and job_description) to store extracted data.
# Changes the working directory to the data folder where the job advertisement documents are located.
# Loops through the subfolders and files in the data directory:
# Skips files named '.DS_Store' or any other non-directory items.
# Changes to the current subfolder (e.g., "Accounting_Finance," "Engineering") for processing.
# For each text file in the current subfolder:
# Extracts the job ID from the filename and appends it to the Job_ID list.
# Appends the current job category (subfolder name) to the Job_Category list.
# Opens and reads the current text file to extract information.
# Extracts and appends the job title, web index, and job description to their respective lists (job_titles, job_webindices, and job_description).
# After processing all text files in a subfolder, the code changes back to the parent data directory.
# Finally, it prints the extracted data for verification, including lists of job IDs, job categories, job titles, web indices, and job descriptions.
# The data is now organized and stored in appropriate data structures, ready for further processing and analysis.
# 

# In[3]:


# Initialize lists to store extracted data
Job_ID = []
Job_Category = []
job_titles = []
job_webindices = []
job_description = []


# In[4]:


# Change working directory to the data folder
os.chdir('/Users/antonyrosario/Documents/RMIT/RMIT Projects/Sem 3/Advance Prog/Assignment 2/a2-milestone1/data/')

# Loop through folders and files in the data directory
for folder in sorted(os.listdir()):
    # Skip '.DS_Store' files or any other non-directory items
    if not os.path.isdir(folder) or folder.startswith('.DS_Store'):
        continue

    os.chdir(folder)  # Change to the current subfolder
    
    # Loop through text files in the current folder
    for filetxt in sorted(os.listdir()):
        if filetxt.endswith(".txt"):
            # Extract job ID and category from the file and append to respective lists
            Job_ID.append(filetxt.split('.')[0])
            Job_Category.append(folder)
            
            # Open and read the current text file
            file = open(filetxt, 'r', encoding='utf-8')
            lines = file.readline()
            
            while lines:
                pos1 = lines.strip().split(' ')
                
                if pos1[0] == 'Title:':
                    # Extract job title and append to the list
                    pos = lines.strip().split('Title:')
                    job_titles.append(pos[1])

                if pos1[0] == 'Webindex:':
                    # Extract web index and append to the list
                    pos = lines.strip().split('Webindex:')
                    job_webindices.append(pos[1])

                if pos1[0] == 'Description:':
                    # Extract job description and append to the list
                    pos = lines.strip().split('Description:')
                    job_description.append(pos[1])  

                lines = file.readline()
            
    os.chdir('..')  # Change back to the parent directory
os.chdir('..')  # Change back to the main data directory


# In[5]:


# Print extracted data for verification
print("Job IDs:\n", Job_ID)
print("Job Categories:\n", Job_Category)
print("Job Titles:\n", job_titles)
print("Web Indices:\n", job_webindices)
print("Job Descriptions:\n", job_description)


# ### 1.2 Pre-processing data

# #### A) Tokenzation

# In[6]:


# Tokenization and sentence splitting
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain

def tokenizeDescription(raw_description):
    """
    Tokenizes a job description by converting words to lowercase,
    segmenting into sentences, and tokenizing each sentence.
    Returns a list of tokens.
    """        
    job_description = raw_description.lower()  # Convert all words to lowercase
    
    # Segment into sentences
    sentences = sent_tokenize(job_description)
    
    # Tokenize each sentence using regex pattern
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 

    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # Merge sentence tokens into a single list of tokens
    tokenized_description = list(chain.from_iterable(token_lists))
    return tokenized_description

# Tokenize job descriptions
tk_description = [tokenizeDescription(r) for r in job_description]

# to save this file for task 2
tk_descripn = [tokenizeDescription(r) for r in job_description]


# In[7]:


# Function to compute and print basic statistics about the tokenized data
def stats_print(tk_description):
    words = list(chain.from_iterable(tk_description))  # Combine tokens into a single list
    vocab = set(words)  # Calculate the vocabulary (unique words)
    lexical_diversity = len(vocab) / len(words)
    
    print("Vocabulary size:", len(vocab))
    print("Total number of tokens:", len(words))
    print("Lexical diversity:", lexical_diversity)
    print("Total number of job descriptions:", len(tk_description))
    lens = [len(article) for article in tk_description]
    print("Average job description length:", np.mean(lens))
    print("Maximum job description length:", np.max(lens))
    print("Minimum job description length:", np.min(lens))
    print("Standard deviation of job description length:", np.std(lens))


# In[8]:


print("Raw Job Descriptions:\n", job_description)


# In[9]:


print("Tokenized Description:\n",tk_description)


# In[10]:


stats_print(tk_description)


# #### B) Removing words with length less than 2

# In[11]:


st_list = [[w for w in job_description if len(w) < 2 ]                       for job_description in tk_description]
list(chain.from_iterable(st_list))


# In[12]:


# filter out one character tokens
tk_description = [[w for w in job_description if len(w) >= 2]                       for job_description in tk_description]


# In[13]:


print("Tokenized Description:\n",tk_description)


# #### C) Removing Stop words

# In[14]:


from nltk.corpus import stopwords

# Initialize an empty list to store stopwords
stop_words = []

# Open the stopwords file and read stopwords into the list
with open('stopwords_en.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stop_words.append(line.strip())  # Remove leading/trailing whitespace and append to the list


# In[15]:


# Filter out stopwords from tokenized descriptions
tk_description = [[w for w in job_description if w not in stop_words] for job_description in tk_description]


# In[16]:


# Print tokenized descriptions after removing stopwords
print("Tokenized Descriptions after removing stopwords:\n", tk_description)


# In[17]:


stats_print(tk_description)


# #### D) Remove the words that appear only once in the document collection, based on term frequency

# In[18]:


from nltk.probability import *
from itertools import chain


# In[19]:


# Calculate the frequency distribution of words in the entire corpus
words = list(chain.from_iterable(tk_description))
term_fd = FreqDist(words)


# In[20]:


# Identify less frequent words (hapaxes)
lessFreqWords = set(term_fd.hapaxes())
lessFreqWords


# In[21]:


len(lessFreqWords)


# In[22]:


# Remove less frequent words from tokenized descriptions
def removeLessFreqWords(description):
    return [w for w in description if w not in lessFreqWords]

tk_description = [removeLessFreqWords(description) for description in tk_description]


# In[23]:


# Print tokenized descriptions after removing less frequent words
print("Tokenized Descriptions after removing less frequent words:\n", tk_description)


# #### E) Remove the top 50 most frequent words based on document frequency.

# In[24]:


# Calculate document frequency for each unique word/type
words_2 = list(chain.from_iterable([set(description) for description in tk_description]))
doc_fd = FreqDist(words_2)

# Identify the most frequent words by document frequency (top 25 in this case)
df_words = set(w[0] for w in doc_fd.most_common(25))


# In[25]:


# Remove the most frequent words by document frequency from tokenized descriptions
def removeMostFreqWords(description):
    return [w for w in description if w not in df_words]

tk_description = [removeMostFreqWords(description) for description in tk_description]

# 


# In[26]:


# Print tokenized descriptions after removing the most frequent words by document frequency
print("Tokenized Descriptions after removing the most frequent words by document frequency:\n", tk_description)


# In[27]:


stats_print(tk_description)


# #### F) Build a vocabulary of the cleaned job advertisement descriptions

# In[28]:


# generating the vocabulary

# Create a vocabulary from the cleaned tokenized descriptions
words = list(chain.from_iterable(tk_description))
vocab = sorted(list(set(words)))  # Compute the vocabulary as a sorted list of unique words

# Print vocabulary statistics
print("Vocabulary size:", len(vocab))


# ## Saving required outputs

# In[29]:


# Save the vocabulary to a text file 'vocab.txt'
output_file = open("./vocab.txt", 'w')
for ind in range(0, len(vocab)):
    output_file.write("{}:{}\n".format(vocab[ind], ind))
output_file.close()


# # Saving job advertisement and information

# In[30]:


# Save tokenized job descriptions to 'Description.txt'
output_file = open("./Description.txt", 'w')
for tokens in tk_descripn:
    output_file.write(' '.join(tokens) + '\n')
output_file.close()


# In[31]:


# Save other information to respective text files
output_file = open("./Info.txt", 'w')
for tokens in tk_description:
    output_file.write(' '.join(tokens) + '\n')
output_file.close()

output_file = open("./WebIndex.txt", 'w')
for index in job_webindices:
    output_file.write(''.join(index) + '\n')
output_file.close()

output_file = open("./Categories.txt", 'w')
for tokens in Job_Category:
    output_file.write(''.join(tokens) + '\n')
output_file.close()

output_file = open("./Title.txt", 'w')
for tokens in job_titles:
    output_file.write(''.join(tokens) + '\n')
output_file.close()


# ## Summary
# 
# Data Examination and Loading:
# The dataset consists of approximately 770 job advertisements categorized into four subfolders: "Accounting_Finance," "Engineering," "Healthcare_Nursing," and "Sales."
# Each job advertisement is stored as a text file named "Job_<ID>.txt" and includes essential information such as job title, web index, and job description.
#     
# Data Preparation:
# The code initializes empty lists to store extracted data, such as job IDs, job categories, job titles, web indices, and job descriptions.
# It systematically traverses the data directory, skipping non-directory items, and collects relevant information from text files.
# The extracted data is organized into structured data structures for further processing.
#     
# Text Preprocessing:
# Tokenization: The job descriptions are tokenized, converting all words to lowercase and segmenting text into sentences using regular expressions and NLTK's tokenizer.
# Removal of Short Words: Words with a length less than 2 characters are removed from the tokenized descriptions.
# Stopword Removal: Stopwords from the provided list are removed to focus on meaningful content.
# Removal of Infrequent Words: Words that appear only once in the entire collection (hapaxes) are eliminated.
# Removal of Common Words: The top 50 most frequently occurring words based on document frequency are removed to reduce noise.
#     
# Building Vocabulary:
# A vocabulary is generated from the cleaned and preprocessed tokenized job descriptions, providing a comprehensive list of unique words.
# Saving Outputs:
# The vocabulary is saved to a text file named "vocab.txt" in alphabetical order, with word-index pairs.
# Tokenized job descriptions are saved to "Description.txt."
# Other relevant information, including web indices, job categories, and job titles, is stored in respective text files.
# Overall, these steps prepare the data for subsequent analysis and text mining tasks, ensuring that it is clean, structured, and ready for further exploration.
