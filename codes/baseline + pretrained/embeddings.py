from transformers import BioGptTokenizer, BioGptForCausalLM, AutoTokenizer, BioGptModel, pipeline, set_seed, BioGptForCausalLM, AutoModel
import numpy as np
import re
import string
import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv
from gensim.models import Word2Vec, FastText, Doc2Vec
import gensim
import torch
from transformers import BertTokenizer, BertModel
import logging
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
import torch.nn as nn
tf.compat.v1.disable_eager_execution()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dim = 15
#print("test")

# This line prevents warnings from popping up
import warnings
warnings.filterwarnings("ignore")



# this function gives us the weight vector for each sentence 
# Note this code was adopted from Bing Wen's code
def get_weights(model,textdata):
    sentence_weights = []
    # checking whether the word is present in the word-to-index dictionary
    for text in textdata:
        word_arr = text.split()
        current_sum = np.zeros(dim)
        count = 0
        # if word inside the word-to-index dictionary count them in
        for word in word_arr:
            if word in model.wv.key_to_index:
                count = count + 1
                current_sum = current_sum + model.wv[word]
        current_sum = current_sum
        sentence_weights.append(current_sum)
    #print(len(sentence_weights))
    return (sentence_weights)

# this function gets the embeddings for continous bag-of-words 
# Note: This code was adopted from Bing Wen's code
def get_cbow(input_data,all_text,window_size):
    #print("Getting cbow embeddings")
    # implementing the model. 
    model = Word2Vec(input_data, vector_size=dim, window=window_size, min_count=1)
    sentence_weights = get_weights(model,all_text)
    # getting the array 
    X = np.array(sentence_weights).astype(float)
    #data = pd.DataFrame(data=X)
    #data.to_csv('cases_details_cbow.csv',index=False)
    return (X)

# this function gets the embeddings for doc-2-vec
# note: this code was adopted from Bing Wen's code
def get_doc2vec(input_data,all_text,window_size):
    #print("Getting doc2vec embeddings")
    # assign tags to individual sentences in the input text data using gensim.models.doc2vec.TaggedDocument
    def create_tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    train_data = list(create_tagged_document(input_data))
    # implementing the model. 
    model = Doc2Vec(train_data, vector_size=dim, window=window_size, min_count=1)
    sentence_weights = get_weights(model,all_text)
    # getting the array 
    X = np.array(sentence_weights).astype(float)
    #data = pd.DataFrame(data=X)
    #data.to_csv('cases_details_doc2.csv',index=False)
    return (X)

# this function gets the embeddings for fast-text
# Note: this code was adopted from Bing Wen's code
def get_fasttext(input_data,all_text,window_size):
    # identifying the fast text model
    model = FastText(input_data, vector_size=dim, window=window_size, min_count=1)
    # gettting the weights
    sentence_weights = get_weights(model,all_text)
    # getting the embeddings
    X = np.array(sentence_weights).astype(float)
    #data = pd.DataFrame(data=X)
    #data.to_csv('cases_details_fasttext.csv',index=False)
    return X

# note this code gets the embeddings from Glove. 
def get_glove(input_data):

    # Getting glove embeddings
    # traning with 'glove.6B.300d.txt'. If you want a pre-trained model of smaller dimensions, please go to https://nlp.stanford.edu/projects/glove/
    # Load pre-trained word embeddings from file
    glove = {}
    # storing the word as the key and its associated word vector - the word vector is extracted from the pretrained "glove.6B.300d.txt". 
    # if you wish to use a larger pre-trained glove model, go to https://nlp.stanford.edu/projects/glove/ 
    with open("glove.6B.300d.txt", encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            glove[values[0]] = np.asarray(values[1:], dtype='float32')

    # run through each document and each word -- see if the word is in the pre-trained Glove -- if it is include it in our embeddings. 
    sentence_vectors = []
    for sentence in tqdm(input_data):
        vec = np.zeros(300)
        count = 0
        # extract each word and see if it is in the pre-trained glove
        for word in sentence:
            if word in glove:
                # if word is in the pre-trained glove, we include it in our vector
                vec += glove[word]
                count += 1
        # average the vectors out
        if count != 0:   
            vec /= count
        sentence_vectors.append(vec)

    # converting to array
    X=np.array(sentence_vectors)
    return (X)

# compute the mean pooling of the token embeddings
def mean_pooling(token_embeddings, attention_mask):
    # Defining the batch size, please adjust accordingly if needed
    batch_size = 8
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    n = token_embeddings.size(0)
    pooled_embeddings = []

    # Pooling in batches to avoid crashing
    for i in (range(0, n, batch_size)):
        # Getting the current batch
        batch_embeddings = token_embeddings[i:i+batch_size]
        batch_mask = input_mask_expanded[i:i+batch_size]

        # If CUDA is available, move the tensors to GPU
        if torch.cuda.is_available():
            batch_embeddings = batch_embeddings.to("cuda")
            batch_mask = batch_mask.to("cuda")

        # Taking the sum
        sum_embeddings = torch.sum(batch_embeddings * batch_mask, 1)
        sum_mask = torch.clamp(batch_mask.sum(1), min=1e-9)
        batch_pooled_embeddings = torch.div(sum_embeddings, sum_mask)
        pooled_embeddings.append(batch_pooled_embeddings)

    # Gathering the embeddings that were trained in batches.
    pooled_embeddings = torch.cat(pooled_embeddings, dim=0)

    # If CUDA is available, move the embeddings back to CPU
    if torch.cuda.is_available():
        pooled_embeddings = pooled_embeddings.to("cpu")

    return pooled_embeddings




# compute the mean pooling of the token embeddings
def get_biogpt_embeddings(texts, batch_size=64, max_length=128):
    notes = list(texts)
    # Load the tokenizer and model
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt")

    # Check if CUDA is available and if so, move the model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Tokenize texts and create input IDs and attention masks
    input_ids, attention_masks = [], []
    for text in notes:
        # please change max_length according to task if needed [NOTE: max_length may differ across task]
        encoded_input = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True,
                                              return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_input['input_ids'])
        attention_masks.append(encoded_input['attention_mask'])

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # If CUDA is available, move the tensors to GPU
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_masks = attention_masks.to("cuda")

    # Batch size for processing
    batch_size = 64
    # Initialize an empty list to store embeddings
    embeddings = []
    # Process input IDs and attention masks in batches
    for i in tqdm(range(0, len(input_ids), batch_size)):
        # print("Progress", (i/len(input_ids))*100)
        input_ids_batch = input_ids[i:i+batch_size]
        attention_masks_batch = attention_masks[i:i+batch_size]

        with torch.no_grad():
            # Get the model's output
            output = model(input_ids=input_ids_batch, attention_mask=attention_masks_batch)

        # Retrieve the last_hidden_state from the output
        last_hidden_state = output.last_hidden_state

        # Optionally, you can obtain the pooled_output by averaging or max-pooling the last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)

        # If CUDA is available, move the embeddings back to CPU
        if torch.cuda.is_available():
            pooled_output = pooled_output.to("cpu")

        # Add the embeddings to the list
        embeddings.extend(pooled_output.numpy())

    embeddings = np.array(embeddings)
    return embeddings



def get_clinicalBERT(texts):
    notes = list(texts)
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

    # Check if CUDA is available and if so, move the model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Tokenize texts and create input IDs and attention masks
    input_ids, attention_masks = [], []
    for text in notes:
        # please change max_length according to task if needed [NOTE: max_length may differ across task]
        encoded_input = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True,
                                              return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_input['input_ids'])
        attention_masks.append(encoded_input['attention_mask'])

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # If CUDA is available, move the tensors to GPU
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_masks = attention_masks.to("cuda")

    # Batch size for processing
    batch_size = 64
    # Initialize an empty list to store embeddings
    embeddings = []
    # Process input IDs and attention masks in batches
    for i in tqdm(range(0, len(input_ids), batch_size)):
        # print("Progress", (i/len(input_ids))*100)
        input_ids_batch = input_ids[i:i+batch_size]
        attention_masks_batch = attention_masks[i:i+batch_size]

        with torch.no_grad():
            # Get the model's output
            output = model(input_ids=input_ids_batch, attention_mask=attention_masks_batch)

        # Retrieve the last_hidden_state from the output
        last_hidden_state = output.last_hidden_state

        # Optionally, you can obtain the pooled_output by averaging or max-pooling the last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)

        # If CUDA is available, move the embeddings back to CPU
        if torch.cuda.is_available():
            pooled_output = pooled_output.to("cpu")

        # Add the embeddings to the list
        embeddings.extend(pooled_output.numpy())

    embeddings = np.array(embeddings)
    return embeddings


def get_bioclinicalBERT(texts):
    notes = list(texts)
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Check if CUDA is available and if so, move the model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Tokenize texts and create input IDs and attention masks
    input_ids, attention_masks = [], []
    for text in notes:
        # please change max_length according to task if needed [NOTE: max_length may differ across task]
        encoded_input = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True,
                                              return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_input['input_ids'])
        attention_masks.append(encoded_input['attention_mask'])

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # If CUDA is available, move the tensors to GPU
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_masks = attention_masks.to("cuda")

    # Batch size for processing
    batch_size = 64
    # Initialize an empty list to store embeddings
    embeddings = []
    # Process input IDs and attention masks in batches
    for i in tqdm(range(0, len(input_ids), batch_size)):
        #print("Progress", (i/len(input_ids))*100)
        input_ids_batch = input_ids[i:i+batch_size]
        attention_masks_batch = attention_masks[i:i+batch_size]

        with torch.no_grad():
            # Get the model's output
            output = model(input_ids=input_ids_batch, attention_mask=attention_masks_batch)

        # Retrieve the last_hidden_state from the output
        last_hidden_state = output.last_hidden_state

        # Optionally, you can obtain the pooled_output by averaging or max-pooling the last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)

        # If CUDA is available, move the embeddings back to CPU
        if torch.cuda.is_available():
            pooled_output = pooled_output.to("cpu")

        # Add the embeddings to the list
        embeddings.extend(pooled_output.numpy())

    embeddings = np.array(embeddings)
    return embeddings


# getting the list of words and preparing the train_text
def worded_list(texts):
    texts=list(texts)
    train_text = []
    all_text = []
    text = list(texts)
    for sentence in text[0:]:
        if isinstance(sentence, str):
            #print(sentence.lower())
            all_text.append(sentence.lower())
            str_arr = sentence.split()
            #print(str_arr)
            train_text.append(str_arr)
        else:
            all_text.append(str(sentence))
            train_text.append(str(sentence))
    return(all_text,train_text)

# this gives allows us to extract a proper window size -- in word embeddings, window size is width of 'neighbouring' words in which we consider contextually when extracting embeddings
def get_window_size(df, text_col):
    window_size = (df[text_col].str.count(' ') + 1).quantile(0.9)
    return(window_size)

# this function identifies the embeddings that are being called and passes them into the specified functions. 
def identify_embeddings(embedding, df, text_col):
    texts=df[text_col]
    all_text,train_text=worded_list(texts)
    window_size=get_window_size(df, text_col)
    if embedding=="cbow":
        X=get_cbow(train_text, all_text,window_size)
    elif embedding=="doc2vec":
        X=get_doc2vec(train_text, all_text,window_size)
    elif embedding=="fasttext":
        X=get_fasttext(train_text, all_text,window_size)
    elif embedding=="glove":
        X=get_glove(texts)
    elif embedding=="elmo":
        X=get_elmo(texts)
    elif embedding=="bioGPT":
        X=get_biogpt_embeddings(texts)
    elif embedding=="clinicalBERT":
        X=get_clinicalBERT(texts)
    elif embedding=="bioclinicalBERT":
        X=get_bioclinicalBERT(texts)
    else:
        X="Invalid, please choose from the following: cbow, doc2vec, fasttext, bioGPT, glove, elmo, bioclinicalBERT, clinicalBERT"
    return(X)