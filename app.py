import os
import tarfile

import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle

# Compute Cosine Similarity
def cosine_similarity(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    dot_product = np.dot(x_arr, y_arr)
    norm_x = la.norm(x_arr)
    norm_y = la.norm(y_arr)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

def format_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embeddings_dict[word] = embedding
    return embeddings_dict


# glove_file_path = 'C:/Users/Lenovo/Desktop/assignment_01/glove.6B.50d.txt'
glove_file_path = 'glove.6B.50d.txt'
with tarfile.open('glove.tar.gz') as tar:
    tar.extractall('.')


formatted_embeddings = format_glove_embeddings(glove_file_path)


def save_embeddings_to_pickle(embeddings_dict, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

# Example usage:
# pickle_path = 'C:/Users/Lenovo/Desktop/assignment_01/embeddings.pkl'
pickle_path = 'embeddings.pkl'
save_embeddings_to_pickle(formatted_embeddings, pickle_path)


# def load_glove_embeddings(glove_path='C:/Users/Lenovo/Desktop/assignment_01/embeddings.pkl'):
def load_glove_embeddings(glove_path='embeddings.pkl'):
    with open(glove_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict

# Get Averaged Glove Embedding of a sentence
def averaged_glove_embeddings(sentence, embeddings_dict):
    words = sentence.split(" ")
    glove_embedding = np.zeros(50)
    count_words = 0

    for word in words:
        if word.lower() in embeddings_dict:
            glove_embedding += embeddings_dict[word.lower()]
            count_words += 1

    return glove_embedding / max(count_words, 1)

# Load glove embeddings
glove_embeddings = load_glove_embeddings()

# Gold standard words to search from
gold_words = ["flower", "mountain", "tree", "car", "building"]

# Text Search
st.title("Search Based Retrieval Demo")
st.subheader("Pass in an input word or even a sentence (e.g. jasmine or mount adams)")
text_search = st.text_input("", value="")

# Find closest word to an input word
if text_search:
    input_embedding = averaged_glove_embeddings(text_search, glove_embeddings)
    cosine_sim = {}
    for index in range(len(gold_words)):
        cosine_sim[index] = cosine_similarity(input_embedding, glove_embeddings[gold_words[index]])

    # Sort the cosine similarities
    sorted_cosine_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)

    st.write("(My search uses glove embeddings)")
    st.write("Closest word I have between flower, mountain, tree, car and building for your input is: ")
    st.subheader(gold_words[sorted_cosine_sim[0][0]])
    st.write("")

