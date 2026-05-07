
# KaStack Conversation Memory RAG System

## Overview

This project implements a conversational memory system using Retrieval-Augmented Generation (RAG).

The system:
- processes conversations chronologically,
- detects topic changes dynamically,
- creates topic checkpoints,
- generates semantic summaries,
- extracts user persona,
- performs semantic retrieval using FAISS,
- answers user queries through a Streamlit chatbot.

---

## Features

- Chronological Message Processing
- Semantic Topic Segmentation
- Topic Checkpoints
- 100 Message Checkpoints
- Persona Extraction
- FAISS Semantic Retrieval
- Streamlit Chatbot UI

---

## Architecture

Conversation Dataset
↓
Chronological Parsing
↓
SentenceTransformer Embeddings
↓
Semantic Topic Segmentation
↓
Topic Summaries
↓
100 Message Checkpoints
↓
Persona Extraction
↓
FAISS Vector Search
↓
Semantic Chatbot Retrieval
↓
Streamlit UI

---

## Topic Detection Logic

Messages are processed chronologically.

Each message embedding is compared against a rolling topic embedding using cosine similarity.

When semantic similarity falls below a threshold, a new topic checkpoint is created.

Additional improvements:
- short message filtering,
- minimum topic size enforcement,
- rolling embedding averaging.

---

## Retrieval Logic

The system uses FAISS vector search for semantic retrieval.

Steps:
1. Convert user query into embedding
2. Search nearest topic embeddings
3. Retrieve semantically relevant topic summaries
4. Return relevant conversational memory

---

## Persona Extraction

Persona extraction uses:
- keyword heuristics,
- communication patterns,
- conversational signals,
- regex-based personal fact extraction.

Extracted:
- habits,
- personality traits,
- communication style,
- personal facts.

---

## Tech Stack

- Python
- SentenceTransformers
- FAISS
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK

---

## Example Questions

- What kind of person is this user?
- What habits does the user have?
- How does the user communicate?
- career goals
- hobbies
- relationships
- future plans

---

## Run Locally

pip install -r requirements.txt

streamlit run app.py
