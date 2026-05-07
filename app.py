
import streamlit as st
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(

    page_title="AI Conversation Memory System",

    layout="wide"
)

# -----------------------------------
# LOAD FILES
# -----------------------------------

with open(

    "topic_summaries.json",

    "r"

) as f:

    topic_summaries = json.load(f)

with open(

    "persona.json",

    "r"

) as f:

    persona = json.load(f)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# -----------------------------------
# CREATE FAISS INDEX
# -----------------------------------

summary_texts = [

    topic["summary"]

    for topic in topic_summaries
]

topic_embeddings = model.encode(
    summary_texts
)

dimension = topic_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(topic_embeddings)
)

# -----------------------------------
# RETRIEVAL FUNCTION
# -----------------------------------

def retrieve(query, top_k=5):

    q_emb = model.encode([query])

    distances, indices = index.search(

        np.array(q_emb),

        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(topic_summaries):

            results.append(

                topic_summaries[idx]
            )

    return results

# -----------------------------------
# CHATBOT
# -----------------------------------

def chatbot(query):

    query_lower = query.lower()

    # persona

    if "person" in query_lower \
    or "personality" in query_lower:

        return {

            "type": "persona",

            "data": persona
        }

    # habits

    elif "habit" in query_lower:

        return {

            "type": "habits",

            "data": persona["habits"]
        }

    # communication

    elif "talk" in query_lower \
    or "communication" in query_lower:

        return {

            "type": "communication",

            "data":
            persona["communication_style"]
        }

    # semantic retrieval

    else:

        retrieved = retrieve(query)

        return {

            "type": "retrieval",

            "data": retrieved
        }

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title(
    "AI Memory System"
)

st.sidebar.markdown(
    """
    ### Features

    - Semantic Topic Segmentation
    - Topic Checkpoints
    - 100 Message Summaries
    - Persona Extraction
    - FAISS Semantic Retrieval
    - Conversational Memory
    """
)

st.sidebar.metric(
    "Total Topics",
    len(topic_summaries)
)

# -----------------------------------
# MAIN TITLE
# -----------------------------------

st.title(
    "AI Conversation Memory System"
)

st.markdown(
    """
    Semantic conversational memory system
    with RAG retrieval and persona extraction.
    """
)

# -----------------------------------
# EXAMPLE QUESTIONS
# -----------------------------------

st.subheader(
    "Example Questions"
)

st.markdown(
    """
    - What kind of person is this user?
    - What habits does the user have?
    - How does the user communicate?
    - career goals
    - relationships
    - hobbies
    - future plans
    """
)

# -----------------------------------
# INPUT
# -----------------------------------

query = st.text_input(
    "Ask a question"
)

# -----------------------------------
# ANSWER
# -----------------------------------

if query:

    result = chatbot(query)

    st.subheader("Response")

    # persona

    if result["type"] == "persona":

        st.json(result["data"])

    # habits

    elif result["type"] == "habits":

        st.write(result["data"])

    # communication

    elif result["type"] == "communication":

        st.json(
            result["data"]
        )

    # retrieval

    elif result["type"] == "retrieval":

        retrieved = result["data"]

        for topic in retrieved:

            with st.expander(

                f"Topic {topic['topic_id']}"

            ):

                st.write(

                    topic["summary"]
                )

                st.write(

                    f"Messages: "

                    f"{topic['start_msg']} "

                    f"to "

                    f"{topic['end_msg']}"
                )
