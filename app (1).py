
import streamlit as st
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Advanced AI Memory System",

    page_icon="🧠",

    layout="wide"
)

# =====================================================
# LOAD FILES
# =====================================================

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

with open(

    "checkpoints.json",

    "r"

) as f:

    checkpoints = json.load(f)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_model()

# =====================================================
# CREATE FAISS INDEX
# =====================================================

@st.cache_resource
def create_index():

    summary_texts = [

        topic["summary"]

        for topic in topic_summaries
    ]

    embeddings = model.encode(

        summary_texts,

        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    # cosine similarity retrieval
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        np.array(embeddings)
    )

    return index

index = create_index()

# =====================================================
# RETRIEVAL FUNCTION
# =====================================================

def retrieve(query, top_k=5):

    q_emb = model.encode(

        [query],

        normalize_embeddings=True
    )

    scores, indices = index.search(

        np.array(q_emb),

        top_k
    )

    results = []

    for i, idx in enumerate(indices[0]):

        if idx < len(topic_summaries):

            topic = topic_summaries[idx]

            results.append({

                "topic_id":
                topic["topic_id"],

                "summary":
                topic["summary"],

                "start_msg":
                topic["start_msg"],

                "end_msg":
                topic["end_msg"],

                "sample_messages":
                topic["sample_messages"],

                "score":
                round(
                    float(scores[0][i]),
                    4
                )
            })

    return results

# =====================================================
# AI RESPONSE GENERATION
# =====================================================

def generate_response(query, retrieved):

    summaries = [

        item["summary"]

        for item in retrieved
    ]

    combined_context = " ".join(
        summaries
    )

    habits = ", ".join(
        persona["habits"]
    )

    traits = ", ".join(
        persona["personality_traits"]
    )

    style = persona[
        "communication_style"
    ]

    query_lower = query.lower()

    # personality questions
    if (

        "person" in query_lower

        or

        "personality" in query_lower
    ):

        return f"""
The user appears to be {traits}.

The conversations suggest interests in
career growth, hobbies, emotional discussions,
future planning, and personal experiences.

Common habits include:
{habits}.

The communication style is mostly
{style['tone']} with medium-length responses.
"""

    # habits
    elif "habit" in query_lower:

        return f"""
The user frequently discusses:
{habits}.

These habits appeared repeatedly
across multiple conversations.
"""

    # communication
    elif (

        "communication" in query_lower

        or

        "talk" in query_lower

        or

        "speak" in query_lower
    ):

        return f"""
The communication style is
{style['tone']}.

Average message length:
{round(style['average_message_length'], 2)} words.

Emoji usage count:
{style['emoji_usage']}.
"""

    # checkpoint questions
    elif "checkpoint" in query_lower:

        checkpoint_text = ""

        for cp in checkpoints[:5]:

            checkpoint_text += (

                f"\nCheckpoint {cp['checkpoint_id']}:\n"

                f"{cp['summary']}\n"
            )

        return checkpoint_text

    # generic retrieval response
    else:

        return f"""
Based on the retrieved conversations:

{combined_context}

The retrieved memories suggest recurring themes
related to the query, including goals,
personal experiences, hobbies, and conversations.
"""

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "🧠 AI Memory System"
)

st.sidebar.markdown(
    """
    ### Features

    ✅ Semantic Topic Segmentation

    ✅ Topic Checkpoints

    ✅ Persona Extraction

    ✅ FAISS Vector Retrieval

    ✅ Conversational Memory

    ✅ AI Response Generation
    """
)

st.sidebar.metric(
    "Topics",
    len(topic_summaries)
)

st.sidebar.metric(
    "Checkpoints",
    len(checkpoints)
)

st.sidebar.metric(
    "Habits",
    len(persona["habits"])
)

# =====================================================
# MAIN TITLE
# =====================================================

st.title(
    "🧠 Advanced Conversation Memory RAG"
)

st.markdown(
    """
    AI-powered conversational memory system
    using semantic retrieval and persona extraction.
    """
)

# =====================================================
# EXAMPLE QUESTIONS
# =====================================================

st.subheader(
    "💡 Example Questions"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        - What kind of person is this user?
        - What habits does the user have?
        - How does the user communicate?
        """
    )

with col2:

    st.markdown(
        """
        - career goals
        - future plans
        - relationships
        - hobbies
        """
    )

# =====================================================
# USER INPUT
# =====================================================

query = st.text_input(
    "Ask a question"
)

# =====================================================
# RESPONSE SECTION
# =====================================================

if query:

    st.divider()

    retrieved = retrieve(query)

    ai_response = generate_response(

        query,

        retrieved
    )

    st.subheader(
        "🤖 AI Response"
    )

    st.success(ai_response)

    st.subheader(
        "🔍 Retrieved Memory"
    )

    for topic in retrieved:

        with st.expander(

            f"Topic {topic['topic_id']}"

        ):

            st.write(

                f"Similarity Score: "

                f"{topic['score']}"
            )

            st.write(
                topic["summary"]
            )

            st.write(

                f"Messages: "

                f"{topic['start_msg']} "

                f"to "

                f"{topic['end_msg']}"
            )

            st.markdown(
                "### Sample Messages"
            )

            for msg in topic[
                "sample_messages"
            ]:

                st.write(
                    f"- {msg}"
                )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Built using SentenceTransformers, FAISS, Streamlit, and semantic RAG retrieval."
)
