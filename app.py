
import streamlit as st
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Conversation Memory System",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# LOAD FILES
# =====================================================

with open("topic_summaries.json", "r") as f:
    topic_summaries = json.load(f)

with open("persona.json", "r") as f:
    persona = json.load(f)

with open("checkpoints.json", "r") as f:
    checkpoints = json.load(f)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

model = load_model()

# =====================================================
# CREATE FAISS INDEX
# =====================================================

@st.cache_resource
def create_faiss_index():

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

    return index

index = create_faiss_index()

# =====================================================
# RETRIEVAL FUNCTION
# =====================================================

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

# =====================================================
# CHATBOT
# =====================================================

def chatbot(query):

    query_lower = query.lower()

    # personality

    if (

        "person" in query_lower

        or

        "personality" in query_lower
    ):

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

    elif (

        "talk" in query_lower

        or

        "communication" in query_lower

        or

        "speak" in query_lower
    ):

        return {

            "type": "communication",

            "data":
            persona["communication_style"]
        }

    # checkpoints

    elif "checkpoint" in query_lower:

        return {

            "type": "checkpoints",

            "data": checkpoints[:5]
        }

    # semantic retrieval

    else:

        retrieved = retrieve(query)

        return {

            "type": "retrieval",

            "data": retrieved
        }

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

    ✅ 100 Message Summaries

    ✅ Persona Extraction

    ✅ FAISS Semantic Retrieval

    ✅ Conversational Memory
    """
)

st.sidebar.metric(
    "Total Topics",
    len(topic_summaries)
)

st.sidebar.metric(
    "Total Checkpoints",
    len(checkpoints)
)

st.sidebar.metric(
    "Detected Habits",
    len(persona["habits"])
)

# =====================================================
# MAIN TITLE
# =====================================================

st.title(
    "🧠 AI Conversation Memory System"
)

st.markdown(
    """
    Semantic conversational memory system
    using RAG and FAISS retrieval.
    """
)

# =====================================================
# EXAMPLE QUESTIONS
# =====================================================

st.subheader(
    "💡 Example Questions"
)

st.markdown(
    """
    - What kind of person is this user?
    - What habits does the user have?
    - How does the user communicate?
    - career goals
    - future plans
    - hobbies
    - relationships
    """
)

# =====================================================
# INPUT
# =====================================================

query = st.text_input(
    "Ask a question"
)

# =====================================================
# RESPONSE
# =====================================================

if query:

    result = chatbot(query)

    st.divider()

    st.subheader("📌 Response")

    # =================================================
    # PERSONA
    # =================================================

    if result["type"] == "persona":

        p = result["data"]

        st.markdown(
            "## 👤 Personality Overview"
        )

        traits = ", ".join(
            p["personality_traits"]
        )

        st.write(

            f"The user appears "

            f"{traits}."
        )

        habits = ", ".join(
            p["habits"]
        )

        st.write(

            f"They enjoy "

            f"{habits}."
        )

        style = p["communication_style"]

        st.write(

            f"Their communication "

            f"style is "

            f"{style['tone']} "

            f"with an average "

            f"message length of "

            f"{round(style['average_message_length'], 2)} words."
        )

        st.markdown(
            "### 📍 Personal Facts"
        )

        unique_facts = list(
            set(
                p["personal_facts"]
            )
        )

        for fact in unique_facts[:10]:

            if len(fact.strip()) > 4:

                st.write(
                    f"- {fact}"
                )

    # =================================================
    # HABITS
    # =================================================

    elif result["type"] == "habits":

        st.markdown(
            "## 🧩 Habits"
        )

        for habit in result["data"]:

            st.write(
                f"✅ {habit}"
            )

    # =================================================
    # COMMUNICATION
    # =================================================

    elif result["type"] == "communication":

        style = result["data"]

        st.markdown(
            "## 💬 Communication Style"
        )

        st.write(
            f"Tone: {style['tone']}"
        )

        st.write(

            f"Average message length: "

            f"{round(style['average_message_length'], 2)} words"
        )

        st.write(
            f"Emoji usage count: {style['emoji_usage']}"
        )

    # =================================================
    # CHECKPOINTS
    # =================================================

    elif result["type"] == "checkpoints":

        st.markdown(
            "## 📚 Checkpoints"
        )

        for cp in result["data"]:

            with st.expander(

                f"Checkpoint {cp['checkpoint_id']}"

            ):

                st.write(

                    f"Messages: "

                    f"{cp['start_msg']} "

                    f"to "

                    f"{cp['end_msg']}"
                )

                st.write(
                    cp["summary"]
                )

    # =================================================
    # RETRIEVAL
    # =================================================

    elif result["type"] == "retrieval":

        retrieved = result["data"]

        st.markdown(
            "## 🔍 Retrieved Topics"
        )

        for topic in retrieved:

            with st.expander(

                f"Topic {topic['topic_id']}"

            ):

                st.write(

                    f"📌 Summary: "

                    f"{topic['summary']}"
                )

                st.write(

                    f"📨 Messages: "

                    f"{topic['start_msg']} "

                    f"to "

                    f"{topic['end_msg']}"
                )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Built using SentenceTransformers, FAISS, Streamlit, and semantic RAG retrieval."
)
