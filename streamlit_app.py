import sys
from pathlib import Path

import streamlit as st


# Make app/ importable
APP_DIR = Path(__file__).parent / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.indexer import Indexer
from rag.retriever import Retriever
from services.rag_service import RAGService


st.set_page_config(
    page_title="LLM Engineering Studio",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def create_rag_service():

    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    indexer = Indexer(
        embedding_model,
        vector_store
    )

    indexer.index("data/documents/sample.txt")

    retriever = Retriever(
        embedding_model,
        vector_store
    )

    return RAGService(retriever)


st.title("LLM Engineering Studio")

st.write(
    "Experiment with Retrieval-Augmented Generation "
    "(RAG) and inspect retrieved context and evaluation metrics."
)

st.sidebar.header("RAG Settings")

top_k = st.sidebar.slider(
    "Number of retrieved chunks (top_k)",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.write(
    f"Retrieving the top {top_k} relevant chunks."
)

question = st.text_input(
    "Ask a question about the documents"
)

if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            rag_service = create_rag_service()

            result = rag_service.answer_question(
                question,
                top_k=top_k
            )

        st.subheader("Answer")

        st.write(result["response"])

        st.subheader("Retrieved Context")

        for i, context in enumerate(
            result["retrieved_context"],
            start=1
        ):

            with st.expander(f"Chunk {i}"):

                st.write(context)

        st.subheader("Evaluation")

        evaluation = result.get("evaluation", {})

        if evaluation:

            cols = st.columns(len(evaluation))

            for col, (metric, details) in zip(
                cols,
                evaluation.items()
            ):

                if isinstance(details, dict):

                    col.metric(
                        metric.capitalize(),
                        f"{details['score']:.2f}"
                    )

                    with col.expander("Reason"):

                        st.write(details["reason"])

        st.subheader("Request Metadata")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latency",
            f"{result['latency']:.2f}s"
        )

        col2.metric(
            "Prompt Tokens",
            result["prompt_tokens"]
        )

        col3.metric(
            "Total Tokens",
            result["total_tokens"]
        )