import sys
from pathlib import Path

import streamlit as st


# Make app/ importable
APP_DIR = Path(__file__).parent / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from bootstrap import create_rag_service
from llm.client import LLMError


st.set_page_config(
    page_title="LLM Engineering Studio",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def get_rag_service():

    return create_rag_service()


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

            rag_service = get_rag_service()

            try:

                result = rag_service.answer_question(
                    question,
                    top_k=top_k
                )

            except LLMError as e:

                st.error(f"The language model request failed: {e}")
                st.stop()

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

        if "error" in evaluation:

            st.warning(evaluation["error"])

        metrics = {
            metric: details
            for metric, details in evaluation.items()
            if isinstance(details, dict)
        }

        if metrics:

            cols = st.columns(len(metrics))

            for col, (metric, details) in zip(
                cols,
                metrics.items()
            ):

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