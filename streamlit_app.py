import os

import httpx
import streamlit as st
from dotenv import load_dotenv


# The UI is a thin client: all retrieval, generation and evaluation
# happens in the API service (app/api/main.py). Start the API first:
#
#     uvicorn api.main:app --app-dir app
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Answering with evaluation makes two LLM calls, which can be slow
REQUEST_TIMEOUT_SECONDS = 120


st.set_page_config(
    page_title="LLM Engineering Studio",
    page_icon="🤖",
    layout="wide"
)


def api_error_message(error: Exception) -> str:

    if isinstance(error, httpx.ConnectError):
        return (
            f"Cannot reach the API at {API_URL}. "
            "Start it with: uvicorn api.main:app --app-dir app"
        )

    if isinstance(error, httpx.HTTPStatusError):
        try:
            detail = error.response.json().get("detail")
        except ValueError:
            detail = error.response.text

        return f"API error {error.response.status_code}: {detail}"

    return f"Request failed: {error}"


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

evaluate = st.sidebar.checkbox(
    "Evaluate answer (LLM judge)",
    value=True,
    help="Makes a second LLM call to score the answer."
)

st.sidebar.header("Documents")

uploaded_file = st.sidebar.file_uploader(
    "Add a .txt document",
    type=["txt"]
)

if uploaded_file is not None and st.sidebar.button("Upload"):

    try:

        response = httpx.post(
            f"{API_URL}/documents",
            files={
                "file": (uploaded_file.name, uploaded_file.getvalue())
            },
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()

        body = response.json()

        st.sidebar.success(
            f"Added {body['chunks_added']} chunks "
            f"({body['total_chunks']} total)."
        )

    except httpx.HTTPError as e:

        st.sidebar.error(api_error_message(e))

question = st.text_input(
    "Ask a question about the documents"
)

if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Generating answer..."):

        try:

            response = httpx.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                    "evaluate": evaluate
                },
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()

        except httpx.HTTPError as e:

            st.error(api_error_message(e))
            st.stop()

    result = response.json()

    st.subheader("Answer")

    st.write(result["answer"])

    st.subheader("Retrieved Context")

    for i, context in enumerate(
        result["retrieved_context"],
        start=1
    ):

        with st.expander(f"Chunk {i}"):

            st.write(context)

    if evaluate:

        st.subheader("Evaluation")

        if result["evaluation_error"]:

            st.warning(result["evaluation_error"])

        metrics = result["evaluation"] or {}

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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latency",
        f"{result['latency_seconds']:.2f}s"
    )

    col2.metric(
        "Prompt Tokens",
        result["prompt_tokens"]
    )

    col3.metric(
        "Total Tokens",
        result["total_tokens"]
    )

    col4.metric(
        "Model",
        result["model"].split("/")[-1]
    )
