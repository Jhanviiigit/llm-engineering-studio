from app.prompts.rag import build_rag_prompt


def test_build_rag_prompt():
    question = "What is RAG?"

    context = [
        "RAG combines retrieval with language models.",
        "RAG provides retrieved information as context."
    ]

    prompt = build_rag_prompt(question, context)

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with language models." in prompt
    assert "RAG provides retrieved information as context." in prompt
    assert "Use only the information contained in the context" in prompt