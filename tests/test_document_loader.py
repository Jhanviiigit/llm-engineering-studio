from app.rag.document_loader import load_text_file


def test_load_text_file():
    content = load_text_file("data/documents/sample.txt")

    assert "Retrieval-Augmented Generation" in content