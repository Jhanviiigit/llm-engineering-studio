from evaluation.case_loader import load_test_cases


def test_load_test_cases():

    cases = load_test_cases(
        "data/evaluation/rag_test_cases.json"
    )

    assert len(cases) == 5
    assert "question" in cases[0]
    assert "reference_answer" in cases[0]