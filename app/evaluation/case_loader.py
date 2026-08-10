import json
from pathlib import Path


def load_test_cases(file_path: str) -> list[dict]:
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)