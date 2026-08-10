def build_paraphrase_prompt(text: str) -> str:
    return f"""
You are an expert editor.

Paraphrase the following text while preserving its meaning.

Text:
{text}

Paraphrased Version:
"""