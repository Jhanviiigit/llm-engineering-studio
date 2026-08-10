def build_translation_prompt(text: str, language: str) -> str:
    return f"""
You are a professional translator.

Translate the following text into {language}.

Rules:
- Preserve the meaning.
- Do not add explanations.
- Return only the translated text.

Text:
{text}
"""