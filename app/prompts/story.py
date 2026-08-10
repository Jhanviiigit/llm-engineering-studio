def build_story_prompt(topic: str) -> str:
    return f"""
You are a creative storyteller.

Write a short story about:

{topic}

The story should be around 150 words.
"""