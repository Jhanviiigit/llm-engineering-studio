def parse_response(response):
    """
    Convert responses from different models into a common format.
    """

    message = response.choices[0].message

    text = message.content

    if text is None:
        text = "[No response text returned.]"

    usage = response.usage

    return {
        "response": text,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }