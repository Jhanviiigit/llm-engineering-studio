from pathlib import Path
from datetime import datetime


OUTPUT_DIR = Path("output")


def save_experiment(
    result,
    prompt,
    context=None,
    evaluation=None
):

    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = datetime.now().strftime(
        "experiment_%Y%m%d_%H%M%S.txt"
    )

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("LLM Engineering Studio\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Prompt:\n{prompt}\n\n")

        file.write(f"Model: {result['model']}\n")
        file.write(f"Temperature: {result['temperature']}\n")
        file.write(f"Latency: {result['latency']:.2f} seconds\n")
        file.write(f"Prompt Tokens: {result['prompt_tokens']}\n")
        file.write(f"Completion Tokens: {result['completion_tokens']}\n")
        file.write(f"Total Tokens: {result['total_tokens']}\n\n")

        if context:

            file.write("Retrieved Context\n")
            file.write("-" * 60 + "\n")

            for i, chunk in enumerate(context, start=1):
                file.write(f"[{i}] {chunk}\n\n")

        file.write("Response\n")
        file.write("-" * 60 + "\n")
        file.write(result["response"] + "\n\n")

        if evaluation:

            file.write("Evaluation\n")
            file.write("-" * 60 + "\n")

            for metric, details in evaluation.items():

                if isinstance(details, dict):

                    file.write(
                        f"{metric.capitalize()}: "
                        f"{details['score']:.2f}\n"
                    )

                    file.write(
                        f"Reason: {details['reason']}\n\n"
                    )

    return filepath