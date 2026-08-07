def display_menu():
    options = [
        "Translate",
        "Paraphrase",
        "Summarize",
        "Sentiment Analysis",
        "Exit"
    ]

    print("\n" + "=" * 40)
    print("LLM Engineering Studio")
    print("=" * 40)

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    return options


def get_user_choice(max_option):
    while True:
        choice = input("\nSelect an option: ")

        if choice.isdigit():
            choice = int(choice)

            if 1 <= choice <= max_option:
                return choice

        print("❌ Invalid choice. Please enter a valid option.")


def handle_choice(choice, options):
    selected_option = options[choice - 1]

    if selected_option == "Exit":
        print("\n👋 Thank you for using LLM Engineering Studio!")
        return False

    print(f"\n✅ You selected: {selected_option}")
    print("🚧 This feature is under development.")

    return True


def main():
    running = True

    while running:
        options = display_menu()
        choice = get_user_choice(len(options))
        running = handle_choice(choice, options)


if __name__ == "__main__":
    main()