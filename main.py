from agent import get_response

def main():
    print("Welcome to the Emotion-Aware Chatbot!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response, "\n")

if __name__ == '__main__':
    main()
